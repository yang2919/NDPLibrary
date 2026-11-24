# Multi-device execution and tensor parallelism

Each Session owns one backend connection, allocation namespace and execution
queue. DeviceGroup assigns an ordinal to each Session. Engine.PIM and
Engine.PNM select compute units within that device. The same submitting
Python thread can enqueue work on every device.

## Layer placement

LlamaDecodeStack accepts a device list and one ordinal per source layer.
Weights and per-request KV history stay with their layer. Adjacent layers
on the same device pass resident tensors. A device boundary uses a blocking
dense host-staged copy. The model boundary accepts and returns CPU FP16
[1,1,hidden_size] tensors. On a failed token, reset_history clears the entire
stack before a new sequence is submitted.

~~~python
stack = ndp.nn.LlamaDecodeStack(
    model.model.layers[:2],
    devices=[{"backend": "gem5_pim", "worker": worker} for _ in range(2)],
    placement=[0, 1], max_tokens=128,
)
y = stack(token)
~~~

This implements sequential layer placement for one sequence. Microbatch
scheduling and overlapping requests remain future work.

## Output and input channel splitting

For PyTorch weight W[N,K], split="output" partitions N. Each device receives
all K input features and computes a disjoint set of output features.
split="input" partitions K. Each device reads only its input features and
computes an FP32 partial of all N outputs.

~~~python
devices = ndp.DeviceGroup([{"backend": "gem5_pim", "worker": worker} for _ in range(2)])
gate = ndp.nn.TensorParallelLinear(source.gate_proj, devices=devices, split="output")
down = ndp.nn.TensorParallelLinear(source.down_proj, devices=devices, split="input")

gate_shards = gate.forward_distributed(token)
partials = down.forward_distributed(gate_shards)
output = devices.reduce_partials(partials.tensors, destination=0)
replicas = partials.replicated()
~~~

This example illustrates projection composition. A complete SwiGLU additionally
applies SiLU and multiplies the corresponding up-projection output before down.

Both modules accept CPU FP16 vectors of rank 1, 2 or 3, with one vector per
call. forward returns the complete CPU output. forward_distributed accepts
CPU input or a DistributedRow and preserves resident results. A resident
input consumes existing device tensors and submits no activation upload.
For input splitting, matching feature ranges pass directly into local GEMVs;
replicated input is sliced on each worker. For output splitting, sharded input
is first all-gathered. Bias-free evaluation-mode Linear is required.

The split uses contiguous ranges. Extra elements go to lower ordinals.
Every device must receive at least one feature. Each worker keeps only its
local weight shard; the Python module keeps a full CPU checkpoint weight.
state_dict uses the ordinary Linear weight key. load_state_dict reconstructs
the resident shards. After direct weight modification, refresh is required.

## Tensor-parallel SwiGLU and GQA attention

```python
devices = ndp.DeviceGroup([{"backend": "gem5_pim", "worker": worker} for _ in range(2)])
layer = model.model.layers[0]  # FP16, eval-mode Hugging Face Llama layer
attention = ndp.nn.TensorParallelLlamaAttention(
    layer.self_attn, devices=devices, max_tokens=128)
mlp = ndp.nn.TensorParallelSwiGLU(layer.mlp, devices=devices)
with torch.inference_mode():
    attention_output = attention(normalized_token)
    mlp_output = mlp(normalized_mlp_input)
attention.reset_history()  # Start a different sequence.
```

The caller supplies the normalized attention and MLP inputs. These modules do
not include RMSNorm or residual addition. Both accept one CPU FP16 token
([K], [1,K], or [1,1,K]) and return the same leading shape. No manual upload,
download or Session calls are needed for this interface. Execution requires
the native gem5_pim worker with FP32 partial support.

| Stage | Device-local work | Communication |
|---|---|---|
| SwiGLU gate/up | Matching output-channel shards, SiLU(gate) * up | One input replica per device, shared by both projections |
| SwiGLU down | Input-sharded projection of local products | FP32 output partials reduced on host |
| Attention Q/K/V | Whole KV groups and their associated Q heads | One input replica per device, shared by Q/K/V |
| RoPE and attention | Local split-half RoPE, resident KV append and causal attention | No KV exchange or context all-gather |
| Attention output | Input-sharded o_proj of local head outputs | FP32 output partials reduced on host |

Attention partitions KV heads into contiguous groups; extra KV heads go to
lower device ordinals. The associated Q heads stay on the same device.
SmolLM2's 9 Q heads and 3 KV heads become Q=(6,3), KV=(2,1) on two devices.
The number of devices cannot exceed the number of KV heads. Linear's optional
ranges argument defines these matching feature intervals explicitly.
SwiGLU distributes intermediate channels evenly, including uneven dimensions.

Attention supports one sequence, one token per call, full-head unscaled
split-half RoPE, bias-free Llama projections, and a fixed max_tokens capacity.
Every device retains only its own KV heads. A failed token requires
reset_history() before continuing; reset discards the entire sequence.
Loading an attention state dict also resets history. The projection keys
match the source module (q_proj.weight, gate_proj.weight, etc.). Direct weight
edits require refresh() on the affected projection and attention history reset.

For composition with other resident operations, forward_distributed accepts
a DistributedRow and returns FP32 partial rows. A replicated input requires
no activation transfers during computation. Sharded inputs are first gathered.
The caller must reduce the result before consuming it as a completed activation.
In the attention distributed API, local work finishes and history advances
before the caller performs reduction; a subsequent reduction failure requires
reset_history() as well. Ordinary forward handles that failure automatically
by rejecting subsequent tokens until reset.

The two sublayers can also be composed through TensorParallelLlamaDecodeBlock
below. Multi-sequence scheduling and device-side collectives remain future work.
The explicit whole-model adapter is described below.

## DistributedRow contract

DistributedRow contains the DeviceGroup, global row width, ordered local
tensors, layout and feature ranges. Construction checks each tensor's owner,
shape and dtype against the metadata. The row dimension is 1 for decode and T for prefill chunks; all local tensors
must have the same positive row count. The partition axis is the last feature axis.

| Layout | Local tensor | Range metadata | Meaning |
|---|---|---|---|
| sharded | FP16 [T,range length] | Consecutive, disjoint ranges covering [0,width) | Completed output features |
| replicated | FP16 [T,width] | Empty | Complete row on every device |
| partial | FP32 [T,width] | Empty | Unreduced contributions to each output feature |

A partial cannot be used as a Linear input. Reduce it first.
replicated() gathers shards or reduces partials and copies the complete FP16
row to all devices. DeviceGroup.all_gather_rows remains available for ordered
FP16 row handles. Replica equality is a producer contract; metadata validation
does not read tensor contents to compare replicas.

## FP32 partials and reduction

The native gem5_pim worker advertises feature bit 2 for FP32 partials.
L2 linear_partial takes resident FP16 input and returns an immutable dense FP32
Tensor. Its default PIM engine consumes prepared single-token weight shards;
Engine.PNM supports dense multi-row weight shards for prefill. The worker allocates role P using its FP32
accumulator representation and skips the usual per-projection FP16 rounding.
U and D carry four little-endian bytes per element for role P. Shape and size
metadata distinguish these tensors from normal FP16 activations.

DeviceGroup.reduce_partials validates one matching FP32 tensor per device,
downloads them in ascending ordinal order and accumulates in FP32 on the host.
It rounds once to FP16, uploads the result to the selected destination and
returns only after completion. Partial tensors remain valid after reduction.
The implementation uses whole-tensor host staging and sequential transfers.
Session staging and destination allocation limits still apply.

L2 supports FP32 dense storage and copies on the partial-capable native
gem5_pim backend. Generic elementwise kernels, mutable DeviceBuffer operations
and packed layouts continue to require FP16. The ELF capture path rejects
FP32 partial launches until its GEMV ABI and target executor support FP32
outputs. Rebuild the native worker when upgrading this protocol.

Reordering the dot product into multiple FP32 partials can change the final
rounding. Input-split results should be compared with explicit tolerances;
output splitting preserves complete per-channel dot products.

## Verification

Build the native worker and Python adapter from the same source revision.

~~~sh
make -C tools/gem5/pim_kernel fp16
cmake --build build-python
ctest --test-dir build-python --output-on-failure
PYTHONPATH=build-python/python python examples/python/llama_stages.py /path/to/SmolLM2-135M \
    --worker build/pim-kernel-worker-fp16
PYTHONPATH=build-python/python python examples/python/llama_tensor_parallel.py /path/to/SmolLM2-135M \
    --worker build/pim-kernel-worker-fp16
~~~

Regression tests cover uneven partitions, ownership and metadata rejection,
checkpoint replacement, resident projection chaining, FP32 inter-device
copy, and cancellation of +80,000 and -80,000 partials before FP16 conversion.
The checkpoint example compares four real gate and down inputs with an
unsharded worker and verifies zero PyTorch transfers when submitting the
resident gate output directly into the input-split down projection.

With the pinned SmolLM2-135M checkpoint, the first layer's output-split gate
used two 768-channel shards of a 576-input projection. Its gathered output
matched the unsharded worker exactly. Four input-split down projections,
using 768 of 1,536 input features per worker and 576 FP32 partial outputs,
had a maximum absolute difference of 4.76837158203125e-7 from the unsharded
worker. The acceptance tolerance was atol=0.02, rtol=0.02. The resident
gate-to-down submission recorded zero ndp::upload and ndp::download events;
the subsequent host reduction necessarily transfers the FP32 partials.

### SwiGLU and GQA checkpoint verification

```sh
PYTHONPATH=build-python/python python examples/python/llama_tensor_parallel_blocks.py \
    /path/to/SmolLM2-135M --worker build/pim-kernel-worker-fp16
```

The example compares four first-layer tokens with Hugging Face eager attention
and SwiGLU, including causal history and split-half RoPE. On the local
SmolLM2-135M checkpoint, maximum absolute differences were 3.05175781e-5
for attention and 9.765625e-4 for SwiGLU (atol=0.02, rtol=0.02).
Resetting history reproduced all four attention outputs exactly.

A relay inspects actual native-worker U/D frames. With input replicas already
resident, both sublayers emitted zero intermediate activation frames. Final
reduction downloaded 2,304 bytes of FP32 partials from each worker, uploaded
1,152 bytes of FP16 output to device 0, and downloaded 1,152 bytes for the CPU
result. These counts exclude initial input, weight and RoPE uploads. They do
not measure simulated memory traffic inside a device or execution performance.
Regression tests also cover uneven MLP shards, independent GQA reference
results, cache capacity, checkpoint reset and recovery after an injected failure.
These checks use native numerical workers; extended gem5 FS was not run.

## Complete tensor-parallel decoder block

```python
block = ndp.nn.TensorParallelLlamaDecodeBlock(
    model.model.layers[0], devices=devices, max_tokens=128)
with torch.inference_mode():
    output = block(token)  # CPU FP16 [1,1,hidden_size]
block.reset_history()
```

The block includes both RMSNorms and residual additions. Device 0 owns the
complete residual row and runs these operations using resident PNM primitives.
Each normalized row is copied to the other devices for the following sublayer.
Attention and SwiGLU run with the partitions described above. Each output
projection produces FP32 partials, which the host reduces and uploads to
device 0 before residual addition. Ordinary forward uploads the input only to
device 0 and downloads the final result only from device 0.

`forward_distributed(token)` returns a completed replicated DistributedRow.
Another block in the same DeviceGroup can consume it directly. This avoids
an intermediate CPU PyTorch tensor; transfers for replication and reduction
still use host staging. The distributed output copies the final row to every
device; ordinary forward avoids that extra replication. Sharded completed
inputs are also accepted and gathered; unreduced partial inputs are rejected.

Checkpoint keys match the source Llama layer. Loading a checkpoint resets
history. If any stage fails after attention may have appended KV data, the
entire block rejects new tokens until reset_history(). Reset all blocks in a
manually composed sequence after a downstream failure. This API remains
single-sequence decode, with one [1,1,hidden_size] CPU token per call.

The checkpoint example above now also runs the complete first decoder block
for four SmolLM2 tokens. Its maximum absolute difference from Hugging Face was
0.03125, passing the elementwise criterion `abs(actual-reference) <=
0.02 + 0.02*abs(reference)`. This is a combined absolute/relative tolerance,
not a claim that every absolute error is below 0.02. History reset reproduced
the four outputs exactly. Native regression checks compare against the
single-device block and cover checkpoint reload, resident composition, and
an injected MLP failure after attention history has advanced.

For width 576, the relay verified these activation frames for each complete
CPU-in/CPU-out token (initial weights and RoPE tables excluded):

| Device | U payload bytes, in order | D payload bytes, in order |
|---|---|---|
| 0 | 1152 input, 1152 attention reduction, 1152 MLP reduction | 1152 normalized attention input, 2304 attention partial, 1152 normalized MLP input, 2304 MLP partial, 1152 final output |
| 1 | 1152 normalized attention input, 1152 normalized MLP input | 2304 attention partial, 2304 MLP partial |

These are native worker protocol checks. Extended gem5 FS and performance
validation remain external verification tasks.

## Multiple layers and complete causal language model

`TensorParallelLlamaDecodeStack` wraps consecutive Llama layers in one
DeviceGroup. With the devices argument, all blocks share device 0 as their residual owner. The output
of a block is passed directly to the next block's resident input without
replicating the residual or creating an intermediate CPU tensor. Each block
still replicates its two normalized rows and performs two host reductions.
The stack uploads/downloads only at its outer CPU boundary.

```python
stack = ndp.nn.TensorParallelLlamaDecodeStack(
    model.model.layers, devices=devices, max_tokens=128)
with torch.inference_mode():
    hidden = stack(embedded_token)  # CPU FP16 [1,1,hidden_size]
stack.reset_history()
```

`forward_ndp` on the block or stack accepts and returns an FP16
[1,hidden_size] NdpTensor owned by device 0. Stack reset clears every layer's
KV history and token position. A failure in any layer prevents reuse of the
whole stack until reset, including layers that completed that token earlier.

For token IDs and generation, use the full model adapter:

```python
# source is a CPU FP16, eval-mode Hugging Face LlamaForCausalLM.
model = ndp.nn.TensorParallelLlamaForCausalLM(
    source, devices=[{"backend": "gem5_pim", "worker": worker} for _ in range(2)],
    max_tokens=128)
ids = torch.tensor([[1, 42, 30, 198]], dtype=torch.int64)
output_ids = model.generate(ids, max_new_tokens=8)

model.reset_history()
logits = model(ids[:, :1])  # Append one token; CPU FP32 [1,1,vocab_size].
```

| Component | Execution location |
|---|---|
| Token embedding | Host PyTorch, FP16 |
| All decoder layers | TP device group; root-device RMSNorm and residuals |
| FP32 partial reductions and device-to-device copies | Host staging |
| Final RMSNorm | Device 0 |
| LM head | Host PyTorch, FP16 computation, returned as FP32 logits |
| Greedy argmax and sequence assembly | Host PyTorch |

The host embedding and LM head are deliberate initial placement choices.
The decoder and final norm use the existing device kernels. This adapter does
not claim that all model computation is offloaded, nor does converting logits
to FP32 restore precision lost during FP16 computation.

`forward` accepts only one CPU int64 token [1,1] and advances internal history.
`generate` resets history, processes a nonempty [1,prompt_length] prompt one
token at a time, then generates greedily. It returns prompt plus new IDs.
It does not use a batched prefill kernel, sampling, beam search, padding,
attention masks, Hugging Face GenerationMixin or external past_key_values.
Use new_request() for independent sequences sharing one adapter, as described
below. Calls without a request use one default history. Concurrent calls on
one instance are unsupported.

`max_tokens` bounds tokens actually executed. To request N new tokens, reserve
prompt_length + N - 1 entries. The final emitted token is not yet in KV history;
pass it to forward if continuing manually. `eos_token_id=None` uses the source
model EOS setting; an empty list disables EOS stopping. An integer or list
selects explicit stop tokens. `max_new_tokens=0` returns the prompt and resets
history without processing it. Generation rejects a request whose worst-case
length exceeds capacity even if EOS could stop it sooner.

Source checkpoint keys are preserved, including model.layers, model.norm,
model.embed_tokens and lm_head. `load_state_dict(source.state_dict())` reloads
resident weights and resets history. Embedding and LM head are independent
frozen CPU copies; the adapter does not preserve shared-parameter identity of
tied source weights. Reload both from a coherent source state dict after changes.
Failures in final normalization or the host LM head also invalidate the stack.

### Whole-model verification

```sh
PYTHONPATH=build-python/python python examples/python/llama_tensor_parallel_model.py \
    /path/to/SmolLM2-135M --worker build/pim-kernel-worker-fp16 --steps 8
```

The example compares every layer and logits on a common token trajectory,
then compares independent greedy generation and exact reset reproducibility.
It reports numeric drift separately from greedy-token agreement. Debug layer
captures intentionally download every layer's output during verification;
normal execution has no such downloads. The ordinary inference API is checked
separately by the native regression tests.

See [whole-model validation results](tensor-parallel-model-validation.md) for
measured precision and the limits of this checkpoint smoke test.


### Llama normalization precision

The Llama adapter now performs `FP16(normalize_fp32(x)) * weight`, with the
multiplication result stored as FP16. It composes L2 RMSNorm with a resident
unit weight and a separate resident multiply, matching the Hugging Face cast
boundary. Both the single-device and TP adapters, including final norm, use
this composition. Unit and gamma weights are prepared once and rebuilt on
checkpoint load; this adds one local multiply per norm, without per-token
host transfers. The general L2 RMSNorm kernel retains its fused semantics.

Run `llama_tensor_parallel_model.py --devices 1` and `--devices 2` with the
same checkpoint and steps to compare unsplit and TP execution. The separate
`llama_norm_precision.py` example holds all norm inputs equal to the reference,
so downstream accumulation cannot conceal the local rounding difference.
Updated measurements are recorded in the full-model validation document.


## Shared staging for broadcasts

`DeviceGroup.broadcast(tensor)` returns a tensor on every device in ordinal
order. The source may belong to any device in the group. Its original
allocation is reused; all other devices receive immutable dense copies.
The source is downloaded once, and that host buffer is reused for each
sequential upload. The call waits for all copies. FP16 and supported FP32
dense tensors are accepted. A one-device group performs no transfer.

The following paths use broadcast automatically:

- Decoder normalization outputs replicated for attention and SwiGLU.
- DistributedRow replication after FP32 reduction.
- The completed row replicated by all_gather_rows.
- Explicit replicated block outputs.

Previously, copying the same source to D-1 destinations downloaded it D-1
times. Broadcast reduces that to one download. The number of uploads is
unchanged. Two-device communication volume is unchanged; savings begin at
three devices. No device-side collective or direct peer access is assumed.

For L decoder layers, hidden width H, D >= 2 devices, and S=2H FP16 bytes,
a complete CPU-token-in/logits-out model step has the following activation
payload volume (excluding preparation and packet headers):

| Path | Previous repeated-copy implementation | Shared staging |
|---|---|---|
| Root uploads | (2L+1)S | (2L+1)S |
| Root downloads | (2L(D+1)+1)S | (6L+1)S |
| Each other device uploads | 2LS | 2LS |
| Each other device downloads | 4LS | 4LS |

The two downloads of FP32 projection partials per layer remain necessary for
the existing host reduction. The optimization saves 2L(D-2)S bytes per token
by eliminating repeated normalized-row downloads. It does not alter arithmetic.
Interlayer residual rows already remain on device 0 and require no extra
activation transfer.

### Full-model transfer and memory measurement

```sh
PYTHONPATH=build-python/python python examples/python/llama_transfer_profile.py \
    /path/to/SmolLM2-135M --worker build/pim-kernel-worker-fp16 --devices 3
```

This example loads the full model once, runs two tokens through the former
repeated-copy path, resets history, then runs them through broadcast. It checks
bitwise-equal logits and the expected U/D payload totals on every worker.
Preparation traffic is excluded. There are no diagnostic layer-output downloads.
The same test can be run with --devices 2 to confirm unchanged volume.

Memory snapshots include live_bytes, peak_bytes, allocations, staging_bytes
and workspace_bytes from L2. They describe runtime-managed allocations, not
worker-process RSS, all libpim internal allocations, PyTorch CPU weights or Python
objects. Peak values are cumulative from session creation, including weight
preparation and both phases; phase peaks must not be interpreted as independent
measurements. Post-token staging_bytes is not a peak staging-memory metric.
This is a native protocol measurement, not a simulated bandwidth/latency test.


On the pinned SmolLM2-135M checkpoint (30 layers, H=576, D=3, two tokens,
max_tokens=2), both tokens produced bitwise-identical logits across the two
copy strategies. Per-token measured payloads were:

| Device | Upload bytes, either strategy | Old download bytes | Broadcast download bytes |
|---|---|---|---|
| 0 | 70,272 | 277,632 | 208,512 |
| 1 | 69,120 | 138,240 | 138,240 |
| 2 | 69,120 | 138,240 | 138,240 |
| Total upload + download | — | 762,624 | 693,504 |

The reduction is 69,120 bytes per token (about 9.1%). Preparation live bytes
were 243,949,824 on device 0 and 243,809,280 on each other device. After KV
allocation, live bytes were 243,965,184 and 243,824,640 respectively, unchanged
between strategies and between the two completed tokens. End-of-token staging
and workspace bytes were zero. No device-memory reduction or speedup is claimed.

Native regression tests cover source ordinal 1, three-device FP16/FP32
broadcasts, the one-device no-transfer case, two-device U/D counts, rejection of
foreign tensors, and a single download of the assembled all-gather row. The
full native suite passed 26/26 after integrating shared staging.


## Layer placement across tensor-parallel groups

The same stack and full-model classes accept device_groups plus placement.
Use this form instead of devices when successive layers should reside in
different groups. Each group owns independent Sessions; within a group, every
layer still uses tensor-parallel attention and SwiGLU.

```python
# Four native worker connections: two groups with two devices each.
options = {"backend": "gem5_pim", "worker": worker}
groups = [[options, options], [options, options]]
layer_count = len(source.model.layers)
midpoint = layer_count // 2
model = ndp.nn.TensorParallelLlamaForCausalLM(
    source, device_groups=groups,
    placement=[0]*midpoint + [1]*(layer_count-midpoint), max_tokens=128)
output_ids = model.generate(input_ids, max_new_tokens=8)
```

Group entries can be DeviceGroup objects or lists of Session objects/backend
option dictionaries. Each dictionary occurrence creates a distinct connection.
For external workers, construct independent Sessions for their endpoints and
pass them in the same grouping. One Session cannot appear in multiple groups.
Each listed group must own at least one layer. placement must contain one
integer group ordinal per source layer; devices and device_groups cannot be
specified together. The single-group devices API remains unchanged.

The decoder layers execute in source order. Adjacent layers on the same group
reuse the completed row on that group's device 0. On a group change, one FP16
[1,hidden_size] row is downloaded from the old root and uploaded to the new
root through the existing blocking L2 copy. No KV cache or weight migrates.
The destination's normalized inputs are then broadcast within its own group.

Embedding input is uploaded to the first assigned group's root. Final RMSNorm
runs on the last assigned group's root and only its normalized result is
returned to the CPU LM head. Group numbers need not match execution order;
placement=[1,0] is valid. Returning to an earlier group is also allowed, but
adds another boundary transfer. Group sizes may differ if each assigned layer
can distribute its KV heads and projection channels across that many devices;
changing TP degree can change FP16 rounding.

The stack's input_session and output_session expose the first and last roots
for explicit resident integration. forward_ndp accepts a row owned by the
input_session and returns a row owned by the output_session. Its devices
attribute covers all group Sessions in group order; device_groups preserves
the grouping. Ordinary forward/generate do not require these low-level handles.

History and checkpoint ownership remain model-wide. If a boundary copy or a
later group fails, earlier layers may already have appended their KV entries.
The whole stack rejects new tokens until reset_history clears every group's
history. Checkpoint keys are independent of placement; loading the source
state dict prepares each layer's weights on its assigned group and resets all
history. Changing placement requires constructing a new adapter.

This provides sequential stage placement plus TP within each stage. It does
not overlap multiple tokens, requests or microbatches, and does not implement
a pipeline scheduler or distributed Python processes. No throughput gain is
claimed from the placement feature alone.

### Hybrid verification

```sh
PYTHONPATH=build-python/python python examples/python/llama_hybrid_parallel.py \
    /path/to/SmolLM2-135M --worker build/pim-kernel-worker-fp16
```

The example compares the full model on one two-device group against two
consecutive two-device groups. All layer partitions remain TP degree two.
It executes two tokens in each configuration, checks bitwise-equal logits,
and verifies U/D payload totals for every worker. The first and second stage
receive half of the source layers. Weight preparation traffic is excluded;
there are no layer-debug downloads. L2 live/peak bytes are reported separately
for each worker. Measurements use native numerical workers, not gem5 FS.

A boundary adds one FP16 download plus one upload, or 4H payload bytes for
hidden width H. The verification checks those boundary bytes in addition to
existing within-group broadcasts and FP32 reduction traffic. Native tests
also cover reversed group order, source checkpoint reload, greedy generation,
invalid/overlapping placements and a failed handoff after the first group
has advanced its history.

For the pinned SmolLM2-135M checkpoint (30 layers, hidden width 576), native
verification used placement=[0]*15+[1]*15 and two workers per group. Both
input tokens [1,42] produced logits bitwise equal to the one-group TP baseline.

| Configuration | Total U/D payload per token | Maximum per-device live bytes after token |
|---|---|---|
| One TP group, 2 devices, all 30 layers | 486,144 | 327,538,944 |
| Two stages, 2 devices each, 15 layers per stage | 488,448 | 163,770,624 |

The extra 2,304 bytes are exactly one 1,152-byte boundary download and upload.
The layer split approximately halves the largest per-device live allocation;
it does not halve total model storage. These are L2-managed bytes for a
max_tokens=2 run, excluding CPU weights and untracked worker internal memory.
The full native regression suite passed 26/26 with the grouped placement API.


## Independent decode requests sharing model weights

The TP model and stack now separate request history from the shared model.
One adapter owns prepared projections, normalization weights, RoPE tables and
worker connections. Each DecodeRequest owns its layer-local KV caches, token
position, capacity and failure status. This is an L3/PyTorch change; L2 remains
a generic tensor runtime.

```python
with model.new_request(max_tokens=64) as a, model.new_request(max_tokens=32) as b:
    logits_a = model(torch.tensor([[1]], dtype=torch.int64), request=a)
    logits_b = model(torch.tensor([[42]], dtype=torch.int64), request=b)
    logits_a = model(torch.tensor([[30]], dtype=torch.int64), request=a)
    print(a.position, b.position)  # 2, 1
    a.reset()                     # Only A's KV history is discarded.
    # B can continue with its original position and caches.
```

Requests are created through model.new_request(), not by constructing
DecodeRequest directly. TensorParallelLlamaDecodeStack supports the same factory
and request= argument on forward and forward_ndp. A request belongs to its
creating stack and cannot be used with another model, even with shared Sessions.
The existing calls without request= retain a separate default single-sequence
history for compatibility. Explicit requests never temporarily replace the
model's default cache fields; state is passed down to each layer and attention
call directly.

| API or property | Meaning |
|---|---|
| new_request(max_tokens=None) | Create empty history; capacity defaults to the model limit |
| model(token, request=req) | Append one token to that request and return logits |
| model.generate(ids, request=req, ...) | Reset only that request and run the existing greedy loop |
| req.position | Number of complete successful model/stack token calls |
| req.max_tokens | Fixed capacity, positive and no larger than the model's RoPE capacity |
| req.failed | A token failed after device work may have begun; reset is required |
| req.reset() | Discard this request's caches and restore position zero |
| req.close() | Release this request's cache references permanently; idempotent |
| req.closed | Closed requests cannot be decoded or reset |

KV allocation is lazy. A request's first token allocates the full configured
KV capacity in each assigned layer/device. It is not a paged cache. Creating
an empty request uploads no weights and allocates no device cache. Reset and
close release only that request's buffers; shared prepared weights remain.
The context manager calls close on exit, including exceptions. Dropping all
references to a request also releases its cache objects; the model tracks
requests weakly. L2 retains any buffers still required by submitted commands
until that work completes.

A layer, group handoff, final norm or LM-head failure marks only the selected
request as failed. Earlier layers may already have advanced their local KV
state, so the entire request must reset before reuse. Other requests retain
their own histories. This isolates request bookkeeping; a disconnected or
otherwise unusable shared backend can still affect all requests using that
Session. Out-of-range token IDs and exhausted capacity are rejected before
history advances. A capacity rejection does not mark an explicit request as
failed.

model.reset_history() resets only the default history. Use req.reset() or
model.reset_history(request=req) for an explicit request. Whole-model or
whole-stack load_state_dict closes all existing explicit requests before
reloading resident weights, even if that load later fails. Create new requests
after a reload. Request state is absent from state_dict; checkpoint keys are
unchanged. Direct submodule reload/weight mutation while requests are active
is unsupported: close all requests and use a whole-model reload.

Calls must still be sequential on the owning Python thread. This feature
provides independent histories for manual interleaving, not a scheduler,
continuous batching, concurrent forward calls or overlapping pipeline work.
The sequential GenerationScheduler now uses this ownership contract; see
[request scheduling](request-scheduling.md) for admission, streaming and cancellation.

### Request isolation verification

```sh
PYTHONPATH=build-python/python python examples/python/llama_requests.py \
    /path/to/SmolLM2-135M --worker build/pim-kernel-worker-fp16
```

The full-model example compares each request run alone against alternating
tokens from two requests on one shared-weight model. It checks exact logits,
independent positions, lazy allocation, reset isolation and return to the
prepared-weight memory baseline after close. Native unit tests additionally
cover group-boundary failures, final-norm failures, generation isolation,
checkpoint invalidation, foreign-model requests and capacity validation.

Measured with SmolLM2-135M, 30 layers, two workers, model capacity 4 and two
requests of capacity 2 each (input sequences [1,42] and [30,198]):

| L2 live bytes | Device 0 | Device 1 |
|---|---|---|
| Prepared model, no request cache | 327,523,584 | 301,824,000 |
| Two requests after interleaved decode | 327,585,024 | 301,854,720 |
| After resetting one request | 327,554,304 | 301,839,360 |
| After closing both requests | 327,523,584 | 301,824,000 |

The two requests added 61,440 and 30,720 KV bytes on devices 0 and 1,
respectively, consistent with the local KV head counts (2 and 1). Interleaved
and isolated logits matched bitwise, and replay after request reset also
matched. Native regression tests passed 26/26. These are native runtime
measurements, not an extended gem5 FS or concurrency validation.


## Serving scheduler

`ndp.serving.GenerationScheduler` alternates one input token per active request,
admits waiting requests in FIFO order and closes histories on completion,
cancellation or failure. It works with both single-group and staged models.
See [request scheduling](request-scheduling.md) for API examples and validation.
GenerationScheduler remains sequential. The separate
[PipelineGenerationScheduler](pipeline-execution.md) overlaps stages across requests.


## Overlapped stage execution

Use `ndp.pipeline.PipelineGenerationScheduler(source, device_groups=...,
placement=..., max_tokens=..., max_active_requests=...)` to overlap contiguous
stages across independent requests. It creates Sessions on dedicated owning
threads, so it takes backend options rather than pre-existing Sessions.
The original staged model and GenerationScheduler remain sequential.
See [pipeline execution](pipeline-execution.md) for ordering, cleanup and native
concurrency validation. Prompt processing is still token-by-token.


## Chunked prefill

`model.prefill(ids, request=...)` appends multi-token chunks using PNM GEMM for
projections and bulk local KV appends. `generate` and both schedulers accept
prefill_chunk_size (default 1). Decode remains PIM GEMV. See [prefill](prefill.md)
for matrix layouts, FP32 partial output support, chunk limits and packet-level
verification. Causal attention remains the existing query/head composition.
