# Using NDP Library from PyTorch

## Inference with ordinary PyTorch tensors

Load the weights into the source PyTorch modules and put them in eval mode.
The adapters prepare resident weights during construction and accept CPU FP16
tensors in `forward`. They own references to the runtime and manage transfers
at module boundaries. Default adapters on the same Python thread share the
single-device `ndp` command backend.

```python
import torch
import ndp

# mha is a pretrained torch.nn.MultiheadAttention in CPU FP16.
attention = ndp.nn.DecodeSelfAttention(mha.eval(), max_tokens=2048)
with torch.inference_mode():
    for hidden in tokens.split(1, dim=1):    # tokens: [1, sequence, embed_dim]
        output = attention(hidden)           # [1, 1, embed_dim], on CPU
    attention.reset_history()                # start the next independent request
```

This adapter runs causal self-attention for one sequence with one token per
call. It splits the source module's packed Q/K/V projection, uploads the token
once, and keeps the projected K/V history on the device. Bias-free projections
use PIM; projections with bias use PNM. Attention uses PNM. The result is the
projected output tensor. The adapter's state dict contains the four separate
projections. Loading it clears sequence history. Use a separate adapter for
each concurrently active sequence.

For native gem5_pim decode, build the current FP16 worker with
`make -C tools/gem5/pim_kernel profile fp16` and select it at construction:

```python
attention = ndp.nn.DecodeSelfAttention(
    mha.eval(), max_tokens=2048, backend="gem5_pim",
    worker="build/extended-gem5/pim-kernel-worker-fp16",
)
output = attention(token)
```

The worker keeps both KV buffers between calls and copies only the appended
range and the requested prefix. The L3 attention implementation composes
per-head gather, transpose, scaled FP32 matmul, last-axis softmax and a second
matmul. Each primitive stores FP16 output. Projection, cache and attention
intermediates stay within the worker. To capture arithmetic launches as well,
pass `capture_worker`, `gemv_elf` and `attention_elf` from the
[PNM launch guide](pnm-launch-abi.md).

For a bias-free SwiGLU MLP, pass three eval-mode CPU FP16 projections from
your model. Gate, up, SiLU, multiply and down share resident intermediates.

```python
mlp = ndp.nn.SwiGLU(source.gate_proj, source.up_proj, source.down_proj)
output = mlp(hidden)
```

SwiGLU accepts a single vector of rank one through three and preserves its
leading axes. To select the N4 native gem5 capture configuration, pass
`backend="gem5_pim"`, `worker`, `capture_worker`, `gemv_elf`, and
`pointwise_elf` as constructor keywords. For a single projection use
`ndp.nn.Linear.from_float(source.down_proj)`. Explicit runtime arguments
remain available for callers managing device lifetimes. See the
[runnable module example](../examples/python/decode_modules.py).

## Decode one complete SmolLM2 block

Convert a loaded CPU FP16 Hugging Face Llama decoder layer in evaluation mode:

```python
block = ndp.nn.LlamaDecodeBlock(model.model.layers[0], max_tokens=128,
                                backend="gem5_pim", worker="build/extended-gem5/pim-kernel-worker-fp16")
with torch.inference_mode():
    for token in hidden_states.split(1, dim=1):
        output = block(token)  # CPU FP16 [1, 1, hidden_size]
block.reset_history()
```

The adapter prepares seven bias-free PIM projections, both normalization
weights, and RoPE tables at construction. Q and K use standard Llama
split-half rotation. The signed sine table incorporates the rotation sign
once and GQA stores only the KV heads. PNM runs normalization, attention,
pointwise operations and residual adds. Projection outputs, the KV prefix,
and both residuals stay in the worker. Each call accepts one token for one
sequence at the next contiguous position. Start a new sequence with
`reset_history()`. The adapter preserves the source layer's state dict keys;
loading weights also clears history. This interface uses unscaled full-width
RoPE and a bias-free SwiGLU MLP.

### Reproduce the checkpoint comparison

Using the FP16 worker build above and the pinned local SmolLM2-135M checkpoint:

```sh
make -C tools/gem5/pim_kernel profile fp16
PYTHONPATH=build-python/python python examples/python/llama_block.py /path/to/SmolLM2-135M \
    --worker build/extended-gem5/pim-kernel-worker-fp16
```

For launch capture, rebuild the decoder-capable attention ELF using the
cross compiler in the [PNM launch guide](pnm-launch-abi.md) and additionally pass
`--capture-worker build-python/ndp_pnm_test_worker --gemv-elf build/pnm/pim_gemv.elf
--pointwise-elf build/pnm/pointwise.elf --attention-elf build/pnm/attention.elf`.

For layer 0 (576 hidden units, 9 Q heads, 3 KV heads, head dimension 64),
the verifier embeds token IDs [1, 42, 30, 198] from the checkpoint. Its CPU
reference runs the original Transformers eager layer once with an explicit
causal mask. The NDP module decodes the same four tokens sequentially.
The maximum absolute differences across all tokens and features were
0.000244140625 after the attention output projection, 0.03125 after the MLP
down projection and 0.03125 at the final residual. All three comparisons
passed the elementwise criterion `abs(error) <= 0.02 + 0.02 * abs(reference)`
in both native reference and launch-capture runs.

A separate replay after cache reset compared outputs exactly and measured one
activation upload and one output download per token. The native worker stream
also showed one U and one D frame with 1,152 tensor bytes each per token.
Construction uploads and diagnostic reads are outside that measurement.
These measurements cover the first block; target PNM/PIM execution, later
layers and model logits are outside this verification.

## Explicit runtime API

Select the command backend with `Session(backend="ndp")`. Pass `config` to override its default NdpSpec. See [the command backend guide](ndp-backend.md).

The extension exposes the following runtime features:
Python ownership/lifecycle bindings, CPU FP16 transfer, resident L2 operations, and
an explicit Python decoder example. This is an eager, inference-only adapter, not a
native torch device. `NdpTensor` is an opaque device handle rather than a subclass
of torch.Tensor. L2 has no PyTorch dependency.

## Convert selected projections for inference

Load the checkpoint into an eval-mode CPU FP16 model. Pass a registered module
path to `ndp.to_ndp` to select a single Linear or the projections in a subtree.
An explicit empty path selects the model's children.

```python
target = "model.layers.0.mlp.down_proj"
for entry in ndp.plan_to_ndp(model, target=target):
    print(entry.path, entry.rule or entry.reason)

with ndp.to_ndp(model, target=target) as accelerated, torch.inference_mode():
    output_ids = accelerated.generate(**inputs, do_sample=False, max_new_tokens=16)
```

The preview opens no session and uploads no weights. The conversion uses one
owned `ndp` session and prepares all eligible weights before replacing any
modules. The scope yields the same model and restores the original module objects
on exit, including when inference raises. Use the model on the entering thread
and load checkpoints before entering the scope. Only unmodified, bias-free
`torch.nn.Linear` projections with materialized CPU FP16 weights are eligible.
The preview reports exclusions such as shared modules, tied weight storage and
hooks. Entering a scope with no eligible projections raises an error.

In `mode="projections"`, the default, single-vector projections use PIM and
multi-vector inputs use CPU Linear. Transformers manages generation and its KV
cache. With Transformers 4.57.6 and eager Llama attention, selecting all three
eligible Q/K/V projections under the attention module automatically shares their
input upload for single-token calls. The preview reports `llama_shared_qkv` at
the attention path. Prefill, non-eager attention, and calls that need projection
forward hooks use the regular projection path. The attention object and
`state_dict` paths are preserved.

For a larger subtree, choose a backend and capacity that fit the prepared
weights. The default `ndp` profile fits SmolLM2's first `down_proj`. The entire
first block requires a larger profile via `config`, or a backend with more
capacity. Set `backend="gem5_pim", worker=worker_path` on `to_ndp` to use a native
FP16 libpim worker. Reproduce the comparison with the pinned SmolLM2 checkpoint
on the functional backend from the repository root:

```sh
PYTHONPATH="$PWD/build-python/python" python examples/python/llama_decode.py "$CHECKPOINT" \
    --verify-to-ndp --tokens 504 3575 282 4649 314 7042 30 --max-new-tokens 4
```

Expect `NDP_TO_NDP_PASS` for the first-layer `down_proj` and for all seven
first-block projections. Each case checks exact logits for prefill and two decode
steps, greedy generation IDs, and original module restoration against manual
conversion on the same backend.

## Keep a Llama SwiGLU MLP resident

With Transformers 4.57.6, select an unmodified Llama SwiGLU MLP directly in
`resident` mode. The original model still accepts ordinary CPU inputs and
manages attention, generation, and KV cache.

```python
with ndp.to_ndp(model, target="model.layers.0.mlp", mode="resident",
                backend="functional") as accelerated, torch.inference_mode():
    output_ids = accelerated.generate(**inputs, do_sample=False, max_new_tokens=16)
```

The conversion checks PIM Linear, PNM Linear, SiLU and multiply before
preparing weights. Each of the three weights is uploaded once and prepared for
both projection engines. A single input row uses PIM GEMV. Multiple rows,
such as a multi-token prompt, use PNM GEMM. SiLU and multiply use PNM in both
cases. Gate, up and the product remain resident. Each MLP call uploads one
input tensor and downloads one output tensor. The runtime checks the actual
input extent against backend and memory limits during execution.

SmolLM2-135M's complete first MLP exceeds the default `ndp` profile's physical
layout capacity. Use the functional backend as shown above, pass a larger
`config` to the `ndp` backend, or select the native worker.

`plan_to_ndp(..., mode="resident")` reports `llama_resident_swiglu` for an
eligible MLP. Pass its module path, for example `model.layers.0.mlp`. The scope
restores the original module on exit. Keep module and hook registrations fixed
during the scope. For lower-level use, `ndp.nn.SwiGLU(..., prefill=True)` enables
the same multi-row behavior.

```sh
PYTHONPATH="$PWD/build-python/python" python examples/python/llama_decode.py "$CHECKPOINT" \
    --verify-resident-mlp --tokens 504 3575 282 4649 314 7042 30 --max-new-tokens 4
```

Expect `NDP_RESIDENT_MLP_PASS`. The example compares the resident output with
CPU FP16 on identical MLP activations from prefill and two cached decode calls.
It checks one upload and one download per call, compares greedy IDs, and verifies
module restoration. Add `--worker` with the native FP16 libpim worker path to
exercise the native worker backend.

## Keep a Llama decoder block and its KV cache resident

With Transformers 4.57.6, select one Llama decoder layer in `resident` mode.
Use eager attention, an unpadded batch of one, and contiguous positions with
the model's standard unscaled split-half RoPE. `max_tokens` bounds the number
of tokens processed per request, including the prompt. If omitted, the limit is
128 tokens or the model's context limit, whichever is smaller.

```python
with ndp.to_ndp(model, target="model.layers.0", mode="resident",
                max_tokens=128, backend="functional") as accelerated, torch.inference_mode():
    output_ids = accelerated.generate(**inputs, do_sample=False, max_new_tokens=16)
```

The selected block keeps normalization, RoPE, GQA, SwiGLU, residuals and its K/V
buffers within the runtime. Multi-token prompt chunks use PNM GEMM and
single-token calls use PIM GEMV for projections. Within a chunk, each query
attends through its own position. The block uploads one input activation and
downloads one output activation per call.

`generate()` uses a Transformers `DynamicCache`. For explicit `forward` calls,
start each request with a fresh `DynamicCache` or with `use_cache=True` and no
prior cache. Pass the returned `past_key_values` on subsequent calls. Each
request holds a distinct NDP allocation for the selected layer, and the other
layers use their own Transformers cache entries. Length queries such as
`get_seq_length()` work for the selected layer. The scope restores the original
decoder and releases the device cache buffers on exit. Start fresh after an
inference failure or when entering another conversion scope.

This path supports greedy generation and causal forward calls. Static or
offloaded caches, padding, custom masks or position offsets, batch expansion,
cache rearrangement or export, and attention or hidden-state recording are
outside its supported interface. Choose a profile that fits the packed weights,
KV capacity and activations. The first SmolLM2 block exceeds the default `ndp`
profile. The functional backend, a larger `ndp` config, or the native FP16
libpim worker can be used for the example below.

```sh
PYTHONPATH="$PWD/build-python/python" python examples/python/llama_decode.py "$CHECKPOINT" \
    --verify-resident-block --tokens 504 3575 282 4649 314 7042 30 --max-new-tokens 4
```

Expect `NDP_RESIDENT_BLOCK_PASS` for two prompt chunks followed by two cached
decode calls. The example checks cache positions, one upload and one download
per call, block outputs against CPU FP16, and greedy generation IDs. It reports
full-model logits differences separately. Add `--worker` with the native FP16
libpim worker path for the same checks on that backend.

## Convert one Linear with an explicit session

Load an eval-mode CPU FP16 model, then convert a bias-free projection once per session.

```python
with torch.inference_mode(), ndp.Session(backend="gem5_pim", worker=worker) as runtime:
    mlp = model.model.layers[0].mlp
    mlp.down_proj = ndp.nn.Linear.from_float(mlp.down_proj, runtime=runtime)
    prefill = model(prompt_ids, use_cache=True)
    step = model(next_id, past_key_values=prefill.past_key_values, use_cache=True)
```

Use the converted model inside its session lifetime. For a remote worker use `ndp.Session.connect(host, port)`. A single vector of rank one through three uses `torch.ops.ndp.linear_pim`, returning a CPU FP16 tensor with the original leading dimensions. Other compatible shapes use CPU Linear. The module stores its checkpoint under `weight` and rebuilds its prepared representation on `load_state_dict`. After editing a weight directly, call `refresh()`. FakeTensor metadata execution is available for the operator. On Python 3.11 and Torch 2.9.1, fixed-shape operator capture with torch.compile(fullgraph=True, backend="eager") matched eager results on the functional and native libpim backends.

## Build and install

Use a Python environment with PyTorch 2.9 or newer, setuptools, wheel, a C++20
compiler, and CMake 3.20+. The implementation was exercised locally with CPython
3.14 and PyTorch 2.9.1 on macOS arm64. The CI configuration targets CPython 3.11
and CPU PyTorch 2.9.1 on Ubuntu. Other combinations are not established by these tests.
The wheel uses the ordinary LibTorch ABI: rebuild it when changing PyTorch or Python.

```sh
# Use the Torch installation in the active environment, without downloading another one.
python -m pip install . --no-build-isolation --no-deps
python examples/python/linear.py
python examples/python/decoder_block.py --ndp-config configs/ndp-small.ini
```

Config paths are explicit filesystem paths; bundled repository examples/configs are
not installed as package resources. Run the examples from the repository root.
CMake and the build prerequisites must already be available for the command above.
The normal C++ build remains unchanged and does not require Python or Torch.

For development and the complete C++/Python tests:

```sh
cmake -S . -B build-python -DNDP_BUILD_PYTHON=ON -DNDP_PYTHON_TESTING=ON \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build-python -j 2
ctest --test-dir build-python --output-on-failure
PYTHONPATH=build-python/python python examples/python/decoder_block.py
```

CMake copies the Python package into `build-python/python/ndp` by default.
`NDP_PYTHON_OUTPUT_DIR` can override this destination. `NDP_PYTHON_TESTING` enables
an internal delayed-failure backend solely for error/GIL/lifetime tests; wheels
explicitly disable it. Tests for that backend are skipped for production wheels.

## Resident libpim projections from external PyTorch

Build the FP16 native reference worker with `make -C tools/gem5/pim_kernel profile fp16`. Use the Python adapter built from the same checkout. On POSIX, `worker` is the path to the executable; each Session starts one process and closes it on exit. This backend supports PIM Matvec and single-token, bias-free Linear.

```python
import torch
import ndp

source = torch.nn.Linear(3, 2, bias=False, dtype=torch.float16).eval()
with ndp.Session(backend="gem5_pim", worker="./build/extended-gem5/pim-kernel-worker-fp16") as s:
    projection = ndp.nn.NdpLinear.from_torch(source, s, ndp.Engine.PIM)
    for x in (torch.ones(1, 3).half(), torch.zeros(1, 3).half()):
        y = projection.forward_ndp(s.upload(x))
        print(s.download(y))
```

To chain compatible layers, pass `y` directly to the next projection's `forward_ndp`. Weight preparation occurs during module construction. The worker uses FP16 operands, FP32 accumulation and a rounded FP16 output at each operation boundary. To include the native worker integration cases in CTest, configure the Python build with `-DNDP_GEM5_PIM_WORKER="$PWD/build/extended-gem5/pim-kernel-worker-fp16"`.

## Session and transfer boundaries

```python
import torch
import ndp

with ndp.Session(backend="ndp", config="configs/ndp-small.ini") as s:
    x = s.upload(torch.tensor([[1, 2, 3]], dtype=torch.float16))
    w = s.upload(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16))
    weight = ndp.llm.prepare_weight(s, w)
    options = ndp.OpOptions(ndp.Engine.PIM, ndp.LayoutPolicy.RequirePrepared)
    y = s.linear(x, weight.tensor(), options)
    y = s.silu(y)  # PNM by default; receives the dense PIM result directly.
    result = s.download(y)
```

`Session()` defaults to the functional backend with 64 MiB off-chip capacity and
4096 bytes of scratchpad. `capacity_bytes` and `scratchpad_bytes` customize that
backend; an NDP Session uses the resources in its NdpSpec.
`max_staging_bytes` controls the runtime's upload staging budget for either backend.
There is no process-global device selection. Engine is selected for each operation.

Upload requires a CPU strided float16 torch.Tensor without requires_grad. Sparse,
quantized, other devices/dtypes, scalar, and empty inputs are rejected. Convert
explicitly before upload. Noncontiguous tensors and lazy negative views are resolved
into contiguous logical values. Upload copies into native-owned staging before
returning; changing or deleting the CPU source afterward is safe.

Download waits and returns a newly allocated CPU FP16 torch.Tensor. It never exposes
worker-written storage before completion. `s.download_async(x)` returns a
HostTransfer; `.wait()` creates the completed CPU tensor. These are copying APIs,
not zero-copy/DLPack boundaries. `NdpTensor.dtype` is `ndp.DType.F16`; its shape is a
list of dimensions. `desc` is native TensorDesc metadata.

## Resident operations and preparation

| API group | Available calls |
|---|---|
| Matrix/vector | `matmul(a,b,options,factor=1.0)`, `matvec(w,x,options)`, `linear(x,w,options,bias=None)` |
| Elementwise/normalization | `add`, `mul`, `rms_norm(x,gamma,epsilon=1e-5,options=...)`, `softmax(x,axis=-1,options=...)`, `silu` |
| Data operations | `reshape(x,shape)`, `gather(x,indices,shape)`, `scale(x,factor)`, `transpose(x)` |
| Planning/layout | `query_operation`, `query_layout`, `prepare_layouts`, `materialize_layout` |
| Mutable storage | `allocate(TensorDesc)`, `copy_into`, `snapshot`, `clear` |
| Completion/inspection | Tensor/Event `wait`, `ready_event`, `synchronize`, `memory_stats`, `capabilities`, `close` |

OpOptions defaults to PNM and AllowMaterialize; matvec defaults to PIM.
RequirePrepared rejects a missing layout instead of creating one at submission.
`ndp.llm.prepare_weight` is L3 policy layered over generic L2 layout preparation;
its `.tensor()` is the ordinary Tensor accepted by L2 linear. Its default engines
are PNM and PIM; an explicit engine list and max_resident_bytes are accepted.

The NDP backend supports PIM matvec and bias-free M=1 linear. Run SiLU on PNM. Unsupported
combinations raise NdpError. The wrapper does not silently select another engine.
Matmul's finite `factor` scales the FP32 accumulator before FP16 output conversion.
Cached attention uses this for `1/sqrt(head_dim)`, avoiding overflow caused solely
by storing an unscaled score first. Scores are still FP16 after scaling, so this
is not a full FP32 or fused-softmax attention implementation.

Tensor results are immutable and ordinarily returned before device execution ends.
Layout preparation waits as the native API requires. Shape, broadcasting, and
layout restrictions are those of the existing C++ runtime, not full ATen semantics.

A model-independent buffer example:

```python
with ndp.Session() as s:
    x = s.upload(torch.tensor([1, 2, 3], dtype=torch.float16))
    buffer = s.allocate(ndp.TensorDesc([6]))
    s.copy_into([ndp.TensorCopy(x, buffer, 0, 2, 3)])
    saved = s.snapshot(buffer, ndp.TensorDesc([6]))
    s.clear([buffer]).wait()
    print(s.download(saved))  # [0, 0, 1, 2, 3, 0]; snapshot owns an independent copy.
```

Copy offsets and counts are FP16 elements in flattened dense order. Snapshot and
reshape do not expose PyTorch views or aliasing mutations. Cross-Session operands
are rejected before entering the native operation.

## Implement layers directly in Python

[decoder_block.py](../examples/python/decoder_block.py) shows the full layer sequence:
normalization, Q/K/V projections, head reshape, RoPE, request-cache append, causal
attention, output projection/residual, and SwiGLU FFN/residual. It does not invoke
the C++ decoder_block function. Native L3 helpers implement RoPE and cache policy
using generic L2 operations.

Initialize weights and layouts once. Call `layer.new_cache()` for each request and
pass its cache explicitly into `forward_ndp`. The example prefills two tokens on
PNM and performs three PIM-projection decode steps. Intermediate NdpTensor objects
stay resident; another layer can receive the output handle directly. Only explicit
downloads cross back to CPU. Current native RoPE composition uploads a small sign
vector per call; residency checks establish no repeated weight uploads or intermediate
downloads, not the absence of every small native upload.

The example fixes width=4, query_heads=2, kv_heads=1, head_dim=2, FFN width=6,
interleaved RoPE, and no batch dimension. It demonstrates the runtime and is not a
checkpoint loader or arbitrary Hugging Face model adapter. Cache append is not
transactional with the whole layer; reset/recreate request state after step failure.

## Replace a selected nn.Linear

```python
source = torch.nn.Linear(16, 8, bias=False, dtype=torch.float16).eval()
with ndp.Session() as s:
    layer = ndp.nn.NdpLinear.from_torch(source, s, engine=ndp.Engine.PIM)
    x = torch.randn(1, 16, dtype=torch.float16)
    cpu_result = layer(x)  # Convenience upload/compute/download boundary.
    resident_result = layer.forward_ndp(s.upload(x))  # Compose more NDP work here.
```

The adapter owns detached CPU checkpoint buffers and separately prepared device
weights. `state_dict()` serializes CPU buffers. `load_state_dict()` rebuilds resident
weights, including loading through a parent module. Buffer replacement is detected by object identity, and ordinary in-place changes
are detected via Tensor version counters; call `refresh()` to rebuild before forward.
Do not mutate storage through `.data` or external raw pointers: those changes can
bypass version counters. Shape/copy errors during loading leave the adapter unusable
until refresh or a successful reload. PIM bias and training mode are rejected.

Use the source module's `.to(...)` before constructing the adapter. The adapter does
not support later dtype/device migration or a magic `.to("ndp")` conversion. It is
an inference component, not a trainable nn.Parameter replacement.

## Ownership, threading, and errors

Each tensor, buffer, event, transfer, prepared weight, and cache retains a shared
native Session owner. Dropping the Python Session variable does not invalidate its
handles. This ownership does not require a Python reference cycle. Explicit close
drains work and makes future submissions fail; existing completion handles may still
be waited on. Destruction performs native RAII cleanup without Python callbacks;
use explicit close/context management to observe shutdown errors.

Use a Session and its handles only on the creating thread. Inspection and close are
also checked. A serving scheduler should marshal requests to that thread. Blocking
waits, downloads, preparation, synchronization, and close release the GIL. No second
Python submission thread or worker-side Python callback is introduced. Free-threaded
Python and subinterpreters are not supported by this adapter.

NdpError derives from RuntimeError and carries `.code` (ErrorCode) and `.operation`.
Submission errors appear at the call site; worker errors appear at wait/download/
synchronize/close. Context manager exit preserves an existing exception and attaches
shutdown diagnostics instead of masking it. Native Session errors remain reportable
after cache reset; reset is not a rollback or a general Session error clear.

## Validation and remaining scope

`tests/python` checks all FP16 encodings through transfer, noncontiguous/lazy views,
source mutation, ownership and cross-thread rejection, delayed execution errors and
GIL release, generic buffers, every bound primitive, layout preparation, PIM-to-PNM
chaining, module reload, and decoder intermediate values. Decoder tests use an
independent PyTorch reference with explicit FP16 primitive boundaries, the default and small NDP profiles and the functional backend, causal masking, cache reset/isolation, and
Python transfer counts. These are small numerical correctness tests, not performance
or full-model accuracy claims.

## Llama checkpoint decode with one PIM projection (P5)

The [local checkpoint example](../examples/python/llama_decode.py) loads a CPU FP16 Hugging Face LlamaForCausalLM in eager evaluation mode. Each run starts with a fresh, unpadded batch of one prompt on CPU. Before prefill the example converts the first layer’s MLP down_proj using ndp.nn.Linear.from_float. A multi-token prompt runs this projection on CPU. The next two single-token steps use the registered PIM operator. Hugging Face owns RoPE, normalization, causal masking, KV cache and the LM head. The projection keeps its prepared weight in the Session and transfers one activation and one result per call. Its input and output have shape [1, 1, features].

With Transformers installed alongside the extension, run the following from the repository root. Select a checkpoint that fits host RAM and whose selected projection fits the backend's weight and staging limits. Token IDs must be valid for that checkpoint.

```sh
PYTHONPATH=build-python/python python3 examples/python/llama_decode.py /path/to/local/llama \
    --worker "$PWD/build/extended-gem5/pim-kernel-worker-fp16" --tokens 1 2 3 4
```

Omit --worker to use the functional backend. The example compares both the projection outputs and the full vocabulary logits with an independent CPU run on the same tokens, reports maximum absolute error, and fails if torch.testing.assert_close exceeds atol=0.02 and rtol=0.02. These thresholds apply to this example; check the errors for each checkpoint before using it in serving.

### Pretrained checkpoint acceptance

Tested the base HuggingFaceTB/SmolLM2-135M LlamaForCausalLM checkpoint at revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` on CPU using PyTorch 2.9.1 and Transformers 4.57.6. Load the BF16 checkpoint in FP16 for both the reference and offloaded runs. Only layer 0's bias-free MLP down projection (1536 inputs, 576 outputs) is converted. [Pinned model files](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2)

The tokenizer encodes “The capital of France is Paris.” as seven tokens with no added BOS. The first five are the prompt and the last two are teacher-forced decode inputs. Each row compares against a fresh CPU run with the same KV cache history. Both runs use batch size one, an unpadded prompt and the model's default causal masking.

| Backend | Decode input ID | Projection max absolute error | Logits max absolute error | Greedy next ID (CPU and NDP) |
|---|---:|---:|---:|---:|
| functional | 7042 | 0 | 0 | 30 |
| functional | 30 | 0.00000762939453125 | 0 | 198 |
| native libpim | 7042 | 0 | 0 | 30 |
| native libpim | 30 | 0 | 0 | 198 |

From the repository root, set CHECKPOINT to a local copy of the pinned config.json and model.safetensors. Run both commands with the PyTorch extension and Transformers available:

```sh
export PYTHONPATH="$PWD/build-python/python${PYTHONPATH:+:$PYTHONPATH}"
python3 examples/python/llama_decode.py "$CHECKPOINT" --tokens 504 3575 282 4649 314 7042 30
python3 examples/python/llama_decode.py "$CHECKPOINT" --tokens 504 3575 282 4649 314 7042 30 \
    --worker "$PWD/build/extended-gem5/pim-kernel-worker-fp16"
```

### First-block attention projections

Append `--target attention` to either pretrained acceptance command above to convert layer 0's q_proj, k_proj, v_proj and o_proj together. The default target retains the down_proj baseline. Both runs use the pinned checkpoint, prompt, decode IDs and tolerances above. Each run creates an independent KV cache. RoPE, cache updates and attention execute in Hugging Face. The input to the post-attention RMSNorm captures the residual sum before normalization.

Measured maximum absolute errors versus CPU FP16:

| Output (shape at decode) | Functional step 1 | Functional step 2 | Native libpim step 1 | Native libpim step 2 |
|---|---:|---:|---:|---:|
| Q (1,1,576) | 0 | 0 | 0 | 0 |
| K (1,1,192) | 5.96e-8 | 7.63e-6 | 5.96e-8 | 0 |
| V (1,1,192) | 0 | 0 | 0 | 0 |
| O (1,1,576) | 9.54e-7 | 3.05e-5 | 9.54e-7 | 0 |
| Attention residual (1,1,576) | 0 | 3.05e-5 | 0 | 0 |
| Logits (1,1,49152) | 0 | 0.05322265625 | 0 | 0 |

Native libpim passed. Functional failed the existing step-2 logits criterion for 5,906 of 49,152 entries. Both backends selected the CPU greedy IDs 30 and 198. In four additional functional runs, offloading O alone reproduced the same maximum logits error. Offloading Q, K or V individually gave zero step-2 logits error. With identical step-2 O inputs and weights, two of 576 functional outputs differed from CPU by one FP16 ULP. The FP64 dot products rounded to FP16 matched CPU on all 576 outputs.

### Seven projections in block 0

Use `--target block` with the same two pretrained acceptance commands, tokens and checkpoint. This mode records three fresh-cache runs: CPU, Q/K/V/O on the selected backend, and Q/K/V/O plus gate/up/down on that backend. It checks that all seven packed weight objects are reused across decode calls. CPU executes RoPE, attention, normalization, activation, gating and residual addition. Both comparisons use the existing elementwise atol=0.02 and rtol=0.02. The example reports every comparison before returning a failing exit status if any check fails.

| Backend | Decode step | Block output max error vs CPU | Logits max error vs CPU | Logits max error vs attention-only | Logits outside tolerance vs attention-only |
|---|---:|---:|---:|---:|---:|
| Functional | 1 | 0.00390625 | 0.04296875 | 0.04296875 | 40 / 49152 |
| Functional | 2 | 0.00048828 | 0.03991699 | 0.03515625 | 162 / 49152 |
| Native libpim | 1 | 0.00390625 | 0.03125 | 0.03125 | 0 / 49152 |
| Native libpim | 2 | 0.00003052 | 0.04199219 | 0.04199219 | 356 / 49152 |

Both backends matched the CPU greedy tokens 30 and 198. Against the attention-only run, Q/K/V/O and the attention residual matched at both steps. At step 1 the gate matched and the up projection differed by 0.00012207 on both backends. The down projection and final block output differed by 0.00390625. The MLP projections and block output passed the elementwise tolerance. The new logits failures are recorded separately from the prior attention-only functional discrepancy.

### First-block routing and transfers

Use `--verify-routing` in place of `--target block` with the pinned checkpoint and seven token IDs above. The example checks the CPU profiler event counts for ndp::upload, ndp::pim_projection and ndp::download in each successful phase. These are logical transfers at the PyTorch/runtime boundary. Physical bank transfers, worker protocol frames and simulator timing require different instrumentation.

| Phase | Uploads | PIM Linear calls | Downloads |
|---|---:|---:|---:|
| Prepare seven weights once | 7 | 0 | 0 |
| Five-token CPU prefill | 0 | 0 | 0 |
| First cached decode token | 7 | 7 | 7 |
| Second cached decode token | 7 | 7 | 7 |
| Fresh one-token prompt | 7 | 7 | 7 |

The functional and native libpim backends produced the same counts. The one-token prompt starts with an empty cache and reuses the prepared weights. It takes the same single-vector route as cached decode.

### Greedy generation with seven first-block projections

Run the two commands below with tokenizer.json from the same pinned revision alongside the model. Each prompt starts with an empty KV cache on each path. Each model feeds its own argmax back until EOS or the specified number of new tokens. The first decision uses the final prefill logit vector. A one-token prompt uses PIM for that decision as well.

```sh
python3 examples/python/llama_decode.py "$CHECKPOINT" --verify-greedy --max-new-tokens 24
python3 examples/python/llama_decode.py "$CHECKPOINT" --verify-greedy --max-new-tokens 24 \
    --worker "$PWD/build/extended-gem5/pim-kernel-worker-fp16"
```

The acceptance criterion is identical generated token IDs and stopping position. The command prints both trajectories and the maximum absolute logits difference across decisions with identical preceding tokens. A first mismatch triggers fresh-cache replay through that decision, reports both top-two candidates and margins, and causes the verification to fail after all prompts have run. The existing teacher-forced numerical tolerance is checked independently by the block comparison.

Measured maximum absolute logits differences for the five prompts (24 generated IDs per prompt, all matched CPU on both backends):

| Prompt | Functional | Native libpim |
|---|---:|---:|
| The capital of France is | 0.0703125 | 0.0703125 |
| Water freezes at | 0.1171875 | 0.0625 |
| The first three prime numbers are | 0.111328125 | 0.078125 |
| Once upon a time | 0.078125 | 0.21484375 |
| Hello | 0.1328125 | 0.064453125 |

### Shared Q/K/V input

In an eager attention implementation call `q, k, v = ndp.nn.project_qkv(x, q_proj, k_proj, v_proj)` with three converted ndp.nn.Linear modules in one Session. For a single vector the dispatcher uploads x once, executes three PIM projections with their existing packed weights and returns three separate CPU tensors. K/V output widths can differ from Q. The uploaded activation belongs to this call. Multi-vector inputs follow CPU Linear. The helper calls the kernels directly; individual projection module forward hooks are bypassed.

For the pinned SmolLM2 checkpoint, append `--reuse-qkv` to the routing command:

```sh
python3 examples/python/llama_decode.py "$CHECKPOINT" --verify-routing --reuse-qkv \
    --tokens 504 3575 282 4649 314 7042 30 --worker "$PWD/build/extended-gem5/pim-kernel-worker-fp16"
```

Omit `--worker` for the functional backend. The example compares independent
projections, manual Q/K/V sharing, and `to_ndp()` with fresh caches. Each shared
path uses five activation uploads, seven PIM projections and seven downloads
per cached token and per fresh one-token prompt. Five-token prefill has zero of
all three. It checks exact equality of all 21 projection outputs and four logits
tensors across the three paths. `ndp::pim_projection` counts individual GEMVs,
including the three inside `ndp::qkv_pim`. Use `--verify-to-ndp` to compare the
generated IDs with the independent projection path.

## Place consecutive decoder layers on independent devices

`LlamaDecodeStack` consumes a single CPU FP16 hidden state shaped
`[1, 1, hidden_size]` per call. Each entry in `placement` selects the device
ordinal for the corresponding source layer. A device dictionary creates one
Session and backend, even when two dictionaries specify the same executable.

```python
devices = [{"backend": "gem5_pim", "worker": worker_path} for _ in range(2)]
stack = ndp.nn.LlamaDecodeStack(
    model.model.layers[:2], devices=devices, placement=[0, 1], max_tokens=128,
)
with torch.inference_mode():
    for token in hidden_states.split(1, dim=1):
        output = stack(token)
stack.reset_history()
```

Weights, rotary tables and each layer's KV cache stay on its assigned device.
Adjacent layers on the same device pass an immutable handle directly. At a
change of device, L2 waits for the source, downloads dense FP16 into host
staging, uploads those bytes to the destination and waits for completion.
Only the first and last stages exchange PyTorch tensors with the application.
A failed token can leave earlier caches advanced. The stack rejects further
tokens until `reset_history()` clears history across all stages. Resetting
starts a new sequence; reconstruct a previous prefix by replaying its tokens.
Each instance currently serves one sequence on its creating Python thread.
Execution is sequential within that sequence. This stage covers layer
placement; scheduling overlapping microbatches remains a separate extension.
Tensor-parallel projection contracts are described in [Parallelism](parallelism.md).

The lower-level `ndp.DeviceGroup([session_a, session_b])` also accepts existing
sessions. `group.copy_to(tensor, 1)` returns a ready tensor owned by session B.
A same-device copy returns the same allocation. Session ownership checks and
per-session staging and allocation budgets apply. On failure the original
tensor remains owned by its source. The C++ API is in
[device_group.hpp](../include/ndp/l2/device_group.hpp).

To compare the first two SmolLM2 layers on one native worker and on two workers:

```sh
PYTHONPATH=build-python/python python examples/python/llama_stages.py /path/to/SmolLM2-135M \
    --worker build/extended-gem5/pim-kernel-worker-fp16
```

The verifier compares every output element for four consecutive tokens
(IDs 1, 42, 30, 198) and observes the v2 upload/download frames per worker,
excluding initialization. For hidden size 576, the expected tensor payload
per frame is 1,152 bytes. Placement [0,0] produces U,D on one worker.
Placement [0,1] produces U,D on A and U,D on B: A's download and B's upload
carry the single inter-device activation.

## Split a single Linear across devices

For one decode token, split a bias-free FP16 projection along the output
channel dimension. Each device prepares its own contiguous range of weight
rows once. Every device receives the full input. Each output channel therefore
uses the same full dot product as the single-device implementation.

```python
devices = ndp.DeviceGroup([{"backend": "gem5_pim", "worker": worker_path}
                           for _ in range(2)])
projection = ndp.nn.TensorParallelLinear(source.gate_proj, devices=devices)
with torch.inference_mode():
    y = projection(x)  # CPU FP16, original leading axes and full output width
```

The module accepts a CPU FP16 vector shaped [K], [1,K] or [1,1,K].
It submits the local GEMVs to separate Session queues before downloading
the output shards and concatenating them on the CPU. A single full CPU
checkpoint weight is retained for state_dict. Each worker owns only its
assigned weight rows. Loading a standard Linear state dict rebuilds all
resident shards. An uneven split assigns the extra rows to the lowest device
ordinals; each device must have at least one row.

For a consumer that operates on device tensors, call
`shards = projection.forward_shards(x)` in place of the CPU forward.
Each handle is a [1,local_width] row on its corresponding device.
`devices.all_gather_rows(shards)` returns a complete [1,N] tensor on every
device. The L2 implementation gathers into device 0 and copies the assembled
row to the other devices using bounded host staging. For a consumer that
accepts the partitioned representation, use the local shards directly.

To verify a real checkpoint with two native workers:
```sh
PYTHONPATH=build-python/python python examples/python/llama_tensor_parallel.py \
    /path/to/SmolLM2-135M --worker build/extended-gem5/pim-kernel-worker-fp16
```

This example uses the first layer's gate projection and four real gate inputs
from a causally masked checkpoint forward. It checks the CPU result and both
device replicas against the unsharded native worker. Input-dimension splitting is available with split="input", using FP32 partials
and host reduction followed by one FP16 conversion. DistributedRow allows
resident tensors to flow between projections. See [Parallelism](parallelism.md)
for the complete API and precision contract. Pipeline scheduling and performance measurements are
independent of this numerical test.


## Tensor-parallel Llama sublayers

Use `ndp.nn.TensorParallelSwiGLU(layer.mlp, devices=devices)` and
`ndp.nn.TensorParallelLlamaAttention(layer.self_attn, devices=devices, max_tokens=128)`
with an FP16 eval-mode source and a shared DeviceGroup. Call each module with
one normalized CPU FP16 token under `torch.inference_mode()`; it returns a
CPU FP16 token without explicit Session, upload or download calls. Attention
retains per-device KV history until `reset_history()` is called.

Gate/up activations and grouped attention heads remain resident on their
assigned devices. Only the down and output projections produce FP32 partials
for host reduction. `forward_distributed` exposes those partials for resident
composition. Normalization and residual addition remain the caller's work.
See [Parallelism](parallelism.md) for supported shapes, failure/reset semantics,
partition rules, the checkpoint example and measured transfer counts.


For the complete layer, including RMSNorm and residual addition, use:

```python
block = ndp.nn.TensorParallelLlamaDecodeBlock(
    model.model.layers[0], devices=devices, max_tokens=128)
with torch.inference_mode():
    output = block(token)  # [1,1,hidden_size], CPU FP16
block.reset_history()
```

Normalization and residual addition run on device 0. Normalized inputs are
replicated for the tensor-parallel sublayers; FP32 projection partials are
reduced on the host. `forward_distributed` returns completed replicas for
another block in the same group. Loading a source layer state dict resets
KV history. A failed token requires resetting the whole composed sequence.


## Whole tensor-parallel model and generation

```python
model = ndp.nn.TensorParallelLlamaForCausalLM(
    source,  # CPU FP16 eval-mode Hugging Face LlamaForCausalLM
    devices=[{"backend": "gem5_pim", "worker": worker_path} for _ in range(2)],
    max_tokens=128)
output_ids = model.generate(input_ids, max_new_tokens=8)
```

The adapter runs all decoder layers and final RMSNorm on NDP. Embedding,
LM head and greedy token selection run on the CPU. Inputs are CPU int64 token
IDs, and generate returns prompt plus generated IDs. No explicit Session,
upload or download calls are required. Generation is single-sequence greedy
decode and processes the prompt one token at a time.

For manual streaming, call `model.reset_history()` then `model(token_ids)` for
each [1,1] token. Each call returns CPU FP32 logits [1,1,vocab_size]. These are
FP16 LM-head results converted to FP32. `generate` resets history itself.
The final emitted token has not been processed into KV history. Checkpoint
reload resets all layers, and any execution failure requires resetting the
whole model. This API is not Hugging Face GenerationMixin.

`TensorParallelLlamaDecodeStack` is available when embedding and output head
are managed by the caller. It accepts one CPU FP16 [1,1,hidden_size] vector,
keeps all interlayer rows on device 0, and returns the final hidden vector.
See [parallelism](parallelism.md) for placement, capacities and EOS semantics,
and [validation results](tensor-parallel-model-validation.md) for measured drift.


Llama RMSNorm adapters preserve the reference's intermediate FP16 conversion:
normalize in FP32, store FP16, then multiply by the FP16 normalization weight.
The existing L2 fused RMSNorm operation is unchanged. This distinction affects
model precision even though both paths return FP16 tensors; see the
[precision measurements](tensor-parallel-model-validation.md).


Multi-device normalization inputs now use a shared-staging broadcast: one source
download is reused for uploads to all other devices. This is automatic in the
TP adapters and changes no user calls or arithmetic. It reduces communication
for three or more devices; two-device volume stays the same. See
[parallelism measurements](parallelism.md) for packet counts and memory scope.


## Combine layer placement and tensor parallelism

Pass device_groups and a layer-to-group placement list to the existing adapter:

```python
model = ndp.nn.TensorParallelLlamaForCausalLM(
    source, device_groups=[[options, options], [options, options]],
    placement=[0]*15 + [1]*15, max_tokens=128)  # A 30-layer model.
output_ids = model.generate(input_ids, max_new_tokens=8)
```

Each option dictionary creates an independent worker Session. Each group runs
its assigned layers with TP; a group boundary passes only the completed hidden
row through host staging. Embedding and LM head stay on the CPU, and final
RMSNorm runs on the last group's root. TensorParallelLlamaDecodeStack accepts
the same arguments for callers managing embedding/head themselves.

Use devices for a single group, or device_groups with placement for multiple
groups. They cannot be mixed. All groups must be used and Sessions must be
disjoint. Checkpoint keys and reset semantics are unchanged. This is sequential
stage placement, not microbatch pipeline scheduling. See [parallelism](parallelism.md)
for the boundary protocol, measured traffic and the full-model comparison example.


## Share one model across independent decode requests

```python
with model.new_request(max_tokens=64) as request_a, model.new_request(max_tokens=64) as request_b:
    logits_a = model(token_a, request=request_a)
    logits_b = model(token_b, request=request_b)
    logits_a = model(next_token_a, request=request_a)
```

Each request owns independent KV caches and positions while weights remain
shared. Calls are sequential on the model's owning thread. No explicit upload,
download or Session calls are needed. Closing a request releases its cache
references; resetting it starts a new sequence without affecting other requests.
`generate(prompt, request=request_a, max_new_tokens=8)` resets and uses only A.
Calls without request= keep the existing default history.

Use the whole-model state dict to reload weights: reload closes all outstanding
explicit requests to prevent stale KV histories. A failed request must reset
before continuing, while healthy requests retain their state (subject to shared
backend health). Request scheduling and concurrent execution are not included.
See [request lifecycle and verification](parallelism.md) for the complete contract.


## Queue several generation requests

```python
with ndp.serving.GenerationScheduler(model, max_active_requests=2) as scheduler:
    a = scheduler.submit(prompt_a, max_new_tokens=8)
    b = scheduler.submit(prompt_b, max_new_tokens=8)
    scheduler.run()
    if a.status == "completed":
        print(a.token_ids)
    else:
        print(a.status, a.error)
```

The scheduler owns request histories and releases them on completion or cancel.
Use step() to receive per-token events and cancel(handle) between steps. Model
calls remain sequential; max_active_requests limits resident histories. See
[request scheduling](request-scheduling.md) for failure handling and EOS rules.


## Overlap stages across requests

`ndp.pipeline.PipelineGenerationScheduler` accepts a source Llama model,
backend-option groups and contiguous layer placement. Each stage owns a thread
and creates its Sessions there. submit/run/cancel use the same request-handle
style as the sequential scheduler, but different requests may execute on
different stages concurrently. One request still has only one token in flight.
See [pipeline execution](pipeline-execution.md) for a complete example and
measured native overlap. Existing Sessions cannot be passed across stage threads.


## Multi-token prefill

Use `model.prefill(prompt_ids, request=req)` to append a [1,T] prompt chunk and
receive [1,T,vocab_size] logits. For automatic prompt chunking, pass
`prefill_chunk_size=16` to model.generate or either scheduler constructor.
Multi-token projections use PNM GEMM; subsequent one-token decode uses PIM GEMV.
The default chunk size remains 1. See [prefill](prefill.md) for cache ordering,
FP32 partial outputs and native-worker limitations.

## Dispatcher contract and graph capture


The CPU dispatch kernels for `ndp::linear_pim` and `ndp::qkv_pim` take CPU FP16 tensors and explicit `LinearPackedParams` objects. Each packed object holds its Session and prepared weight. Fake execution reads the logical input and output widths. A real launch checks Session lifetime and thread ownership. Q/K/V requires three weights from the same Session and returns separate CPU tensors.

The module serializes the CPU weight in its state dictionary and rebuilds the packed representation after checkpoint loading. Direct edits require refresh before execution. A backend failure propagates to the caller.

### Verified dispatcher behavior

U1–U4 passed their native acceptance checks. The functional and native libpim backends passed the two-step tiny Llama checkpoint comparison.

U3 passed all four torch.library.opcheck tests on PyTorch 2.9.1: schema, autograd registration, FakeTensor and dynamic AOT dispatch. Packed handles compare by native identity and raise TypeError when probed for sequence length. The fake kernel uses immutable shape metadata, including after the runtime closes.

On macOS arm64 with Python 3.11.18 and PyTorch 2.9.1, torch.compile(fullgraph=True, backend="eager") captured the fixed-shape linear_pim operator and matched eager execution for two inputs on both the functional and native libpim backends. Torch warns during capture that it retains the original session-bound handle when deepcopy is unavailable. Coverage applies to operator graph capture; whole-model compilation and Inductor code generation require separate validation.

Run the regression with the same Python interpreter used to build the extension:

```sh
PYTHONPATH=build-python/python NDP_GEM5_PIM_WORKER="$PWD/build/extended-gem5/pim-kernel-worker-fp16" \
    python -m unittest discover -s tests/python -p test_dispatch.py -v
```
