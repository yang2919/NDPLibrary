# Multi-token PNM prefill

The TP Llama adapter can append a contiguous prompt chunk in one model call.
For chunks with more than one token, its seven projections per layer run as
PNM Linear/GEMM operations. Single-token calls continue to use PIM GEMV.
Attention remains a composition of existing causal query/head operations;
this is not a fused or batched attention kernel.

```python
with model.new_request(max_tokens=128) as request:
    logits = model.prefill(prompt_ids, request=request)
    # prompt_ids: CPU int64 [1,T]; logits: CPU FP32 [1,T,vocab_size].
    next_id = logits[:, -1].argmax(-1, keepdim=True)
    next_logits = model(next_id, request=request)  # Existing PIM decode path.
```

prefill appends to the selected history; it does not reset it. The chunk must
fit the remaining request capacity. Without request=, the default history is
used. A one-token prefill chunk is valid and uses the decode projection path.
Only one sequence is represented by each chunk; [batch,T] with batch > 1 is
not supported.

For greedy generation, chunking is explicit and defaults to the previous
one-token behavior:

```python
output_ids = model.generate(prompt_ids, max_new_tokens=8, prefill_chunk_size=16)
```

The prompt is split into chunks no larger than prefill_chunk_size. A final
one-token remainder uses PIM. Generated tokens always use the single-token
decode path. Larger chunks increase temporary activation and logits memory;
select a size compatible with backend tensor/frame limits. There is no
automatic memory-budget-based chunk sizing.

## Projection, layout and cache behavior

- Output-sharded Q/K/V and gate/up weights keep the same partitions as decode.
  Their PNM projections return FP16 matrices [T,local_output_width].
- Input-sharded o_proj and down_proj return FP32 [T,hidden_size] partials. L2
  reduces them on the host in device order and converts once to FP16.
- PNM uses the dense representation already retained by prepared weight shards;
  no separate full-model prefill weights or per-token weight uploads are added.
- RoPE gathers the correct table entries for every position from the request's
  starting position through start+T-1, independently for each local head.
- Each local KVCache receives one append containing [T,local_kv_heads,head_dim].
  For query t, causal attention reads only through start+t; future rows from the
  same appended chunk cannot contribute.
- RMSNorm normalizes each row independently and repeats resident gamma values
  across the rows. Residuals, activations and interstage transfers preserve the
  token dimension.
- A successful chunk advances the request by T positions. If a later operation
  fails after some layers have appended, the entire request must reset. A
  capacity rejection happens before model execution and does not poison history.

DistributedRow now accepts FP16 sharded/replicated or FP32 partial matrices with
matching positive row counts. Its rows property exposes that count. The feature
partition remains along the last axis. DeviceGroup.all_gather_rows assembles
feature shards row by row and broadcasts the complete matrix. Existing
single-row callers are unchanged.

## L2 and backend changes

`Session.linear_partial(x, weight, engine=Engine.PIM)` keeps the previous default.
Selecting Engine.PNM permits a bias-free multi-row Linear with FP16 inputs and
FP32 output. General elementwise operations still require FP16. The functional
and generic NDP backends continue to reject unsupported FP32 output kernels.

The native gem5_pim backend binds the PNM output as role P and sends the existing
A dense command with Linear kind and its M/N/K dimensions. Its payload framing
and command IDs are unchanged. The existing native FP32-capable worker stores
that role without FP16 output rounding. Host reductions operate on the complete
partial matrix. No new custom ISA is defined by this change.

The ELF/capture path still rejects FP32 partial execution. The full TP prefill
path therefore remains a native-worker capability until the target executor
and ABI support PNM FP32 outputs. No extended gem5 FS execution is claimed.

## Scheduler integration

Both schedulers accept prefill_chunk_size, defaulting to 1:

```python
with ndp.serving.GenerationScheduler(
    model, max_active_requests=2, prefill_chunk_size=16,
) as scheduler:
    request = scheduler.submit(prompt_ids, max_new_tokens=8)
    scheduler.run()
```

PipelineGenerationScheduler accepts the same constructor argument together
with its source model, stage groups and placement. One chunk per request can
be in flight across the stages. Stage boundaries transfer the complete hidden
matrix for that chunk. Generated-token dispatch remains one token at a time.

GenerationStep.input_tokens reports the number of input tokens processed in
that event (for pipeline events, by the completed stage). A prefill event may
consume multiple prompt tokens but emits at most the first generated token,
and only after the prompt's final chunk reaches the output. Earlier prompt
logits are not sampled. The sequential scheduler alternates by chunk rather
than by individual prompt token when chunk size is greater than one; this is
an explicit fairness/throughput tradeoff. Decode scheduling remains token-wise.

## Native verification

```sh
cmake --build build-python
ctest --test-dir build-python --output-on-failure
PYTHONPATH=build-python/python python examples/python/llama_prefill.py \
    /path/to/SmolLM2-135M --worker build/pim-kernel-worker-fp16
```

On the pinned SmolLM2-135M checkpoint (30 layers, two workers), the four-token
prompt [1,42,30,198] produced exactly 210 PNM Linear commands per device, all
with M=4, and zero PIM GEMV commands. The subsequent one-token decode produced
210 PIM GEMVs per device and zero PNM Linear commands. Counts were read from
the actual worker command stream, excluding preparation.

| Comparison | Maximum absolute logit difference |
|---|---|
| Four-token prefill vs token-by-token prompt | 0.109375 |
| First decode after those histories | 0.03125 |
| Two chunks of two tokens vs one four-token chunk | 0 |

The two checked greedy next-token choices matched. These measurements retain
FP16/accumulation differences rather than requiring bitwise equality between
PNM GEMM and PIM GEMV. They are a short functional check, not an accuracy or
performance benchmark.

Regression tests cover nonzero cache offsets, causal-prefix invariance,
chunk/decode continuation, request capacity and failure/reset, three-row FP32
partials of +80,000/-80,000 reduced before FP16 conversion, matrix all-gather,
and chunk progression through both schedulers.

Final validation covered all 26 native CTest entries. The first combined run
and an initial Python retry hit elapsed-time limits; a diagnostic Python run
completed with only the sandbox-denied localhost test failing. The final
permission-enabled Python CTest rerun passed in 14.23 seconds. No timeout or
numerical tolerance was increased to obtain that result.
