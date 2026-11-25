# Sequential request scheduling

GenerationScheduler manages greedy requests on one shared-weight
TensorParallelLlamaForCausalLM. It supports both one TP group and the existing
multi-group layer placement. All model calls run sequentially on the creating
Python thread. The active-request limit bounds resident histories, not concurrent
GPU-style launches or overlapping pipeline stages.

```python
with ndp.serving.GenerationScheduler(model, max_active_requests=2) as scheduler:
    first = scheduler.submit(prompt_a, max_new_tokens=8)
    second = scheduler.submit(prompt_b, max_new_tokens=8)
    third = scheduler.submit(prompt_c, max_new_tokens=8)
    scheduler.run()
    for request in (first, second, third):
        if request.status == "completed":
            print(request.token_ids)  # CPU int64 prompt plus generated tokens.
        else:
            print(request.status, request.error)
```

Each prompt is a nonempty CPU int64 tensor [1,prompt_length]. submit copies the
IDs, validates vocabulary and capacity, and returns a GenerationRequest handle.
Request IDs are monotonically increasing within a scheduler. No device cache
is allocated merely by queueing a request.

## Scheduling policy

1. Admit FIFO waiting requests until max_active_requests histories are active.
2. Select the oldest ready request and process exactly one input token.
3. If it is still live, append it to the end of the ready queue.
4. If it completed or failed, close its DecodeRequest immediately. Another
   waiting request can enter at the next step.

Prompt tokens and generated tokens use the same one-token model forward path.
Thus a long prompt cannot consume multiple consecutive model calls while
another active request is ready. This initial prefill is token-by-token; it is
not a multi-token PNM GEMM or batched attention implementation.

The last prompt-token call also produces the first generated token. Later calls
consume the previously generated token and emit the next. The final emitted
token is not consumed into KV history, matching model.generate. The scheduler
reserves prompt_length + max_new_tokens - 1 KV entries for that request, capped
by the model's configured capacity. Actual allocation remains lazy until the
first model call. Admission is limited by request count, not a byte-level memory
budget; differing prompt lengths and limits can require different amounts of
memory. Waiting prompts still consume ordinary host memory.

## Stepping, streaming and cancellation

```python
with ndp.serving.GenerationScheduler(model, max_active_requests=2) as scheduler:
    first = scheduler.submit(prompt_a, max_new_tokens=8)
    second = scheduler.submit(prompt_b, max_new_tokens=8)
    while scheduler.pending:
        event = scheduler.step()
        if event.token_id is not None:
            print(event.request_id, event.token_id)
        if event.request_id == first.request_id and event.token_id is not None:
            scheduler.cancel(second)
```

`step()` performs at most one model call and returns a frozen GenerationStep,
or None when no work remains. `run()` drains pending work by repeatedly calling
step; it returns None. To stream output, use step and inspect its events.

| Event field | Meaning |
|---|---|
| request_id | Scheduler-local handle ID |
| phase | prefill or decode for the consumed input; admission on an admission failure |
| token_id | Newly emitted ID, or None when the prompt is not yet complete |
| status | Request state after this step |
| error | Failure text or None |

Request status is queued, prefill, decode, completed, cancelled or failed.
The handle also exposes done, generated_tokens, error and token_ids. token_ids
returns a fresh tensor, so callers cannot mutate the scheduler's internal token
list. Terminal handles retain CPU results and status, but no device history.
The scheduler does not keep an archive of completed requests.

Cancel may target queued or active requests between step calls. It releases
cache references and marks the handle cancelled. Cancelling a terminal handle
again is harmless. Cancellation does not interrupt an in-flight model call or
emit a separate step event. Closing the scheduler cancels all pending requests;
the context manager guarantees this on scope exit. Scheduler calls must not be
concurrent or reentrant.

## Completion and failures

- max_new_tokens=0 completes immediately with the original prompt and no KV cache.
- EOS uses the source model setting by default. Pass an integer or list to
  choose explicit IDs, or [] to disable EOS stopping. The emitted EOS is kept.
- EOS or the requested generation length marks a request completed and releases
  its cache immediately.
- Invalid submissions raise without entering the queue.
- Admission or execution exceptions mark the affected request failed and close
  its cache. The error is stored as text without retaining a traceback, which
  could otherwise retain tensor references. Other requests remain scheduled.
- run does not raise per-request execution errors; inspect the returned handles.
  Interruptions such as KeyboardInterrupt propagate after cancelling the selected
  request; context-manager cleanup cancels the remaining work.
- A failed shared backend may cause other requests to fail too. There is no
  automatic retry, reconnect or request replay.

The scheduler uses only explicit DecodeRequests and does not reset the model's
default history or other manually managed requests. Close the scheduler before
reloading model weights or changing placement; queued work must not silently
switch checkpoints. Sampling, priorities, paged-cache admission, continuous
batching and overlapping pipeline execution are outside this implementation.

## Verification

```sh
cmake --build build-python
ctest --test-dir build-python --output-on-failure
PYTHONPATH=build-python/python python examples/python/llama_scheduler.py \
    /path/to/SmolLM2-135M --worker build/extended-gem5/pim-kernel-worker-fp16
```

The full-model example submits three prompts to a scheduler with two active
slots, compares their generated IDs against independent model.generate calls,
and checks that completed requests release all KV allocations. Native tests
cover round-robin order, queued and active cancellation, EOS, zero-token requests,
invalid capacity/IDs, failures with a healthy peer, admission failure,
interruption and reentrant-call rejection. They also verify that new work can
be submitted after draining the queue.


On the pinned SmolLM2-135M checkpoint (30 layers, two native workers), prompts
[1,42], [30] and [198,504] each requested two new tokens with EOS disabled.
The input-token execution order was [0,1,0,1,0,2,2,2]. All three generated
sequences exactly matched independent generation. Request 2 waited until an
active slot became free, while requests 0 and 1 alternated.

L2 live allocations before submission and after all completions were identical:
327,523,584 bytes on device 0 and 301,824,000 on device 1. Submission added no
device allocation. This verifies terminal KV reclamation, not process RSS or
throughput. The native regression suite passed 26/26; no gem5 FS execution was
performed.


## Overlapping pipeline stages

For concurrent progress across stages, use the separate
[PipelineGenerationScheduler](pipeline-execution.md). It constructs its own
thread-owned stage adapters from a source model and backend options. Its step
events include a stage ordinal and represent a stage completion; sequential
GenerationScheduler events have stage=None and represent a whole input-token
call. This distinction preserves the existing sequential API and Session thread
ownership contract.


## Prompt chunks

Set prefill_chunk_size on the constructor to process multiple prompt tokens in
one turn. The default 1 preserves the token-wise policy above. Larger chunks
use PNM GEMM and turn round-robin admission into chunk-wise prefill scheduling;
decode remains one token per turn. GenerationStep.input_tokens reports the
actual count. See [prefill](prefill.md) for the API and numerical checks.
