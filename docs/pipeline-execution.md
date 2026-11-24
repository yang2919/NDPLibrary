# Overlapping tensor-parallel stages

PipelineGenerationScheduler runs each contiguous group of decoder layers on
its own host thread. That thread creates and exclusively uses the group's
Sessions, prepared weights and request-local KV histories. Different requests
can occupy different stages at the same time. Each stage still executes its
own token calls sequentially and uses TP inside its device group.

```python
# source: CPU FP16, eval-mode Hugging Face LlamaForCausalLM.
options = {"backend": "gem5_pim", "worker": worker_path}
n = len(source.model.layers)
placement = [0]*(n//2) + [1]*(n-n//2)
with ndp.pipeline.PipelineGenerationScheduler(
    source,
    device_groups=[[options, options], [options, options]],
    placement=placement, max_tokens=128, max_active_requests=2,
) as scheduler:
    a = scheduler.submit(prompt_a, max_new_tokens=8)
    b = scheduler.submit(prompt_b, max_new_tokens=8)
    scheduler.run()
    for request in (a, b):
        if request.status == "completed":
            print(request.token_ids)
        else:
            print(request.status, request.error)
```

Unlike the sequential GenerationScheduler, this constructor takes the source
checkpoint model, not an already connected NDP adapter. Existing Sessions
cannot migrate across threads. device_groups must contain backend-option
dictionaries; Session and DeviceGroup objects are rejected. Every dictionary
occurrence opens a separate connection on its owning stage thread. If supplying
external worker endpoints or descriptors, they must represent independent
connections. The pipeline owns and closes these connections.

At least two nonempty groups are required. placement has one ordinal per source
layer, uses every group, and must be contiguous in increasing stage order:
[0,0,1,1] is valid; [1,0] and [0,1,0] are not. The sequential staged model API
continues to support its more general placement rules. Reconfiguration and
checkpoint reload require closing and rebuilding the pipeline.

## Ordering and communication

The coordinator keeps at most one token in flight per request. A token must
complete all stages before that request can start the next prompt/decode token.
This preserves causal KV order in every layer. Across requests, stage queues
are FIFO and the coordinator advances whichever stage completion is available.
Waiting requests are admitted FIFO up to max_active_requests; this is not a
strict round-robin completion order or a byte-level memory budget.

For example, after A's token leaves stage 0, stage 0 may process B's token while
stage 1 processes A's token. With only one active request, there is no such
overlap. Prefill remains one token per call, with no microbatch fusion.

| Component | Execution owner |
|---|---|
| CPU embedding | Coordinator thread |
| TP layers and their cache | Assigned stage thread and its Sessions |
| Within-stage broadcast/reduction | Existing host-staged L2 operations on that stage thread |
| Stage boundary | Previous root downloads FP16 hidden row; next root uploads it |
| Final RMSNorm | Last stage's root device |
| CPU LM head | Last stage thread, using its frozen checkpoint copy |
| Greedy argmax and result assembly | Coordinator thread |

Boundary tensors are immutable CPU tensors passed through futures. No native
NdpTensor or Session is operated on from another stage thread. The L1 packet
protocol and PNM/PIM numerical operations are unchanged. There is no device-side
peer transfer, cross-stage all-reduce or new ISA command.

## Progress, termination and cleanup

submit, run, cancel and result handles follow the sequential scheduler's token,
capacity and EOS conventions. A zero-new-token request completes immediately.
An execution failure marks only its request failed and other requests remain
scheduled, subject to the health of shared backend connections. There is no
automatic retry or replay.

`step()` waits for and processes one completed **stage**, not a whole input token.
It returns GenerationStep with stage set to the zero-based stage ordinal.
Intermediate-stage events have token_id=None. A newly generated ID appears only
after the last stage has processed the final prompt token or a decode token.
Sequential GenerationScheduler events keep stage=None. Use a step loop to stream
IDs or submit/cancel work between calls. Background stages can continue running
between coordinator calls, but forwarding to the next stage requires step/run.

Cancellation does not preempt a native call. It removes the request from future
routing, waits for already queued stage work, and closes that request's history
on each owning thread. Terminal status is set after these cleanup calls finish.
Cleanup may therefore wait behind other queued work. Completion and failure use
the same release ordering. A hung backend still needs its own timeout/recovery;
this scheduler provides no hard cancellation deadline.

Use the context manager or explicit close. Shutdown drains in-flight work,
releases histories and prepared handles on their owners, closes Sessions, and
joins all stage threads. Initialization failures also clean up stages already
created. Coordinator interruption closes the pipeline instead of leaving a
request without a tracked future. All public scheduler operations must be on
its creating thread and must not be reentrant. Final shutdown errors are
reported after attempting all stage cleanups.

## Measurement and verification

```sh
cmake --build build-python
ctest --test-dir build-python --output-on-failure
PYTHONPATH=build-python/python python examples/python/llama_pipeline.py \
    /path/to/SmolLM2-135M --worker build/pim-kernel-worker-fp16
```

The example compares two independent generation requests with the existing
sequential two-group adapter, then checks that stage execution intervals for
different requests overlap. Each request emits two new tokens. Both paths use
the same TP degree and layer placement.

Optional record_timing=True records request ID, input-token index, stage,
and start/end perf_counter timestamps. timings returns a copy of these records.
Recording is disabled by default because an unbounded trace costs host memory.
These intervals span host-side stage calls, including native waits/transfers
and the final CPU head. They are evidence of overlapping host execution, not
simulated cycles, individual hardware utilization or throughput speedup.

memory_stats() is available only when no requests are active. It asks each
owning thread for per-device live/peak L2-managed bytes. Peaks include preparation
and are cumulative; CPU checkpoint copies, process RSS and all backend internal
allocations are not included.

### Observed native results

On the pinned SmolLM2-135M checkpoint: 30 layers, two stages of 15 layers, two
workers per stage, max_tokens=3 and two active requests. Prompts [1,42] and
[30,198] produced [1,42,198,198] and [30,198,198,504], exactly matching the
sequential staged reference. Twelve stage calls completed. Five pairs of
cross-request/cross-stage host intervals overlapped, with a maximum intersection
of about 1.115 seconds in this run. This duration is machine/run dependent.

| Stage | Device | Prepared live bytes | Live bytes after all requests completed |
|---|---|---|---|
| 0 | 0 | 163,756,800 | 163,756,800 |
| 0 | 1 | 150,908,160 | 150,908,160 |
| 1 | 0 | 163,759,104 | 163,759,104 |
| 1 | 1 | 150,908,160 | 150,908,160 |

Native tests use a two-party barrier between stage 0 of request B and stage 1
of request A: this cannot pass if the stages share one serial execution thread.
They also check per-request token/stage order, stage failures with a healthy
peer, cancellation, cache reclamation, EOS, initialization failure, interruption
and thread shutdown. These results use native workers, not extended gem5 FS.

The complete native CTest suite passed 26/26. The pipeline-specific tests were
also rerun after shutdown-cleanup hardening and passed all four cases.


## Multi-token prefill

The optional prefill_chunk_size constructor parameter defaults to 1. A larger
value passes prompt matrices through the same ordered stages, using PNM GEMM
projections and bulk KV appends. The one-flight-per-request invariant then
applies to a chunk. Events and optional timing records include input_tokens;
generated tokens remain single-token PIM decode. See [prefill](prefill.md).
