# External Python client and a persistent SE worker

This adds an external-client path without booting Linux. The client and simulator
are separate processes. One TCP connection owns one worker session and its tensor
handles. The existing v2 worker protocol is unchanged.

```text
Mac Python / ndp.Session.connect()
  -> loopback TCP (Docker port forwarding)
  -> Python relay in the container
  -> host FIFOs backing the SE process's stdin/stdout
  -> simulated RISC-V worker -> libpim -> PIM controller
```

This is host-assisted SE I/O, not simulated guest networking. gem5 can block on
host FIFO reads while waiting for the next client request. Host waiting and TCP
latency are not modeled device latency. The dedicated configuration selects the
same 16-channel FP16/FP32 profile and VLEN=256 as the previous tests.

## Two-device decoder, request state and layer transfer

The user confirmed that this suite passed in both SE Atomic and Timing modes.

Start the two endpoints using the terminal A and B commands in the Linear
procedure below. Wait until both are ready. In terminal C run:

```sh
PYTHONPATH="$PWD/build-python/python" python3 tools/gem5/pim_kernel/verify_tp_remote.py --suite decoder
```

The client runs three checks in order on one connection per endpoint:

1. Split a one-layer decoder (4 query heads, 2 KV heads) across both workers.
   Compare the reduced attention output, block output and logits against the
   same checkpoint on rank 0 for a two-token prefill and one decode token.
2. Compare four-token prefill with two chunks of two, including the fifth
   token's decode logits. Interleave two requests, close the first, and compare
   the surviving request with its isolated run.
3. Place one layer on each worker. Compare the activation received by layer 2,
   the final block output and logits with both layers on rank 0.

Comparisons use absolute tolerance 0.005 and relative tolerance 0.03.
Expect `NDP_TP_DECODER_BLOCK_PASS`, `NDP_TP_DECODER_STATE_PASS`,
`NDP_LAYER_TRANSFER_PASS`, then `NDP_TP_DECODER_CLIENT_PASS devices=2`.
Require the PIM trace and both close markers on each endpoint. After Atomic,
restart both endpoints with Timing and repeat the client. The layer transfer
check measures functional correctness across independent simulator instances.

## Two-device tensor-parallel Linear

Start each endpoint in its own terminal from the repository root. Wait for
`NDP_SE_HOST_READY` in terminal A before starting terminal B so the shared
build steps finish in order. Terminal B publishes its container on Mac port 45241.

```sh
# Terminal A
STAGE=STREAM CPU=atomic PORT=45240 ./tools/gem5/pim_kernel/run.sh
```

```sh
# Terminal B
STAGE=STREAM CPU=atomic PORT=45241 ./tools/gem5/pim_kernel/run.sh
```

Once both listeners are ready, run the client in terminal C:

```sh
PYTHONPATH="$PWD/build-python/python" python3 tools/gem5/pim_kernel/verify_tp_remote.py
```

Require `NDP_TP_LINEAR_CLIENT_PASS devices=2` in terminal C and
`NDP_SE_PIM_TRACE`, `NDP_SE_SESSION_CLOSED` and `NDP_SE_TRANSPORT_CLOSED`
in **each** server terminal. Repeat all three commands with `CPU=timing` on
both servers. Each rank runs in an independent gem5 instance.

The client checks an uneven output-feature split against PyTorch Linear.
For an input-feature split, it reads both FP32 partials and then reduces the
original device handles through the production host reduction path. The expected
partials are [80000, 600] and [-80000, 1400]. Their sum [0, 2000] must be
returned in FP16. Two local worker processes passed these checks. The user
confirmed that the same client passed with two SE endpoints in both Atomic
and Timing modes.

## Accepted gate: request-local KV cache isolation

The production decoder gate passed both SE modes. This next test shares one
model, one connection and one device between two live requests. It checks
sequential interleaving, not concurrent execution or thread safety.

Terminal A (repository root, Docker Desktop running):

```sh
STAGE=STREAM CPU=atomic ./tools/gem5/pim_kernel/run.sh
```

After `NDP_SE_HOST_READY`, terminal B:

```sh
PYTHONPATH="$PWD/build-python/python" python3 tools/gem5/pim_kernel/verify_decoder_remote.py --suite isolation
```

After Atomic passes, restart terminal A with `CPU=timing` and repeat terminal B.
Each run requires a fresh endpoint: do not run another client on it first.

Terminal B must print:

```text
NDP_REMOTE_ISOLATION_BASELINES_PASS
... max_absolute_logit_error ...
NDP_SE_REQUEST_ISOLATION_CLIENT_PASS
```

Terminal A must print `NDP_SE_PIM_TRACE`, `NDP_SE_SESSION_CLOSED` and
`NDP_SE_TRANSPORT_CLOSED`. Require both terminal verdicts. Request/PIM counts are
not those of the original two-GEMV client.

The test first executes A `[1, 2, 3]` and B `[4, 5, 6, 7]` independently to obtain
reference logits. It then creates two live requests and checks:

1. A prefills two tokens, B decodes one, then A decodes its third token.
2. A exceeds capacity and is rejected; neither request advances or becomes failed.
3. B continues with its second token.
4. A resets and replays its first token; B continues with its third token.
5. A closes and rejects further use; B completes its fourth token.

There are 15 successful target token evaluations, including the reference runs.
Tokenwise replay must match exactly. A's prefill/decode comparison uses
`atol=0.005, rtol=0.03` because its reference uses tokenwise execution.
All outputs must be finite. Native results had zero error in all seven comparisons.
The references use the same model/backend; this tests state isolation rather than
independently proving decoder mathematics. It does not establish memory-leak
freedom, multiple-device isolation or separate PNM-core execution.

Developer-only native check (no gem5):

```sh
PYTHONPATH="$PWD/build-python/python" python3 tools/gem5/pim_kernel/verify_decoder_remote.py \
  --native-worker build/extended-gem5/pim-kernel-worker-fp16 --suite isolation
```

The user confirmed SE Atomic and Timing acceptance for this isolation suite. Client markers contain `SE` even in native
mode; only a simulator run plus its PIM trace provides SE evidence.

## Accepted gate: production PyTorch decoder adapter

The two-GEMV and production decoder clients passed Atomic and Timing, as reported
by the user. The decoder client
uses the actual `TensorParallelLlamaForCausalLM` adapter with one device, one layer,
hidden width 4, two query heads, one KV head, FFN width 6 and vocabulary 8. The
checkpoint is generated deterministically; no Transformers install or model
download is required. The TP-capable class is used with a single device; this is
not multi-device acceptance.

Terminal A (Mac, repository root):

```sh
STAGE=STREAM CPU=atomic ./tools/gem5/pim_kernel/run.sh
```

After `NDP_SE_HOST_READY`, terminal B:

```sh
PYTHONPATH="$PWD/build-python/python" python3 tools/gem5/pim_kernel/verify_decoder_remote.py
```

After Atomic passes, restart terminal A with `CPU=timing` and run the same client
again. Do not run the old GEMV client on this connection first: each endpoint
serves exactly one session.

Terminal B must print:

```text
NDP_REMOTE_DECODER_PREFILL_DECODE_PASS
NDP_REMOTE_DECODER_RESET_PASS
... max_absolute_logit_error ...
NDP_SE_DECODER_CLIENT_PASS
```

Terminal A must print `NDP_SE_PIM_TRACE`, `NDP_SE_SESSION_CLOSED` and
`NDP_SE_TRANSPORT_CLOSED`. Request and PIM counts will differ from the old
22-request two-GEMV example. Both terminals' verdicts are required.

The client checks seven target token evaluations in one persistent session:

- Two-token prefill followed by one-token decode, including request positions.
- An over-capacity token is rejected without advancing the request position.
- Reset, then the same three tokens one at a time.
- Reset again, then replay the first token with exact agreement to its prior result.

Logits are compared to a local functional-backend `LlamaDecodeBlock`, CPU embedding,
final normalization and LM head. This is an integration oracle with partly shared
implementation, not an independent mathematical proof of Llama. Finite outputs
and `atol=0.005, rtol=0.03` are required for cross-path comparisons; reset replay
requires exact agreement. The observed native logit errors were all zero.

The embedding and LM head execute on the external host. Decoder general arithmetic
executes on the simulated worker CPU and single-row PIM projections on the PIM
controller. This does not establish separate PNM-core execution. The client uses
the production PyTorch request/cache adapter, unlike the earlier embedded protocol
fixtures. It does not validate pretrained checkpoints or multi-device reductions.

For developer-only native verification (no gem5):

```sh
PYTHONPATH="$PWD/build-python/python" python3 tools/gem5/pim_kernel/verify_decoder_remote.py \
  --native-worker build/extended-gem5/pim-kernel-worker-fp16
```

This command passed locally and ends with `NDP_NATIVE_DECODER_RELAY_PASS`.
The user reported both SE Atomic and Timing acceptance, with all three reported
maximum absolute logit errors equal to zero.

## Terminal A: start the SE endpoint

From the NDP repository root on the Mac, with Docker Desktop running:

```sh
STAGE=STREAM CPU=atomic ./tools/gem5/pim_kernel/run.sh
```

The runner builds the target worker and simulator incrementally. It starts one
relay, publishes only Mac loopback port 45240, and prints:

```text
NDP_SE_HOST_READY port=45240 (relay listening; worker handshake pending)
```

This means the relay accepts a client, not that the simulated worker has completed
initialization. The client's handshake is the first readiness check. The relay
waits for terminal B without a connection deadline by default; Ctrl+C cancels.
Request deadlines remain 240 seconds after connection. To bound the initial wait,
set `CONNECT_TIMEOUT=600` in terminal A. Worker exit during the wait is reported
instead of waiting indefinitely. Do not probe the
port with another connection; the endpoint accepts exactly one session.

## Terminal B: run external PyTorch

Use the Python environment containing PyTorch and the previously built NDP binding:

```sh
PYTHONPATH="$PWD/build-python/python" python3 tools/gem5/pim_kernel/verify_remote.py --label NDP_SE
```

If that Python cannot import the binding, use the same Python executable used to
build it. This is the existing remote-client example, now with an SE result label.
It creates one resident weight, evaluates two inputs and checks exact FP16 results:

```text
GEMV [[6.0, 1.0]]
GEMV [[-1.0, -4.0]]
NDP_SE_CLIENT_PASS
```

Terminal A must then report `NDP_SE_PIM_TRACE` with nonzero WrGb, AbMac and RdReg
counts, followed by `NDP_SE_SESSION_CLOSED` and `NDP_SE_TRANSPORT_CLOSED`.
The client verdict proves the numeric comparisons; the relay verdict proves
framing, clean worker exit and target PIM trace presence. Require both.

For Timing use `CPU=timing` in terminal A after Atomic acceptance. For a different
Mac port, use `PORT=45241` in terminal A and `--port 45241` in terminal B.
Do not run other verification jobs concurrently against the shared build cache.

## Failure handling and evidence

Find the current directory through `build/extended-gem5/latest-run.txt`:

- `driver.log`: build/orchestration output.
- `stream.log`: relay readiness, failure or close verdict.
- `worker.log`: simulator stdout/stderr.
- `sim/guest.stderr`: simulated worker diagnostics.
- `sim/pim.trace`, `sim/stats.txt`, `sim/config.ini`: target execution evidence.
- `result.txt`: final orchestration result.

On disconnect, invalid framing or timeout, the relay terminates its owned
worker/simulator and removes its FIFOs. It never retries the uncertain operation
or reconnects using the old handles. Restart a new session explicitly. A valid
worker error reply is forwarded unchanged and can leave the session usable.
Diagnostic output never shares the binary reply stream.

## Validation scope

Native tests passed for two projections using one persistent worker, fragmented
request delivery, invalid frame length, and disconnect cleanup. The dedicated SE
configuration has been checked statically but not run by the assistant. The user
runs SE/FS validation. Native relay success does not establish gem5 FIFO syscall
behavior or real PIM execution; those are the next user acceptance checks.

The preceding FS smoke and SE Atomic/Timing boundary and lifecycle tests passed
according to the user. The user subsequently confirmed both endpoint and external-client verdicts for
Atomic and Timing, with 22 requests and WrGb 32 / AbMac 32 / RdReg 4 in each
reported session. This covers the two-projection fixture, not arbitrary models.
