# F3: persistent emulator execution

F2 FS validation was reported complete by the user. F3 adds an executing receiver
and a synchronous client, while preserving the F2 metadata-only probe. Native
validation has passed; **F3 FS verification has been reported complete by the user**. See the
standalone [user verification procedure](stock-gem5-verification.md).

## Responsibilities

`execution::Client` serializes upload, download and submit requests over a borrowed
Unix socket. `execution::serve` owns one existing emulator transport per connection.
The emulator retains shared memory, GPR, scratchpad, Global Buffer and MAC state
across requests. Commands are decoded using F1 and executed by the existing
emulator; the receiver does not reconstruct Tensor metadata or choose kernels.

The standalone guest probe forks a receiver, performs multiple requests, and
compares downloaded results with both explicit numerical expectations and a
separate local emulator. Both processes execute inside Linux in an FS run. This
is software emulation on the simulated CPU, not PNM/PIM hardware simulation.

F3 does not connect this client to `Session` or replace the public NDP backend
factory. The client deliberately does not implement `ICommandTransport`: uncertain
completion still needs process ownership, termination/reaping and buffer lifetime
integration in F4. The socket is borrowed and must be closed by its owner. The
probe owns its child and terminates/reaps it on failure.

## Execution protocol

The existing 32-byte NIP1 framing, payload CRC and 8 MiB payload limit are reused.
Three explicit frame kinds are added: ExecuteRequest=4, ExecuteComplete=5 and
ExecuteError=6. F2's Accepted=2 is never accepted as execution completion. An older
F2-only reader rejects these new kinds instead of treating them as metadata.

Epoch/sequence fields must match. Sequences begin at 1 with Hello and advance for
every correctly identified request, including requests that fail execution.
Identity violations terminate the receiver after an error response; malformed
framing terminates the connection. There is one outstanding request at a time.
The probe uses a deterministic epoch, not a production reconnect/session service.

Each request payload starts with a one-byte operation:

| ID | Operation | Remaining request payload | Successful response payload |
|---|---|---|---|
| 1 | Hello | codec version u16, numeric profile u16, mapping ID u16, seven geometry u64 fields | Exact Hello payload |
| 2 | Upload | space u8, offset u64, byte count u64, raw data | Operation byte only |
| 3 | Download | space u8, offset u64, byte count u64 | Operation byte followed by raw data |
| 4 | Submit | F1 command batch | Operation byte only |
| 5 | Close | None | Operation byte only |

Integers are little-endian with no padding. Space IDs are Shared=1, Gpr=2,
Scratchpad=3. Hello requires codec version 2, FP16 profile 1 and mapping ID 1
(`[row][channel][bank][byte]`). Geometry fields, in order, are channels, banks,
rows, row_bytes, gb_bytes, gpr_bytes and scratchpad_bytes. The receiver compares
all fields with its configuration before creating the emulator. The Hello payload
is 63 bytes. A later Hello is rejected as an invalid operation.

ExecuteComplete is sent after the emulator call returns and its writes are
visible. Close completion follows device teardown. ExecuteError contains the
operation byte, error ID u16 and at most 256 message bytes. Error IDs are explicit:
InvalidArgument=1, UnsupportedOperation=2, OutOfMemory=3, BackendFailure=4. Other
internal errors map to BackendFailure. This acknowledgement describes a completed
failed operation; it does not promise rollback of arbitrary execution failures.
The existing whole-batch preflight behavior is preserved.

Maximum upload size is 8 MiB minus 18 descriptor bytes. Maximum download size is
8 MiB minus its response operation byte. A Submit's F1 batch must fit in 8 MiB
minus one byte, with at most 65,536 commands. Oversized requests fail; F3 does not
automatically split or retry them. Downloads are copied into caller storage only
after a complete, correctly identified response with the expected size.

A transport failure, timeout, wrong response kind/identity or malformed completion
faults the client. Further requests and close fail locally, without retransmission.
A timeout does not prove remote quiescence. The owner must stop/reap the receiver
before assuming remote work has ceased. A valid ExecuteError allows subsequent
requests because the receiver has completed that failed operation. Successful
close is idempotent on the client.

Program loading and real RISC-V execution remain unsupported by this protocol.
PNM requests use the existing functional kernels; F3 does not silently replace an
ELF launch with another operation.

## Native verification

```sh
cmake -S . -B build -DNDP_BUILD_PYTHON=OFF -DBUILD_TESTING=ON
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

The suite now contains 23 tests. Run F3 checks alone with:

```sh
ctest --test-dir build -R '^execution_' --output-on-failure
./build/ndp_execution_probe
```

Coverage includes:

- Persistent PIM MAC accumulation across two submits: 32 then 64.
- PNM Add consuming the resident PIM result: 65, without re-uploading that result.
- PNM 2x2 GEMM with explicit expected outputs 19, 22, 43 and 50.
- Relayout round-trip, shared/GPR/scratchpad transfers, copy and fill.
- Invalid batch preflight preserving existing data, invalid download preserving
  caller storage, and valid operations after acknowledged errors.
- Geometry mismatch, invalid identity, malformed/oversized requests, wrong response
  kind/sequence/length, timeout faulting, close and child reaping.

The same fixture is run through a local emulator and remote receiver, with exact
FP16 byte comparison. It is a primitive-level fixture, not a decoder or Session
lifetime test. General regression and ASan/UBSan suites passed natively on macOS.
The guest source subset was also linked and executed with the native compiler;
this does not establish Linux cross-compilation or FS success.

## Files and F4 boundary

- `src/l1/gem5/execution.hpp/.cpp`: client, execution protocol and persistent receiver.
- `tools/gem5/guest/execution_probe.cpp`: two-process numerical/state fixture.
- `tests/execution_test.cpp`: negotiation, malformed requests and client fault tests.
- `tools/gem5/fs/run.sh`, `run-container.sh`, `boot.py`: user-run `STAGE=F3` support.

F4 implements process ownership and connects the execution client to Session's
blocking completion, dependency and lifetime contracts; see
[F4 Session integration](gem5-f4.md). F5 remains the guest LLM decoder validation.
FS execution is owned by the user for both stages.
