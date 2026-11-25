# F2: metadata-only guest IPC

Status (2026-11-05): implementation and native macOS validation passed. **The user has reported FS validation complete.** No gem5 build, download, boot or polling was performed
by the assistant for this milestone. The guest cross-compilation command is provided for the user;
it has not been executed by the assistant.

## Execution model

`ndp_metadata_probe` creates an `AF_UNIX/SOCK_STREAM` socket pair and forks a
receiver process. The parent is the sender. In FS, both processes and both socket
endpoints live inside the simulated Linux guest. There is no socket connection
from the Mac into gem5, pathname socket, background daemon or new SimObject.

The sender uses the existing F1 command codec. The receiver:

1. Reassembles a bounded frame across partial reads.
2. Checks framing, payload CRC, epoch and request sequence.
3. Decodes the F1 batch and checks command metadata ranges against its fixed spec.
4. Re-encodes the decoded commands and returns the bytes with `Accepted`.

The sender requires exact response identity and byte equality. A message being
written to a socket is not proof of receipt. Conversely, an `Accepted` response
proves metadata validation only: no command emulator is created, no device memory
is allocated, no arithmetic is executed, and no output Tensor is marked ready.
The probe does not implement `ICommandTransport` or connect to `Session`.

The small fixture contains PIM setup/MAC/read commands, PNM Add and relayout,
transfer/copy/fill commands. The large fixture appends 300 fill descriptors to
exceed 4 KiB. These are explicit command fixtures, not a captured decoder run.
The receiver does not rebuild Tensor or Layout objects or choose a tile schedule.
A socket message does not correspond to one gem5 memory Packet. F2 establishes
software IPC delivery, not NDP controller delivery or a hardware timing model.

## Protocol and checks

The outer header is 32 bytes, little-endian, with no padding:

| Offset | Width | Meaning |
|---|---|---|
| 0 | 4 | ASCII `NIP1` |
| 4 | 2 | Envelope version 1 |
| 6 | 2 | Request=1, Accepted=2, Rejected=3 |
| 8 | 8 | Connection epoch |
| 16 | 8 | Request sequence |
| 24 | 4 | Payload byte count, maximum 8 MiB |
| 28 | 4 | CRC-32 of payload (reflected polynomial 0xedb88320, initial/final XOR 0xffffffff) |

Request payloads are [F1 batches](command-transport.md). Accepted responses contain
the canonical re-encoded batch; rejected responses have an empty payload. CRC is
an accidental-corruption check, not authentication. Header identity is validated
separately. Bad framing, CRC, truncation or I/O errors terminate the connection.
A well-framed request with invalid command metadata receives `Rejected`.

Sequences start at 1. A request matching the expected epoch/sequence consumes its
sequence even if its command metadata is invalid. Duplicate, out-of-order and
wrong-epoch requests are rejected without advancing the expected sequence. The
probe uses a fixed epoch for its deterministic fixture, not a production session
identity or a reconnect protocol. There is one outstanding request at a time.

The receiver spec is fixed for this probe: channels=1, banks=1, rows=32,
row_bytes=32, gb_bytes=32, gpr_bytes=256, scratchpad_bytes=256. This is not a
configurable remote backend or negotiated geometry. The F1 payload already carries
the numerical-profile ID. Future executing integration must agree on geometry and
capabilities before accepting workload commands.

Metadata checks cover explicit address ranges, FP16 operand spans, channel/bank
mask bounds, MAC row/GB bounds and overflow-safe relayout byte spans. They are not
full execution validation: kernel dimensions, initialized MAC/GB state, aliasing,
program availability and numerical semantics remain execution-layer checks for
F3. Zero-length or otherwise semantically invalid operations may still be
structurally accepted; acceptance never means they are executable.

Each send/receive frame has an I/O deadline covering its header and payload.
Partial transfers, EINTR and temporary backpressure are handled. Peer disconnects
and timeout fail explicitly; no automatic resend occurs. SIGPIPE is suppressed on
both macOS and Linux. The probe requests chunks of at most 17 bytes to exercise
fragmentation independent of kernel buffering. Its timeout is 30 seconds per
frame (guest time in FS). The outer FS runner separately bounds host run time.

## Native checks

From the repository root:

```sh
cmake -S . -B build -DNDP_BUILD_PYTHON=OFF -DBUILD_TESTING=ON
cmake --build build -j 4
ctest --test-dir build --output-on-failure
./build/ndp_metadata_probe
```

The suite contained 21 tests at F2 completion (23 after F3). To run only F2 checks:

```sh
ctest --test-dir build -R '^metadata_' --output-on-failure
```

Native tests cover separate sender/receiver processes, multi-frame persistence,
exact byte echoes, fragmented transfers, duplicate/out-of-order/stale requests,
unknown opcodes and invalid ranges, malformed/truncated/oversized frames, CRC
corruption, wrong acknowledgements, read/write deadlines and disconnects.
Existing 19 tests remain passing. ASan/UBSan also passed; see the sanitizer build
commands in [F1 documentation](command-transport.md). The guest source subset was
also compiled and executed directly with the native compiler to check that it
links independently of the full library. Native macOS success is not Linux/FS
verification.

## User validation

Follow the standalone [F2 user verification guide](stock-gem5-verification.md)
for prerequisites, execution commands, expected output and failure reporting.
The user has reported completion of this FS verification.

## Files and next boundary

- `src/l1/gem5/metadata.hpp/.cpp`: bounded framing, probe/receiver and range checks.
- `tools/gem5/guest/metadata_probe.cpp`: deterministic two-process fixture.
- `tests/metadata_test.cpp`: malformed-input and failure-boundary checks.
- `tools/gem5/fs/run.sh`, `run-container.sh`, `boot.py`: user-run stage selection.

F3 now adds [persistent emulator execution](gem5-f3.md) and upload/download request semantics.
F2 intentionally supplies neither data transfer into emulated device memory nor
execution completion records. Its Accepted/Rejected responses must not be reused
as successful `IBackend::execute` results.
