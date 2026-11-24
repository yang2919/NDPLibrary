# Command transport and codec (F1)

F1 introduces a private synchronous transport boundary and a versioned command
batch format. The public `IBackend`, L2 `Session`, L3 API, layouts and command
builder remain unchanged. No Tensor descriptors are reconstructed by the codec.

## Objects and responsibilities

| Object | Responsibility |
|---|---|
| `NdpBackend` | Own allocation ranges, validate plans, build commands and call a transport |
| `ICommandTransport` | Upload/download addressed bytes, execute ordered commands, shut down |
| `EmulatorTransport` | Own one persistent `Emulator` and forward synchronous calls |
| `wire::encode` | Serialize existing commands into a bounded byte vector |
| `wire::decode` | Validate framing and field encodings, return typed commands without executing them |
| `wire::Limits` | Bound encoded input/output bytes and command count |

The implementation is private under `src/l1/ndp/`: `transport.hpp/.cpp` and
`codec.hpp/.cpp`. The existing factory selects `EmulatorTransport`. It does not
serialize local calls unnecessarily; the codec is available for the F2 IPC path.
There is no socket, remote factory, process manager, new queue or worker in F1.

All calls are serialized by the owner. A successful `submit` means computation
and writes are complete, not merely that metadata was accepted. On return or
exception, no work may continue to access caller buffers. `shutdown` is idempotent;
the emulator adapter releases device state and rejects later I/O/submission with
`SessionClosed`. Destroying the adapter also releases its synchronous device.
Existing backend allocation bookkeeping still supports release after shutdown.

The transport preserves emulator preflight and execution errors. A structural
codec success says nothing about address ranges, memory geometry, valid kernel
shapes or a loaded program. Those remain receiver/emulator responsibilities.
Encoding `LaunchPnmProgram` does not implement a RISC-V executor. Program loading
and unloading remain emulator-specific control operations; they are not added to
this transport or protocol by F1.

## Version 2 command byte format

All integers are unsigned little-endian; records have no implicit padding.
Fields are serialized individually, never by copying C++ objects. This is a
library protocol, not the vendor AiM ISA encoding and not a gem5 Packet format.

### Batch header (16 bytes)

| Offset | Width | Field |
|---|---|---|
| 0 | 4 | Magic: ASCII `NDP1` (`4e 44 50 31`) |
| 4 | 2 | Protocol version: 2 |
| 6 | 2 | Numerical profile: 1, current FP16 extension semantics |
| 8 | 4 | Total byte count including this header |
| 12 | 4 | Number of command records |

An empty batch is valid and contains only the header. Version or numerical-profile
mismatches are rejected; BF16 is not silently substituted. Profile 1 identifies
the existing FP16 storage/FP32 arithmetic rules, not a new floating-point model.

Each command begins with `opcode:u16`, `reserved:u16` (must be zero), and
`payload_bytes:u32`. Records are executed in vector order only after a caller
explicitly submits the decoded batch. The decoder requires exact batch and record
lengths; no trailing bytes are accepted.

### Field types

- `Address`: `space:u8`, `offset:u64`. Spaces are Shared=1, Gpr=2, Scratchpad=3.
  GlobalBuffer=4 is a Transfer destination only. Its byte offset is channel × gb_bytes + local_offset.
- `Operand`: `Address`, `elements:u64` (17 bytes).
- All `size_t` command fields become `u64`; a receiver rejects values that cannot
  fit its local `size_t`. Channels, banks masks, program IDs and offsets are `u64`.
- Boolean fields are one byte, exactly 0 or 1.
- Float parameters are four bytes containing the IEEE-754 binary32 bit pattern.
  Signed zero and NaN payloads are preserved; numerical validity is checked by the
  execution layer, not the codec.
- Operand/index arrays have a `u32` element count followed by entries in order.

### Payloads, in field order

| Opcode | Command | Payload |
|---|---|---|
| 1 | Transfer | source Address, destination Address, bytes u64 |
| 2 | InitializeMac | channels u64, banks u64, gpr_offset u64 |
| 3 | Unused in v2 | — |
| 4 | MacAllBanks | channels u64, row u64, column u64, bursts u64 |
| 5 | ReadMac | channels u64, gpr_offset u64 |
| 6 | LaunchPnmProgram | program u64, arguments Address, argument_bytes u64, stack Address, stack_bytes u64, instruction_limit u64 |
| 7 | PnmCompute | kernel u16, input count u32, input Operands, output Operand, rows u64, width u64, inner u64, transpose_b bool, factor f32, epsilon f32, index count u32, indices u64[] |
| 8 | PnmRelayout | source Address, destination Address, rows u64, columns u64, banks u64, tile_elements u64, row_bytes u64, pack bool |
| 9 | CopyBytes | source Address, destination Address, bytes u64 |
| 10 | FillBytes | destination Address, bytes u64, value u8 |

PNM kernel IDs: Matmul=1, Add=2, Mul=3, RMSNorm=4, Softmax=5, SiLU=6, Copy=7,
Gather=8, Scale=9, Transpose=10. Wire IDs are explicitly mapped independently of
C++ enum values and variant order. Future incompatible field changes require a
protocol version change.

Default limits are 8 MiB per batch and 65,536 commands. Callers may supply other
limits, but wire lengths/counts remain `u32`. Nested array counts must fit the
remaining payload before allocation. Invalid input returns `InvalidArgument`;
allocation failures return `OutOfMemory`. Decode failure returns no partial batch
and does not change device state.

This format has no checksum, authentication, request epoch, sequence number or
completion record. Valid-looking payload corruption is not detectable by this
codec. F2 now adds [metadata-only IPC framing and identity](gem5-f2.md) around this
codec, with partial I/O handling and a fixed receiver geometry. F3 now adds [upload/download and execution-completion messages](gem5-f3.md). An executing endpoint must
validate geometry and profile and distinguish receipt from execution completion. Do not use the codec alone as a remote backend.

## Native verification

From the repository root, without Docker or gem5:

```sh
cmake -S . -B build -DNDP_BUILD_PYTHON=OFF -DBUILD_TESTING=ON
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

Run just the new suite when diagnosing a codec or transport failure:

```sh
ctest --test-dir build -R '^ndp_transport_contracts$' --output-on-failure
```

The executable prints `Command transport and codec tests passed`. The full suite
contained 19 tests at F1 completion (21 after F2). Existing backend and decoder tests run through the
new transport without API changes. The new suite checks literal wire fixtures for
all ten commands, all PNM IDs, float bits, mixed batches, header/payload truncation,
invalid fields/counts, byte limits, PNM/PIM numerical results after decoding,
persistent memory, preflight rejection and shutdown behavior.

Native ASan/UBSan verification:

```sh
cmake -S . -B build-f1-sanitize -DNDP_BUILD_PYTHON=OFF -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer'
cmake --build build-f1-sanitize -j 4
ctest --test-dir build-f1-sanitize --output-on-failure
```

## FS handoff

F1 is a native implementation milestone. There is no F1-specific FS executable
or guest transport to run yet. The [F0 harness](gem5-fs.md) remains available for
user-run baseline checks, but rerunning it does not validate this codec or its
transport. F2 now supplies the [guest sender/receiver and user-run procedure](gem5-f2.md),
including the expected metadata report and failure evidence. FS execution and acceptance remain owned by
the user. No gem5 build, download, boot or polling is part of F1 validation.
