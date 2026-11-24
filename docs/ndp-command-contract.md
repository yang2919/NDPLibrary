# NDP command contract and implementation status

Status: A1–A6 implemented with A4b changed to functional PNM emulation. Hardware
encoding, RISC-V instruction execution, and timing simulation remain out of scope.
The decisions below were confirmed by the user. Detailed command, ABI, and
arithmetic rules are implementation proposals, not verified hardware contracts.
Scope: one NDP device extending the AiM submission path with PNM commands.

## Confirmed decisions

- Use FP16 first. BF16 support is a subsequent task. This is an AiM-derived
  extension, not the numerical behavior of the BF16 device in the Hot Chips paper.
- Submit PIM, PNM, and transfer commands through one ordered device interface.
- A4b now uses explicit functional PNM commands, as requested after A4a. Kernels
  execute on emulator-owned memory using address/size/scalar arguments.
- Retain the RV64/RVV ELF loader as an independent, inactive path. Binary launches
  still fail explicitly; functional commands do not pretend to execute a binary.
- PNM can directly access shared PIM memory and GPR. Global Buffer access requires
  an explicit transfer; it is not a PNM load/store address range.
- Preserve public Session/L3 APIs and the blocking IBackend contract initially.

## Ownership and execution boundary

| Component | Owns or decides | Must not do |
|---|---|---|
| L2 Session | Tensor metadata, allocation lifetimes, layout readiness, dependencies | Encode vendor opcodes or schedule DRAM commands |
| Host target implementation under L1 | Target layout rules, tile traversal, temporary use, command generation and address binding | Create a second independent tensor/planning model |
| Driver/transport | Data and command submission, completion | Infer tensor operations |
| Device emulator | Addressable memory, registers, command decoding and execution | Accept OperationSpec/KernelPlan or choose tensor tiling |
| Functional PNM engine | Execute explicit kernel commands on device storage | Rebuild Tensor/KernelPlan objects or claim RISC-V execution |

TensorDesc, LayoutDesc, OperationSpec, and KernelPlan remain authoritative host
objects. IBackend::execute(plan, inputs, output) is a host entry point. It builds
commands and submits them; it does not transmit plan to the device model.
AllocationHandle resolves to a device range through a host allocation table.
The table stores range, ownership, and lifetime information, not another TensorDesc.

The emulator consumes command operands and bytes. Dimensions or strides in a
functional command or binary argument block are execution arguments, not an instruction to reconstruct
a Tensor or to plan the operation again.

## Semantic commands versus hardware encoding

The first interface uses typed semantic records. Names below are library contract
names, not assigned AiM opcodes. No new binary opcode numbers are defined by A1.
Keep the future hardware encoder separate from the semantic executor.

The paper separates host-generated ISR instructions, DMA expansion, and controller
DRAM commands. Its example ISR_MAC_ABK with OPSIZE=63 expands into 64 requests.
Our semantic record uses a positive count, not an encoded OPSIZE. An eventual
encoder must apply each instruction's verified encoding rule. It must not apply
count-minus-one to all instructions based on that one example.

### Common operand rules

- Counts are positive element/burst counts; byte sizes and offsets are explicit.
- Logical channel mask bit i selects channel i in the semantic interface. A wire
  encoder translates hardware mask ordering. Reject zero or out-of-range masks.
- Device addresses contain an address-space tag and a 64-bit byte offset. They
  are never host pointers. Reject overflow, misalignment, and out-of-range access.
- The memory profile defines one mapping between shared byte addresses and
  channel/bank/row/burst locations. PNM loads and PIM bank access use the same
  backing bytes. Do not maintain separate unsynchronized copies of weights.
- Bank operands use explicit row and burst-column indices. One MAC command cannot
  cross a row; the host builder splits it. The mapping format is versioned and
  shared by packing and address resolution, not independently inferred by each.

### Initial command set

| Semantic command | Required operands | Effect |
|---|---|---|
| Transfer | Source/destination space and offset, byte count | Copy over an explicitly supported route |
| InitializeMac | Channel/bank selection, GPR source | Initialize selected accumulators from FP16 values, including explicit zeros |
| MacAllBanks | Channel mask, row, burst column, burst count | Accumulate selected bank/GB products |
| ReadMac | Channel mask, GPR destination | Store selected accumulator results in documented channel/bank order |
| LoadPnmProgram | Immutable ELF payload and declared requirements | Validate and load a program; return an opaque program handle |
| LaunchPnmProgram | Program handle, argument address/size, stack range, execution limit | Enter the binary and wait for success or failure |
| UnloadPnmProgram | Program handle | Release loaded code after all users finish |

InitializeMac, MacAllBanks, and ReadMac represent the initial
AiM-derived semantic subset. Exact register transfer encodings require a hardware
specification. Transfer routes involving GPR/shared memory are extension semantics
until their physical implementation is established; do not invent vendor commands.
Initial supported routes are shared memory <-> GPR, shared memory <-> scratchpad,
and Shared/GPR -> a single channel’s GB via Transfer. Other routes are rejected until specified.
Overlapping copies are rejected initially. Transfer preserves bytes, not dtype.

Activation/LUT and arbitrary PIM elementwise commands are not part of this initial
subset. The existing sampled SiLU table is not evidence for an AiM AF contract.
Layout conversion now runs through PnmRelayout commands on device storage. The
original A3 host packing path has been replaced.

## Memory spaces and lifetime

| Space | Contents | Access and ownership |
|---|---|---|
| Shared off-chip memory | Weights, inputs, public outputs, argument blocks | Host-managed allocations; PNM and PIM share bytes |
| GPR | Bias, partial sums, intermediate results | Device storage; PNM direct access and explicit transfers |
| Global Buffer | PIM input vector tiles | PIM/transfer commands only; not PNM load/store accessible |
| MAC registers | Live partial sums | Command-visible device state, not persistent public Tensor storage |
| PNM scratchpad | Kernel working data and stack | PNM-addressable device memory; reserved through completion |
| Program storage | Loaded binary segments | Device-owned loaded-program lifetime |

Use byte offsets at the command boundary. The PNM executor maps shared memory,
GPR, scratchpad, and program storage into non-overlapping guest address windows.
The profile supplies window bases/sizes; never hardcode host addresses into ELF
arguments. Exclude GB and MAC registers from guest load/store mapping.
Enforce permissions and bounds rather than exposing the entire host process.

The existing PNM workspace cannot also stand for GPR and GB capacity. Validate
these resource limits separately. Initial execution is serial, so reusable private
temporary regions need no concurrent allocator or new public Tensor class.

Within an operation, keep partial sums in MAC registers or GPR. Initially place
public operation outputs in dense shared memory before completion. This preserves
the existing API and avoids a host round trip. Cross-operation GPR residency is
deferred. Host download occurs only through the explicit download API.

GB and MAC state persist across commands. A command must not implicitly clear them.
The host builder initializes every value it will read, including padded vector
lanes, padded weights, and accumulators for reused output groups. Inactive banks
do not contribute output values. Select a deterministic ascending channel/bank
order for ReadMac and scatter only valid logical outputs.

## PNM binary and launch ABI (retained, inactive path)

The following is a proposed library execution ABI, not a claim about an existing
board firmware interface. It is the baseline for the future executor adapter.

- ELF64, little-endian RISC-V, statically linked executable with explicit entry;
  initially reject dynamic linking and unsupported relocations.
- RV64 with V 1.0 is the baseline. A manifest must additionally declare scalar
  extensions, floating-point/FP16 extensions, and vector requirements. RVV 1.0
  alone does not establish support for every FP16 instruction.
- Reject a binary whose declared requirements exceed the executor profile.
  Fixed guest memory windows and program load ranges must not overlap.
- The host launch setup provides a 16-byte-aligned stack, a0 = guest address of
  the argument block, and a1 = argument-block size in bytes. Other registers are
  unspecified unless the executor ABI explicitly initializes them.
- The argument block is versioned, little-endian, and kernel-specific. Addresses
  are guest addresses, not AllocationHandle values. It contains only explicit
  operands such as pointers, dimensions, strides, dtype, and scalar parameters.
- Entry behaves as a callable kernel. The adapter sets ra to a reserved return
  trampoline; a normal return reports a0 = 0 for success or nonzero for failure.
  There is no operating-system syscall ABI in the initial environment.
- Traps, invalid memory access, unsupported instructions, and instruction-budget
  exhaustion fail execution. The executor stops before returning the failure.
- Program code and argument/stack/data allocations remain live until completion.
  Unload rejects a program in use. No automatic host-function fallback is allowed.

The exact scalar extension list, compiler flags, vector capacity, guest window
values, and return-trampoline integration are adapter deliverables before actual
binary execution is enabled. The A4a loader below does not establish a working toolchain.
The current vector_lanes config must not be reinterpreted as an RVV VLEN contract.

## Numerical profile

The initial functional extension uses FP16 storage and FP32 arithmetic state.
It intentionally does not claim vendor bit-exactness. Specify the reference path:

1. Convert FP16 operands to FP32 and form lane products.
2. Reduce each burst using a fixed balanced FP32 tree, padding unused lanes with
   zero. Apply round-to-nearest, ties-to-even at arithmetic steps; do not fuse
   multiplication and addition or enable reassociation/fast-math.
3. Add the burst result to the selected FP32 MAC accumulator in command order.
4. ReadMac converts to FP16, round-to-nearest, ties-to-even, into GPR. Reloading a
   spilled partial sum therefore introduces FP16 rounding; keeping it in MAC does
   not. The host schedule can affect numerical results.

Preserve subnormals in the functional reference. Propagate NaNs without promising
payload preservation; compare NaNs by classification in tests. Overflow follows
the selected IEEE format. Validate finite normal and boundary inputs separately.
Functional PNM matrix kernels use FP32 FMA, apply Matmul scaling before FP16 output
conversion, and add optional Linear bias before conversion. Elementwise and reduction
kernels use FP32 intermediates and explicit FP16 writes. A future binary executor
would instead derive numeric behavior from the binary instructions.
BF16 and any verified hardware accumulation rules require a new numerical profile.

## Ordering, completion, and errors

One submission executes commands in order, each completing before the next starts.
This is an initial emulator contract, not a statement about hardware throughput.
PIM writes, transfers, and PNM writes are visible to the next command without an
emulated cache. Real hardware adapters must supply equivalent visibility rules.

Keep the existing Session worker and Event behavior. IBackend methods return only
after accesses stop, including error paths. No second PNM worker or hardware FIFO
manager is introduced. Completion means all submitted writes are visible, not just
that a command was enqueued.

Preflight command kinds, capabilities, address ranges, ELF requirements, and launch
availability before executing a submission. Reject binary PNM launches before any command
runs if no executor is installed. Execution faults can still leave partial writes:
stop remaining commands, return failure, and preserve failed output readiness.
No rollback is promised. Host dependencies must not consume a failed output.
Future submissions must explicitly initialize transient state they use.

## Configuration and code changes after A1

Add a versioned NDP profile with shared memory geometry/mapping, GPR/GB/scratchpad
sizes, transfer alignment, guest address windows, numeric profile, and PNM execution
capabilities.

- Keep IBackend as the host facade; split command construction and submission inside
  a new src/l1/ndp implementation. Keep command types private until a consumer needs
  a public interface.
- Reuse LayoutDesc.format to identify the mapping. Define the actual packing and
  inverse mapping together in A3; do not introduce a parallel AimLayout hierarchy.
- Update Planner's unconditional FP32 accumulator restriction to validate the
  target numerical contract. Preserve dense public outputs in the first milestone.
- Keep existing functional arithmetic as a reference backend.

## Revised milestones and acceptance

| Stage | Deliverable | Acceptance |
|---|---|---|
| A1 | This semantic contract and confirmed choices | Host/device responsibilities and remaining hardware unknowns are explicit |
| A2 | Command decoder and device memory/register state | Handwritten streams test bounds, ordering, reset/reuse, and failures |
| A3 | Host-generated PIM GEMV streams | Multi-tile/tail GEMV; shared-memory/bank mapping round trips; no device-side tensor planning |
| A4a | ELF loader and launch ABI boundary | Validate images, segments, handles, arguments; missing executor fails explicitly |
| A4b, replaced | Functional PNM command engine | Explicit kernels operate on device addresses; no binary substitution |
| A5 | Session integration and resource lifetime | Existing API behavior and failures validated for supported commands |
| A6 | Mixed PIM/PNM decoder and cleanup | Requires A4b; PIM-to-GPR-to-PNM handoff without host download |

A4b and A5 use functional execution by explicit user choice. A6 completes the synthetic decoder integration described below. Actual RISC-V instruction execution must not be claimed from these tests.

## Evidence and limits

### A2 implementation status

The private implementation in `src/l1/ndp/` now accepts typed commands directly.
The A3 facade described below connects it to Session through a separate factory.
A2 originally had no binary wire decoder. F1 now provides the private
[versioned command codec](command-transport.md). A PNM instruction executor and
timing model remain unimplemented.
The ELF loader is described under A4a. Unsupported PNM launch fails during
whole-stream preflight.

The initial Profile is configured in C++; a versioned config-file reader remains
future work. Its functional shared-memory mapping is
`(((row * channels + channel) * banks + bank) * row_bytes + byte_offset)`.
The original A2 bank-major mapping was changed to this row-stripe mapping in A3
so dense allocation ranges also have contiguous addresses for future PNM access.
This is an explicit emulation mapping, not vendor address encoding. Bursts contain
16 FP16 elements (32 bytes). MAC burst column j reads GB bytes `[32*j,32*j+32)`.
Each GB Transfer writes one channel from Shared or GPR;
InitializeMac consumes compact values in ascending selected channel/bank order.
ReadMac writes all banks of selected channels in ascending order. MAC initialization
and GB load state are checked before updating a selected command's destinations.
All memories start at zero, but GB and MAC still require explicit initialization.
No command implicitly resets a live accumulator, including ReadMac.

Transfer supports the routes listed above at byte granularity; GB loads require
32-byte source/destination offsets and sizes. MAC initialization/result destinations
require 2-byte alignment. Host upload/download are synchronous staging/inspection
hooks for shared memory, GPR, and scratchpad; they are not guest PNM execution.
The emulator is single-owner and has no internal threads or synchronization.

`ndp_command_contracts` covers distinct channel/bank/row/column patterns, persistent
accumulation, masked reuse, GPR/shared/scratchpad transfers, whole-stream preflight,
partial writes on state errors, bounds/geometry overflow, channel bit 63, and FP16
spill/reload rounding. Functional PNM-to-GPR access is tested under A4b; binary execution remains deferred.

### A3 implementation status and usage

`make_ndp_backend(NdpSpec)` in `ndp/l1/ndp/backend.hpp` creates the new command-driven
backend. Pass the resulting backend to `Session::create`, then use the existing
`from_host`, `materialize_layout`, `matvec`, and `to_host` APIs.
No L2/L3 source changes are required for A3. For example, given a Session and
resident FP16 tensors W[N,K] and x[K], call
`session.matvec(W, x, {Engine::PIM})` and check its Result before consuming the output.
The executable contract example is `tests/ndp_backend_test.cpp`.

At the A3 milestone the backend supported only PIM Matvec. A4b/A5 below adds PNM
operations and device-side relayout/fill/copy. Binary execution remains unsupported. NdpSpec is a C++ configuration
object. A6 adds a separate versioned
file schema through load_ndp_spec and --ndp-config.

Weights use LayoutDesc format `ndp.fp16.stripe.v1` with geometry suffixes. The
serialized order is `[output group][K tile][channel/bank][row byte]`. Each bank holds
one output row within a group. Tile width is min(row_bytes, gb_bytes)/2 FP16 values.
All padded output rows, lanes, and unused row bytes are zeroed during packing.
A4b replaced the original host staging implementation with device-side PnmRelayout.

The host builder uses existing KernelPlan metadata and allocation ranges. It clears
vector padding, transfers input tiles to GPR, loads GB, and emits one MacAllBanks
per K tile. Partial sums remain in MAC registers until the output group completes.
ReadMac writes GPR, followed by a transfer of only valid outputs to dense shared
memory. GEMV execute never downloads inputs/results to calculate or assemble them
on the host. The emulator receives commands, not the KernelPlan.

The initial allocator reserves whole row stripes across all banks for both dense
and packed allocations. Row zero is reserved as immutable zero data for commands.
Allocation handles own ranges only; they do not duplicate tensor metadata. Dense
allocations have contiguous shared addresses, allowing a future PNM binary to use
ordinary pointers. Release reclaims stripes and handles are not reused.
This deliberately simple allocator can exhaust stripes before Session's logical
byte accounting reaches capacity; memory statistics do not include stripe padding.
It is not yet a serving-oriented suballocator.

`ndp_backend_contracts` checks Session GEMV with multiple K tiles and output groups,
odd dimensions, repeat execution, a GB smaller than a row, independent packed-byte
expectations and round trips, forged plans/layouts, allocation reuse/exhaustion, and
the number/order of emitted MAC commands. Functional PNM integration is covered below.

### A4a ELF loading and launch preparation

Private `Emulator::load_program` and `unload_program` are synchronous control
operations on the same single-owner device instance. Loading returns a monotonic
program handle. They are not yet exposed through Session/IBackend or encoded as
wire instructions. `LaunchPnmProgram` remains in the common command stream.
The program store owns copies of segment bytes and declared requirements, not
tensor metadata. Future transport integration must preserve this control ordering.

The loader accepts static little-endian RISC-V ELF64 ET_EXEC with bounded PT_LOAD
segments, an aligned entry in executable file-backed bytes, and supported manifest
requirements. It zero-fills BSS, retains permissions, and rejects overlapping ranges,
dynamic linking, TLS, relocation sections, writable executable segments, unknown
ELF flags, and unsupported program headers. File tables and ranges are checked
before allocating segment storage. Extended ELF table numbering is unsupported.
ELF attributes are not interpreted and instructions are not scanned: the explicit
manifest is a declaration to validate, not proof of actual binary ISA usage.

The initial private profile declares RV64/RVV 1.0, VLEN 128, and I/M/A/F/D/V.
FP16 instruction extensions are not implicitly supported. Program storage is
0x10000000–0x10ffffff with a 32 MiB aggregate loaded-byte budget. Simultaneously
resident ET_EXEC images must use disjoint load ranges. This is a loader profile,
not a claim that an executor implements these extensions.

Launch preparation uses separate guest windows for shared memory (base 0x40000000),
GPR (0x80000000), and scratchpad (0xc0000000), each with at most 1 GiB of actual
backing memory. GB/MAC registers are not mapped. Return address 0x1ffff000 is
reserved for the future executor's return trampoline; no trampoline executes yet.
The initial windows are private ABI constants, not configurable board addresses.

Arguments must already be resident in shared memory or GPR before launch preflight.
The argument block is 8-byte aligned, starts with little-endian uint32 version=1
and uint32 total byte size, and fits its backing space. The stack is a nonempty,
16-byte-aligned scratchpad range; SP points to its upper end. An instruction limit
is mandatory. `prepare_launch` returns PC/SP/RA/A0/A1 and a retained program reference.
It validates backing ranges, not Session allocation ownership. Retaining this
state prevents program unload; released/stale handles are rejected.

Valid launch submissions still return UnsupportedOperation before any earlier
command in that submission executes. Malformed launches return their validation
error instead. There is no CPU-kernel substitute, instruction execution, guest
permission enforcement during execution, or Session-level binary launch API yet.

`ndp_program_contracts` uses synthetic ELF fixtures, not cross-compiled execution
tests. It checks truncation, table/range overflow, segment permissions/overlap,
unsupported dependencies, memory budgets, BSS, image ownership, stale/pinned
handles, argument/stack validation, register setup, and rejected-launch atomicity.

ELF loading rules were checked against the
[System V ELF program-loading specification](https://gabi.xinuos.com/elf/07-pheader.html)
and [RISC-V psABI](https://riscv-non-isa.github.io/riscv-elf-psabi-doc/).
The argument header, guest windows, and trampoline are project-specific ABI choices.

### A4b/A5 functional PNM and Session integration

PnmCompute carries a kernel identifier, device operand addresses and element counts,
rows/width/inner dimensions, transpose/bias conventions, scalar options, and optional
Gather indices. It never carries TensorDesc, LayoutDesc, OperationSpec, or KernelPlan.
The host builder resolves the existing plan to these operands; the device validates
bounds, counts, overflow, aliasing, and kernel support before the stream executes.

Supported PNM operations are Matmul, Matvec, Linear (with optional bias), Add/Mul
(equal shape or last-axis broadcast), RMSNorm, Softmax, SiLU, Reshape, Gather, Scale,
and Transpose. PIM supports Matvec and bias-free single-row Linear. Unsupported PIM selections do not fall back
silently to PNM. GEMM uses two FP16 scratchpad slots (four bytes), not an RVV timing
or tile-performance model; insufficient scratchpad is rejected. Matrix command
operands may reside in shared memory or GPR, not the reserved scratchpad staging area.
Other commands can access scratchpad where their explicit operands permit it.

PnmRelayout converts dense and banked storage in-device and zeros padding. CopyBytes
has memmove semantics; FillBytes initializes or clears allocation bytes. Compute
and relayout source/output aliases are rejected. These extensions operate directly
on shared/GPR/scratchpad backing bytes. GB remains inaccessible through PNM operands.
The retained ELF loader is independent and unused by this execution path.

Existing Session events, dependencies, allocation lifetimes, materialize_layout,
DeviceBuffer copy/clear/snapshot, and close/error handling work without L2/L3 source
changes. Query methods use immutable profile data; device mutation remains on the
existing serial backend worker. PIM output can feed PNM without a host download or
explicit user wait. Whole-stream static errors are rejected before writes, while
existing Session semantics propagate asynchronous allocation/execution failures.

`ndp_pnm_contracts` compares all PNM operation families with the existing functional
reference and tests pre-rounding Matmul scaling, PIM-to-PNM chaining, direct GPR
consumption, pending-input lifetimes, immutable snapshots, clear/copy, allocation
exhaustion and scratchpad errors. Handwritten streams test invalid kernels, counts,
addresses, aliases and rejection before earlier writes. Row-stripe allocator padding remains a limitation. A6 adds a versioned configuration
file reader for the C++ command backend.

### A6 decoder completion

PIM single-row, bias-free Linear now uses the same GEMV command builder with
operand order resolved on the host. PreparedWeight can therefore request both
engine layouts through its existing Linear query; no L2/L3 implementation changes
are needed. Biased or batched PIM Linear remains explicitly unsupported.

The existing explicit-layer example accepts `--ndp` and `--ndp-config`. The separate
`configs/ndp-small.ini` schema defines version, channels, banks, rows, row_bytes,
gb_bytes, gpr_bytes and scratchpad_bytes. Parsing is strict; construction checks
geometry/resources.

The decoder oracle suite now runs on both reference and command backends. It covers
two requests, PNM prefill, three PIM decode steps per request, reset, all-PNM versus
mixed execution, one-time weight packing, no internal host downloads, exact engine
call counts, and request-memory reclamation. CTest also runs the explicit-layer
example with both default and file-configured command profiles.

See [the backend guide](ndp-backend.md) for runnable commands and configuration.
The older emulator and inactive ELF loader are retained for compatibility; neither
is used as a hidden fallback. Full serving, Python backend selection, hardware
encoding and timing remain outside this milestone.

Primary reference: SK hynix, System Architecture and Software Stack for GDDR6-AiM,
Hot Chips 34 (2022), DOI 10.1109/HCS55958.2022.9895629; user-provided PDF.
Pages 5–6 describe BF16 and the GB/MAC path; 9–12 describe DMA/controller boundaries;
15 and 17–19 describe host instruction generation, emulation, and tiling schedules.
PNM binary commands, direct GPR access, and the FP16 arithmetic profile above are
project extension decisions. Hardware opcode encoding, exact register transfer
formats, cache/coherence behavior, and vendor arithmetic remain unverified.
