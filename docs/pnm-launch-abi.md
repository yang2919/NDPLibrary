# Separate PNM core launch ABI (N0 proposal)

The guest service receives a launch from L1 and starts a separate PNM RISC-V core. Both example programs execute on that core. Only the GEMV program issues PIM instructions. The profile and bytes below are a review fixture for the service and gem5 implementers. All integers use little endian, offsets use bytes, and fields have no implicit padding.

## Launch body

| Byte offset | Type | Value |
|---:|---|---|
| 0 | u64 | Session-local loaded program handle |
| 8 | u32 | Argument template size S |
| 12 | u32 | Binding count B |
| 16 | S bytes | Kernel-specific argument template |
| 16 + S | B × 32 bytes | Binding records |

Each binding contains a u32 argument pointer offset at 0, u32 access at 4 (1 read, 2 write), u64 buffer handle at 8, u64 offset within the buffer at 16, and u64 access extent at 24. The enclosing protocol chooses the Launch operation and supplies framing in N1. The argument ABI version below is separate from that protocol version.

The program manifest declares pointer slots, alignment and access. The service validates a whole request against live buffers and the loaded manifest before changing device state. Require one binding per slot, zeroed 8-byte aligned pointer placeholders, checked address arithmetic, valid ranges and compatible layout. Patch each pointer with a PNM-visible address. Only relocated argument bytes are read by the kernel. Allocation metadata and bank tables are service-owned. Keep descriptors and their referenced storage pinned until completion.

## Two example entry points

| Arg offset | Type | RVV vector add | PNM-issued PIM GEMV |
|---:|---|---|---|
| 0 | u32 | Argument ABI 1 | Argument ABI 1 |
| 4 | u32 | Total size 40 | Total size 40 |
| 8 | u32 | Element count 5 | Matrix rows 2 |
| 12 | u32 | Reserved, zero | Matrix columns 3 |
| 16 | u64 | A, dense FP16, read | Matrix slice descriptor, read |
| 24 | u64 | B, dense FP16, read | X, dense FP16, read |
| 32 | u64 | Y, dense FP16, write | Y, dense FP16, write |

**Vector add.** Buffer handles 11, 12 and 13 have fixture bases 0x40001000, 0x40002000 and 0x40003000. A starts two bytes into handle 11. All three extents are ten bytes. With A=[1,2,3,4,5] and B=[5,4,3,2,1], expect Y=[6,6,6,6,6]. The program manifest must declare the vector FP16 arithmetic features it uses. For executor acceptance also run lengths around the actual VLMAX.

**GEMV.** Handles 21, 22 and 23 resolve to 0x40004000, 0x40005000 and 0x40006000 with extents 16, 6 and 4. The matrix binding refers to immutable metadata for an already prepared matrix. Descriptor offset 0 is u32 version 1 and offset 4 is u32 slice count. Starting at 8 each pair of u32 values gives first physical row and contiguous all-bank row count. Reserve row slices across all participating banks with one common row index. Concatenate slices in logical row order and map logical row index group * number_of_K_tiles + K_tile through those slices. Reject slices overlapping reserved rows or another allocation. The fixture table has one slice at row 7 of length 1. The negotiated LPDDR6 layout maps bank/unit/channel and column fields. It has 128 bank lanes across eight channels and 2048 bytes per row. W=[[1,2,3],[-1,0,2]], X=[1,1,1] yields [6,1]. X=[2,0,-1] reuses W and yields [-1,-4]. The program writes input tiles to GB, issues all-bank MAC, collects FP32 accumulators and stores rounded FP16 outputs. The descriptor is a proposed freestanding ABI, independent of Linux libpim object layout.

## Program and device lifecycle

Use static ELF64 little-endian RISC-V executables with an explicit entry and no Linux syscall dependencies. Validate segments, zero BSS, check the manifest against RV64/RVV 1.0, VLEN, numeric format, layout and optional PIM instruction support. The current ProgramStore and prepare_launch provide native validation and initial register state. Their sample map uses program base 0x10000000, shared 0x40000000, GPR 0x80000000 and scratchpad 0xc0000000. The example assumes a reserved return trampoline at 0x1ffff000. Shared addresses above are fixture values. Agree actual ranges and protection with the device team.

Set PC=e_entry, sp to the top of a private 16-byte-aligned scratchpad stack, a0 to the device argument address, a1=40 and ra to the return trampoline. Permit one active image and launch per session initially. On return, a0=0 means success. Retire writes and drain outstanding PIM commands before reporting success. Reject unload/release of live operands. On timeout or trap, report failure and keep resources pinned until the core and controller are quiescent.

## Reproduce the host-side byte checks

```sh
dir=$(mktemp -d)
python3 tools/gem5/pnm/launch_fixture.py "$dir"
python3 tools/gem5/pnm/check_launch.py "$dir"
ctest --test-dir build-python -R 'ndp_(program|pnm)_contracts' --output-on-failure
```

Each .launch is 152 bytes. Each .args is the expected 40-byte relocated argument block. gemv.matrix contains the 16-byte slice table. The Python checker decodes these two fixtures and rejects corrupt handles, ABI version, placeholders, access flags, offsets and duplicate slots. It does not execute either kernel. The existing C++ tests separately exercise the emulator's ELF/launch and functional PNM contracts.

## Joint review before target acceptance

Confirm the executable and data maps, advertised extensions/VLEN, physical row allocator and layout identity, segment installation and instruction-cache visibility, core start/stop/return/trap mechanism, and PNM↔PIM memory ordering. Agree completion evidence for a timeout with pending bank commands. N1 can implement the host/worker protocol against the fixture while these target details are under review. N2/N3 require traces from the separate PNM core and PIM controller.

## N1 native worker and host client

The independent v3 capture endpoint uses one ordered request/reply stream per Session. The existing v2 reference GEMV worker still serves the existing PyTorch backend. The host selects the endpoint when starting the process. Requests and replies have a four-byte little-endian body length, at most 16 MiB. A request starts with a one-byte ASCII opcode. A reply starts with status 0 (success) or 1 (remaining bytes are a diagnostic). A validly framed rejected operation leaves the stream available for the next request. EOF closes the Session and releases its handles. A partial frame ends the process with failure.

| Opcode | Payload after opcode | Success payload |
|---|---|---|
| H | u8 version 3, required first | u8 version 3, u8 executor 0 (capture) |
| P (LoadProgram) | u32 profile, u32 template size, u32 slot count, template bytes, slot declarations, remaining bytes are a static ELF | u64 program handle |
| A | u64 logical bytes, u32 allowed access mask | u64 buffer handle |
| J (Launch) | N0 body specified above | u64 ELF entry, then relocated argument bytes |
| U (UnloadProgram) | u64 program handle | Empty |
| F | u64 buffer handle | Empty |

A slot declaration is 20 bytes: u32 patch offset, u32 access, u32 alignment, u64 required extent. The N1 manifest binds a fixed scalar template and exact extents at load time. Each launch can use different buffer handles and offsets for this signature. Profiles 1 and 2 assume the N0 RV64/RVV test profile with Zvfh for both kernels and additionally the LPDDR6 FP16 PIM extension for profile 2. The loader uses the existing ProgramStore to validate ELF segments, entry, overlap and memory budget. The two test ELF images contain a return stub at different addresses. Executor 0 records entry and bindings and performs no kernel arithmetic. The allocated ranges are virtual test ranges. Gem5 device access, tensor upload and actual PNM execution belong to the target adapter.

Run from the repository root with CMake testing enabled on a POSIX build host:

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target ndp_pnm_test_worker
ctest --test-dir build -R '^pnm_worker_protocol$' --output-on-failure
```

The host uses [Client](../tools/gem5/pnm/client.py) with buffered binary streams for the child process. The integration test loads both ELF fixtures, launches each twice using one program handle, compares returned bytes with N0, and rejects malformed ELF/profile, duplicate or invalid bindings and stale handles. Single-threaded synchronous dispatch keeps the image and buffers live through construction of the capture receipt. A target executor must retain them until its core and controller have actually stopped.

## N2 separate-core vector-add handoff

The [kernel](../tools/gem5/pnm/vector_add.S) is a freestanding RV64 ELF with real vector instructions. It checks the argument header, uses e16/m1 strip-mining, sets round-to-nearest-even, adds input vectors and returns through ra. The linker places the entry at 0x10000000. This target requires vector FP16 arithmetic (Zvfh), FP/vector state enabled, vstart=0 at entry, and the N0 sample map. Initial VLEN for this fixture is 128 bits. Source and linker script can be rebuilt for another agreed target profile.

Build on a host with RISC-V-capable LLVM clang and LLD:

```sh
make -C tools/gem5/pnm
cmake --build build-python --target ndp_pnm_test_worker
python3 tools/gem5/pnm/vector_add_case.py prepare build-python/ndp_pnm_test_worker \
    build/pnm/vector_add.elf build/pnm/cases --vlen 128
```

Alternatively, the macOS native preparation was verified with Zig 0.14.1 as the LLVM driver:

```sh
make -B -C tools/gem5/pnm PNM_CC='zig cc' \
    ISA_FLAGS='-target riscv64-freestanding -mcpu=generic_rv64+m+a+f+d+v+zvfh'
```

The N1 capture worker loads the real ELF and relocates each case. Its reply supplies the entry and argument bytes. Each directory contains the ELF, register setup and memory locations (state.json), args.bin, initial guarded input/output storage (a.bin, b.bin, y.bin) and the host golden output (expected.bin). Cases cover lengths 5, VLMAX−1, VLMAX, VLMAX+1 and 2×VLMAX+1. The input pattern includes FP16 rounding ties.

**Extended gem5 team procedure.** Load the ELF PT_LOAD segments into the separate PNM core's executable window. Copy args.bin to the address specified by a0 and the complete a.bin, b.bin, y.bin to their respective buffer_bases, including both guard bytes at each end. Install the target's return trap at ra and provide writable stack space below sp. Apply register state, start the PNM core, retire its return and wait for shared-memory writes to become visible. Capture an issuer-tagged instruction trace with entry PC, vsetvli, vle16.v, vfadd.vv, vse16.v and return. Read the complete output allocation to output.bin. Record the observed values in completion.json:

```json
{"issuer":"pnm","entry_pc":268435456,"return_pc":536866816,"status":0,"quiescent":true,"vlen_bits":128}
```

Use simulator observations for these fields; the verifier compares the report to the case and does not parse the simulator's instruction trace. Keep that trace with the evidence. Run the following for each case:

```sh
python3 tools/gem5/pnm/vector_add_case.py verify build/pnm/cases/n9 /path/to/run/n9
```

Target acceptance also requires an injected trap and a deadline expiration with evidence of core quiescence before buffer reuse. The supplied snapshot has a guest RISC-V CPU and PIM controller sources; a separate-core start/stop adapter and its actual control interface are still needed for these runs. The v3 capture worker continues to report executor 0 and only returns a relocated launch receipt.

## N3 PNM-issued GEMV

The freestanding [GEMV kernel](../tools/gem5/pnm/gemv.c) takes the same 40-byte N0 argument block. It traverses the row-slice descriptor in logical group/tile order, clears the bank accumulators, transfers each dense X tile to each unit's GB, issues all-bank MAC and reads valid accumulators back as FP32. It narrows each valid result once into dense FP16 Y. Short transfers advance by the actual byte count returned by the supplier's instruction wrapper. The kernel uses preallocated storage and the supplier's inline instruction wrappers.

**Fixture assumptions.** The sample LPDDR6 map uses 128 bank lanes and 1024 FP16 elements per row. Its dedicated PIM instruction aperture starts at PIM_LAYOUT_WINDOW_BASE_FS (0x800000000). N0 dense buffers and the slice descriptor remain in the sample PNM shared window. Confirm both address mappings before a device run. Start with FP/vector enabled, vstart=0, frm=RNE. The allocator must supply an immutable descriptor of exactly 8 + 8 × slice_count bytes with disjoint, owned rows. Validate this at allocation/launch on the target; N1 capture validates binding extents only. Input/weight writes must already be device-visible; collect outputs after PNM and PIM quiescence. The entry and exit fences supplement the target cache/visibility contract. Both sample ELFs use the same entry window; unload the vector-add image before installing the GEMV image.

### Build and native verification

```sh
make -C tools/gem5/pnm test-gemv
make -C tools/gem5/pnm gemv
cmake --build build-python --target ndp_pnm_test_worker
python3 tools/gem5/pnm/gemv_case.py prepare build-python/ndp_pnm_test_worker \
    build/pnm/pim_gemv.elf build/pnm/gemv-case
```

Use a RISC-V-capable clang/LLD for the second command. With Zig 0.14.1 use the following replacement:

```sh
make -B -C tools/gem5/pnm gemv PNM_CC='zig cc' \
    ISA_FLAGS='-target riscv64-freestanding -mcpu=generic_rv64+m+a+f+d+v+zvfh'
```

The native test runs the same kernel loop against the supplier's FP16 reference primitives. It packs bank data independently from the documented LPDDR6 bit positions, checks two different vectors against dense double dot products rounded to FP16, checks output guards and rejects a reserved row before issuing MAC. At VLEN=128 the measured all-bank MAC counts for two launches were 16 (2×3), 256 (128×1024) and 544 (129×1025). The last case covers both tails and uses physical rows [16,17], [10], [7]. These are functional request counts.

### Extended gem5 user verification

Use build/pnm/gemv-case/state.json for addresses and initial registers. Install the ELF PT_LOAD segment, configure the N2 stack/return trap and apply the files in writes once, including the two packed weight rows. On each of the two launches load x1.bin or x2.bin at x, reset the complete guarded output allocation from y.bin, restore the entry registers and start the separate PNM core. Keep the same matrix and program across launches. Save the entire guarded output as output1.bin or output2.bin in the evidence directory. Expected logical outputs are [6,1] and [-1,-4].

Capture an issuer-tagged trace of PNM PCs and corresponding controller activity. Record per-launch instruction count deltas (one GB write and one all-bank MAC per unit for this six-byte tile). Supply observed completion1.json and completion2.json in the following format:

```json
{"issuer":"pnm","entry_pc":268435456,"return_pc":536866816,"status":0,"quiescent":true,"vlen_bits":128,"wr_gb":8,"ab_mac":8}
```

```sh
python3 tools/gem5/pnm/gemv_case.py verify build/pnm/gemv-case /path/to/evidence
```

The checker compares the two reported summaries and raw output bytes. Check the original traces for PNM attribution and bank execution. Actual extended-gem5 acceptance remains pending the separate-core execution adapter.

### Capture a PyTorch Session launch through L1

A native verification session can use two workers. L1 uses the existing libpim
worker for weight placement, data transfers and numerical results. For each
supported PIM Matvec or bias-free single-token Linear, L1 first submits the
same operation's dimensions and operand bindings to the v3 capture worker.
A failed capture fails the operation before numerical execution.

From the repository root, after building the Python extension, FP16 reference
worker, N3 GEMV ELF and `ndp_pnm_test_worker`, run:

```python
import torch
import ndp

with ndp.Session(backend="gem5_pim", worker="build/extended-gem5/pim-kernel-worker-fp16",
                 capture_worker="build-python/ndp_pnm_test_worker",
                 gemv_elf="build/pnm/pim_gemv.elf") as session:
    layer = torch.nn.Linear(3, 2, bias=False, dtype=torch.float16).eval()
    with torch.no_grad():
        layer.weight.fill_(1)
    projection = ndp.nn.NdpLinear.from_torch(layer, session, ndp.Engine.PIM)
    x = session.upload(torch.ones(1, 3, dtype=torch.float16))
    print(session.download(projection.forward_ndp(x)))  # [[3, 3]] from libpim
```

The first submission loads the ELF with a shape-specific manifest. Subsequent
submissions of the same shape reuse the program. A shape change unloads it and
registers a new manifest. Capture buffer IDs follow L1 allocation lifetimes.
The bindings are a 16-byte single-slice matrix descriptor, a dense FP16 input,
and a dense FP16 output. L1 verifies the returned entry PC, argument header
and pointer alignment. The v3 worker reserves addresses only. Descriptor
contents, packed weight rows, input bytes and numerical outputs live outside
the capture protocol. Running the ELF on the PNM core still requires the
extended-gem5 execution adapter.

To include this case in the existing Python-enabled CTest build:

```sh
cmake -S . -B build-python \
  -DNDP_GEM5_PIM_WORKER="$PWD/build/extended-gem5/pim-kernel-worker-fp16" \
  -DNDP_PNM_GEMV_ELF="$PWD/build/pnm/pim_gemv.elf"
cmake --build build-python
ctest --test-dir build-python -R python_contracts --output-on-failure
```

## N4 mixed PIM/PNM MLP

Prepare gate, up and down as bias-free FP16 `ndp.nn.NdpLinear` projections
with `ndp.Engine.PIM`. Open the capture Session shown above with the additional
keyword `pointwise_elf="build/pnm/pointwise.elf"`. One decode token follows
this path (x has shape [1, hidden]):

```python
x = session.upload(host_x)
gate = gate_proj.forward_ndp(x)
up = up_proj.forward_ndp(x)
activation = session.silu(gate)
product = session.mul(activation, up)
host_y = session.download(down_proj.forward_ndp(product))
```

The three projections use the PIM-issuing GEMV ELF. SiLU and equal-shape
multiply use the PNM pointwise ELF with profile 1. Its little-endian 40-byte
argument block uses u32 fields at offsets 0, 4, 8 and 12 for ABI version 1,
size 40, FP16 element count and operation (1 SiLU, 2 multiply). Offset 16
binds the first dense input. For multiply, offset 24 binds the second dense
input; SiLU leaves those eight bytes zero and omits that slot. Offset 32
binds the output. Each dense binding spans exactly twice the element count
in bytes and requires two-byte alignment. Each operation stores an FP16
result before the next operation reads it.

L1 preserves buffer handles across ELF switches and checks each live handle's
relocated address across launches. Native reference worker requests S
(destination, input) and M (destination, left, right) use little-endian u32
handles after the opcode. These operations use worker-local numerical staging.
Rebuild the FP16 reference worker before using the mixed path.

```sh
make -C tools/gem5/pim_kernel profile fp16
make -C tools/gem5/pnm gemv pointwise test-pointwise
cmake -S . -B build-python \
  -DNDP_GEM5_PIM_WORKER="$PWD/build/extended-gem5/pim-kernel-worker-fp16" \
  -DNDP_PNM_GEMV_ELF="$PWD/build/pnm/pim_gemv.elf" \
  -DNDP_PNM_POINTWISE_ELF="$PWD/build/pnm/pointwise.elf"
cmake --build build-python
ctest --test-dir build-python -R python_contracts --output-on-failure
```

The ELF targets use the same cross compiler and ISA flags as N3. The native
pointwise test checks all 65,536 FP16 input bit patterns for SiLU and for
multiply with a fixed FP16 right operand. The PyTorch integration test runs
two tokens through five launches each, compares each intermediate with
PyTorch and checks one activation upload and one output download per token
after weight preparation. Diagnostic reads are counted separately.

For extended-gem5 acceptance, repeat those two inputs and weights from
`test_resident_mixed_mlp` on the separate core. Record the issuing PC and
completion of each launch, shared addresses at producer/consumer boundaries,
and PIM controller drain before the next PNM read. Confirm the dense PNM
product is visible to the down projection and compare intermediate FP16
values and the final MLP output. Native capture verifies binding continuity;
target traces and memory observations are still required for device coherence.

## Single-token multi-head attention

Build `attention.elf` with the same cross compiler and flags as N3, selecting
`make -C tools/gem5/pnm attention`. Run `make -C tools/gem5/pnm test-attention`
for the native entry-point test. To include both native worker and capture
tests in the Python CTest configuration shown above, also set
`-DNDP_PNM_ATTENTION_ELF="$PWD/build/pnm/attention.elf"`.

The L3 KV cache uses two worker-owned mutable FP16 buffers. Each append writes
one range per buffer. Each attention call snapshots the live K/V prefix and
composes gather, transpose, scaled matmul, softmax and matmul for each head.
The result is assembled in head order and passed directly to the output
projection. This reference implementation materializes prefix and per-head
tensors on the worker. It does not implement a fused attention kernel.

### Dense PNM launch

The v3 profile-1 argument block is 64 bytes. All fields are little endian.
L1 supplies exact extents in the launch bindings. Slots with a zero pointer
and no operand have no binding. Switching between biased and bias-free Linear
changes the manifest even for identical dimensions.

| Byte offset | Type | Field |
|---:|---|---|
| 0, 4 | u32 each | ABI version 1, size 64 |
| 8 | u32 | Operation: 1 matmul, 2 linear, 3 transpose, 4 softmax, 5 RMSNorm, 6 residual add |
| 12, 16, 20 | u32 each | m, n, k; k is zero for non-dot operations |
| 24, 28 | f32, u32 | Accumulator scale (RMSNorm epsilon for operation 5), reserved zero |
| 32 | u64 | Left input, read |
| 40 | u64 | Right input for binary operations, read |
| 48 | u64 | Optional linear bias [n], read |
| 56 | u64 | Output, write |

Matmul reads [m,k] and [k,n]. Linear reads [m,k] and [n,k]. Both store [m,n]
after FP32 accumulation. Transpose reads [m,n] and stores [n,m]. Softmax
treats the input as m rows of n elements. RMSNorm reads gamma[n], accumulates the row mean square in FP32 and applies epsilon before the inverse square root. Add reads two equal [m,n] operands. All data pointers are aligned to
two bytes and refer to dense FP16. The capture worker validates arithmetic
launches. The native reference worker performs all data movement and produces
the numerical results. Target verification still requires the separate PNM core.

### Native FP16 worker extension

After the v2 geometry query, V returns one byte. Bit 0 enables the operations
below and allocation role B for mutable FP16 storage. Bit 1 adds RMSNorm and residual add on A and permits stand-alone native SiLU and multiply. Rebuild the attention ELF and reference worker together for full-block capture. Older workers can reject
V and continue serving GEMV. Reshape uses the existing whole-buffer T request.
All offsets and counts in B and K use FP16 elements. Each field following the
opcode is little-endian u32 unless otherwise specified.

| Opcode | Request fields in order | Success payload |
|---|---|---|
| V | none | u8 feature bits |
| F | destination, fill byte in low 8 bits | empty |
| B | destination, source, source offset, destination offset, count | empty |
| K | destination, source, one source index per output element | empty |
| A | destination, left, right or zero, bias or zero, operation, m, n, k, f32 scale | empty |

The worker validates buffer extents, gather indices and copy ranges before
writing. The library keeps cache contents, gathered data and attention scores
behind the worker boundary. Fill, copy, reshape and gather are data movement
requests on the reference stream; they do not launch a captured arithmetic ELF.
