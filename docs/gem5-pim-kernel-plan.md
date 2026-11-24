# Kernel launch plan for the RISC-V gem5 PIM backend

## Target architecture: PNM issues PIM instructions

The device has a separate RV64/RVV PNM core that can issue PIM instructions. The Linux guest CPU runs a resident service. External PyTorch submits work to that service through L1. The service loads programs and launches the PNM core. A kernel may compute on RVV, drive the PIM controller, or do both. The existing Linux-worker GEMV path below is a reference and transport baseline.

The [N0 launch ABI and golden byte fixtures](pnm-launch-abi.md) specify the vector-add and GEMV examples for joint review.

**Launch contract.** Build one freestanding, static, little-endian ELF per initial entry point on the build host. Declare required scalar, vector, FP and PIM ISA features, minimum VLEN, numerical format, argument ABI and compatible layout profile. Negotiate a new protocol version. Keep program loading, launch, unload, buffer allocation and transfer distinct. Return opaque session-scoped handles. For the first implementation allow one synchronous launch at a time. A launch carries the program handle, a versioned argument template and explicit buffer bindings (patch offset, buffer handle, byte offset, access extent). Resolve each binding on the service/device side after bounds and alignment checks. Use e_entry, a 16-byte-aligned stack, a0 for the argument address, a1 for its byte count, and ra for the return trampoline. Return status only after device writes are visible. Keep buffers/programs pinned until completion or confirmed quiescence after an error.

**Memory ownership.** The device allocator is authoritative for physical placement. An allocation handle resolves to a device representation with dtype, logical shape, physical extent, layout and access permissions. A prepared banked matrix may need multiple slices; pass a device-readable slice descriptor rather than assuming one dense pointer. The target profile defines program, argument, stack, scratchpad, GPR and shared-bank windows. PNM loads/stores can reach shared banks and GPR. Global Buffer access uses explicit transfers. Define controller drain and any required cache maintenance for PIM-write/PNM-read and PNM-write/PIM-read. Reuse resident representations between launches; insert a device-side conversion where producer and consumer layouts differ.

| Stage | Library / service work | gem5 team work | Completion evidence |
|---|---|---|---|
| N0. Joint ABI | Derive a target profile and linker script from the agreed map. Specify typed kernel arguments and buffer bindings. Review the inactive ELF validator and register setup. | Confirm the map, supported extensions and VLEN, control interface, trap/return delivery, cache and controller completion semantics. | Both sides decode the same golden launch and reject a mismatched profile. |
| N1. Host and guest protocol | Implement version negotiation, load/launch/unload, pinned resources and two static ELF fixtures. Exercise the native transport with a test executor and malformed or stale bindings. | Supply control-interface contract for loading segments, setting PC/registers and obtaining completion. | One upload of the program serves repeated launches; invalid requests fail before start. |
| N2. PNM execution | Build an RVV vector-add kernel; send argument descriptors and compare output with the CPU reference. | Load PT_LOAD segments, zero BSS, enforce disjoint fixed program ranges or activate one image per launch, start the separate PNM core, capture return/trap and stop it on failure. | PNM PC hits ELF entry, vector/tail results match, trap and timeout release resources only after quiescence. |
| N3. PIM from PNM | Port the supplier GEMV issue loop to a freestanding target shim using preallocated descriptors. | Decode PIM instructions issued by PNM, route to the controller, implement ordering and expose issuer-tagged traces. | Two GEMVs reuse one matrix handle. Trace shows PNM PC, GB transfer, bank MAC and result visibility; FP16 outputs match the fixture. |
| N4. Mixed device region | Chain PIM gate/up GEMV, PNM SiLU/multiply, and PIM down GEMV. Select programs through L1 and use L2/L3 for resident tensors and layouts. | Verify cross-engine visibility, shared-buffer addressing and completion under repeated launches. | Compare intermediate results and final logits where applicable, count one activation upload and one output download for a CPU-boundary MLP with resident weights. |

The N2 vector-add ELF and per-length memory fixtures have passed native load/relocation checks. See the [separate-core handoff](pnm-launch-abi.md) for target execution and evidence collection.

N3 now has a freestanding GEMV ELF, a native supplier-reference test and a two-launch target fixture. See the [N3 build, measured counts and target verification procedure](pnm-launch-abi.md). Target execution and issuer-tagged traces are pending.

N4 has a native mixed-MLP capture and numerical reference test, plus a freestanding pointwise ELF. See the [N4 ABI, native test and target acceptance procedure](pnm-launch-abi.md). Separate-core execution and cross-engine visibility remain open for extended gem5.

The Linux libpim memory setup uses /dev/mem and mmap, and some tensors rely on process-local backing. The PNM shim needs device pointers and explicit workspace. Port instruction emission and tiling independently of Linux allocation. A PNM core running the pinned program can first be demonstrated with a static fixture before connecting external PyTorch. Native and stock x86 FS tests validate protocol handling; instruction execution on the target core and controller requires extended gem5.

## What the delivered code executes

The snapshot comes from `riscv-only` at `6b242c1ce1`. It contains selected simulator and library sources, headers, two example programs and golden outputs. Building the simulator requires the full fork and Ramulator 2.1. The background document describes the original commands and subsequent errata. Use the implementation and the ISA/API drafts from the same revision when resolving differences. [Snapshot inventory](../pim-isa-library-2026-11-10/README.md) · [Background and errata](../pim-isa-library-2026-11-10/docs/pim/pim_background_and_spec.md) · [ISA](../pim-isa-library-2026-11-10/docs/pim/pim-isa-spec.md) · [C API](../pim-isa-library-2026-11-10/docs/pim/pim-library-api.md)

| Stage | Owner | Artifact or action |
|---|---|---|
| Prepare | NDP application/build frontend | Build the resident worker for one target profile. At runtime, select a supported logical operation and bind its tensors. |
| Compile and link | Build host, which can be the gem5 team's Linux machine | Cross-compile a RISC-V Linux ELF with the matching layout header and static libpim archive. The example uses rv64gcv_zicbom, lp64d and disabled FP contraction. |
| Launch | gem5 run script | In SE, pass the ELF to `--binary`. In FS, boot RISC-V Linux, provision the ELF and inputs, and invoke it inside the guest. The supplied FS boot path needs an added workload delivery and invocation step. |
| Execute | Simulated RISC-V CPU and PIM memory controller | The CPU runs the ELF. libpim emits custom instructions; gem5 decodes them and Ramulator executes the modeled memory operations. |
| Complete | Workload and runner | Capture output, status and simulation evidence. The application's completion event resolves after output visibility and successful execution. |

## Ownership and deployment

External PyTorch owns model execution. NDP L3 chooses which weights to prepare and reuse; L2 owns logical descriptors, tensor lifetimes, engine requests and completion. The gem5_pim L1 adapter maps those decisions to remote handles and operations. The resident worker owns the libpim objects for one session. libpim owns physical allocation, layout, tiling, coherence and instruction emission; gem5/Ramulator models execution and timing. The simulated RISC-V CPU runs the worker. The separate PNM-core path is specified in N0–N4 above.

Build one profile-specific worker with GEMV linked in. Compile during deployment and bind new shapes and handles at runtime. A user-defined kernel compiler or artifact service would be a separate feature. [Example Makefile](../pim-isa-library-2026-11-10/tests/test-progs/pim-gemv/src/Makefile) · [SE/FS launcher](../pim-isa-library-2026-11-10/configs/example/gem5_library/riscv-pim.py)

Start with bias-free `y = W[N,K] x[K]`. The guest allocates `W` as `PIM_MATRIX_AB`, x as `PIM_VECTOR_GB` and y as `PIM_VECTOR_SB`. It uploads W once, updates x for each launch, calls `pim_gemv(y,W,x)` and downloads y on request. The vector GB tensor has a host-side backing inside the workload; libpim writes each tile to the units’ GBs. Matrix row slices, unit/bank placement, register initialization, GB loads, all-bank MAC and result collection are already implemented in libpim. Use its tensor and synchronization APIs. [GEMV example](../pim-isa-library-2026-11-10/tests/test-progs/pim-gemv/src/pim_gemv.c) · [Command generation](../pim-isa-library-2026-11-10/util/pim/src/ops.c)

Keep the existing packet emulator and its CENT tests as an independent backend. Add an opt-in `src/l1/gem5_pim/` implementation with a public factory under `include/ndp/l1/gem5_pim/`. It consumes the logical operation and layout requirements from L2. Bypass the current NDP1 command builder, GB Transfer packets, Shared offsets and prepacked stripe bytes for this target. The supplier's layout header and libpim allocator own physical placement. Existing NIP1/NDP1 binaries remain useful for stock-FS IPC regression; they do not produce the new custom instructions.

| Proposed object | Contents and lifetime |
|---|---|
| Worker executable | Build once for the selected target, libpim archive and layout header. Verify the worker profile at session open. |
| GEMV request | Operation and resident output, weight and input IDs. Validate shapes, roles and types registered at tensor creation. |
| Handle ownership | L1 associates an NDP allocation/representation with a session-scoped remote ID. Only the worker maps that ID to a libpim object. Release it on last use and clean up on disconnect. |
| Completion | Per-launch status, libpim error detail and output readiness. For a standalone SE job also check process exit and output length. After uncertain termination, quiesce the job before buffer reuse. |

**L2 issue to resolve before coding the adapter.** Current `allocate(bytes)` cannot by itself select a matrix, GB vector or result vector; several shapes and roles have the same byte count. For a first GEMV path, retain logical dense inputs and lazily materialize the appropriate libpim tensor when `transform` or `execute` supplies the descriptor. Cache prepared weights by handle, role and layout, invalidate on writes and release with the handle. Account for staging and physical row slices separately; surface allocation failure before marking a weight prepared. Test that two equal-size inputs of different roles cannot alias. If exact eager capacity reservation or L2 layout validation cannot be expressed this way, add an optional descriptor-aware allocation hook with a byte-allocation default for existing backends. Audit `reserve_buffer`, planner validation and layout queries before changing the public API. [IBackend](../include/ndp/l1/backend.hpp) · [Session](../src/l2/session.cpp) · [Planner](../src/l2/planner.hpp)

**Numerical contract.** First reproduce the supplied default BF16 operand / FP32 accumulator workload against its golden output. For the NDP-facing backend use an explicitly generated FP16/FP32 layout and matching runtime configuration. libpim's copy API exchanges host FP32 arrays: widen incoming NDP FP16 values for upload, retain the target accumulator precision during GEMV and round at the NDP FP16 output boundary. Verify this conversion with a target-specific reference before advertising support. Include the memory standard, policy, channel interleave, geometry, numerical formats and layout header in the artifact identity. The layout guard and simulator configuration checks must agree. [Layout selection, numeric formats and limitations](../pim-isa-library-2026-11-10/docs/pim/pim-library-api.md)

## Implementation order and acceptance

| Stage | Deliverable | Acceptance |
|---|---|---|
| K0 · Reproduce supplier baseline | Pin full fork/toolchain and LPDDR6 default profile. Build and run the supplied GEMV natively with libpim_ref; simulation team runs the same fixture under SE. | Match the supplied 131 × 1031 reference and successful exit. Capture command, layout and output. |
| K1 · Native worker and FP16 contract | The native Python/worker prototype already demonstrates BF16 GEMV and handle reuse. The native FP16 worker now links a separate LPDDR6 reference archive and checks the negotiated profile. Specify conversion at each logical FP16 output, including outputs consumed while resident. | Compare non-BF16-representable inputs, rounding-sensitive outputs and repeated launches against a profile-matched reference. Exercise two shapes with the same worker binary. |
| K2 · L1 operation integration | Implement gem5_pim binding, role-specific materialization, release and blocking completion behind IBackend. Support PIM Matvec and compatible bias-free single-token Linear. Keep physical placement inside libpim. | Public C++ Session prepares one weight, launches twice and reads matching results. Verify allocation failure, stale handles, shape tails and cleanup. |
| K3 · External PyTorch integration | Expose K2 through the existing Python Session binding and exercise it with PyTorch CPU FP16 tensors. Reuse the native worker protocol and lifetime from K1. | An actual PyTorch caller uploads one weight, updates x twice and obtains two reference results from one worker session. Verify upload counts and distinct results. |
| K4 · FS handoff | Package the same kernel ELF and fixture into the RISC-V guest; document deployment, invocation, output retrieval, permissions for Way 1 /dev/mem and access to the reserved PIM region. Team/user executes FS. | Confirm workload execution beyond kernel boot, numerical match, and device-side PIM request counts or traces. Separate simulation startup, compilation, copies and kernel ROI when reporting timing. |
| K5 · Extend coverage | Add elementwise Add/Mul only after defining paired tensor allocation and the NUM_REGS=1 constraint. Inventory decoder operations and assign an explicit execution engine to each. | Compare each new kernel to libpim_ref. Full decoder acceptance requires an agreed path for unsupported operations, including normalization, softmax and PNM GEMM. |

For K0 in the complete fork, the example below uses the supplied default layout. Use an isolated build directory or serialize custom layout builds because the example reuses the name libpim_custom.a. [Build recipe](../pim-isa-library-2026-11-10/util/pim/README.md) · [Reference input](../pim-isa-library-2026-11-10/tests/gem5/pim/ref/pim-gemv/lpddr6.args) · [Reference output](../pim-isa-library-2026-11-10/tests/gem5/pim/ref/pim-gemv/lpddr6.out)

```sh
make -C tests/test-progs/pim-gemv/src TARGET=host PIM_STD=lpddr6
tests/test-progs/pim-gemv/bin/host/lpddr6/pim-gemv 131 1031
make -C tests/test-progs/pim-gemv/src TARGET=riscv PIM_STD=lpddr6
build/RISCV_RAMULATOR2/gem5.opt configs/example/gem5_library/riscv-pim.py \
  --mode se --memory lpddr6 \
  --binary tests/test-progs/pim-gemv/bin/riscv/lpddr6/pim-gemv --args 131 1031
```

## Decisions needed from the gem5 team

For target runs, confirm the full fork/build environment, matching FP16 layout and device counters. For FS, agree on worker ELF deployment, image permissions and the connection from the external host to the guest. The external client and resident-worker split is settled. Native K1–K3 can proceed before the FS transport is selected.

## Native external Python client checkpoint

Run from the repository root with SCons, a native C/C++20 toolchain and Python 3:

```sh
make -C tools/gem5/pim_kernel
python3 tools/gem5/pim_kernel/client.py build/pim-kernel-worker
```

The Python process owns the worker subprocess and exchanges length-prefixed messages over its stdin/stdout. Each four-byte little-endian length covers the following body. Version 1 starts with H + version byte; the reply identifies the LPDDR6/BF16/FP32 profile. C creates a role-specific tensor, U uploads FP32 values, G launches GEMV using output/weight/vector handles, D downloads FP32 values, R releases a handle and Q closes the session. Replies start with status byte 0 (success) or 1 (diagnostic text). EOF also releases all tensors. This native transport keeps the FS deployment choice open.

The example uploads one 2×3 weight and reuses all three handles for two launches. Expected results are (6, 1) and (−1, −4). The supplied 131×1031 native reference workload also matched the committed LPDDR6 golden output. The BF16 worker retains its original protocol.

## K1 native FP16 acceptance

```sh
make -C tools/gem5/pim_kernel fp16
python3 tools/gem5/pim_kernel/test_fp16.py build/pim-kernel-worker-fp16
```

The FP16 worker uses protocol version 2 and little-endian binary16 upload/download payloads. libpim widens the interface to host float arrays and accumulates in FP32. On a successful GEMV the worker copies the FP32 SB output through a local float array into a GB-vector tensor with FP16 storage. Download and a subsequent GEMV both consume this rounded representation. It stays in the worker process; the conversion incurs local data movement and the GB vector has host-side backing in libpim. The test covers two matrix shapes, tie-to-even, two half-ULPs accumulating before narrowing, FP16 values distinguishable from BF16, repeated launches and profile rejection.

The supplier layout generator imports Ramulator Python modules absent from this environment. Native acceptance uses a header derived from the supplied LPDDR6 layout with identical address geometry, FP16 operands of the same width and a separate link guard. For gem5 target acceptance, generate the full matching header in the complete fork. The K2 integration below uses this native reference layout.

## K2 native C++ Session acceptance

Build the optional worker and register the integration test with its absolute path:

```sh
make -C tools/gem5/pim_kernel fp16
cmake -S . -B build -DNDP_GEM5_PIM_WORKER="$PWD/build/pim-kernel-worker-fp16"
cmake --build build --target ndp_gem5_pim_tests
ctest --test-dir build -R '^gem5_pim_session$' --output-on-failure
```

Link the POSIX target `ndp_gem5_pim` and pass the backend returned by `make_gem5_pim_backend(worker_path)` to `Session::create`. Each Session owns one FP16 worker. Upload creates a remote dense GB-vector representation; preparing a weight creates a matrix tensor and converts it locally in the worker. Subsequent GEMVs reuse that matrix handle. GEMV outputs expose the worker's rounded FP16 representation to the next operation. The supported operations are PIM Matvec and bias-free, single-token Linear.

The v2 worker additionally accepts I (three little-endian uint32 values: banks across all units, row bytes, usable row count) and T (destination and source handles: copy logical values between equal-size representations). L2 accounts for dense logical bytes and for the full row slices of a prepared matrix. The worker owns physical placement and internal accumulator/temporary storage; Session memory statistics do not measure its process memory. Transport failure terminates and reaps the owned worker before the L1 call returns. K3 exposes this target through the external PyTorch Session binding.

## K3 external PyTorch acceptance

Select `backend="gem5_pim"` and pass the FP16 worker executable via the keyword `worker`. The integration test creates two PIM Linear projections from CPU torch.float16 modules, runs two distinct inputs in the same session and checks both final results against a PyTorch reference with an FP16 rounding boundary between layers. Python API transfer counts cover one upload per weight, one upload per input and one download per final output. The intermediate stays in the worker. The native process path is exercised here; target FS deployment and connectivity remain K4. [Usage](pytorch-integration.md#resident-libpim-projections-from-external-pytorch) · [Integration test](../tests/python/test_gem5_pim.py)

## K4 stock FS reference path

External PyTorch can open a blocking TCP connection and pass its descriptor as `worker_fd`. The stock x86 FS configuration bridges that session to one guest reference worker through m5 file operations. Run the [two-terminal acceptance procedure](gem5-k4-stock-fs-user-verification.md) to verify both GEMV results and guest shutdown. Execution of RISC-V PIM instructions and device timing still require the target fork.

## K4 target handoff after stock FS acceptance

For the new `gem5-ParallelPIM-main` delivery, use the updated build/profile
instructions in [V0 integration](extended-gem5-v0.md). The commands below describe
the older supplier snapshot (`util/pim`) and must not be used unchanged for the
new fork (`libpim`). Both paths require target execution evidence.

The user verified the stock x86 FS session, including both FP16 outputs and guest worker shutdown. The next checkpoint runs those same projections through the RISC-V instruction-emitting libpim build. A file fixture allows the gem5 team to validate this step independently of its guest networking and boot-script integration.

On the build host, start in the NDP repository root. Set `PIM_ROOT` to the absolute path of the complete supplier fork with its Ramulator layout generator dependencies. Install SCons and a Linux RISC-V C/C++20 cross toolchain supporting rv64gcv_zicbom and static lp64d libraries.

```sh
export PIM_ROOT=/absolute/path/to/full-gem5-fork
bundle="$PWD/build/riscv"
mkdir -p "$bundle"
python3 "$PIM_ROOT/util/pim/gen_pim_layout.py" --standard lpddr6 \
    --num-fmt fp16 --acc-fmt fp32 -o "$bundle/lpddr6_fp16.h"
make -C tools/gem5/pim_kernel riscv-fp16 PIM_ROOT="$PIM_ROOT" \
    PIM_LAYOUT="$bundle/lpddr6_fp16.h" CROSS_COMPILE=riscv64-linux-gnu-
python3 tools/gem5/pim_kernel/target_fixture.py "$bundle"
file "$bundle/pim-kernel-worker-fp16"
```

Expect a statically linked 64-bit RISC-V ELF. Use the generated header for the simulator's `--layout` with `--memory lpddr6 --num-fmt fp16 --acc-fmt fp32`. Match the generated channel geometry and OS memory reservation. In the target FS image, provision the ELF with execute permission plus requests.bin and expected.bin in one directory. Use the team's guest startup hook or shell after boot. The guest needs access to /dev/mem and the layout's reserved FS window outside System RAM. From that directory run:

```sh
PIM_MODE=fs ./pim-kernel-worker-fp16 < requests.bin > actual.bin &&
    cmp expected.bin actual.bin && echo K4_TARGET_NUMERIC_PASS
```

The comparison covers the profile, all handle operations, both GEMV results and the close acknowledgement. Capture the ELF and layout hashes, simulator config, guest exit status, and PIM controller trace or counter deltas over this workload. Identify the all-bank MAC activity for the two launches before accepting device execution. The exact counter names depend on the target fork. After target acceptance, connect the same worker stream to the external client and proceed to K5 kernel coverage.
