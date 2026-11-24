# Extended gem5 PIM integration review and validation plan

Status: the initial review was static only. V0 native build/profile integration
and focused native checks were subsequently completed on 2026-11-17; see
[V0 results and commands](extended-gem5-v0.md). RISC-V worker and smoke ELF builds
and static inspection subsequently passed; [target verification](extended-gem5-smoke-verification.md)
is prepared. Subsequent user-run simulator checkpoints are recorded below.
Existing native and stock-FS results are not evidence
that this extended simulator executes the NDP kernels correctly.

Active implementation and validation target gem5. Deferred board integration
requirements are recorded in the [VHK158 handoff](vhk158-integration-requirements.md).

## User verification checkpoints

- FS smoke: user reported exact replies, guest exit and controller completions
  (WrGb 32, AbMac 32, RdReg 4).
- SE Atomic boundary: user reported 8 cases / 24 GEMVs passing, with WrGb 15,936,
  AbMac 15,936 and RdReg 3,108 accesses. This is not Timing acceptance.
- Lifecycle/error recovery: native suite passes 11 expected rejections and four
  successful GEMVs in one session. The user reported SE Atomic acceptance
  (WrGb 64, AbMac 64, RdReg 8), followed by Timing acceptance.
- The user reported `NDP_PIM_SE_TIMING_PASS suites=boundary,lifecycle`.
  Both fixture suites now have Timing acceptance. External-client SE integration
  passed Atomic and Timing as reported by the user; see
  [the client procedure](gem5-se-external-client.md). No later gate is implicitly
  accepted by these results.

Current arithmetic gate: the user reported chain SE Timing acceptance (18 GEMVs,
WrGb 5,568 / AbMac 5,568 / RdReg 2,594). A new `decoder` fixture passed 11 native
checks for worker-CPU arithmetic plus a final PIM GEMV. The user reported its SE Timing
acceptance (WrGb 16, AbMac 16, RdReg 1); see the current user-verification manual. This does not
complete the separate-PNM execution or full-decoder/KV gates.

The next `decoder_state` fixture passed native checks for a tiny one-head block:
prefill/decode, tokenwise execution and cache reset (3 schedules, 7 tokens,
33 exact result checks). The user reported SE Timing acceptance with WrGb 688,
AbMac 688 and RdReg 86 completions. This is worker-protocol graph
validation, not production PyTorch adapter or separate-PNM acceptance.

Production PyTorch adapter: a tiny one-layer, single-device GQA model passed
native TCP relay tests (prefill/decode, tokenwise comparison, capacity rejection
and reset; seven target token evaluations). The user reported both SE Atomic
and Timing acceptance in `verify_decoder_remote.py`, with zero logit error.

Request isolation: `verify_decoder_remote.py --suite isolation` passed native TCP
validation (two live requests sharing one model/session, 15 successful token
evaluations, seven zero-error comparisons). It checks interleaving, capacity
rejection, reset and close against isolated request baselines. The user confirmed SE Atomic and Timing acceptance.

Two-device Linear passed with two native workers. The user also confirmed SE
Atomic and Timing acceptance with two independent simulator endpoints for output
sharding and host reduction of FP32 partials. See [the client procedure](gem5-se-external-client.md).

Two-device decoder, request state and layer transfer passed on two native workers.
The user confirmed that all three checks also passed in SE Atomic and Timing
with two independent gem5 instances. See [the three checks](gem5-se-external-client.md).

Extended FS external client: the guest ELF passed cross-build and static inspection.
Native mailbox tests passed for dependent GEMVs and both disconnect cases.
All five external FS scenarios passed: gemv, chain, partial, lost and decoder.
In standalone GEMV run `run-FSSTREAM-3lWGRvQZ`, the client reported [6, 1]
and [-1, -4] and printed `NDP_FS_CLIENT_PASS`. The worker closed after 22
requests and two GEMVs. The guest exited with code 0 and the runner printed
`NDP_FS_EXTERNAL_PASS case=gemv` with WrGb 32, AbMac 32 and RdReg 4.

FS chain passed in run `run-FSSTREAM-w47xBUSO`. The saved client verdict
covers final values 11 and 2. The worker closed after 24 requests and four
GEMVs, the guest exited with code 0, and the runner reported
`NDP_FS_EXTERNAL_PASS case=chain` with WrGb 64, AbMac 64 and RdReg 6.

The user confirmed FS partial-frame disconnect acceptance. The relay received
12 of 13 GEMV body bytes. The worker completed six setup requests and zero GEMVs.
The Timing trace had zero completed AbMac operations, the guest exited with
code 0, and the runner printed `NDP_FS_EXTERNAL_PASS case=partial`. The lost-reply case also passed in run `run-FSSTREAM-nh8QFgai`: seven requests,
one completed GEMV, Timing completions WrGb 16 / AbMac 16 / RdReg 2, guest exit
code 0 and `NDP_FS_EXTERNAL_PASS case=lost`.

The user confirmed completion of the external single-device FS decoder check.
Run `run-FSSTREAM-MX0xnOgS` closed after 1,178 requests and 35 GEMVs, with
560 WrGb, 560 AbMac and 140 RdReg completions, guest exit code 0 and
`NDP_FS_EXTERNAL_PASS case=decoder`. The separate native TCP check had zero
maximum absolute error over seven target token evaluations.
See the [FS client procedure](extended-gem5-fs-client.md).

The [Qwen3 single-layer timing procedure](qwen3-layer-timing.md) runs the dense
and MoE workload with synthetic data in SE. Both user Timing runs completed at batch 1 and context 16. The dense layer took
106,903,400,589 ticks and the MoE layer took 27,628,120,224 ticks.

## Reviewed delivery

The supplied directory is `gem5-ParallelPIM-main`. Its README identifies the
upstream baseline as `f5c5a6e390`; this is a documentation claim, not a verified
revision identity for the delivered directory. Record a source revision or
archive hash before producing acceptance results.

Primary sources:

- [Fork overview](../gem5-ParallelPIM-main/README.md).
- [libpim API, allocation, layout and coherence](../gem5-ParallelPIM-main/libpim/README.md).
- [RISC-V SE/FS configuration](../gem5-ParallelPIM-main/configs/example/gem5_library/riscv-pim.py).
- [PIM memory parameters](../gem5-ParallelPIM-main/src/mem/ramulator2/PimRamulator2.py)
  and [controller implementation](../gem5-ParallelPIM-main/src/mem/ramulator2/pim_ramulator2.cc).
- [LPDDR6 layout](../gem5-ParallelPIM-main/include/gem5/pim/pim_layout_lpddr6.h).
- [Supplier test driver](../gem5-ParallelPIM-main/tests/gem5/pim/run_p4r.py).
- [Supplier LLM workload](../gem5-ParallelPIM-main/workload/llm/README.md).

The normal instruction path is:

```text
RISC-V Linux/SE program
  -> libpim placement, cache maintenance and custom instruction wrappers
  -> RISC-V decoder and CPU memory request path
  -> PimRamulator2
  -> Ramulator2 PIM sequencing, timing and value model
```

The fork provides RVV-based custom scalar/vector PIM instructions, bank memory,
Global Buffers and registers. `PimSystemMemory` combines OS memory and PIM memory.
LPDDR6, DDR5, GDDR7 and HBM4 configurations and several address policies exist.
The initial NDP integration should select one profile, not test every combination.

The example config creates a RISC-V processor with `--cores` cores and an optional
boot-to-detailed CPU switch. Its presence does not establish a separate guest
service CPU plus an independently loaded PNM core. The reviewed configuration,
PIM headers and memory implementation do not establish the NDP LoadProgram /
LaunchProgram service, scratchpad map, or core start/return adapter.

The supplier's LLM example uses core 0 to issue PIM commands and other CPU threads
for RVV work. It offloads attention GEMVs while projections largely run on RVV;
this is different from our current decoder placement. Its success would be useful
supplier evidence, but would not validate our decoder, worker or PNM launch ABI.

## Compatibility findings and required preparation

| Area | Current NDP integration | New delivery and required action |
|---|---|---|
| Build location | Worker Makefile uses `PIM_ROOT/util/pim` | Library is `PIM_ROOT/libpim`. Adapt build paths and target instructions before execution. Do not silently replace the old reference baseline. |
| Geometry | Old LPDDR6 header has 8 channels | New default has 16 channels, 16 banks per channel, 2 KiB rows, 32 GiB PIM capacity. Regenerate artifacts and references; old hard-coded physical addresses and instruction counts cannot be reused unchanged. |
| Number format | NDP tensors use FP16; reductions may retain FP32 partials | Supplier defaults to BF16 operands / FP32 accumulators. Generate an explicit FP16/FP32 header and use the same profile in the archive, worker and simulator. |
| Mapper | Old fixtures include profile-specific packed addresses | New default is `RoRaBaChCo`, 2 KiB channel interleave. Use the generated header and `pim_addr.h`; keep physical packing in libpim/device adapter, not L2. |
| Addresses | PNM fixtures have provisional code/shared/stack addresses | New default PIM base is `0x800000000` in SE and `0x1000000000` in FS. These do not automatically match the separate-core ABI. Reconcile maps before running those fixtures. |
| Worker API | Uses `pim_tensor_create/destroy`, `pim_copy_to/from`, `pim_gemv` | These entry points exist in the new public header. This is source-level overlap, not proof of build or behavioral compatibility. Verify error, allocation and numeric behavior. |
| Profile handshake | `LPDDR6/FP16/FP32` string | Does not identify channels, mapper, VLEN or layout hash. At minimum bind these to a run manifest; add a versioned profile identity before supporting arbitrary remote configurations. |
| Completion | L2 expects blocking completion and visible output | PIM store-like requests can be acknowledged at acceptance (`posted`). Verify libpim fence/cache maintenance and readback ordering under timing execution. A packet response alone is insufficient evidence. |
| Execution engine | Native worker can compute dense operations in C++ | Running that worker on a simulated RISC-V Linux CPU is a CPU reference/integration path. It is not proof of execution on the intended separate PNM core. |
| Multiple devices | NDP DeviceGroup uses independent sessions | Channels and CPU cores are not independent NDP devices. Establish disjoint device/controller ownership before accepting TP/PP results on a shared simulated system. |

For the new default LPDDR6 mapper, the generated header states:

```text
PIM-relative byte offset:
row[34:19] | bank[18:17] | bank-group[16:15] | channel[14:11]
           | column[10:5] | byte-in-column[4:0]
```

This is a profile-specific offset map, not a universal host physical address map.
Adding the selected SE/FS window base gives the corresponding physical address
for Way 1. Way 2 additionally requires correct page-backed buffer placement.

## Previously planned checks that remain open on the real target

| Existing plan | Existing evidence | Remaining target acceptance |
|---|---|---|
| K0–K3 supplier/native worker, C++ and external Python | Prior native reference and protocol checks | Repeat essential cases with the new profile and actual instruction-emitting RISC-V binary. |
| K4 FS worker | User reported stock x86 FS acceptance | Extended RISC-V FS worker, custom instruction/controller evidence, profile-matched outputs and clean exit. |
| K5 decoder kernel coverage | Native GEMV, dense arithmetic, attention and FP32 partial support | Verify actual target support and execution location of every operation used by the decoder. Reject unsupported routes explicitly. |
| N0–N1 program ABI and capture | Launch contract and native/capture checks | Agree target map, loader, feature negotiation, start/return/trap and quiescence semantics. |
| N2 vector add | Freestanding fixture/native checks | Execute ELF on the separate PNM core, verify registers, output guards and completion. |
| N3 PNM-issued GEMV | Native reference and repeated-launch fixture | PNM-PC/issuer evidence, custom instruction and controller activity, correct results for two inputs with resident weights. |
| N4 mixed MLP | Native mixed PIM/PNM chain | PIM gate/up -> PNM SiLU/multiply -> PIM down, layout compatibility and cross-engine visibility without host intermediate round trips. |
| Attention, KV and prefill | Native decoder/cache and chunked-prefill checks | Actual target KV append/read, RoPE, causal attention, PNM GEMM for multi-row input and PIM GEMV for one row. |
| TP, layer placement, requests and pipeline | Native host reduction, isolation and overlap checks | Actual device isolation, FP32 partial transfer/reduction, request-local KV, and simulated-stage overlap. |

See [kernel plan](gem5-pim-kernel-plan.md), [PNM ABI](pnm-launch-abi.md),
[parallelism](parallelism.md), [model validation](tensor-parallel-model-validation.md),
[prefill](prefill.md), [request scheduling](request-scheduling.md) and
[pipeline execution](pipeline-execution.md). Historical commands that reference
`util/pim` must be adapted; they are not ready-to-run commands for this delivery.

## Recommended validation sequence

Each gate requires saved evidence before proceeding. Native preparation is library
work; the user/gem5 team runs FS validation, as previously agreed. SE validation
is a small diagnostic step before FS, not a replacement for FS acceptance.

| Gate | Work and inputs | Acceptance and evidence |
|---|---|---|
| V0: profile and build adaptation | Pin sources/toolchain; adapt `libpim` path; generate one LPDDR6 FP16/FP32 profile; build matching native reference and RISC-V worker; audit ISA requirements and target feature flags. | Manifest includes source/header/archive/ELF hashes, geometry, mapper, numeric formats, VLEN, CPU/memory/cache configuration. No reference-backend macro in the target executable. Header/archive layout guard and simulator profile agree. |
| V1: supplier SE smoke | Run a small supplied GEMV with the supplier default first, then a matched FP16 case. Start with one core and a bounded instruction/tick budget. | Correct exit and outputs; custom PIM instructions and nonzero corresponding controller activity. Failure here blocks NDP worker debugging. |
| V2: NDP worker SE fixture | Adapt/reuse `tools/gem5/pim_kernel/target_fixture.py`: one 2x3 weight, two inputs, release and close; then boundary cases below. | Logical results `[6,1]` and `[-1,-4]`; exact binary replies for this exactly representable fixture; weights survive both launches. Capture MAC activity for each launch and verify no reference fallback. |
| V3: RISC-V FS memory and completion | Boot a suitable kernel; use Way 1 first and reserved PIM window, root `/dev/mem` access, correct `PIM_MODE=fs`, working cache maintenance. Repeat V2 and dirty-cache/reuse cases on a timing CPU. | Correct guest exit/replies, visible results after completion, no stale cache values or overlapping allocation. Save kernel/config, console, request/reply files, trace and controller statistics. |
| V4: external host session | Connect external Python -> L1 -> guest worker over the chosen bridge; preserve the binary stream and keep diagnostics separate. Exercise partial frames, two launches, release, close and disconnect. | Same results as V3; one persistent weight; no duplicate execution on transport failure; uncertain operations are not automatically retried or their buffers prematurely reused. |
| V5: decoder operation coverage | Tiny deterministic block; per-operation outputs, KV growth and reset; one-token decode and multi-token prefill. Include FP32 partial output and its transfer. | Every operation has a declared execution location. Compare intermediate tensors, not only logits. PNM-labeled CPU reference results remain labeled as such until the separate-core track passes. |
| V6: separate PNM execution | Resolve topology/loader prerequisite, then N2 -> N3 -> N4 and attention/GEMM kernels. Regenerate fixtures for the agreed profile. | ELF load/start/return/trap, guards, repeated launches, PNM-issued PIM traces and cross-engine visibility. No success acknowledgment before required writes and PIM work are complete. |
| V7: multi-device and serving | Two small disjoint devices first; TP partial reduction, stage transfer, two interleaved requests, cancellation and chunked prefill. Add pipeline overlap last. | Results match isolated runs; request/device data cannot alias; FP32 reduction stays on the external host; real controller ownership and simulated overlap are recorded. Multiple sockets alone do not pass this gate. |
| V8: representative model and timing | Small layer subset, then bounded full-model cases only after correctness gates. Freeze CPU/cache/controller settings and use a defined ROI. | Numerical report and execution placement are acceptable; report simulated ticks separately from wall time and network/boot/loading costs. No speedup claim from native threads or Atomic results. |

V0–V5 can advance on the provided Linux/SE CPU path, subject to integration work.
V6 is a separate architectural gate: successful Linux worker execution does not
waive it. V7 may first test isolated simulator instances for functionality, but
shared-system performance requires an explicit multi-device topology and timing
model. These are different experiments and must be labeled separately.

### Small correctness matrix

Use the selected profile to derive dimensions. Let `L` be the number of matrix
output rows placed across all banks in a row slice, and `T` the operand elements
per bank row. For the new default LPDDR6 FP16 profile, `L = 16 * 16 = 256` and
`T = 2048 / 2 = 1024`.

- GEMV `(M,K)`: `(2,3)`, `(L-1,17)`, `(L,17)`, `(L+1,17)`, `(3,T-1)`,
  `(3,T)`, `(3,T+1)`, `(L+1,T+1)`. These cover row groups, column tiles and both tails.
- Use zero, signed, one-hot and bounded random inputs; include FP16 values not
  exactly representable in BF16. Verify padding does not contribute and guard
  bytes are unchanged. Reuse weights with a changed input and then mutate weights.
- Test encode/decode and element placement at channel/bank/row boundaries, at
  contiguous-piece and 4 KiB boundaries, and near reserved register rows. Do not
  allocate the entire 32 GiB merely to test a boundary.
- Check valid and invalid shapes, stale handles, allocation exhaustion, live
  resource release and deterministic cleanup. Run fatal simulator negative cases
  separately from reusable-session tests.
- Verify host write -> PIM read, PIM write -> CPU read, and, for V6, PNM write ->
  PIM read / PIM write -> PNM read. Repeat with dirty cached data and back-to-back
  dependent operations. An uncached PIM request does not itself flush CPU caches.
- For prefill use 4 tokens versus chunks of 2+2, then one decode token. Check KV
  positions, causal masking, projection dimensions and intended GEMM/GEMV routing.
- For TP use two devices and values sensitive to premature FP16 rounding. Check
  FP32 partials before the host sum and final FP16 conversion.

Start with a few selected cases, not the Cartesian product of every CPU, VLEN,
mapper and memory standard. Add one timing CPU early; expand to O3, alternate
VLEN, Way 2 and alternate mapping only when those are required by the target.

### Reuse supplier tests selectively

The supplier driver has `config`, `decode`, `path`, `cache`, `cpuless`, `fs-boot`,
`fs-way1`, `fs-protocol`, `fs-way2`, `sv48` and other groups. Select explicit cases
from the driver with `--only` after reviewing their prerequisites. In that driver,
“native” can mean the base `build/RISCV/gem5.opt` simulator, not our native C++
reference worker. Do not confuse these evidence categories.

Use `config` for profile agreement, `decode` for instruction encodings, `path`
and `cache` for request/coherence behavior, and focused FS groups for Linux
mapping and cache-maintenance permissions. CPU-less controller tests help isolate
memory issues but cannot prove the CPU instruction path. The supplier's large
Qwen workload and full/release acceptance suites are not initial integration gates.

## Numeric acceptance and evidence

Use a profile-matched native reference and a simple independent mathematical
reference. Exact fixtures, packet bytes, addresses and lifecycle results require
exact agreement. For accumulated outputs, report max absolute error, relative
error away from zero, and the number of out-of-tolerance elements against a
declared reference. Preserve FP32 partials until the defined FP16 boundary.

Do not make a full-model FP16 difference alone a blocker when it follows the
agreed arithmetic/rounding path. Conversely, wrong dtype, mapping, stale data,
missing operations or early completion are correctness failures, not acceptable
FP16 noise. Greedy-token agreement is supplementary evidence, not proof of every
tensor's correctness.

Each case should save its ID, complete command/config, source and artifact hashes,
input/expected/actual data, exit status, simulated ROI statistics and relevant
trace. Separate library request counts, retired custom instructions and controller
commands: vector splitting and bank sequencing mean these counts need not match.
For N3, replace the old 8-channel fixture's counts with expectations derived from
the new geometry and actual VLEN. Prefer trace-derived completion evidence over
manually asserted completion JSON alone.

## Decisions required before the separate-core and multi-device gates

1. Is the fork's ordinary RISC-V CPU intended to represent the PNM core, or will
   a distinct service CPU and PNM core be configured? Define the issuer and OS role.
2. If distinct, who implements ELF installation, scratchpad/GPR/shared mappings,
   entry/start/return/trap and stop/drain? The existing fixture addresses are proposals.
3. How are multiple NDP devices isolated: independent simulator instances or
   distinct controller/address/core groups in one system? Per-process libpim
   allocation must not give two workers ownership of the same physical rows.

These decisions do not prevent preparing V0–V5, but they prevent claiming that
the intended separate-PNM or multi-device architecture has been validated.
