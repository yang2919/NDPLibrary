# Extended gem5 validation record

## Verified scope

The following simulator results were reported from user runs with the project FP16 profile. The worker runs on the simulated RISC-V CPU and calls libpim for PIM GEMV.

| Environment | Coverage | Evidence |
|---|---|---|
| RISC-V FS smoke | Two GEMVs and orderly guest exit | 32 WrGb, 32 AbMac, 4 RdReg completions |
| SE Atomic and Timing | Boundary shapes and session lifecycle | 8 shapes, 24 launches in boundary tests. Lifecycle covers 11 errors and 4 successful launches. |
| SE Timing | Resident chains, decoder operations and KV state | 18 chained GEMVs with zero intermediate downloads. Decoder tests passed 11 operation checks and 33 state checks over 7 tokens. |
| External client, SE Atomic and Timing | Single-device decoder, interleaved requests, two-device linear and decoder, request state and layer transfer | Single-device decoder matched the functional logits. Multi-device tests used two independent simulator endpoints. |
| External client, FS Timing | GEMV, dependent GEMVs, disconnect before dispatch, lost reply and decoder | Paired client and guest checks recorded below |
| Qwen3 SE Timing | One synthetic decoder layer, one token, batch 1, context 16 | Qwen3-32B dense took 106,903,400,589 ticks. Qwen3-30B-A3B took 27,628,120,224 ticks. |

| FS case | Requests / completed GEMVs | Evidence |
|---|---|---|
| gemv | 22 / 2 | `run-FSSTREAM-3lWGRvQZ`, outputs [6, 1] and [-1, -4] |
| chain | 24 / 4 | `run-FSSTREAM-w47xBUSO`, outputs 11 and 2 |
| partial | 6 / 0 | 12 of 13 body bytes received, zero completed AbMac operations |
| lost | 7 / 1 | `run-FSSTREAM-nh8QFgai`, one GEMV completed before disconnect |
| decoder | 1178 / 35 | `run-FSSTREAM-MX0xnOgS`, 560 WrGb, 560 AbMac, 140 RdReg completions |

Reproduce the [build/profile check](extended-gem5-v0.md), [simulator fixtures](extended-gem5-smoke-verification.md), [SE client runs](gem5-se-external-client.md), [FS client runs](extended-gem5-fs-client.md) or [Qwen3 timing measurement](qwen3-layer-timing.md).

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
| Build location | Worker Makefile uses `PIM_ROOT/libpim` | Build the native reference archive and RISC-V worker against the same delivered library and generated layout. |
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

## Remaining target validation

- Separate PNM core execution needs an agreed address map, ELF loader and start/return/trap interface. Validate vector add (N2), PNM-issued GEMV (N3) and the mixed PIM/PNM MLP (N4) with issuer-PC evidence and memory visibility checks. See the [launch ABI](pnm-launch-abi.md).
- Shared-system tensor and pipeline parallel timing needs disjoint device/controller ownership and a modeled interconnect. Independent SE instances already cover host orchestration and numerical checks.
- The Qwen3 measurement covers the attention and FFN regions of one layer. Full-model serving latency and checkpoint accuracy require separate experiments.

## Correctness fixtures

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
