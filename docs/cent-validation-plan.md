# CENT-based execution/layout validation plan

Status: implemented native conformance harness. See the [recorded results](cent-conformance-results.md) and [reproduction procedure](cent-user-verification.md) for executed coverage and limits. The normative
rules are in [execution/layout specification](execution-layout-spec.md). This
work is native-only and does not run gem5 FS, download model checkpoints or
launch CENT's full paper-reproduction scripts.

## 1. Reviewed upstream and important differences

Pin CENT to `4c2aac6ce6aa3ddfe1944fc9c5d656b90d3fc7d2`. Its `aim_simulator` gitlink
is `2eb1ee04f4e8f3c255c9926e7721081726aa4fe4`. Keep the dependency outside core
library builds; record the revision, local patches, Python/Torch versions and
numeric mode in every report.

Relevant upstream code:

- [cent_simulation/aim_sim.py](https://github.com/Yufeng98/CENT/blob/4c2aac6ce6aa3ddfe1944fc9c5d656b90d3fc7d2/cent_simulation/aim_sim.py): functional bank arrays, GB and latches; `WR_BIAS`, `WR_GB`, `MAC_BK_GB`, `RD_MAC`, and single-bank reads/writes.
- [cent_simulation/TransformerBlock.py](https://github.com/Yufeng98/CENT/blob/4c2aac6ce6aa3ddfe1944fc9c5d656b90d3fc7d2/cent_simulation/TransformerBlock.py): higher-level weight placement and GEMV scheduling. Its weight path assigns contiguous groups of output rows to banks, unlike the library's cyclic mapping.
- [README](https://github.com/Yufeng98/CENT/tree/4c2aac6ce6aa3ddfe1944fc9c5d656b90d3fc7d2): separates Python infrastructure and the AiM simulator build/performance workflow.

Use the functional Python primitives for numerical/state checks. A text trace
accepted by a timing simulator is not a numerical result. Optional later timing
replay must be reported separately and does not gate initial correctness tests.

### Compatibility findings

| Aspect | Library | CENT functional implementation | Validation response |
| --- | --- | --- | --- |
| Address | Tagged byte offsets, row/channel/bank order | Bank arrays indexed by row and element-column; helper flattened order differs | Decode coordinates explicitly; never feed a library byte offset to CENT.address() |
| Row size | `row_bytes` | `DRAM_column` counts tensor elements | `DRAM_column = row_bytes/2` for FP16 logical data |
| GB | Configurable byte capacity V | Tensor of DRAM_column elements | Restrict accesses to negotiated min(R,V); do not infer independent GB-size support |
| Burst | 32 bytes, 16 FP16 values | Configurable burst_length | Fix burst_length=16 for this profile |
| Weight placement | Cyclic rows over C×B banks | Higher-level block distribution | Replay low-level coordinates; do not use upstream weight-mapping helper for the main test |
| MAC initialization | FP16 values read from GPR | Bias list into a selected latch | Decode GPR values and call WR_BIAS per channel |
| MAC arithmetic | Explicit FP32 product / balanced tree / FP32 accumulation | `MAC()` uses `(A*B).sum()`; arrays created without explicit dtype | Set/assert CPU float32; separate numerical equivalence from bit-exact tests |
| Readback | Convert MAC values to FP16 in GPR | RD_MAC returns latch values | Adapter performs the project's explicit FP16 boundary conversion |
| PNM | Functional commands, scratchpad, shared memory | No matching project RV64/RVV execution contract in reviewed primitives | Separate adapter extension/reference checks; never label them native CENT PNM execution |

Do not change production bank layout merely to match CENT's high-level mapper.
Different valid placements can produce the same GEMV; the question is whether
**our actual bytes and command addresses obey our specified placement**.

## 2. Validation architecture: test production output, not a replacement planner

```text
Production Session + NDP backend + recording transport
    ├─ actual uploads, relayout results, command batches, output bytes
    └─ offline fixture (profile, addresses, NDP1 batches, raw binary snapshots)
                       ↓
Independent checks: address formula / padding / bounds / command schedule
                       ↓
CENT primitive replay of the same packed bytes and PIM commands
                       ↓
Compare local output, CENT output, and independent arithmetic references
```

The recording transport delegates to the existing emulator while observing calls.
Do not replace the production `build_matvec()` with a Python GEMV command generator.
Export the real packed buffer after the production relayout completes using raw
backend/transport reads; `Session::to_host(..., PimBanked)` unpacks and therefore
is not an appropriate packed-byte export.

The offline manifest may contain logical test shapes, seed, input/output region
labels and expected sizes. This is test provenance, not new runtime/device metadata.
Serialized commands use the existing codec. Fixture integer fields must have
explicit units. Record command and snapshot hashes for reproduction.

Implemented components:

| File | Responsibility |
| --- | --- |
| `tools/cent/export_case.cpp` | Drive real Session/backend paths and record commands/storage |
| `tools/cent/replay.py` | Decode NDP1 and invoke pinned CENT primitives |
| `tools/cent/reference.py` | Independent address and numeric oracles, no production builder imports |
| `tools/cent/run_cases.py` | Bounded matrix, deterministic seeds, per-case process limits |
| `tests/cent/` | Adapter unit tests, malformed fixtures and regression counterexamples |
| `docs/cent-user-verification.md` | Setup and reproduction commands |

`tools/cent/check_outputs.py` also checks logical results directly from host input bytes.

## 3. Primitive replay mapping

Initialize `PIM(args)` on CPU with `only_trace=False`, `op_trace=False`,
`model_parallel=False`, `FC_devices=1`, `threads=1`, `reuse_size=1`. Provide a
per-case temporary trace path, close its file, and explicitly set/assert float32.
Set C/B/row counts from the fixture, not a full LLM default configuration.

For Shared byte offset a:

```text
row = a // (C*B*R)
within = a % (C*B*R)
c = within // (B*R)
b = (within % (B*R)) // R
column_element = (within % R) // 2
```

Split transfers at every bank-row boundary; require even offsets/lengths for the
FP16 numerical replay subset. DIMM index is zero. Do not round an odd byte offset
into an element index. Byte-level arbitrary-fill/odd-transfer behavior remains
covered by native protocol tests unless separately implemented in the adapter.

| Production operation | CENT replay action |
| --- | --- |
| Packed weights upload/snapshot | Decode actual FP16 bytes; store exact values at `(0,c,b,row,col)` with `store_to_DRAM_single_bank` |
| Shared→GPR Transfer | Read the specified bank segments into adapter-owned GPR byte storage |
| GPR→Shared Transfer | Decode written FP16 slots and update the same CENT bank arrays |
| InitializeMac | `WR_BIAS(0,c,C,0,bias_values,False)` for each selected channel |
| Transfer to channel c’s GB | `WR_GB(0,c,C,gb_offset/2,bytes/32,values,False)` |
| MacAllBanks | `MAC_BK_GB(0,c,C,row,burst_column*16,0,bursts,False,"breakdown_sa_weight")` |
| ReadMac | `RD_MAC(0,c,C,0,False)`; ordered FP16 encode into GPR |

Validate masks explicitly. Main production GEMV initializes all banks; arbitrary
partial-bank initialization is not supported by WR_BIAS directly and must either
use an audited preserve-unselected-latches wrapper or fail as out-of-scope, never
reset unselected latches silently. CENT mask-to-text order also differs from a
naive numeric mask interpretation; initial tests disable trace output.

Do not pass partial final data to WR_GB and rely on old lanes. Replay the entire
zero-padded vector tile that the real command requests. The GB command size is
in bursts, not bytes; each data list must exactly match the intended range.

CENT Shared arrays are the single numerical authority during replay. Adapter GPR
and scratchpad are explicit separate spaces. A later PNM extension must read/write
those same Shared arrays; no detached dense Tensor copy may bypass the tested
PIM storage. Packed byte fidelity is audited separately before float decoding;
NaN payload-bit preservation is not established by CENT float tensors.

## 4. Independent checks and numerical pass criteria

### A. Layout and schedule: exact

Independently calculate every valid W[n,k] address from the specification. Compare
actual packed FP16 bits at those addresses, all padding bytes and guard regions.
Verify address uniqueness, region containment, inverse mapping and nonzero bases.
Check MAC row/burst fields, exact command counts, GPR ranges, per-group initialization,
persistent accumulation and final write length. These are exact tests, with no
numeric tolerance and no dependency on matching two GEMV implementations.

Use impulse vectors and sparse weights at each boundary to expose transpose,
bank-order and tile-order mistakes. Refill freed packed/vector ranges with nonzero
sentinels before reuse to detect accidental reliance on zero-initialized memory.

### B. Unmodified CENT arithmetic: structural and mathematical comparison

Quantize all input values to FP16 first, then represent them exactly in CPU float32
CENT storage. Use zeros, ones, one-hot vectors and small powers-of-two patterns
whose sums are exactly representable to require exact final FP16 output equality.

For arbitrary finite/cancellation cases, compare pre-readback CENT accumulators
and the library arithmetic reference against an independent float64 dot product.
Use a documented error bound rather than an unexplained fixed percentage:

```text
u = 2^-24
n = 4*K + 32
bound32 = (n*u / (1-n*u)) * sum_k(abs(w_k*x_k))
```

This is a conservative finite-input FP32 bound for the selected test domain, not
a hardware accuracy guarantee. Require n*u<1 and exclude FP32 overflow/underflow
from this tier. Compare each pre-FP16 accumulator to float64 using bound32; final
outputs must equal that path's specified FP16 rounding. If only final outputs are
available, add the two paths' FP16 rounding errors explicitly. Near-zero outputs
must not use relative error alone. Log max absolute error and ULP distance.

Do not increase tolerance after a failure without classifying the first divergence.
Arithmetic-order differences are not layout failures, and tolerance must never
hide an exact address/padding/schedule mismatch.

### C. Project numeric profile on CENT topology: bit-exact

Use an explicitly named derived adapter mode that overrides only the burst MAC
reduction with the specified FP32 16-lane adjacent-pair tree. Keep CENT bank/GB/
latch dispatch and the real command stream. Readback rounds once to FP16.
An independent Python scalar rounding reference implements the same mathematical
contract, without copying C++ code or using the production helper.

Require finite FP16 output bits to match the local emulator and scalar oracle.
Test NaN/Inf/subnormal/signed-zero cases separately with specified classification
and sign rules; NaN payload bits are not part of cross-model acceptance.
This is **CENT topology with a project arithmetic adapter**, not unmodified CENT
or vendor AiM arithmetic certification. Report modes B and C separately.

## 5. Dimension and profile matrix

Start with small geometries so exhaustive boundary coverage is cheap:

| Profile | C,B | R,V bytes | P,T | Purpose |
| --- | --- | --- | --- | --- |
| tiny | 1,1 | 64,64 | 1,32 | Single-bank/tile behavior |
| asymmetric | 2,3 | 96,64 | 6,32 | Non-power-of-two banks, unused row suffix |
| gb-limited | 2,2 | 128,64 | 4,32 | GB smaller than bank row |
| row-limited | 2,2 | 64,128 | 4,32 | Row smaller than configured GB; used prefix only |
| wide | 2,16 | 2048,2048 | 32,1024 | Wider bank group / K tiles |

These are validation geometries, not claims about a particular CENT hardware SKU.
CENT's effective GB length remains DRAM_column; only the used prefix is compared.
A hardware-geometry profile can be added after reviewing the upstream CLI defaults.

For each profile use deduplicated positive values:

```text
N ∈ {1, P-1, P, P+1, 2P-1, 2P, 2P+1, 3P+5}
K ∈ {1,15,16,17,31,32,33,T-1,T,T+1,2T-1,2T,2T+1,3T+7}
```

Run the full cross-product with exact patterns, then a bounded random sample
(seeds 0,1,7,42) of finite inputs. Include rectangular/prime cases such as
(3,17), (7,65), (17,127), (33,1025). Longer cases such as N=257,K=4097 and
N=1025,K=11009 are a separate memory-budgeted tier, not default smoke tests.

PNM Matmul tests use M∈{1,2,3,7,17}, K∈{1,15,16,17,31,33,65},
N∈{1,3,15,16,17,33}; Linear uses transposed weight addressing and optional bias.
Compare with an independent sequential-FMA reference. CENT does not supply that
PNM contract: label this as library-PNM validation, then separately test PIM→PNM
Add and PNM→PIM vector/weight production against shared replay storage.

Reject N/K/M=0, mismatched K, wrong dtype, insufficient GPR/scratchpad, insufficient
physical rows, overflow shapes, PIM Matmul and multi-row/bias PIM Linear. Invalid
batch tests must show no state changes, not merely a returned exception.

## 6. Stateful tests and bounded resource use

- Run A→B→A with changed vectors, odd tails and output-group counts. A must repeat.
- Reuse packed weights without re-upload; verify command/snapshot hashes unchanged.
- Allocate guard rows around test regions; check no writes beyond logical output.
- Exercise allocations with nonzero row bases, fragmented/free/reused stripes,
  and capacity exactly sufficient versus one stripe short.
- Reset MAC once per output group; mutate/remove an InitializeMac to ensure tests fail.
- Deliberately alter one bank address, burst count, packed value or tail zero to
  prove that the harness detects the intended fault, not just matching zeros.
- Require no growth in live library allocations over repeated cases.

CENT constructors allocate additional inherited Bank/Channel arrays as well as
useful bank storage. Use minimal per-case row counts derived from all live regions
and guards; estimate constructor overhead, not only C*B*rows*columns*4. Set a
proposed 512 MiB per-case memory cap, one active case and a 60-second subprocess
limit; large cases must opt in. Report resource-limit cases separately, never as
numerical passes. Use subprocess isolation to reclaim Torch/device-model storage.

## 7. Implementation stages and exit criteria

| Stage | Work | Exit criterion |
| --- | --- | --- |
| V0 | Pin dependency; minimal CENT initialization and primitive probes | Actual dtype/units/latch persistence confirmed; versions recorded |
| V1 | Production recording/export and independent layout checker | Packing, padding, guards, allocation and schedule boundary matrix passes |
| V2 | Low-level CENT replay, exact-pattern tier | Real production command streams pass all profile/dimension boundary cases |
| V3 | Numerical modes B/C and scalar references | Tolerance-bound tests and project-profile bit checks reported separately |
| V4 | PNM and shared-storage chaining/state tests | GEMM/Linear reference tests, PIM↔PNM visibility and A→B→A pass with explicit coverage labels |
| V5 | Bounded extended sweep and conformance report | Counterexamples triaged; supported shapes/profiles and limitations documented |

For each failure save profile, dimensions, seed, mode, packed snapshot, NDP1 batch,
first divergent command, relevant bank/GB/latch values and reference output. A
failure must distinguish production planning/packing, replay translation, CENT
arithmetic order, unsupported capability and resource limits.

The final report must not say 'CENT supports all dimensions'. It should state the
exact tested profiles/ranges and which rules passed. Timing-model acceptance,
real AiM arithmetic, RISC-V/VPU execution and gem5 device integration remain
separate validation tasks.
