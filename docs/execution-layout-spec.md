# Single-device execution and layout specification

Status: proposed normative baseline distilled from the current command backend.
This document specifies **what to execute and how data is placed**, before the
[transport specification](gem5-communication-spec.md) applies. Native CENT replay results are documented in the [conformance report](cent-conformance-results.md). Differences requiring future code
changes are explicitly identified.

## 1. Scope and ownership

Version 1 uses FP16 storage, one device and explicit engine selection. The initial
operations are PIM GEMV, bias-free single-token PIM Linear, and PNM Matmul/Linear.
Higher-level layers compose these and other existing PNM operations. Multi-device
partitioning, automatic engine selection and optimized RVV kernels are excluded.

| Responsibility | Owner | Input → output |
| --- | --- | --- |
| Model intent / weight reuse | Application or L3 | Layer configuration → generic L2 calls |
| Logical operation validation | L2 | TensorDesc, engine, attributes → OperationSpec |
| Target execution planning | L1 metadata planner | OperationSpec + device geometry → KernelPlan and canonical LayoutDesc |
| Plan validation / readiness | L2 | Plan → required representations, leases, dependencies |
| Physical allocation | L1 allocator | Representation byte count → whole row stripes, handle→Region |
| Data packing rule | L1 layout definition | Logical index → physical byte address |
| Packing execution | Device PNM relayout command in current baseline | Source/destination addresses and layout parameters → packed bytes |
| Tile traversal / command generation | L1 command builder | Validated plan + bound Regions → macro-command sequence |
| Arithmetic / micro-operation execution | Device emulator / target | Commands → memory/register effects |

L2 never guesses the bank mapping. A receiver never generates another KernelPlan.
Layout planning precedes allocation of that representation. Initial dense upload
may occur earlier because the dense input is useful for PNM as well. Packing must
not be decided by an allocator after it has already selected an arbitrary size.

The current baseline uploads dense data, allocates packed storage, and performs
`PnmRelayout`. Host-side direct packing/upload is a possible future optimization,
not a second layout definition. It must implement exactly the same address rule.

## 2. Device profile and supported operations

Use these symbols consistently:

| Symbol | Meaning |
| --- | --- |
| C, B | Channels and banks per channel; P = C×B parallel output lanes |
| R | Bytes in one bank row |
| V | Global Buffer bytes per channel |
| S | Row-stripe bytes = C×B×R |
| L | FP16 elements per 32-byte burst = 16 |
| T | K tile elements = min(R,V)/2 |
| N, K | GEMV weight shape W[N,K] |
| M | Input token/row count for GEMM/Linear |
| G, H | ceil(N/P), ceil(K/T) |

Profile conditions: 1≤C,B≤64; positive rows, scratchpad and GPR sizes; R and V
positive multiples of 32; GPR byte size even. Reserve row zero. GPR must hold
`2T+4P` bytes for the baseline PIM schedule; at least four scratchpad bytes are
required by current PNM matmul. Check all products/ceil divisions for overflow.
Unsupported shape/dtype/profile combinations are errors, never implicit fallbacks.

| Operation | Inputs → output | PIM | PNM |
| --- | --- | --- | --- |
| Matvec | W[N,K], x[K] → y[N] | Supported | Supported |
| Matmul | A[M,K], B[K,N] → Y[M,N] | Reject in v1 | Supported |
| Linear | X[M,K], W[N,K] → Y[M,N] | M=1, no bias only | Supported; optional bias[N] |

All dimensions are positive. PIM weights require `PimBanked`; x and y are dense.
PNM inputs and outputs are dense. Keep engine selection explicit: a large shape
must not silently change the selected engine. `RequirePrepared` rejects missing
input layouts; `AllowMaterialize` schedules their preparation.

## 3. Address and physical storage rules

### 3.1 Shared memory and dense layout

```text
shared(row,c,b,byte) = (((row*C)+c)*B+b)*R + byte
region_base(s)       = s*S
Dense W[n,k]         = base + 2*(n*K+k)
Dense A[m,k]         = base + 2*(m*K+k)
Dense B[k,n]         = base + 2*(k*N+n)
Dense Y[m,n]         = base + 2*(m*N+n)
```

Dense accesses can cross bank/row boundaries. A handle resolves to a region;
it is not itself an address. Every allocation starts at a stripe boundary.
A representation of D bytes consumes ceil(D/S) rows. Requested byte count and
allocator footprint are distinct; whole-stripe rounding can exhaust rows before
L2's byte budget is exhausted. No region may overlap row zero or another live
region. Releases return rows to the allocator but do not promise zero contents.

### 3.2 PIM weight placement

For a weight element W[n,k], define:

```text
g = floor(n/P)             p = n mod P
c = floor(p/B)             b = p mod B
t = floor(k/T)             lane = k mod T
physical_row = weight_first_row + g*H + t
address(W[n,k]) = shared(physical_row,c,b,2*lane)
packed_bytes = G*H*S
```

This is **cyclic output-row distribution across banks**, then output groups,
then K tiles. Physical storage dimensions are `[G][H][C][B][R bytes]`.
There is one canonical format, the existing `ndp.fp16.stripe.v1...` identifier.
A format change requires a new identifier and coordinated packing/lowering.

Before packing, zero the entire packed region. Write valid FP16 elements to the
addresses above. Keep all invalid output lanes, the final K-tile tail and unused
bank-row suffix bytes zero. This applies to reused nonzero storage as well.
Unpacking retrieves only N×K logical values; pack→unpack must preserve their bits.

### 3.3 Concrete irregular example

C=2, B=2, R=64, V=64 gives P=4, T=32, S=256. For W[5,35],
G=2, H=2 and packed storage is 1024 bytes (4 stripes), versus 350 dense bytes.
If the packed allocation starts at row 7:

| Logical weights | Channel / bank | Device row | Valid FP16 positions |
| --- | --- | --- | --- |
| W[0,0:32] / W[0,32:35] | 0 / 0 | 7 / 8 | 0..31 / 0..2 |
| W[1,0:32] / W[1,32:35] | 0 / 1 | 7 / 8 | 0..31 / 0..2 |
| W[2,0:32] / W[2,32:35] | 1 / 0 | 7 / 8 | 0..31 / 0..2 |
| W[3,0:32] / W[3,32:35] | 1 / 1 | 7 / 8 | 0..31 / 0..2 |
| W[4,0:32] / W[4,32:35] | 0 / 0 | 9 / 10 | 0..31 / 0..2 |

The other banks of rows 9/10 are zero. Final-tile lanes 3..31 are zero.
For example W[4,34] is at `10*256 + 2*2 = 2564`, in Shared memory.
Different allocation bases must change addresses, not layout semantics.

## 4. PIM GEMV execution schedule

GPR reservations for this serial schedule:

```text
vector tile:    [0, 2T)
initial MAC:    [2T, 2T+2P)
readback:       [2T+2P, 2T+4P)
```

Copy 2P zero bytes from reserved row zero into the initial-MAC range once.
For g=0..G−1:

1. Initialize all selected MAC accumulators from those FP16 zeros exactly once.
2. For t=0..H−1, let q=min(T,K−tT):
   - Clear the full 2T-byte GPR vector range using reserved zero storage.
   - Copy x[tT:tT+q] into its first 2q bytes.
   - Send one GPR→GB Transfer of 2T bytes to each selected channel's GB starting at byte zero.
   - Execute MAC at row `weight_first_row+gH+t`, burst-column 0,
     burst count `ceil(q/16)`, on every bank of each selected channel.
3. Read MAC results into GPR in channel-major, bank-major order.
4. Copy only `min(P,N−gP)` FP16 values to dense y starting at element gP.

Preserve MAC state across all K tiles in a group. Do not round partial sums to
FP16 or reinitialize between tiles. Inactive output banks may execute because
their weights are zero; never copy their results beyond N.

For current all-channel lowering, expected semantic command counts are:

```text
InitializeMac = G; MacAllBanks = G*H; ReadMac = G
Transfer = 1 + G*((2+C)*H+1)
Total = 1 + G*((3+C)*H+3)
```

These exclude uploads, allocation and layout materialization. Check this schedule
independently from numerical output: a correct result on zeros does not prove a
correct command sequence. Macro burst counts are positive counts, not an encoded
vendor OPSIZE-minus-one. No MAC spans a bank row or exceeds the used GB range.

### PIM arithmetic profile

Decode FP16 operands to FP32. For each 16-element burst, multiply with FP32
rounding, reduce through adjacent-pair balanced additions (16→8→4→2→1), rounding
each addition to FP32; add the burst sum to the FP32 MAC accumulator. Process
bursts and tiles in increasing order. `ReadMac` converts to FP16 once. Use the
existing nearest-even conversion/rounding rules. This is a project profile, not
a vendor BF16 or CENT bit-exact claim.

## 5. PNM GEMM and execution continuity

Current PNM is a **functional baseline**, not a tiled scratchpad/VPU kernel.
For Matmul, compute each Y[m,n] in increasing k using an FP32 accumulator and
`std::fma(FP16(A), FP16(B), accumulator)`. Apply factor, optional supported bias,
then FP16 output conversion. Linear indexes W[n,k] rather than B[k,n].
The existing implementation stages two FP16 operands in four scratchpad bytes;
it does not load full A/B tiles or model VPU throughput.

Keep this baseline explicit. PIM's tree reduction and PNM's sequential FMA need
not be bit-identical on arbitrary inputs; compare each against its own arithmetic
contract. CENT's AiM MAC is not evidence for RVV execution correctness.

A future tiled PNM profile must define (Tm,Tn,Tk), tail masks, DMA ordering and
accumulator lifetime first. At minimum, a single-buffer implementation storing
FP32 partial sums needs `2*Tm*Tk + 2*Tk*Tn + 4*Tm*Tn` scratchpad bytes, plus any
explicitly declared alignment/other workspace. The tile choice belongs to L1;
changing it must not introduce FP16 rounding at K-tile boundaries. This profile
is deferred, not silently assumed by current `PnmCompute`.

PIM outputs are dense Shared values, so PNM can immediately consume y after
completion. Conversely, a PNM-generated vector can feed PIM; a PNM-generated
weight matrix requires materialization into the canonical packed layout first.
Both engines must use the same backing memory, not independent stale copies.

## 6. Allocation and validation obligations

Before enqueueing, validate logical dimensions, engine support, checked packed
size, canonical format and workspace. L2 reserves representation bytes; L1 binds
physical stripes when the task executes. A failed physical allocation must appear
as an Event failure, not a successful Tensor with missing storage.

Before a batch executes, validate addresses, ranges, masks, burst bounds and
command support. Static failure has no effects. Failed execution must obey the
quiescence rules of the transport specification before leases are released.
Validate result writes against the dense output length, including final groups.

## 7. Mapping to current code and planned tightening

| Rule | Existing location | Status / action |
| --- | --- | --- |
| Canonical packed size/format | `weight_layout()` in command_builder.cpp | Implemented; make this specification its conformance target |
| Packing formula and zero fill | `PnmRelayout` in pnm.cpp | Implemented; verify bytes and nonzero-reuse cases independently |
| Allocation/binding | NdpBackend in backend.cpp | Implemented; test stripe exhaustion separately from byte budgets |
| Planning and readiness | query_operation(), L2 Planner/Session | Implemented; retain single authoritative plan |
| GEMV tile traversal/GPR | build_matvec() | Implemented; export real generated commands for testing |
| PIM arithmetic | emulator.cpp | Implemented; independent scalar reference required |
| PNM GEMM | pnm.cpp | Functional four-byte baseline only |
| CENT replay / cross-model checks | `tools/cent/` | Implemented and tested across boundary and long-K fixtures |

Conformance tests record production commands and verify their addresses and output bytes.

See [CENT validation plan](cent-validation-plan.md) and [layer data flow](layer-data-flow.md).
