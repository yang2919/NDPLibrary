# Backend integration handoff

The [command-driven backend](ndp-backend.md) now completes the A6 functional decoder
path. It is an emulator, not a hardware adapter. Its semantic command boundary can
be reused for future transport work; register encodings and coherence still need
hardware-specific validation.

The stock gem5 receiver uses the [command codec](command-transport.md).
The extended gem5 worker uses the [libpim kernel interface](gem5-pim-kernel-plan.md).
See the [backend verification record](extended-gem5-validation-plan.md) for target coverage.

## Required contracts

- DeviceCaps reports usable DRAM and scratchpad separately. Live DRAM reservations
  must fit actual physical allocation requirements, including layout padding.
- query_layout/query_operation are immutable, thread-safe metadata queries. They may
  run while the FIFO worker is executing a transfer. Kernel plans must match shapes,
  dtype, engine, layout, workspace and numerical semantics; unsupported requests
  must return an error without silently using host compute.
- allocate/release, upload/download, transform and execute currently use a blocking
  adapter contract. Return, error and exception all require accesses to caller buffers
  to have stopped. release remains valid after shutdown for surviving Tensor handles.
- Successful execute/transform/copy_region completion must make data visible to the
  next PNM/PIM/DMA operation. A FIFO host call order alone does not establish coherence.
- copy_region must bounds-check byte ranges and complete device-to-device copies.
  KV append consists of separate K/V writes: the L2 batch Event fails all destinations; L3 interprets that as cache failure.
- fill must initialize/overwrite an entire allocation with the supplied byte. Generic
  buffer allocation and clear depend on this operation and on device visibility.
- reserve_workspace allocates a persistent arena independently of DRAM. FIFO commands
  reuse it. Native kernels must honor workspace requirements, firmware reservations,
  alignment and bank placement.
- The functional test packing format is NOT an AiM layout. Real backend format ids,
  bank/channel mapping, vector distribution, reduction-result paths, supported dtype
  and LUT functions must come from that target's specification.
- Native asynchronous submission requires a new ticket/completion interface plus
  dependency/resource tracking. Uncertain completion cannot be reported as a completed
  failure; buffers must remain retained or quarantined until access cessation is known.

## Validation gates before claiming S7 complete

1. Identify actual simulator/version, loading/configuration and host/device I/O.
2. Implement the adapter without modifying logical Session/Tensor semantics.
3. Run native allocation, upload/download, layout mapping and visibility tests.
4. Run supported S4 compute tests using numerical tolerances justified for that target.
5. Run supported S5/S6 primitives, append/reset and decoder prefill/decode.
6. Record unsupported operations/dtypes/layouts and failure/quiescence behavior.
7. Distinguish functional correctness results from timing/performance measurements.

Current tests target FunctionalBackend and instrumented wrappers. They provide
reference cases, not an automatically proven contract for an unknown simulator.
