# NDP Library Object Abstractions

This document describes the **implemented single-device C++ library**. It explains
what each object represents, why it exists, who owns it, and which layer uses it.
It is an abstraction reference; the [usage guide](using-ndp-library.md) follows a
complete example, and the [PyTorch design](pytorch-integration.md) describes proposed
bindings. Python bindings and real FPGA/simulator backends are not implemented.

Headers and implementations are physically separated into `common`, `l1`, `l2`,
and `l3` directories under `include/ndp` and `src`. Shared metadata contracts live
in common; they do not hold runtime or request state. See the
[README directory layout](../README.md#directory-layout).

## 1. Overall model

An application owns one `Session`, persistent weights, request-specific caches, and
Tensor handles. It submits operations using these handles. The runtime validates
operations, manages representations and lifetimes, and invokes a backend.

```text
Application / future Python adapter
  Session + Tensor + PreparedWeight + KVCache + OpOptions
                        |
L3 cache / rope / attention / decoder
                        |
Public L2 calls         |  linear, rms_norm, gather, copy_into, ...
                        v
Runtime
  Planner -> KernelPlan
  State -> Command queue -> worker
  TensorRecord -> Representation -> BufferLease
                        |
                        v
IBackend
  allocate / transfer / transform / execute / release
                        |
                        v
FunctionalBackend today; device-specific backend in the future
```

The public Session interface is L2 and serves L3 callers. `IBackend` is the L1
boundary. The `ndp` target contains L2/backend code; `ndp_llm` contains L3 and links
`ndp`. L2 contains no KVCache, RoPE, or attention operation.

Three distinctions are fundamental:

- **Logical data vs physical representation:** one Tensor describes one value but
  can retain multiple physical layouts of that value.
- **Execution engine vs storage:** PNM/PIM selects a processor for an operation;
  it does not specify where all related data must permanently reside.
- **Submission vs completion:** a successful `Result<Tensor>` can contain a Tensor
  whose computation is still pending. Its Event tracks execution completion.

## 2. Public resource and execution objects

### Session — one device connection and runtime owner

**Represents:** the context in which device data is created and work is submitted.
A Session is neither a model nor an end-user conversation. A model can reuse one
Session across many layers and requests.

**Responsibilities:** accept uploads and compute calls; validate ownership and
operation requirements; manage device/staging/workspace accounting; schedule work;
provide synchronization and orderly shutdown. Backend-specific instructions remain
behind `IBackend`.

**Ownership and lifetime:** created with an owning `unique_ptr<IBackend>` through
`Session::create`. Session is movable, not copyable. It holds shared runtime State.
`close()` stops submissions, drains work, shuts down the backend, and reports errors.
The destructor also closes, but does not give the caller a Result to inspect.

**Contract:** one submitting host thread per Session. Operands from another Session
are rejected. Internal locking does not make the entire public API safe for concurrent
submission. Keeping a Tensor alive after close preserves safe resource release, not
the ability to submit more work with it.

**Why needed:** centralizes device ownership and execution policy so each operation
or layer does not independently manage drivers, allocation, and completion.

### Tensor — logical immutable values backed by managed device storage

**Represents:** a logical shape/dtype and the physical representations containing its
values. The public handle shares a private `TensorRecord`.

**Responsibilities:** expose `desc()`, representation metadata, readiness, and wait.
It has no public raw device pointer, arbitrary stride/view API, or in-place data update.
Operations return new Tensors; adding a representation preserves the logical values.

**Ownership and lifetime:** copying a Tensor copies a shared handle, not its device
contents. Representations hold buffer leases; queued work retains leases independently
of local handle lifetime. No caller-side device `free()` is needed.

**Contract:** storage is FP16 in the current implementation. There is a canonical
dense representation. `reshape()` currently copies device data; it is not an aliasing
view. A pending Tensor can be passed to another operation without a host wait.

**Why needed:** separates application-level values from allocation handles, physical
packing, and readiness. It is not currently a PyTorch Tensor.

### PreparedWeight — reusable projection weight representations

**Represents:** a Tensor whose layouts were prepared for a selected set of engines.
It wraps a Tensor and exposes it through `tensor()`, plus `resident_bytes()`.

**Responsibilities:** make repeated `linear()` calls reuse prepared weight data.
`prepare_weight()` waits for the requested representations before returning.

**Ownership and lifetime:** its private representation map shares existing immutable
buffers with the source Tensor; newly prepared representations belong to that map.
It does not publish new layouts into the original Tensor's map. Retain and reuse the
PreparedWeight rather than repeatedly preparing the original Tensor.

**Contract:** preparation is not training, quantization, or a change of mathematical
values. Multiple layouts consume additional memory. Defaults prepare PNM and PIM.

**Why separate from Tensor:** identifies weights prepared for repeated execution and
keeps preparation cost outside the inference loop. Normalization weights remain
ordinary Tensors because they do not need projection-specific packing here.

### DeviceBuffer — generic mutable dense storage (L2)

Represents a fixed-size, zero-initialized FP16 allocation. Copying a handle shares
its mutable storage. `Session::allocate`, `copy_into`, `snapshot`, and `clear` manage
its lifetime and ordered access. It carries a TensorDesc and readiness Event but
no token counts, request identifiers, or model semantics.

`TensorCopy` describes a source Tensor, destination DeviceBuffer, and flattened
source/destination element offsets and count. A copy batch validates all ranges
before submission, then performs writes in order. Every destination inherits the
same Event; partial failure invalidates readiness without rolling back written bytes.

`snapshot` copies a dense region into an immutable Tensor, so future writes cannot
change its values or invalidate its packed layouts. `clear` overwrites all bytes
with zero and can recover failed writes. Failed allocation cannot be recovered.
This design explicitly separates mutable storage from immutable computed values.

### KVCache — mutable attention history (L3)

Represents fixed-capacity K and V DeviceBuffers for one layer/request. L3 owns
capacity checks, token positions, scheduled length, and completion history. It uses
only public L2 allocation/copy/snapshot/clear APIs, never private leases or backend
handles. Shared `KVState` retains the DeviceBuffer handles and append Events.

`scheduled_length()` counts accepted reservations; `committed_length()` derives the
successful prefix from Events. `poisoned()` inspects write/allocation failures.
These metadata methods and submissions belong to the submitting host thread.
Copying a KVCache handle shares request state; it does not create independent history.

Reset clears both buffers and resets the logical length. A failed append can be
recovered by successful reset; failed allocation requires a new cache. Decoder work
is not transactional, so an error after append may leave the cache advanced.

### Event — execution completion and failure

**Represents:** shared completion state: Pending, Running, Succeeded, or Failed.
`status()` inspects it; `wait()` blocks and returns success or failure. A default
Event is already complete; submitted commands receive pending Events.

**Ownership and lifetime:** owns shared `EventState`. An Event is a completion token,
not an allocation owner or an application-managed hardware command queue.

**Contract:** compute calls usually expose it through their Tensor. Layout conversion,
append, and reset return it directly. Dependency failure prevents dependent command
execution. Waiting after every operation is unnecessary; wait at application boundaries.

**Why needed:** connects pending operations without forcing host synchronization and
keeps execution errors distinct from immediate submission errors.

### HostTensorView, HostTensor, and HostTransfer — host transfer boundaries

| Type | Abstraction and ownership | Relevant contract |
|---|---|---|
| `HostTensorView` | TensorDesc plus non-owning byte span supplied by the caller | Input must remain valid during `from_host`; that call copies into owned staging before returning |
| `HostTensor` | TensorDesc plus an owning byte vector | Contains host FP16 bytes, not float32 values implicitly converted by the library |
| `HostTransfer` | Event plus shared pending host output | Returned by `to_host_async`; `wait()` returns a Result containing host data |

`to_host()` combines submission and waiting. These objects prevent temporary host
buffers from being confused with device data or completed download results.

## 3. Metadata, options, and results

These are value descriptions, not independent devices, storage allocations, or workers.

| Type | Meaning and responsibility | Important limit/default |
|---|---|---|
| `TensorDesc` | Logical shape and dtype | Dtype defaults to F16; positive rank/nonzero dimensions required |
| `DType` | Data-type identifier | F32/BF16 enum values exist, but current Tensor storage supports only F16 |
| `Engine` | Requested execution processor | PNM or PIM; unsupported choices return an error |
| `OpOptions` | Engine and input layout policy for one call | PNM + AllowMaterialize by default; `matvec()` has a PIM default argument |
| `LayoutPolicy` | Permission to prepare missing input representations | AllowMaterialize schedules conversion; RequirePrepared rejects missing/failed layouts but accepts pending preparation |
| `SessionOptions` | Runtime host-staging budget | `max_staging_bytes` defaults to 64 MiB; does not select an engine |
| `WeightPrepareOptions` | Engines to prepare and resident-byte budget | Both engines by default; budget covers weight representations |
| `KVCacheDesc` | Capacity, KV head count, head dimension, dtype | Fixed capacity and F16 storage |
| `RepresentationInfo` | Public snapshot of layout and EventStatus | Metadata only; not a mutable buffer handle |
| `DeviceCaps` | Backend name, memory capacity, scratchpad size | Not a full per-operation support matrix; query_operation supplies operation requirements |
| `MemoryStats` | Live/peak device and staging bytes, allocation count, workspace bytes | Runtime accounting, including reservations; not measured hardware traffic or utilization |
| `Status` / `ErrorCode` | Failure category and descriptive message | Used for both validation and execution errors |
| `Result<T>` | Either a value or a Status | Check before `.value()`; `Result<void>` contains success or a Status |

`OpOptions` has no persistent effect on Session defaults. In the decoder example,
projection options are passed only to `linear()`. Other calls use their own defaults.
PIM currently supports matvec and bias-free single-row linear. The functional LLM
primitives execute on PNM; there is no automatic fallback to another engine.

## 4. Layout and operation descriptions

### LayoutKind and LayoutDesc — physical arrangement requirements

`LayoutKind` identifies a supported layout family. `LayoutDesc` adds the logical
TensorDesc, physical byte count, alignment, and backend format/version string.
A descriptor says how bytes must be arranged; creating one allocates no storage.

`DenseRowMajor` is contiguous logical order. `TestTile4x4` is a padded functional-test
packing format, **not an AiM/FPGA bank/channel mapping**. `dense_layout()` and
`Session::query_layout()` return descriptors; `materialize_layout()` creates an actual
representation and returns its completion Event. Existing successful or pending
representations can be reused.

Physical size can exceed logical byte size because of padding. Device memory and
PNM scratchpad are separate resources; a layout descriptor is not a memory-tier
placement request.

### OpKind, OpAttributes, and OperationSpec — requested mathematical work

- `OpKind` names an operation such as Linear, RMSNorm, Gather, Transpose, or Scale.
- `OpAttributes` carries operation-specific scalar/shape metadata: epsilon, axis,
  reshape dimensions, gather indices, and a scale factor. No token position or cache
  range is stored in L2 operation metadata.
- `OperationSpec` combines kind, engine, input TensorDesc values, and attributes.
  It contains no input allocation handles or output data.

The Session constructs these internally from public calls. `infer_output()` validates
logical operation shapes/attributes and returns the output TensorDesc. This separates
mathematical validation from choosing a backend kernel.

### KernelPlan and Planner — executable requirements and their validation

`KernelPlan` is the backend's description of an implementation: OperationSpec,
kernel identifier, required input/output layouts, workspace size, tile size, and
accumulator dtype. It is not a queued command and owns no operand buffers.

`Planner::build()` asks `IBackend::query_operation()` for a plan and verifies it against
logical inference, backend layout descriptions, and scratchpad capacity. The Planner
is a stateless internal validator; it is not a graph optimizer or serving scheduler.

Current plans require FP32 accumulation and dense output. Therefore a PIM projection
can feed a PNM operation without output relayout in this implementation. Supporting
non-dense PIM output requires backend conversion or a change to this runtime contract.

## 5. Backend boundary

### IBackend — device-specific memory and execution adapter

**Represents:** the contract the runtime uses to control one execution environment.

**Responsibilities:** report capabilities; allocate/release buffers; transfer data;
query layout/kernel requirements; transform layouts; reserve separate scratchpad
workspace; execute kernels; copy buffer regions; fill allocations; shut down.

**Contract:** execution/transfer methods are blocking. They may return, even with an
error or exception, only after associated device accesses have stopped. The runtime
relies on that condition to release buffers safely. Hardware with native asynchronous
queues must wait inside the adapter under the current interface. Metadata queries
are thread-safe and may run concurrently with transfers.

**Lifetime:** `Resources` owns the backend. `shutdown()` stops new I/O, but `release()`
must remain valid for allocations retained by surviving handles.

`AllocationHandle` is an opaque backend allocation identifier (`uint64_t` today),
not a public host/device pointer. Application code uses Tensor handles instead.

### FunctionalBackend — CPU implementation of the contract

Created by `make_functional_backend()`. It models capacity, buffers, test packing,
workspace, and PNM/PIM operation behavior on the CPU. It validates library behavior;
it does not simulate hardware timing, coherence, DMA costs, or real AiM placement.
Its existence lets the runtime and application APIs be exercised before hardware
integration. Replacing it is an L1 task as long as the backend contract remains valid.

## 6. Private runtime objects and resource lifetime

These types are implementation details in [session.cpp](../src/l2/session.cpp), except
L3 KVState in [llm.cpp](../src/l3/llm.cpp).
Applications and Python bindings should not expose or construct them directly.

| Internal type | Responsibility | Ownership / lifecycle |
|---|---|---|
| `State` | FIFO queue, worker, submission/close state, first execution error | Shared by Session; owns Resources and worker; close drains and joins |
| `Resources` | Backend ownership, cached capabilities/options, locks, memory accounting | Shared by State and leases; deliberately holds no reference back to State |
| `BufferLease` | Reserved byte accounting and eventual allocation release | Shared by representations/cache/commands; final destruction releases allocation and reservation |
| `StagingLease` | Host staging-byte accounting | Shared while staged data is needed; destruction returns its budget |
| `Representation` | One LayoutDesc, buffer lease, and readiness Event | Stored in a TensorRecord and captured as needed by work |
| `TensorRecord` | Logical TensorDesc and layout-to-representation map | Shared by Tensor handles; map access belongs to the submission thread |
| `BufferState` | TensorDesc, one buffer lease, allocation and latest-write Events | Shared by DeviceBuffer handles; accessed by the submitting thread |
| L3 `KVState` | K/V DeviceBuffers, scheduled length, append Event history | Lives in `llm.cpp`, outside the runtime; derives committed/failed state from Events |
| `Command` | Completion Event, dependencies, execution closure, optional completion callback | Queued and executed by State; captures resources needed until work finishes |
| `EventState` | Mutex/condition variable, status, optional error | Shared by Event copies; permits waiting for status publication |

The worker checks dependencies, executes a command, updates completion bookkeeping,
and drops command-owned resources before publishing completion. It serializes backend
work; the asynchronous public API does not imply concurrent PNM/PIM execution.

The separate `Resources` object solves a lifetime problem: a Tensor may outlive the
Session's executor while still needing a valid backend to release its allocation.
Leases retain Resources without retaining the worker or creating a State ownership cycle.

## 7. Decoder composition and example-only helpers

`DecoderConfig` groups head counts, head dimension, and normalization epsilon.
`DecoderWeights` groups normalization Tensors and projection PreparedWeights. Neither
object owns an execution thread or independently allocates model storage; its members
retain their underlying data handles.

The L3 `rope()` and `causal_attention()` functions compose generic Session operations.
The L3 `decoder_block()` convenience function composes those functions and
updates the supplied cache. It is optional. The current
[decoder example](../examples/decoder_block.cpp) composes those operations directly
so the layer's data flow is visible.

`ndp::example::tensor`, `weight`, `weights`, `tables`, and `values` in
[toy_decoder.hpp](../examples/toy_decoder.hpp) build synthetic inputs/model data.
The example's local `require()` unwraps Result values or throws; it does not wait.
These helpers are not additional runtime abstractions or required application APIs.

## 8. One projection traced through the objects

For `session.linear(x, weight, {Engine::PIM, LayoutPolicy::RequirePrepared})`:

1. Session checks that `x` and `weight` belong to its Resources.
2. It constructs an OperationSpec from the logical inputs and requested engine.
3. Planner obtains and validates a KernelPlan requiring dense input and packed weight.
4. The runtime resolves those Representations; missing preparation is an error.
5. It reserves a BufferLease for the output and creates a TensorRecord with a dense
   Representation and completion Event.
6. A Command retains operand/output leases and readiness dependencies in State's queue.
7. The worker invokes IBackend allocation/execution using AllocationHandles. The public
   caller can already hold the returned Tensor while this work is pending.
8. A later PNM operation retains that output's lease and Event as its own dependency.
   No host download or per-operation host wait is needed.
9. Once all handles and queued consumers release a buffer lease, its allocation can
   be released. Session close drains remaining work but does not invalidate this
   ownership discipline.

This is the reason for the object boundaries: the caller expresses model operations,
the runtime preserves data and ordering, and the backend implements device behavior.

See [generic L2 migration](generic-l2.md) for API changes, snapshot costs, and FP16
intermediate precision limitations.
