# Generic L2 Runtime and L3 Migration

L2 now manages generic tensor operations and storage. It has no KV cache or token
history knowledge. L3 implements request state and model operations using public L2
APIs. This is an API-breaking change; it does not remove asynchronous execution.

## Build and header boundaries

| Target | Headers | Responsibility |
|---|---|---|
| `ndp` | `ndp/l2/session.hpp`, compute/layout/types/event, backend | Generic runtime and functional backend |
| `ndp_llm` | `ndp/l3/llm.hpp`, `ndp/l3/decoder.hpp` | KVCache, RoPE, causal attention, decoder composition |

`ndp_llm` links `ndp`. Core-only applications and buffer tests link only `ndp`.
RMSNorm, softmax, and SiLU remain stateless numerical primitives; they do not know
about models, requests, positions, or cache state. PreparedWeight remains an optional
layout-preparation helper. `linear` also accepts an ordinary Tensor.

## Physical source organization

Headers and sources are grouped under `common/`, `l1/`, `l2/`, and `l3/` in
`include/ndp/` and `src/`. Shared types, layout/operation contracts, and FP16 helpers
live in common. The backend interface and functional kernels live in L1; Session,
Event, and the private planner belong to L2; model compositions belong to L3.
The namespace and target names are unchanged. Update consumers to the new include
paths; old flat header paths are not retained as compatibility wrappers.

## Explicit mutable storage and immutable values

`DeviceBuffer` is a fixed-size, zero-initialized dense FP16 allocation with ordered
writes. Copying its handle shares that storage. `Tensor` remains an immutable value
with layout representations. A buffer cannot be supplied directly to compute: obtain
an independent Tensor with `snapshot` first. This avoids invalidating packed copies
or changing data underneath already-submitted consumers.

```cpp
auto storage = session.allocate({{8, 4}}); // Zero-initialized [8,4] FP16 storage.
// Check every Result before accessing value().
// Assume rows is an existing [2,4] Tensor in the same Session.
auto written = session.copy_into({
    {rows, storage.value(), 0, 3 * 4, 2 * 4},
});
auto selected = session.snapshot(storage.value(), {{2, 4}}, 3 * 4);
// selected can feed normal matrix/elementwise operations while still pending.
```

`TensorCopy` fields are source Tensor, destination DeviceBuffer, source offset,
destination offset, and count. Offsets/count are **FP16 elements in flattened dense
order**, not bytes or tokens. The runtime always reads the source's dense representation.
Packed layout offsets are not exposed. All ranges and Session identities are checked
before a batch is submitted. Empty/zero-count/out-of-range copies are rejected.

A batch executes copies in the specified order; overlapping writes use that order.
It publishes one Event to all destinations. If any copy fails, all destinations in
that batch have failed readiness, even if some bytes were already written. There is
no byte rollback. Subsequent writes/snapshots depend on that failure.

`clear({buffers...})` fully overwrites the buffers with zeros. It depends on initial
allocation and runs after preceding FIFO work, so it can recover failed writes.
It cannot recover failed allocation. Waiting for clear makes recovery observable;
Session's first execution error remains reportable by synchronize/close.

Snapshot is a device copy. It captures preceding writes and cannot be changed by
future buffer writes or clear. Pending consumers retain the copied Tensor storage.
This ordering relies on the current single FIFO worker. A multi-queue runtime must
implement read/write hazards before relaxing that ordering.

## Generic composition operations

- `gather(tensor, indices, shape)` selects flattened dense element indices, allowing
  repetition/reordering, and writes a dense Tensor of the specified shape. Index count
  must match the output element count. Indices are host metadata, not a device Tensor.
- `transpose(matrix)` copies a rank-two matrix with swapped axes.
- `scale(tensor, factor)` multiplies every element by a finite host scalar.
- `linear(x, weight_tensor, options, bias)` performs `x * weight^T` with ordinary
  Tensor operands. A PreparedWeight overload remains for existing callers.

Gather, transpose, and scale currently use PNM. Matrix/linear engine selection and
layout policies retain their existing meaning. These are a minimal composition
surface, not full NumPy/PyTorch indexing, broadcasting, batching, or arbitrary views.

## L3 responsibilities

`KVCache::create` allocates two DeviceBuffers. `append` checks capacity/shapes and
submits a K/V copy batch. L3 stores end positions with their Events and derives
committed length and failure state from completion. `reset` clears both buffers.
`attention` snapshots the valid prefix and calls the stateless L3 composition.
No private runtime headers, buffer leases, callbacks, or backend handles are used.

`causal_attention` selects each query/head and its causal K/V prefix, computes scaled
scores and softmax, then multiplies by V. `rope` selects rotary pairs and table entries
and composes elementwise math. Its alternating sign vector is uploaded once per
call and broadcasts over the final axis; activation data stays on the device. These functions use public Session operations only.

## API migration

| Previous call | Current call |
|---|---|
| `session.create_kv_cache(desc)` | `ndp::KVCache::create(session, desc)` |
| `session.append(cache, k, v)` | `cache.append(session, k, v)` |
| `session.reset(cache)` | `cache.reset(session)` |
| `session.attention(q, cache, start)` | `cache.attention(session, q, start)` |
| `session.rope(x, sin, cos, positions)` | `ndp::rope(session, x, sin, cos, positions)` |
| Core `CacheState` / `CachePoisoned` | L3 `KVState` / generic `ErrorCode::InvalidState` |

Backends implementing mutable storage must add blocking `fill(handle, byte)` and
support `copy_region`. Fill is used for zero initialization and recovery. The default
fill implementation returns UnsupportedOperation, so missing support fails explicitly.
Backend operation kinds no longer contain Attention or RoPE; they contain generic
Gather, Transpose, and Scale instead.

## Numerical and performance limits

The former fused RoPE/attention kernels rounded only their final outputs. Composition
stores intermediate FP16 tensors and therefore introduces additional rounding.
Tests compare with independent references using tolerances, not bitwise equivalence.
Large dot products can overflow FP16 scores even when a fused FP32-score implementation
would remain finite. This is a documented limitation of the initial composition.

The reference path also copies cache prefixes, creates gather metadata, and submits
per-query/head operations. It is not a claim of efficient long-context serving.
A future optimization should preserve the public separation, for example through
validated fused execution of generic operation sequences and richer accumulation
storage. Hardware-native async, multi-device serving, and PyTorch bindings are not
implemented by this change.
