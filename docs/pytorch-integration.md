# Connecting NDP Library to PyTorch

This is an implementation design, not an available Python API. The repository
currently contains a C++ library, examples, and functional tests; it has no Python
bindings, registered PyTorch operators, autograd formulas, or PyTorch device backend.
The Python snippets below describe the proposed interface and are not runnable yet.

## Assessment and recommended first interface

**Keep the current Session/Tensor core for explicit, inference-only integration.**
Its managed handles, per-operation options, and persistent weights/cache can support
Python orchestration without copying every intermediate to the host. Add an adapter
layer that accepts CPU PyTorch tensors at the boundaries and exposes NDP handles
between operations.

This does **not** make an unchanged Hugging Face model run on NDP. If the requirement
is `model.to("ndp")` followed by ordinary PyTorch operations, the present API is
insufficient. That objective requires a device backend and substantially more tensor
semantics. Choose the intended integration level before committing to that work.

| Integration level | User experience | Fit with the current implementation |
|---|---|---|
| Explicit Python bindings, eager execution | `session.linear(ndp_input, prepared_weight, options)` | Recommended first milestone; preserve the C++ runtime |
| Registered custom operators at a region boundary | A PyTorch Tensor enters an NDP region and a PyTorch Tensor leaves | Feasible additional layer; state and compiler contracts need design |
| Native PyTorch device | Ordinary tensor operations dispatch to NDP | Requires allocator/storage, dispatch, copy, views, device management, and other backend work |

PyTorch distinguishes calling an external function from registering an operator that
participates in its dispatcher and transformation systems. A pybind11 method alone
does not provide those contracts. See the official
[custom operator overview](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html).

## L2/L3 ownership boundary

The core `ndp` target now provides generic immutable Tensor values and mutable
DeviceBuffer storage. The optional `ndp_llm` target provides cache policy, RoPE,
attention, and decoder composition through public L2 APIs. Bind L2 first; L3 can be
wrapped from C++ or implemented in Python on the same generic buffer API.
`copy_into` writes a buffer; `snapshot` returns an independent immutable Tensor.
Do not model a snapshot as a PyTorch view or in-place alias. See
[the migration contract](generic-l2.md) for ordering and failure recovery.

## Mapping the existing objects to Python

| Existing C++ type/API | Proposed Python representation | Adapter responsibility |
|---|---|---|
| `Session` | Owning Python Session wrapper | Own the move-only Session, serialize submissions, expose explicit close/context manager |
| `Tensor` | `NdpTensor` wrapper | Keep the native handle and owning Session wrapper alive; expose shape, dtype, wait |
| `PreparedWeight` | `PreparedWeight` wrapper | Retain prepared device storage; do not upload again on every forward |
| `DeviceBuffer` | Explicit mutable storage wrapper | Expose checked copy/clear and copying snapshot; retain Session owner |
| L3 `KVCache` | Per-request, per-layer cache wrapper | Retain owner, expose lengths/reset, prevent accidental sharing across requests |
| `OpOptions` | Engine and layout-policy enums/options | Translate directly; retain per-operation control |
| `Event`, `HostTransfer` | Completion/transfer wrappers | Translate failures when waiting and retain pending host output |
| `Result<T>` / `Status` | Return a value or raise an exception | Preserve error code and operation context; never expose unchecked `.value()` |

`NdpTensor` is an opaque device handle, **not a subclass of `torch.Tensor`** in this
first interface. `torch.add(ndp_tensor, ...)` is unsupported; callers use
`session.add(...)`. A prepared weight is not an `nn.Parameter`. The library is
inference-only and does not implement optimizer updates or backward computation.

Python wrappers should retain a strong reference to their owning Session wrapper.
The native runtime already retains buffers for queued work; the Python reference
also makes connection lifetime explicit. Explicit `close()` invalidates future
submissions even if handles remain alive. Reject cross-Session operands as C++ does.

## Host Tensor conversion: an explicit copying boundary

Implement two adapter entry points, for example `Session.upload(torch_tensor)` and
`Session.download(ndp_tensor)`. These names are proposed, not existing C++ methods.

### Upload

1. Require a CPU `torch.Tensor`, dtype `torch.float16`, supported nonempty dimensions,
   and `requires_grad == False`. Initially reject CUDA tensors and other dtypes with
   an actionable error; the caller can explicitly convert them.
2. Obtain contiguous row-major data with `.contiguous()` when needed. The current
   NDP API carries shape but not arbitrary strides or storage offsets.
3. Build `TensorDesc` and `HostTensorView` from the contiguous tensor's shape and
   byte span. Validate byte counts and integer conversions.
4. Call `Session::from_host`. It copies into owned staging before returning, so the
   adapter need not keep the temporary CPU tensor alive after that call completes.
5. Return an `NdpTensor`; execution may still be pending.

FP16 storage already matches the library's binary16 contract. Copy bytes, not a
float32 reinterpretation. This boundary is not zero-copy: contiguous conversion
may allocate, staging copies, and backend upload transfers data again.

### Download

Call `Session::to_host`, check completion, allocate a contiguous CPU FP16 PyTorch
Tensor of the returned shape, and copy the returned bytes into its owned storage.
Do not return `torch::from_blob` over a local `HostTensor::bytes` vector: that storage
would be destroyed when the adapter returns. An owning-deleter scheme can be added
later; the first version should use an explicit copy.

A synchronous CPU Tensor return is a clear initial contract. Do not expose a normal
CPU Tensor containing data still being written by the NDP worker. Optional async
download should return a transfer handle whose `wait()` produces the CPU Tensor.

## Express the decoder layer in Python without host round trips

The same operation sequence as [decoder_block.cpp](../examples/decoder_block.cpp)
can be implemented in a Python class. The following is **proposed pseudocode**;
weight loading and validation are omitted. `w` contains prepared projection weights
and uploaded normalization weights; `sin` and `cos` are resident NDP tensors.

```python
class NdpDecoderLayer:
    def __init__(self, session, weights, sin, cos, query_heads, kv_heads, head_dim):
        self.session = session
        self.w = weights
        self.sin, self.cos = sin, cos
        self.query_heads, self.kv_heads = query_heads, kv_heads
        self.head_dim = head_dim

    def forward_ndp(self, x, cache, projection_options):
        s, w = self.session, self.w
        tokens, width = x.shape
        start = cache.scheduled_length()
        positions = list(range(start, start + tokens))

        n = s.rms_norm(x, w.attention_norm)
        q = s.linear(n, w.q, projection_options)
        k = s.linear(n, w.k, projection_options)
        v = s.linear(n, w.v, projection_options)
        q = s.reshape(q, [tokens, self.query_heads, self.head_dim])
        k = s.reshape(k, [tokens, self.kv_heads, self.head_dim])
        v = s.reshape(v, [tokens, self.kv_heads, self.head_dim])
        q = ndp.rope(s, q, self.sin, self.cos, positions)
        k = ndp.rope(s, k, self.sin, self.cos, positions)
        cache.append(s, k, v)
        a = cache.attention(s, q, start)
        a = s.reshape(a, [tokens, width])
        residual = s.add(x, s.linear(a, w.output, projection_options))

        n = s.rms_norm(residual, w.ffn_norm)
        gate = s.silu(s.linear(n, w.gate, projection_options))
        up = s.linear(n, w.up, projection_options)
        return s.add(residual, s.linear(s.mul(gate, up), w.down, projection_options))
```

An `nn.Module` wrapper may call this eager code. Merely inheriting `nn.Module` does
not register the NDP handles as parameters/buffers or enable `.to()`, `state_dict()`,
autograd, or compilation. Those behaviors must be implemented or explicitly rejected.

Use a boundary around several layers or a whole supported decoder stack:

```python
# Proposed eager usage; ndp bindings are not implemented yet.
with torch.inference_mode():
    x = session.upload(cpu_hidden_states)
    for layer, cache in zip(layers, request_caches):
        x = layer.forward_ndp(x, cache, projection_options)
    cpu_output = session.download(x)
```

Upload weights and prepare their layouts once during model initialization. Keep
weights across requests and allocate separate caches for each request/layer. Do not
download after every primitive, or even after every layer, unless CPU computation
actually needs the data. FP16 intermediate handles stay on the NDP device.

For Hugging Face integration, write a model-specific adapter that maps checkpoint
weights, normalization epsilon, RoPE convention, attention heads, and cache semantics.
The current example has no batch dimension, padding-mask interface, model loader,
tokenizer, sampling, or generation API. Supporting an arbitrary architecture or
unchanged `generate()` is a separate project; start with one supported decoder model.
Retain the host checkpoint as the serialization source and recreate device weights
when loading. Native device handles cannot be serialized as checkpoint tensor data.

## Threading, GIL, and failure semantics

The C++ Session permits one submitting host thread. The Python GIL is not an adequate
replacement for an explicit submission policy, especially when waits release it.
For the first adapter, record the Session's creating thread and reject submissions
from other threads; have a serving scheduler marshal requests to that thread. This
preserves the existing contract without adding locks throughout the runtime.

Release the GIL around blocking `wait`, `download`, `prepare_weight`, `synchronize`,
and `close` calls, with arguments and native lifetimes retained. Do not access Python
objects while it is released. Because other Python threads can then run, the owner
thread restriction must cover close/reset/inspection as well as compute submission.
Avoid Python callbacks on the backend worker in the initial design.

Check submission errors immediately and execution errors at explicit wait/download
boundaries. Cache append is stateful and decoder execution is not transactional.
A failed step may have already appended K/V; reset or recreate the request cache
before retrying. Session errors remain reportable after cache reset. Do not translate
a failed operation into an empty or zero-filled torch Tensor.

## Registered operators and `torch.compile`: a separate milestone

A dispatcher operator is useful when a supported NDP region must appear as an
operation in a PyTorch program. For a first **stateless correctness bridge**, a
CPU-Tensor-in/CPU-Tensor-out linear operator can upload operands, invoke NDP linear,
wait, and download. This is easy to compare with PyTorch, but repeated weight uploads
make it unsuitable as the serving architecture.

Register a C++ schema using `TORCH_LIBRARY` and a CPU implementation with
`TORCH_LIBRARY_IMPL` for this boundary: dispatch is CPU because the inputs are CPU
Tensors, even though the implementation offloads work. A Python module method calling
a pybind function is not the same registration. Shape/dtype-only fake implementations
and appropriate validation are needed before claiming compiler support. PyTorch's
[custom C++ operator tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
describes registration, `register_fake`, `opcheck`, and optional autograd integration.

A persistent serving operator needs Session, prepared weights, and request cache state.
Do not put arbitrary pybind instances into a dispatcher schema, or hide mutable cache
state behind an integer handle and declare the operation pure. That conceals effects
from graph transformations and complicates lifetime safety. TorchBind offers registered
C++ custom classes, but an ordinary pybind class is not such a class. See the official
[custom class tutorial](https://docs.pytorch.org/tutorials/advanced/custom_classes.html).

For the initial stateful decoder, keep eager execution and an explicit graph break
around the NDP region if a surrounding model uses compilation. Before capturing it,
design and test state mutation, aliasing, ordering, fake-class behavior where needed,
and lifetime contracts for the selected PyTorch version. Another approach is a
functional cache-in/cache-out operator, but the current mutable `KVCache` API does not
supply that contract. A fake output-shape function alone does not solve cache effects.

Autograd and `vmap` are outside the first milestone. Reject gradient-bearing inputs;
provide no implicit training claim. Register and test each transformation separately
if it becomes a requirement.

## When the C++ structure must change

| Current property | Explicit eager bindings | Requirement that needs further work |
|---|---|---|
| Opaque owned tensors, no exported pointer | Suitable for copying boundaries and resident handles | Zero-copy interoperability needs memory ownership, accessibility, synchronization, and layout contracts |
| Dense contiguous logical tensors, immutable results | Suitable for this decoder composition | Native PyTorch views/strides/aliasing/in-place operations require richer tensor semantics |
| One submitting thread, FIFO backend worker | Suitable for serialized eager execution | Concurrent serving/streams need a scheduler and thread-safe submission design |
| Mutable opaque KVCache | Suitable for explicit request state | Graph capture needs visible effect/alias semantics or a different cache API |
| Dense outputs enforced by Planner | Suitable for the functional backend | Real packed PIM outputs require backend normalization or a runtime representation change |
| Only FP16 and restricted operations | Suitable for the documented model subset | General PyTorch dispatch requires defined dtype, shape, device, and fallback coverage |
| No Python extension build | Add adapter sources and packaging | Build a shared extension, enable PIC for linked static code where required, and pin/test PyTorch ABI/toolchain compatibility |

Native device integration is substantially broader than adding bindings. PyTorch's
[PrivateUse1 guide](https://docs.pytorch.org/tutorials/advanced/privateuseone.html) and
[dispatcher backend guide](https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html)
cover device registration and backend dispatch. A native backend must implement or
explicitly handle factory/copy operations, storage/allocator behavior, device guards,
and required operator semantics. Current Session handles can remain an internal
resource layer, but cannot alone provide ordinary `torch.Tensor` behavior.

Do not add a tensor subclass, raw pointer API, or a PrivateUse1 backend merely to
make the current example callable from Python. If transparent device integration is
the actual acceptance criterion, revisit the Tensor and synchronization design before
expanding the operator set.

## Implementation sequence and acceptance criteria

1. **Bindings and CPU FP16 bridge.** Add an optional extension target/package, enum
   bindings, owning wrappers, error translation, owner-thread checks, and copying
   upload/download. Verify shape/dtype checks, noncontiguous CPU input, Session mismatch,
   safe close, garbage collection with pending work, and failures during waits.
2. **Resident eager decoder.** Bind the primitive calls used in the example and
   implement the Python layer above. Compare prefill and several decode steps with
   both the C++ example and an independent PyTorch reference using explicit FP16
   tolerances. Check intermediate operations, causal masking, and cache reset, not
   only final output shape. Verify weights upload once and no intermediate downloads.
3. **One model adapter.** Load an explicitly supported checkpoint, match its RoPE and
   normalization conventions, retain per-request caches, and validate token-step
   outputs. Define batch/padding limitations and external generation integration.
4. **Optional dispatcher bridge.** Register a stateless boundary first, run `opcheck`,
   and test fake execution/compilation. Keep the stateful eager path until mutation
   semantics are validated. Do not claim compiler support based on eager tests.

These are proposed checks, not completed tests. The repository's existing C++ tests
and CPU functional backend do not establish Python, PyTorch compiler, or FPGA support.
The immediate missing work is an adapter and its tests; explicit eager integration
does not currently justify a rewrite of L2/L3.
