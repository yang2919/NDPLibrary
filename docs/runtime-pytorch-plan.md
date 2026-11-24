# Runtime-to-PyTorch implementation plan

Status: P1–P4 implemented. P5 and dispatcher/native-device milestones remain planned.
See [the current Python usage guide](pytorch-integration.md) for build commands and
validated behavior. The sections below retain the implementation design and gates.
This plan builds on [the integration design](pytorch-integration.md) and the current
single-device L2 Session and configurable L1 emulator.

## First delivery and architecture

Deliver an inference-only Python package named `ndp`, with CPU FP16 PyTorch tensors
at transfer boundaries and resident NDP handles between operations. Users implement
layers explicitly in Python using runtime operations. Preserve the generic L2 core
and keep weight/cache policy in L3.

```text
PyTorch CPU Tensor
    | Session.upload (copy)
Python NdpTensor + Session operations
    | pybind11 / ATen adapter
C++ L2 Session -> IBackend -> functional or NDP command backend
    | Session.download (wait and copy)
PyTorch CPU Tensor
```

An NdpTensor is an opaque wrapper, not a torch.Tensor subclass. The first delivery
supports eager Python orchestration; it does not automatically dispatch `torch.add`,
move arbitrary modules using `.to("ndp")`, or capture stateful work in torch.compile.
These are different integration contracts, not consequences of adding bindings.

## Objects and contracts

| Python abstraction | Native state | Required behavior |
|---|---|---|
| `Session` | Move-only C++ Session held by an owning adapter | Backend/config selection, owner-thread checking, context manager, close, synchronize, memory stats |
| `NdpTensor` | C++ Tensor plus shared Session owner | Read-only shape/dtype, readiness/wait; no exposed pointer, arbitrary strides, or in-place mutation |
| `DeviceBuffer` | C++ DeviceBuffer plus owner | Explicit mutable dense storage; allocate, copy_into, clear, snapshot |
| `TensorCopy` | Source/destination handles and element ranges | Check same Session; offsets/count are FP16 elements |
| `OpOptions` | Engine and LayoutPolicy | Preserve per-operation engine choice; Session selects backend, not a global engine |
| `LayoutDesc` / `KernelPlan` | Native value metadata | Query results and preparation inputs; expose read-only metadata rather than editable kernel plans |
| `Event` | C++ Event plus owner | Expose status/wait; propagate execution failures without a per-op implicit wait |
| `HostTransfer` | C++ HostTransfer plus owner | Optional later async download; wait returns fully initialized CPU Tensor |
| `NdpError` | Translated Status | Preserve error code, operation name, and original message |
| L3 `PreparedWeight` | C++ prepared weight plus owner | One-time layout preparation; expose its Tensor to L2 linear |

Do not expose Result.value() to Python. The adapter either returns a valid object or
raises NdpError. Expose Tensor, buffer, and event constructors only through valid
native results. Session ownership must not depend on the Python variable name
remaining in scope. Queued native work retains its allocations.

## Transfer and lifetime policy

Upload accepts a CPU strided torch.Tensor with dtype float16, supported positive
nonempty dimensions, and no gradient requirement. Reject CUDA, sparse, quantized,
unsupported ranks/dtypes, and requires_grad inputs with explicit errors. Normalize
noncontiguous input to contiguous row-major storage and resolve lazy view flags
before reading its logical values. Check byte-size and shape integer conversions.
Do not silently downcast or move devices.

Construct HostTensorView from the contiguous tensor. Session::from_host copies into
owned staging before returning, so the caller may later modify or release the CPU
source. Return a resident NdpTensor without waiting for device execution.

Download uses Session::to_host and copies into a newly allocated contiguous CPU
FP16 ATen tensor. Do not return a from_blob view of temporary HostTensor bytes. A
normal CPU Tensor is returned only after completion. No zero-copy or DLPack API is
included: packed layouts and opaque allocations do not yet supply that contract.

Session records its creation thread. Reject operations from other threads before
entering native Session methods; the serving scheduler must marshal submissions to
the owner thread. Do not rely on the GIL for serialization. Release the GIL during
native waits/download/preparation/close, retaining native arguments throughout.
Do not touch Python objects or invoke Python callbacks while the GIL is released.

Explicit close drains work and reports errors. Subsequent submissions fail even if
handles remain alive. Context manager exit preserves an existing Python exception
rather than replacing it with a shutdown exception; attach shutdown diagnostics.
Finalization must not raise, invoke Python callbacks, or call owner-thread-checked
Python methods. Native RAII cleanup remains the last safety net; use explicit close
for observable errors. Test finalization after references are dropped on another
thread and while work is pending.

## Operation surface and model composition

Bind the existing L2 methods directly: matmul, matvec, linear, add, mul, rms_norm,
softmax, silu, reshape, gather, scale, transpose; plus query_operation,
prepare_layouts, materialize_layout and buffer operations. Preserve existing shape
restrictions and error behavior. In particular, snapshot copies, reshape is not a
PyTorch aliasing view, and emulator PIM Linear requires M=1 and no bias.

Example usage:

```python
import torch
import ndp

with ndp.Session(backend="ndp", config="configs/ndp-small.ini") as s:
    weight = s.upload(torch.randn(8, 16, dtype=torch.float16))
    prepared = ndp.llm.prepare_weight(s, weight, engines=[ndp.Engine.PNM, ndp.Engine.PIM])
    x = s.upload(torch.randn(1, 16, dtype=torch.float16))
    options = ndp.OpOptions(engine=ndp.Engine.PIM,
                            layout_policy=ndp.LayoutPolicy.RequirePrepared)
    y = s.linear(x, prepared.tensor(), options)
    y = s.silu(y, ndp.OpOptions(engine=ndp.Engine.PNM))
    result = s.download(y)  # First explicit wait in this compute chain.
```

A Python decoder example must show normalization, projections, RoPE, cache writes,
attention, residuals, and FFN directly, matching examples/decoder_block.cpp. It may
use named L3 helpers, but must not hide the entire block in a native forward call.
Weights stay resident across calls; each request/layer owns its buffer/cache state.
Upload once on entry to a supported region and download only when CPU consumers
need the result. Failure after a cache write is not transactional; reset/recreate
request state before retrying.

Add an explicit `NdpLinear.from_torch(module, session, ...)` adapter for selecting
individual nn.Linear modules. Import detached checkpoint weights once and expose
`forward_ndp(NdpTensor)` for resident chaining. Its ordinary `forward(torch.Tensor)`
is a convenience upload/compute/download boundary. Support CPU FP16 inference,
validate training mode, and keep a host checkpoint as the serialization authority.
Rebuild resident weights explicitly after checkpoint reload; never silently retain
stale prepared weights. A broader module-conversion API can follow once this
lifecycle works. Do not override Module.to globally or present the adapter as native
device support.

## File boundaries and build

```text
bindings/pytorch/
  module.cpp           # Extension initialization and enums
  session.cpp          # Owning Session wrapper, thread policy, errors
  tensor.cpp           # Tensor/buffer/event wrappers and transfer bridge
  operations.cpp       # Generic L2 forwarding and metadata queries
  llm.cpp              # Optional L3 bindings only
python/ndp/
  __init__.py
  nn.py                # Explicit module adapters
  llm.py               # Weight/cache helpers and layer composition
examples/python/
  linear.py
  decoder_block.py
tests/python/         # Transfer, ownership, compute, and decoder contracts
```

Keep PyTorch/pybind11 headers and dependencies outside include/ndp/l2 and src/l2.
Use an opt-in CMake extension target with position-independent ndp/ndp_llm libraries.
Package through pyproject.toml and a CMake-capable build frontend; discover Torch
from the active Python environment and use its ABI/build flags. Pin one tested
Python/PyTorch/compiler combination initially; record it in CI rather than promising
binary compatibility across Torch versions. Default C++ builds remain dependency-free
with respect to Python/Torch. Test wheel installation and import outside the source
tree, not just an in-tree build. Avoid adding an extra runtime submission thread.

## Ordered milestones and completion gates

| Stage | Implementation | Acceptance gate |
|---|---|---|
| P1: package and ownership | Optional extension, Session factory, enums, wrappers, errors, thread policy | Clean C++-only build; installed Python package opens/closes both backends; cross-thread and closed-Session calls fail safely |
| P2: tensor transfer | CPU FP16 upload/download, metadata, wait | Exact FP16 bit roundtrip including special values; noncontiguous/view input matches logical values; source mutation after upload cannot change staged data; mismatch and lifetime tests |
| P3: resident primitives | All current L2 compute, plan/layout queries, preparation, buffers | PNM/PIM numerical comparisons against independent PyTorch references; packed weight reuse; PIM-to-PNM chaining; copy bounds, snapshot independence, error propagation |
| P4: explicit Python decoder | L3 weight preparation, request cache support, direct Python layer composition, selected Linear adapter | Prefill plus multiple decode steps match reference within documented per-operation tolerances; caches isolated/reset; weights upload once; no intermediate host downloads |
| P5: one checkpoint adapter | Select one concrete model/config, map weights/RoPE/normalization and supported shapes | Compare intermediate activations and step logits; document supported batch/mask/cache conventions before generation integration |

The implementation and checkpoint results are recorded in the [PyTorch integration guide](pytorch-integration.md). The [dispatcher design and verification](pytorch-user-api-plan.md) covers packed weights, fake execution and graph capture.

## Native device scope

Native `.to("ndp")` is a separate design decision: it requires a PrivateUse1 device,
allocation/storage integration, transfer dispatch, device management, and tensor
view/stride/alias semantics. It also needs a defined policy for unsupported ops and
synchronization. The opaque immutable L2 Tensor can remain an execution resource,
but does not itself implement these torch.Tensor contracts. Training/autograd,
vmap, general torch.compile support, and unchanged Hugging Face generate() are not
part of P1–P4. See [PyTorch's extension overview](https://docs.pytorch.org/tutorials/extension.html).
