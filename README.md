# NDP Library

A C++20 library for executing tensor operations on a single near-data processing
(NDP) device, with LLM inference components built on top of a generic runtime.
The intended device combines a PNM processor with vector execution and scratchpad,
and off-chip PIM memory with limited compute capabilities.

**The current implementation runs on a CPU FunctionalBackend.** It validates API
behavior, memory ownership, layouts, and numerical results. Real FPGA/simulator
execution and Python/PyTorch bindings are not implemented.

Start with the [usage guide](docs/using-ndp-library.md) for a complete L2 matrix example
and a decoder walkthrough matched to the C++ source.

## Architecture

| Layer | CMake target / interface | Responsibility |
|---|---|---|
| L3: model composition | `ndp_llm`, `ndp/llm.hpp`, `ndp/decoder.hpp` | KV cache policy, RoPE, causal attention, decoder composition |
| L2: generic runtime | `ndp`, `ndp/session.hpp` | Tensor operations, memory, layout preparation, submission and completion |
| L1: backend adapter | `ndp/backend/backend.hpp` | Allocation, transfers, layout conversion, kernel execution |

`ndp_llm` depends on `ndp`. The core runtime has no KV cache, token-position, RoPE,
or attention operation. L3 implements these using public L2 APIs; a matrix/vector
application can link only `ndp`.

## Build and run

Requirements: CMake 3.20 or newer, a C++20 compiler, and platform thread support.
No PyTorch installation or device SDK is required for the functional backend.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

To build examples without tests, configure with `-DBUILD_TESTING=OFF`.

| Example | Demonstrates |
|---|---|
| [single_device_linear.cpp](examples/single_device_linear.cpp) | Reusing a prepared weight for PNM/PIM projections |
| [decoder_block.cpp](examples/decoder_block.cpp) | Directly composing a decoder layer, with prefill and incremental decode |

```sh
./build/single_device_linear
./build/decoder_block
```

The decoder example performs two-token prefill followed by three single-token steps:

```text
prefill: committed tokens=2
decode: committed tokens=3
decode: committed tokens=4
decode: committed tokens=5
```

Its code shows normalization, Q/K/V projections, head reshape, RoPE, cache append,
attention, output projection, residual connections, and SwiGLU FFN. It does not call
the prebuilt decoder block helper. Weights and hidden states are synthetic; this is
not a text generation program or a Hugging Face checkpoint runner.

## Use from another CMake project

```cmake
add_subdirectory(path/to/NDPLibrary ndp-library)

add_executable(matrix_app main.cpp)
target_link_libraries(matrix_app PRIVATE ndp)

add_executable(decoder_app decoder.cpp)
target_link_libraries(decoder_app PRIVATE ndp_llm)
```

Include `ndp/session.hpp` for L2, `ndp/llm.hpp` for cache/RoPE/attention, or
`ndp/decoder.hpp` for decoder containers and the optional block helper.
`ndp/fp16.hpp` provides host binary16 conversion helpers.

## Main API objects

| Object | Purpose |
|---|---|
| `Session` | Own one backend connection and coordinate memory, work, and shutdown |
| `Tensor` | Immutable values with logical shape/dtype, physical representations, and readiness |
| `DeviceBuffer` | Mutable dense storage accessed through checked writes and copying snapshots |
| `PreparedWeight` | Optional handle for layouts prepared once and reused across projections |
| `OpOptions` | Select an operation's engine and permission to prepare missing input layouts |
| `Event` | Observe completion or execution failure of submitted work |
| L3 `KVCache` | Manage K/V buffers, capacity, token lengths, and reset for one layer/request |

Use `from_host()` to create a Tensor from FP16 bytes. Use `allocate()` for a
zero-initialized DeviceBuffer, `copy_into()` to write element ranges, `snapshot()`
to obtain an independent Tensor, and `clear()` to overwrite buffers with zeros.
Snapshot is a device copy, not a view. Handles and queued work retain their storage;
applications do not manually free device allocations.

`linear()` accepts either an ordinary Tensor weight or a PreparedWeight. Layout
preparation is optional as an explicit step, but required input representations must
exist or be created according to the operation's layout policy.

## Supported execution

| Operation | Current support |
|---|---|
| GEMM (`matmul`) | PNM |
| GEMV (`matvec`) | PNM or PIM |
| `linear` | PNM; PIM for a single input row without bias |
| Add/multiply, RMSNorm, softmax, SiLU | PNM; restricted shapes/broadcasting documented in the usage guide |
| Reshape, gather, transpose, scale | PNM; results are new dense Tensors |
| L3 RoPE and causal attention | Compositions of generic PNM operations |

In the decoder example, prefill projections use PNM and single-token decode
projections use PIM. The projection engine does not move the entire layer to PIM.
Unsupported engine choices return errors; there is no automatic fallback.

Tensor storage is FP16. Matrix accumulation uses FP32. PIM weights use a padded
`TestTile4x4` representation in the functional backend; this is not an AiM/FPGA
memory mapping. Operation outputs are dense, so the current PIM-to-PNM path does
not require output relayout. Actual hardware must satisfy or extend this contract.

## Execution and failure contracts

- One host thread submits work to each Session. An internal FIFO worker executes
  blocking backend calls serially; asynchronous API returns do not imply PNM/PIM overlap.
- Check `Result<T>` for submission errors, then wait or download to check execution
  errors. Pending Tensors can be passed to subsequent operations without per-call waits.
- A failed copy batch can leave partially written buffers. All its destinations share
  failed readiness; `clear()` can recover failed writes, but not failed allocation.
- L3 cache reset clears both buffers and resets lengths. A decoder step is not
  transactional; reset or recreate request state before retrying a failed step.
- Explicit `Session::close()` drains work and reports errors. The first execution
  error remains reportable even after successful buffer/cache recovery.

## Current limits

This is a single-device functional implementation, not an optimized serving system.
There is no native hardware queue integration, multi-device execution, batching
scheduler, paged cache, model importer, or generation API.

L3 attention copies cache prefixes and constructs per-query/head intermediates.
RoPE and attention store intermediate results in FP16, introducing additional
rounding and possible score overflow relative to fused FP32-intermediate kernels.
Current tests use small independent numerical references and explicit tolerances;
they do not establish performance or accuracy for arbitrary full-scale models.

PyTorch integration is a proposed adapter layer. Calling NDP explicitly from Python
and supporting ordinary PyTorch device dispatch are different scopes; neither is
implemented by the current C++ library.

## Documentation

| Document | Read it for |
|---|---|
| [Usage guide](docs/using-ndp-library.md) | Build/link instructions, L2 examples, and the decoder walkthrough |
| [Object abstractions](docs/library-abstractions.md) | Responsibilities, ownership, lifetimes, and object relationships |
| [Generic L2 migration](docs/generic-l2.md) | Buffer contracts, L2/L3 separation, and old-to-new API names |
| [PyTorch integration design](docs/pytorch-integration.md) | Proposed bindings, data conversion, and compiler/device-backend limitations |
| [Backend integration handoff](docs/backend-handoff.md) | Contracts a real simulator/FPGA adapter must satisfy |

See [the cleanup review](docs/code-review-cleanup.md) for retained and removed code.
Earlier design and milestone documents under `docs/` are historical records;
the usage and abstraction guides describe the current API.
