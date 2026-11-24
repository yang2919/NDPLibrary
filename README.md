# NDP Library

A C++20 library for executing tensor operations on a single near-data processing
(NDP) device, with LLM inference components built on top of a generic runtime.
The intended device combines a PNM processor with vector execution and scratchpad,
and off-chip PIM memory with limited compute capabilities.

| L1 backend | Execution path | Selection |
|---|---|---|
| Functional | In-memory CPU reference for runtime and numerical tests | C++ and Python examples use it by default |
| NDP | Builds commands and executes PNM kernels and PIM GEMV in a local emulator | C++ `--ndp` or `--ndp-config PATH`, Python `Session(backend="ndp", config=...)` |
| gem5 | Uses the NDP command planner and sends commands to a separate software execution process over IPC | C++ gem5 factory and the Linux guest workflow |

CENT is an external validation harness for PIM mapping and arithmetic. See the [NDP backend guide](docs/ndp-backend.md) and [gem5 integration plan](docs/gem5-integration-plan.md).

Start with the [usage guide](docs/using-ndp-library.md) for a complete L2 matrix example
and a decoder walkthrough matched to the C++ source.

## Architecture

| Layer | CMake target / interface | Responsibility |
|---|---|---|
| L3: model composition | `ndp_llm`, `ndp/l3/llm.hpp`, `ndp/l3/decoder.hpp` | Weight preparation, KV cache policy, RoPE, attention, decoder composition |
| L2: generic runtime | `ndp`, `ndp/l2/session.hpp` | Tensor operations, memory, layout preparation, submission and completion |
| L1: backend adapter | `ndp/l1/backend.hpp` | Allocation, transfers, layout conversion, kernel execution |

`ndp_llm` depends on `ndp`. The core runtime has no KV cache, token-position, RoPE,
or attention operation. L3 implements these using public L2 APIs; a matrix/vector
application can link only `ndp`.

## Directory layout

Public headers and implementations use the same layer boundaries:

```text
include/ndp/                 src/
├── common/                 ├── common/
│   types, FP16, layouts,    │   FP16 conversion and metadata validation
│   operation descriptions  │
├── l1/                     ├── l1/
│   backend.hpp             │   common/ (shared CPU implementation)
│   functional/backend.hpp  │   functional/ (reference execution)
│   ndp/backend.hpp         │   ndp/ (commands and emulator)
│   gem5/backend.hpp        │   gem5/ (process transport)
├── l2/                     ├── l2/
│   session.hpp, event.hpp  │   session runtime and private planner
└── l3/                     └── l3/
    weight.hpp, llm.hpp,        weight preparation, cache, RoPE, attention,
    decoder.hpp                decoder composition
```

`include/ndp/l1/backend.hpp` contains only the backend contract. Concrete factories
live under `l1/functional/`, `l1/ndp/`, and `l1/gem5/`. L2 depends only on the contract.

`common` contains shared data/operation contracts rather than runtime or model state.
L1 depends on common contracts, L2 uses the backend interface, and L3 uses public L2
APIs. Private implementation headers stay under `src`. Namespaces and CMake targets
remain `ndp` / `ndp_llm`; public include paths now contain their layer, for example
`#include "ndp/l2/session.hpp"` and `#include "ndp/l3/llm.hpp"`.
The former flat include paths have no forwarding headers.

## Build and run

Requirements: CMake 3.20 or newer, a C++20 compiler, and platform thread support.
No PyTorch installation or device SDK is required for the functional backend.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Select an emulator profile at startup (run from the repository root):

```sh
./build/single_device_linear --ndp-config configs/ndp-small.ini
./build/decoder_block --ndp-config configs/ndp-small.ini
```

Without backend arguments, examples retain the reference backend.

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

Include `ndp/l2/session.hpp` for L2, `ndp/l3/llm.hpp` for cache/RoPE/attention, or
`ndp/l3/decoder.hpp` for decoder containers and the optional block helper.
`ndp/common/fp16.hpp` provides host binary16 conversion helpers.

## Main API objects

| Object | Purpose |
|---|---|
| `Session` | Own one backend connection and coordinate memory, work, and shutdown |
| `Tensor` | Immutable values with logical shape/dtype, physical representations, and readiness |
| `DeviceBuffer` | Mutable dense storage accessed through checked writes and copying snapshots |
| L3 `PreparedWeight` | Optional handle for layouts prepared once and reused across projections |
| `OpOptions` | Select an operation's engine and permission to prepare missing input layouts |
| `Event` | Observe completion or execution failure of submitted work |
| L3 `KVCache` | Manage K/V buffers, capacity, token lengths, and reset for one layer/request |

Use `from_host()` to create a Tensor from FP16 bytes. Use `allocate()` for a
zero-initialized DeviceBuffer, `copy_into()` to write element ranges, `snapshot()`
to obtain an independent Tensor, and `clear()` to overwrite buffers with zeros.
Snapshot is a device copy, not a view. Handles and queued work retain their storage;
applications do not manually free device allocations.

L2 `linear()` accepts Tensor weights only. L3 `ndp::prepare_weight()` (include
`ndp/l3/weight.hpp`, link `ndp_llm`) provides model-specific preparation policy;
pass its result's `.tensor()` to L2. Generic `Session::query_operation()` and
`prepare_layouts()` remain available without L3. Required input layouts must exist
or be created according to the operation's layout policy.

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

## Python adapter

With PyTorch, setuptools, wheel, and CMake already installed in the active environment:

```sh
python -m pip install . --no-build-isolation --no-deps
python examples/python/linear.py
python examples/python/decoder_block.py --ndp-config configs/ndp-small.ini
```

The package uses explicit `Session.upload`, resident `NdpTensor` operations, and
`Session.download`. The decoder example implements the layer directly in Python.
[Build/test instructions and contracts](docs/pytorch-integration.md) include the
optional CMake extension target; the default C++ build has no Torch dependency.

## Current limits

The separate `make_ndp_backend(NdpSpec)` C++ factory now supports command-driven
FP16 PIM GEMV and functional PNM operations through the existing Session API. It uses
device-side layout conversion and memory/GB/MAC/GPR commands. PNM includes matrix,
elementwise, normalization and data operations; it does not execute RISC-V binaries.
The toy decoder now runs PNM prefill and PIM single-token projections with `--ndp`
or `--ndp-config configs/ndp-small.ini`. No argument retains the reference backend.
See [the A3 contract and limitations](docs/ndp-command-contract.md#a3-implementation-status-and-usage).

Each Session manages one device. DeviceGroup supports synchronous dense tensor
transfers between Sessions, and LlamaDecodeStack can place successive layers
on independent workers. See the [PyTorch usage guide](docs/pytorch-integration.md).
TensorParallelLinear supports output and input channel sharding.
TensorParallelSwiGLU and TensorParallelLlamaAttention keep intermediate channels
and whole KV groups resident, with FP32 host reduction at their output projections.
TensorParallelLlamaDecodeBlock adds resident RMSNorm and residual operations
for a complete single-token decoder layer. TensorParallelLlamaDecodeStack
connects all layers through resident rows. TensorParallelLlamaForCausalLM
provides single-sequence greedy generation, with CPU embedding/LM head and
NDP decoder/final RMSNorm. Both stack and model support device_groups plus
layer placement for sequential stages with TP inside each group. Independent
new_request() histories share those weights and can be manually interleaved.
GenerationScheduler provides bounded active histories, round-robin token steps
and automatic terminal cache cleanup; see [request scheduling](docs/request-scheduling.md).
Input-channel Linear preserves FP32 partials through host reduction. See
[parallelism contracts and examples](docs/parallelism.md).
PipelineGenerationScheduler overlaps contiguous TP stages across independent
requests with thread-owned Sessions; see [pipeline execution](docs/pipeline-execution.md).
Multi-token prefill uses PNM GEMM projections with bulk KV appends; decode
remains PIM GEMV. See [prefill](docs/prefill.md).
Microbatch fusion, paged cache, and native hardware
queue integration remain open.

L3 attention copies cache prefixes and constructs per-query/head intermediates.
RoPE and attention store intermediate results in FP16, introducing additional
rounding and possible score overflow relative to fused FP32-intermediate kernels.
Current tests use small independent numerical references and explicit tolerances;
they do not establish performance or accuracy for arbitrary full-scale models.

The optional PyTorch adapter supports explicit eager inference with CPU FP16 transfer
boundaries. Native `.to("ndp")` dispatch, autograd, and torch.compile are not implemented.
See the [Python usage guide](docs/pytorch-integration.md).

## Documentation

| Document | Read it for |
|---|---|
| [Usage guide](docs/using-ndp-library.md) | Build/link instructions, L2 examples, and the decoder walkthrough |
| [Object abstractions](docs/library-abstractions.md) | Responsibilities, ownership, lifetimes, and object relationships |
| [Generic L2 migration](docs/generic-l2.md) | Buffer contracts, L2/L3 separation, and old-to-new API names |
| [PyTorch usage](docs/pytorch-integration.md) | Installation, Python runtime operations, decoder example, and current limits |
| [Backend integration handoff](docs/backend-handoff.md) | Contracts a real simulator/FPGA adapter must satisfy |
| [Command backend usage](docs/ndp-backend.md) | A6 decoder execution, configuration, supported operations and limitations |
| [Execution/layout specification](docs/execution-layout-spec.md) | Layer ownership, bank/tile address rules, PIM schedule and PNM baseline |
| [CENT conformance results](docs/cent-conformance-results.md) | Executed dimension matrix, numerical results and coverage limits |
| [CENT reproduction guide](docs/cent-user-verification.md) | Build the exporter and run native CENT replay |
| [CENT validation plan](docs/cent-validation-plan.md) | Production-command replay, dimension matrix and independent correctness checks |
| [gem5 communication specification](docs/gem5-communication-spec.md) | AiMX-informed macro submission, address/wire schema, completion and target binding contract |
| [Layer data flow](docs/layer-data-flow.md) | Allocation, transfers, command lowering, IPC data and completion across L2/L1/execution |
| [PIM and memory packet reference (English)](docs/gem5-pim-packets.md) / [한국어](docs/gem5-pim-packets.ko.md) | PIM execution and common memory packets, with v2 command field layouts and replies |
| [gem5 implementation handoff (English)](docs/gem5-extension-handoff.md) / [한국어](docs/gem5-extension-handoff.ko.md) | Receiver wire protocol, command bit layouts, response formats and complete packet examples |
| [Stock FS external PyTorch verification](docs/gem5-k4-stock-fs-user-verification.md) | Two-terminal K4 procedure for guest reference GEMV |
| [F5 user verification](docs/gem5-f5-user-verification.md) | Decoder FS procedure and pass criteria (user verification reported complete) |
| [F5 decoder integration](docs/gem5-f5.md) | Explicit layer example, KV reset, reclamation and F0–F5 integration report |
| [F4 user verification](docs/gem5-f4-user-verification.md) | Session lifecycle FS procedure and pass criteria (user verification reported complete) |
| [F4 Session backend](docs/gem5-f4.md) | Owned receiver, existing NDP planner, completion and fault cleanup |
| [F3 user verification](docs/gem5-f3-user-verification.md) | Persistent emulator FS run procedure (user verification reported complete) |
| [F3 execution protocol](docs/gem5-f3.md) | Upload/submit/download, geometry handshake and native validation |
| [F2 user verification](docs/gem5-f2-user-verification.md) | User-only FS run procedure, pass criteria and logs to report |
| [F2 metadata IPC](docs/gem5-f2.md) | Metadata-only IPC; FS verification reported complete by the user |
| [Command transport and codec](docs/command-transport.md) | F1 abstractions, versioned wire fields and native validation |
| [gem5 integration plan](docs/gem5-integration-plan.md) | Pure FS guest IPC and software emulation milestones (F0–F5) |
| [Stock gem5 FS baseline](docs/gem5-fs.md) | User-run FS commands, expected results and failure reporting; assistant validation stays native |
| [NDP command contract and status](docs/ndp-command-contract.md) | Unified PIM/functional PNM commands, A1–A6 status, memory and completion rules |

Earlier design and milestone documents under `docs/` are historical records;
the usage and abstraction guides describe the current API.
