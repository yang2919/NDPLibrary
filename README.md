# NDP Library

NDP Library provides C++ and PyTorch APIs for LLM inference on near-data processing devices. Each device combines a programmable near-memory processor (PNM) with processing-in-memory (PIM) banks. A decoder can keep weights and KV history on the device, use PIM for single-token matrix–vector projections, and use PNM for multi-token projections, normalization and attention.

[한국어 기능 안내](docs/runtime-overview.ko.md) · [C++ usage](docs/using-ndp-library.md) · [PyTorch usage](docs/pytorch-integration.md)

## Library structure

| Component | Responsibility | Code |
|---|---|---|
| PyTorch adapters and serving | Accept CPU FP16 tensors, prepare resident weights, coordinate devices and schedule generation requests. | [Python modules](python/ndp) |
| L3 · Model operations | Compose decoder layers from projections, RoPE, causal attention and feed-forward operations. Keep reusable model weights separate from each request’s KV cache. | [Public headers](include/ndp/l3) · [Implementation](src/l3) |
| L2 · Tensor runtime | A `Session` owns one backend. It validates tensor operations, prepares layouts, queues work and tracks completion. C++ applications can use this API directly for matrix and vector operations. | [Session API](include/ndp/l2/session.hpp) · [Implementation](src/l2) |
| L1 · Device backends | Report supported operations and layouts, allocate device storage, transfer data and execute kernels in the selected environment. | [Backend contract](include/ndp/l1/backend.hpp) · [Implementations](src/l1) |

## How a decoder uses the runtime

The C++ example runs one decoder layer over a two-token prompt and then decodes three tokens individually. It composes the layer explicitly, so each projection, cache update and residual addition can be traced to its runtime call. [Decoder source](examples/decoder_block.cpp)

1. **Prepare weights and storage.** The example opens a session, prepares the physical layouts needed by PNM and PIM, and allocates KV buffers for the request. A `PreparedWeight` retains the prepared representations across tokens. A `DeviceBuffer` supports writes to selected ranges as the cache grows.
2. **Choose an engine per operation.** The example passes `OpOptions` to each projection. Prefill uses PNM matrix–matrix multiplication. Decode uses PIM matrix–vector multiplication for the same weights. RMSNorm, attention primitives and residual arithmetic use PNM.
3. **Pass tensors between operations.** The runtime queries the backend for the input layouts and workspace each kernel requires. It reuses a compatible layout or prepares the missing one when the caller permits conversion. Each output `Tensor` can feed the next operation while execution is pending.
4. **Update history and observe completion.** The layer appends the new keys and values, attends to the causal prefix and runs the feed-forward network. The example waits on the block output before reporting the committed cache length. A caller that needs CPU data can use `to_host()`. [Data flow and ownership](docs/layer-data-flow.md)

**PyTorch integration.** An application can wrap an existing attention module with `ndp.nn.DecodeSelfAttention` or a supported Llama layer with `ndp.nn.LlamaDecodeBlock`. The adapter handles the session and transfers at its boundaries. The application calls the module with ordinary CPU FP16 tensors. [Runnable module example](examples/python/decode_modules.py)

**Multi-device serving.** A device group coordinates independent sessions. Layer placement keeps each layer’s weights and KV state together. Tensor parallelism partitions a linear weight by output or input dimension. Output shards can be concatenated, while input sharding produces FP32 partials that the host sums. Request scheduling tracks independent token histories while sharing model weights. [Parallel execution](docs/parallelism.md) · [Request scheduling](docs/request-scheduling.md)

## Execution paths

| L1 backend | Where work runs | Entry point |
|---|---|---|
| Functional | Native CPU reference | Default C++ example |
| `ndp` | Local emulator of PNM and PIM commands | [Configuration and command execution](docs/ndp-backend.md) |
| `gem5` | Software receiver executes emulator commands in a separate process, including in a stock gem5 guest | [Stock FS verification](docs/stock-gem5-verification.md) |
| `gem5_pim` | A resident worker calls libpim. The native build uses the reference library. The extended gem5 RISC-V build issues PIM instructions and runs supporting arithmetic on the simulated CPU. | [Extended gem5 verification](docs/extended-gem5-smoke-verification.md) |

## Build and run

From the repository root, with CMake 3.20+ and a C++20 compiler:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/decoder_block
./build/decoder_block --ndp-config configs/ndp-small.ini
```

Both commands run the same decoder. The first uses the functional reference and the second selects the local NDP emulator. Each run should report committed token counts of 2, 3, 4 and 5.

For backend specifications, simulator verification, timing results and FPGA integration requirements, see the [documentation index](docs/README.md).
