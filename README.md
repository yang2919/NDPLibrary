# NDP Library

NDP Library runs FP16 tensor operations on near-data processing devices. A device combines a programmable near-memory processor (PNM) with processing-in-memory (PIM) banks. Each L2 `Session` owns one backend. The Python serving layer can coordinate several sessions for layer placement and tensor parallelism.

## Build and run

From the repository root, with CMake 3.20+ and a C++20 compiler:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/decoder_block
./build/decoder_block --ndp-config configs/ndp-small.ini
```

The first run uses the functional reference. The second uses the command-driven emulator. The [decoder example](examples/decoder_block.cpp) composes the layer from projections, RoPE, KV cache operations, attention and a feed-forward network. The [C++ guide](docs/using-ndp-library.md) explains how its tensors, weights and execution options move through L2 and L3.

For Python, start with the [PyTorch guide](docs/pytorch-integration.md). Its inference adapters accept CPU FP16 tensors and manage resident weights and transfers. See [multi-device execution](docs/parallelism.md) for placement and host-side reduction.

## Execution paths

| L1 backend | Where work runs | Entry point |
|---|---|---|
| Functional | Native CPU reference | Default C++ example |
| `ndp` | Local emulator of PNM and PIM commands | [Configuration and command execution](docs/ndp-backend.md) |
| `gem5` | Software receiver executes emulator commands in a separate process, including in a stock gem5 guest | [Stock FS verification](docs/stock-gem5-verification.md) |
| `gem5_pim` | A resident worker calls libpim. The native build uses the reference library. The extended gem5 RISC-V build issues PIM instructions and runs supporting arithmetic on the simulated CPU. | [Extended gem5 verification](docs/extended-gem5-smoke-verification.md) |

L1 implements device allocation, transfers and execution. L2 provides generic tensor operations and tracks dependencies. L3 uses those operations to prepare weights, manage KV history and compose model layers. See the [object contracts](docs/library-abstractions.md) and [layer data flow](docs/layer-data-flow.md).

The [documentation index](docs/README.md) lists the API guides, backend contracts, validation procedures, timing results and FPGA integration requirements.
