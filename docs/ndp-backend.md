# Command-driven NDP backend

This is the A1–A6 functional implementation of a single device combining AiM-derived
PIM commands and functional PNM commands. It runs on the CPU, but all intermediate
operands reside in emulator-owned device memory. It does not simulate timing,
execute RISC-V instructions, or claim vendor bit-exact AiM behavior.

## Run the decoder

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/decoder_block --ndp-config configs/ndp-small.ini
./build/decoder_block --ndp
./build/single_device_linear --ndp-config configs/ndp-small.ini
```

`decoder_block.cpp` implements the layer directly using library operations. It runs
two prefill tokens on PNM, followed by three single-token steps whose seven Linear
projections run on PIM. Normalization, RoPE, attention and FFN activation remain PNM
operations. Expected committed cache lengths are 2, 3, 4 and 5. Inputs are synthetic
hidden states; this is not checkpoint loading or text generation.

No argument selects the original functional reference. `--ndp` selects this command backend with default NdpSpec;
`--ndp-config` selects its versioned configuration. Invalid configuration fails
startup; there is no fallback. These options currently apply to C++ examples only.

## Create a Session

```cpp
#include "ndp/l1/ndp/backend.hpp"
#include "ndp/l2/session.hpp"
#include <stdexcept>

auto spec = ndp::load_ndp_spec("configs/ndp-small.ini");
if (!spec) throw std::runtime_error(spec.error().message);
auto backend = ndp::make_ndp_backend(spec.value());
if (!backend) throw std::runtime_error(backend.error().message);
auto opened = ndp::Session::create(std::move(backend.value()));
if (!opened) throw std::runtime_error(opened.error().message);
auto session = std::move(opened.value());
```

Session, Tensor, DeviceBuffer and PreparedWeight retain their existing APIs.
`prepare_weight` queries PNM and PIM requirements and retains dense and banked
representations. `RequirePrepared` prevents unexpected packing during decoding.

## Host and device responsibilities

| Stage | Action |
|---|---|
| L2 Session | Validate shapes, track storage, layouts, events and dependencies |
| Host L1 facade | Use existing KernelPlan and allocation handles to generate commands |
| PNM relayout command | Convert resident dense weights to the banked representation and zero padding |
| PIM projection | Transfer vector tiles to GPR/GB, initialize MAC, accumulate K tiles, read GPR results |
| PIM output | Transfer valid results to dense shared memory, without downloading to the host |
| PNM consumer | Read that shared output and run the requested functional kernel |

The command engine receives addresses, element counts and kernel arguments, not a
second Tensor/KernelPlan model. The same ordered submission path carries PIM, PNM,
copy, fill and relayout commands. PNM can also read GPR directly; GB is accessed via
explicit transfer commands. Execution is serial and blocking under the existing
Session worker. User code can chain pending tensors without intermediate waits.

## Supported operations

| Engine | Operations |
|---|---|
| PIM | Matvec; bias-free Linear with input shape [1,K] |
| PNM | Matmul, Matvec, Linear with optional bias, Add, Mul, RMSNorm, Softmax, SiLU, Reshape, Gather, Scale, Transpose |
| Device memory commands | Dense/banked relayout, byte copy, allocation initialization and clear |

Batched or biased PIM Linear is rejected; select PNM explicitly. PIM Matmul and
activation are not silently implemented using PNM. Storage is FP16. PIM uses a
fixed FP32 reduction tree and MAC accumulation; PNM matrix kernels use FP32 FMA
and round when storing FP16. Different accumulation orders need numerical tolerances.

## Configuration

The file contains decimal unsigned `key = value` entries and optional `#` comments.
All fields are required; unknown, duplicate, incomplete and malformed entries are
rejected. `load_ndp_spec` parses the schema; `make_ndp_backend` validates geometry
and resources before creating a usable device.

| Field | Meaning |
|---|---|
| version | Must be 1 |
| channels, banks | Channels and banks per channel; each 1–64 |
| rows, row_bytes | Rows per bank and bytes per row; row bytes are a positive multiple of 32 |
| gb_bytes | Per-channel GB size, a positive multiple of 32 |
| gpr_bytes | GPR capacity, even and large enough for vector, initialization and result regions |
| scratchpad_bytes | PNM storage; matrix kernels require at least four bytes |

Geometry determines shared-memory capacity. One whole row across all banks is
reserved as zero data. Allocation uses whole row stripes, including for small dense
tensors; logical Session memory statistics omit this internal padding. Consequently,
allocation may fail before the reported logical byte capacity is exhausted. The
profile is fixed for the lifetime of a Session. This is not a production allocator.

## Validation and remaining scope

The decoder regression runs both reference and command backends against an independent
scalar oracle, using two requests with cache reset and then an all-PNM comparison.
It checks seven one-time weight conversions, 42 PIM projection calls across the
requests, no intermediate host downloads, cache lengths, and restoration of live
memory after request teardown. The command/backend suites separately cover K tiles,
bank tails, packing, PNM operations, failure propagation and resource limits.

A6 completes the toy decoder path. It does not implement multi-device serving,
checkpoint import, hardware encoding, timing,
or binary instruction execution. The A4a ELF loader is retained as an isolated
future integration utility; functional PNM commands do not call it.
See [the command contract](ndp-command-contract.md) for exact command semantics.

## Command transport (F1)

The facade now owns a private `ICommandTransport` instead of an `Emulator`
directly. Its initial adapter forwards synchronous calls to one persistent
emulator; public APIs, planning and layouts are unchanged. A separate bounded
command codec is ready for F2 IPC work. See [transport and wire format](command-transport.md)
for responsibilities, fields, native checks and the user-owned FS handoff.
