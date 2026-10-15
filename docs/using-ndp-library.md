# Using NDP Library: Generic L2 APIs and a Decoder Layer

This guide describes the current C++ API. Start with generic matrix/tensor operations
and device storage, then follow a decoder layer built on those APIs. The decoder
walkthrough matches the numbered comments in
[decoder_block.cpp](../examples/decoder_block.cpp); it does not call the prebuilt
`ndp::decoder_block()` function.

All examples use the **CPU FunctionalBackend**. PNM/PIM choices select different
functional execution/layout paths; they do not execute on an FPGA. Tensor storage
is FP16. Python/PyTorch bindings are not implemented yet.

## 1. Choose the library layer

| Application need | Include | CMake target |
|---|---|---|
| Matrix/vector operations, transfers, layouts, mutable storage | `ndp/l2/session.hpp` | `ndp` |
| KV cache, RoPE, causal attention | `ndp/l3/llm.hpp` | `ndp_llm` |
| Decoder weight/config containers and optional block helper | `ndp/l3/decoder.hpp` | `ndp_llm` |
| Host float-to-FP16 conversion | `ndp/common/fp16.hpp` | `ndp` |

`ndp_llm` links `ndp` transitively. L2 has no cache, token position, or attention
operation. L3 implements those concepts by composing public L2 operations.

```cmake
# After adding this repository as a CMake subdirectory:
add_executable(my_matrix_app main.cpp)
target_link_libraries(my_matrix_app PRIVATE ndp)

add_executable(my_decoder decoder.cpp)
target_link_libraries(my_decoder PRIVATE ndp_llm)
```

To build and run the existing decoder example from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target decoder_block
./build/decoder_block
```

Expected output:

```text
prefill: committed tokens=2
decode: committed tokens=3
decode: committed tokens=4
decode: committed tokens=5
```

The example uses deterministic synthetic weights and **hidden states**, not token IDs.
It runs one layer over two prefill tokens followed by three single-token steps. It
has no tokenizer, checkpoint loader, sampling, or text generation. Decode inputs are
independently generated test data, not outputs sampled from earlier steps.

## 2. Understand the objects before using them

| Object | What it means | What the caller does |
|---|---|---|
| `Session` | Connection to one backend and coordinator of memory/work | Create once; reuse; close when finished |
| `Tensor` | Immutable logical values, shape/dtype, physical layouts, readiness | Upload or receive from an operation; pass to another operation |
| `DeviceBuffer` | Explicit mutable dense storage | Allocate; write regions; obtain independent Tensor snapshots |
| `PreparedWeight` | Optional handle retaining prepared weight layouts | Prepare once and reuse for repeated projections |
| `OpOptions` | Engine and layout policy for one operation | Pass to the specific operation whose behavior you want to control |
| `Event` | Completion or execution failure of queued work | Wait where the application needs completed results |
| L3 `KVCache` | K/V DeviceBuffers plus request-specific length policy | Create per layer/request; append; reset for reuse |

### What Session does

A Session owns the device runtime: it checks operands, asks the backend which kernel
and layouts are required, reserves resources, submits work, and reports errors.
It is not a model or an end-user chat session. One Session can serve several layers
and requests, provided each request uses its own cache state.

All operands must belong to the Session making the call. Its public APIs require
**one submitting host thread**. The internal worker runs blocking backend calls in
FIFO order; it does not execute PNM and PIM concurrently or schedule serving requests.

Tensor and buffer handles retain storage automatically. Queued commands also retain
the resources they need. There is no application-side device `free()`. Copying a
handle shares storage; it does not clone data. Explicit `close()` drains submitted
work and reports errors. Handles surviving close can be destroyed safely, but cannot
be used to submit new work through that closed Session.

### What OpOptions controls

```cpp
ndp::OpOptions options{
    ndp::Engine::PIM,
    ndp::LayoutPolicy::RequirePrepared,
};
```

These two fields apply **only to the call receiving them**. They neither change a
Session-wide engine nor permanently place a Tensor in PNM scratchpad.

| Field/value | Effect |
|---|---|
| `engine = PNM` | Request the PNM implementation |
| `engine = PIM` | Request the PIM implementation; unsupported requests fail |
| `AllowMaterialize` | Permit creation of missing required input layouts |
| `RequirePrepared` | Reject missing/failed input layouts; reuse existing successful or pending preparation |

Defaults are PNM and AllowMaterialize, except `matvec()` defaults to PIM.
RequirePrepared does not prohibit output/workspace allocation and does not force a
host wait for pending input preparation. There is no automatic engine fallback.

`SessionOptions` is different: it limits host staging bytes (64 MiB by default).
`WeightPrepareOptions` selects which engines' weight layouts to prepare and optionally
limits their resident bytes. Neither changes the default execution engine.

### Submission success is different from execution success

A `Result<T>` contains either a value or an error. Check it before accessing `.value()`.
For compute, a successful Result means work was accepted; the returned Tensor may
still be pending. `Tensor::wait()` or `to_host()` checks execution completion.

The decoder example uses a local `require()` helper to unwrap Results or throw with
an operation name. It does not wait and is not part of the library API.

## 3. Run a matrix operation using L2 only

This complete program uploads FP16 data, computes `x * weight^T`, and downloads the
result. It requires only the `ndp` target, not L3 or PreparedWeight.

```cpp
#include "ndp/common/fp16.hpp"
#include "ndp/l2/session.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

template <class T> T require(ndp::Result<T> result) {
  if (!result) {
    throw std::runtime_error(result.error().message);
  }
  return std::move(result.value());
}

int main() {
  try {
    auto session = require(ndp::Session::create(
        ndp::make_functional_backend(1 << 20, 256)));

    std::vector<uint16_t> input_bits;
    for (float value : {1.0f, 2.0f, 3.0f}) {
      input_bits.push_back(ndp::float_to_fp16(value));
    }
    std::vector<uint16_t> weight_bits;
    for (float value : {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}) {
      weight_bits.push_back(ndp::float_to_fp16(value));
    }

    auto x = require(session.from_host({{{1, 3}}, std::as_bytes(std::span(input_bits))}));
    auto w = require(session.from_host({{{2, 3}}, std::as_bytes(std::span(weight_bits))}));
    auto y = require(session.linear(x, w)); // Default PNM; output shape [1, 2].
    auto host = require(session.to_host(y)); // Waits for execution and download.

    for (size_t i = 0; i < host.bytes.size(); i += sizeof(uint16_t)) {
      uint16_t bits;
      std::memcpy(&bits, host.bytes.data() + i, sizeof(bits));
      std::cout << ndp::fp16_to_float(bits) << '\n';
    }
    auto closed = session.close();
    if (!closed) {
      throw std::runtime_error(closed.error().message);
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
```

Output is `14` followed by `32`. Upload expects native-endian IEEE binary16 bytes;
it does not reinterpret or convert float32 values automatically. `from_host()` copies
bytes into owned staging before returning, so the host vector need not survive the
queued transfer. `to_host()` returns an owning FP16 byte vector.

Selected L2 operations:

| Call | Shapes/behavior | Current execution support |
|---|---|---|
| `matmul(a, b, options)` | `[M,K] * [K,N] -> [M,N]` | PNM |
| `matvec(w, x, options)` | `[N,K] * [K] -> [N]` | PNM/PIM |
| `linear(x, w, options, bias)` | `[M,K] * [N,K]^T -> [M,N]`; optional `[N]` bias | PNM; PIM only M=1 without bias |
| `add`, `mul` | Same shape or final-axis vector broadcast | PNM |
| `rms_norm(x, gamma, epsilon)` | Normalize final axis; gamma is a final-axis vector | PNM |
| `softmax(x)`, `silu(x)` | Final-axis softmax / elementwise SiLU | PNM |
| `reshape(x, shape)` | Preserve element count; device copy | PNM |
| `gather(x, indices, shape)` | Flattened element indices into a specified output shape | PNM |
| `transpose(matrix)` | Copy a rank-two matrix with axes swapped | PNM |
| `scale(x, factor)` | Multiply by a finite scalar | PNM |

These APIs cover a supported subset, not arbitrary PyTorch strides, views, broadcasting,
or batched matrix products. Tensor storage currently supports only FP16 even though
other dtype enum values exist.

## 4. Store and update data with DeviceBuffer

Use Tensor for values consumed by computation. Use DeviceBuffer when an application
needs reusable storage whose regions change over time. Compute methods do not accept
a DeviceBuffer directly; `snapshot()` creates an independent Tensor for computation.

The following continuation assumes `session`, `x` with shape `[1,3]`, and `require()`
from the matrix example, before closing the Session:

```cpp
auto storage = require(session.allocate({{4, 3}})); // Twelve zero-initialized elements.

// TensorCopy: source, destination, source offset, destination offset, element count.
require(session.copy_into({{x, storage, 0, 2 * 3, 3}})); // Write row 2.
auto row = require(session.snapshot(storage, {{1, 3}}, 2 * 3));

// Later writes cannot change row: it is a copy, not a mutable view.
auto cleared = require(session.clear({storage}));
auto finished = cleared.wait();
if (!finished) {
  throw std::runtime_error(finished.error().message);
}
// row still contains [1, 2, 3], while storage now contains zeros.
```

Offsets and counts are **FP16 elements in flattened dense order**, not bytes, rows,
or token positions. Snapshot's TensorDesc supplies the output shape. The source's
dense representation is used even if it also has packed layouts.

A copy batch checks every range and Session identity before submission. Copies run
in order; all destinations share one Event. Partial execution failure does not undo
written bytes, and all destinations retain failed readiness. `clear()` fully overwrites
buffers and can recover failed writes, but cannot recover failed allocation. Session
synchronize/close still report the first execution failure after recovery.

Snapshots consume additional device memory and copying work. This separation makes
Tensor layouts safe to reuse: changing a buffer cannot make an old Tensor's packed
representation stale. Buffer handles themselves share mutable storage when copied.

## 5. Prepare reusable weights and understand layouts

For repeated inference, prepare the weight once:

```cpp
// w is the ordinary Tensor weight from the matrix example.
auto prepared = require(session.prepare_weight(w)); // Default: prepare PNM and PIM.
const ndp::OpOptions pim{
    ndp::Engine::PIM, ndp::LayoutPolicy::RequirePrepared};
auto pim_output = require(session.linear(x, prepared, pim));
```

PreparedWeight is optional; `linear(x, w, options)` also accepts a Tensor. Preparation
makes the one-time layout cost explicit and waits for the requested layouts. Retain
the returned prepared handle. It has its own representation map sharing existing
immutable buffers; repeatedly preparing the original Tensor can repeat packing.

**Materializing a layout creates real storage**, not just metadata. The functional
backend uses dense PNM weights and padded `TestTile4x4` PIM weights. This test layout
is not an AiM/FPGA bank mapping. For example, `[2,3]` FP16 weights occupy 12 dense bytes
plus 32 packed bytes when both representations are retained.

PIM outputs currently use the same dense layout as PNM inputs:

```text
Dense activation + packed weight -> PIM linear -> dense Tensor -> PNM operation
```

No output relayout is required by this implementation. The runtime enforces dense
outputs; an actual FPGA backend must satisfy that contract or the contract must be
extended. Movement from off-chip memory into PNM scratchpad is a separate issue from
layout conversion. The functional backend does not validate real hardware transfer costs.

## 6. Follow the decoder example

The sections below match `Step 1` through `Step 7` in the C++ example. The helper file
[toy_decoder.hpp](../examples/toy_decoder.hpp) creates synthetic inputs, prepared
weights, and RoPE tables. These `ndp::example::*` helpers are example utilities, not
required runtime APIs. Replace them with model-specific loading in an application.

### Step 1 — Create one Session

The example creates `make_functional_backend(1 << 20, 256)`: 1 MiB of device memory
and 256 bytes of separate scratchpad. Weights, activations, and buffers consume device
memory. PNM matrix kernels use scratchpad tiles managed by the backend. The caller
does not allocate scratchpad per operation.

### Step 2 — Prepare model data once

The example uses model width 4, two query heads, one KV head, head dimension 2, and
FFN width 6. `DecoderConfig` and `DecoderWeights` are L3 data containers; neither
executes the layer.

| Data | Shape | Handle |
|---|---|---|
| Attention/FFN normalization weights | `[4]` | Tensor |
| Q and attention output projection | `[4,4]` | PreparedWeight |
| K/V projection | `[2,4]` | PreparedWeight |
| FFN gate/up projection | `[6,4]` | PreparedWeight |
| FFN down projection | `[4,6]` | PreparedWeight |
| Sine/cosine tables | `[8,1]` | Tensor |

Weights use `[output_width,input_width]`. The RoPE helper generates test angles;
a real model must supply tables consistent with its interleaved rotary convention.

### Step 3 — Create a per-request L3 cache

```cpp
auto cache = ndp::KVCache::create(
    session, {cache_capacity, config.kv_heads, config.head_dim});
// Check cache before accessing cache.value(), as the full example does.
```

This allocates two zero-initialized DeviceBuffers with shape `[8,1,2]`, 64 bytes in
total. L3 owns the valid token count and capacity checks; L2 sees ordinary allocations
and copy ranges. Use one cache per layer/request and retain it across prefill/decode.
Copying a cache handle shares history; it does not create a new request.

### Step 4 — Upload this step's hidden states

```cpp
const bool is_prefill = step == 0;
const uint64_t token_count = is_prefill ? 2 : 1;
const auto engine = is_prefill ? ndp::Engine::PNM : ndp::Engine::PIM;
auto input = ndp::example::tensor(
    session, {token_count, model_width},
    ndp::example::values(token_count * model_width, static_cast<int>(step + 1)));
```

Prefill uses `[2,4]`; decode uses `[1,4]`. The variable `engine` is used only in the
projection options below; it does not choose an engine for the whole layer. If this
is a later layer in a model, use the preceding layer's output Tensor directly instead
of downloading/uploading it.

### Step 5a — Normalize and project Q, K, V

```cpp
const ndp::OpOptions projection{engine, ndp::LayoutPolicy::RequirePrepared};
auto normalized = require(session.rms_norm(input, weights.attention_norm, config.epsilon),
                          "Attention RMSNorm");
auto query = require(session.linear(normalized, weights.q, projection), "Q projection");
auto key = require(session.linear(normalized, weights.k, projection), "K projection");
auto value = require(session.linear(normalized, weights.v, projection), "V projection");
```

`normalized` is `[T,4]`. Q becomes `[T,4]`; K/V become `[T,2]`, with
`T = token_count`. Only these and subsequent `linear()` calls use `projection`.
Prefill uses PNM GEMM; single-token decode uses PIM matrix-vector projections.
RequirePrepared avoids scheduling missing weight layouts during these calls.

### Step 5b — Split heads and apply RoPE

```cpp
auto query_heads = require(
    session.reshape(query, {token_count, config.query_heads, config.head_dim}), "Q heads");
auto key_heads =
    require(session.reshape(key, {token_count, config.kv_heads, config.head_dim}), "K heads");
auto value_heads = require(
    session.reshape(value, {token_count, config.kv_heads, config.head_dim}), "V heads");
const uint64_t start_position = cache.value().scheduled_length();
std::vector<uint64_t> positions;
for (uint64_t token = 0; token < token_count; ++token) {
  positions.push_back(start_position + token);
}
auto rotated_query =
    require(ndp::rope(session, query_heads, sine_table, cosine_table, positions), "Q RoPE");
auto rotated_key =
    require(ndp::rope(session, key_heads, sine_table, cosine_table, positions), "K RoPE");
```

Reshape produces Q `[T,2,2]` and K/V `[T,1,2]`. It copies data rather than
creating views. Read the starting position **before append**: positions are `[0,1]`
for prefill, then `[2]`, `[3]`, and `[4]` for decode.

`ndp::rope()` is L3. It composes generic gather/multiply/add operations and uploads a small
last-axis sign vector to rotate Q/K; V is unchanged. There is no Session RoPE method or backend RoPE kernel.

### Step 5c — Append K/V and compute causal attention

```cpp
require(cache.value().append(session, rotated_key, value_heads), "KV append");
auto attended = require(cache.value().attention(session, rotated_query, start_position),
                        "Cached attention");
```

`KVCache::append()` submits a two-copy L2 batch into K and V DeviceBuffers.
It reserves positions immediately; completion is tracked through the batch Event.
No host wait is required before the following attention call.

`KVCache::attention()` snapshots the current valid prefix, then calls L3
`causal_attention()`. For each query/head, that function composes:

```text
gather query and causal K/V prefix
  -> transpose K -> matmul(Q, K^T) -> scale(1/sqrt(head_dim))
  -> softmax -> matmul(probabilities, V) -> assemble output
```

The first prefill token sees token 0; the second sees tokens 0–1. Each decode token
sees its entire preceding prefix including itself. Two query heads share one K/V
head (GQA). The output is `[T,2,2]`. All computation in this composition uses PNM.
The L3 helper already calls softmax; the example does not call it a second time.

### Step 5d — Project attention output and add the residual

```cpp
auto merged = require(session.reshape(attended, {token_count, model_width}), "Merge heads");
auto projected = require(session.linear(merged, weights.output, projection),
                         "Attention output projection");
auto residual = require(session.add(input, projected), "Attention residual");
```

Merging returns `[T,4]`. The output projection uses the selected projection
engine, while `add()` uses PNM. The original input Tensor remains available for the
residual because operations return new values.

### Step 5e — Run the SwiGLU FFN and add the residual

```cpp
auto ffn_input =
    require(session.rms_norm(residual, weights.ffn_norm, config.epsilon), "FFN RMSNorm");
auto gate = require(session.linear(ffn_input, weights.gate, projection), "Gate projection");
auto up = require(session.linear(ffn_input, weights.up, projection), "Up projection");
auto activated = require(session.silu(gate), "SiLU");
auto gated = require(session.mul(activated, up), "SwiGLU multiply");
auto down = require(session.linear(gated, weights.down, projection), "Down projection");
auto output = require(session.add(residual, down), "FFN residual");
```

Gate and up are `[T,6]`; SwiGLU computes `SiLU(gate) * up`. The down
projection restores `[T,4]`, and the residual addition produces the layer output.
The three projections use the selected engine; RMSNorm, SiLU, mul, and add use PNM.
No host transfer is needed between these operations.

### Step 6 — Wait at the application boundary

```cpp
auto completed = output.wait();
if (!completed) {
  throw std::runtime_error("Decoder execution failed: " + completed.error().message);
}
```

The example waits once per layer step to report committed token count. All earlier
`require()` calls checked submission only. Pending Tensor/Event dependencies keep the
operation sequence ordered without waiting between every call.

For CPU access, call and check `session.to_host(output)`, which waits and returns
owned FP16 bytes. For another NDP layer, pass `output` directly. `to_host_async()`
returns a HostTransfer whose `wait()` yields the completed host result.

`scheduled_length()` counts accepted append reservations. `committed_length()` derives
the successful prefix from append Events. After these successful waits, the example
reports 2, 3, 4, and 5 tokens. Cache metadata access belongs to the submitting thread.

### Step 7 — Reset request state and close

```cpp
auto reset = cache.value().reset(session);
// Check reset, then wait and check reset.value().wait().
auto closed = session.close();
// Check closed for shutdown and earlier execution failures.
```

The full example checks each Result. Reset calls L2 `clear` on both K/V buffers:
it zeroes their contents and resets logical lengths while keeping the allocations.
Wait for successful reset before reusing a failed cache. Keep Session and weights
alive if more requests follow; this example closes because the program is exiting.

A failed append can leave partially written bytes. Subsequent dependent work fails,
and the cache reports poisoned state. Reset or recreate that request's cache before
retrying. A whole decoder step is not transactional: failure after successful append
can leave history advanced. Session synchronize/close retain the first execution
error even after a successful recovery. Failed initial allocation requires a new cache.

## 7. Current costs and limitations

- The public API is asynchronous, but one FIFO worker executes blocking backend calls
  serially. There is no PNM/PIM overlap, multi-device routing, or request scheduler.
- Tensor reshape and DeviceBuffer snapshot copy data. L3 attention copies cache
  prefixes and creates per-query/head intermediates; it is not optimized long-context
  serving. No zero-copy views or paged KV cache are implemented.
- L3 RoPE/attention store FP16 intermediate results. Compared with fused FP32-intermediate
  kernels, rounding can differ and large attention scores can overflow before softmax.
  Current decoder tests use numerical tolerances, not bitwise equivalence.
- PIM currently supports matvec and bias-free M=1 linear. Dense output and test packing
  are functional-backend contracts, not verified FPGA behavior.
- The example has no batching/padding-mask interface, model loader, or generation loop.
  PyTorch integration remains proposed work.

For object ownership details, see [Object abstractions](library-abstractions.md).
For old-to-new API names and the buffer contract, see [Generic L2 migration](generic-l2.md).
For proposed Python bindings and their limits, see [PyTorch integration](pytorch-integration.md).
