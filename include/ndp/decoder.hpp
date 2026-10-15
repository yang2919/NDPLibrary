// Expose decoder configuration, prepared weights, and the single-block inference entry point.
#pragma once
#include "ndp/llm.hpp"

namespace ndp {
struct DecoderConfig {
  uint64_t query_heads, kv_heads, head_dim;
  float epsilon = 1e-5f;
};

struct DecoderWeights {
  Tensor attention_norm, ffn_norm;
  PreparedWeight q, k, v, output, gate, up, down;
};

// One pre-norm decoder block, interleaved RoPE, GQA, SwiGLU and residuals.
// Inputs [T, query_heads*head_dim]. Uses cache.scheduled_length() as positions.
// Cache mutation is not transactional with the whole block: reset after block failure.
Result<Tensor> decoder_block(Session &, const Tensor &, const DecoderWeights &, KVCache &,
                             const Tensor &sin, const Tensor &cos, DecoderConfig,
                             Engine projection_engine = Engine::PNM);
} // namespace ndp
