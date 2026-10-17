// Build deterministic FP16 inputs, weights, and RoPE tables for decoder examples and tests.
#pragma once

#include "ndp/common/fp16.hpp"
#include "ndp/l3/decoder.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ndp::example {

// Deterministic values make example runs and reference comparisons reproducible.
inline std::vector<float> values(size_t count, int seed) {
  std::vector<float> data;
  data.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const int centered = static_cast<int>((i * 7 + seed) % 11) - 5;
    data.push_back(static_cast<float>(centered) / 16);
  }
  return data;
}

inline Tensor tensor(Session &session, std::vector<uint64_t> shape,
                     const std::vector<float> &host_values) {
  std::vector<uint16_t> bits;
  bits.reserve(host_values.size());
  for (float value : host_values) {
    bits.push_back(float_to_fp16(value));
  }

  auto uploaded = session.from_host({{std::move(shape)}, std::as_bytes(std::span(bits))});
  if (!uploaded) {
    throw std::runtime_error("Tensor upload failed: " + uploaded.error().message);
  }
  return uploaded.value();
}

inline PreparedWeight weight(Session &session, uint64_t rows, uint64_t cols, int seed) {
  auto raw_weight = tensor(session, {rows, cols}, values(rows * cols, seed));
  auto prepared = ndp::prepare_weight(session, raw_weight);
  if (!prepared) {
    throw std::runtime_error("Weight preparation failed: " + prepared.error().message);
  }
  return prepared.value();
}

inline DecoderWeights weights(Session &session) {
  // Model width 4, two query heads, one KV head, head dimension 2, FFN width 6.
  return {
      tensor(session, {4}, {1, 0.875f, 1.125f, 1}), // Attention normalization.
      tensor(session, {4}, {0.875f, 1, 1, 1.125f}), // FFN normalization.
      weight(session, 4, 4, 1),                     // Query projection.
      weight(session, 2, 4, 2),                     // Key projection.
      weight(session, 2, 4, 3),                     // Value projection.
      weight(session, 4, 4, 4),                     // Attention output projection.
      weight(session, 6, 4, 5),                     // FFN gate projection.
      weight(session, 6, 4, 6),                     // FFN up projection.
      weight(session, 4, 6, 7),                     // FFN down projection.
  };
}

inline std::pair<Tensor, Tensor> tables(Session &session, uint64_t tokens) {
  std::vector<float> sine;
  std::vector<float> cosine;
  sine.reserve(tokens);
  cosine.reserve(tokens);
  for (uint64_t position = 0; position < tokens; ++position) {
    const float angle = static_cast<float>(position) * 0.25f;
    sine.push_back(std::sin(angle));
    cosine.push_back(std::cos(angle));
  }

  // One sine/cosine value per position for a head dimension of two.
  return {
      tensor(session, {tokens, 1}, sine),
      tensor(session, {tokens, 1}, cosine),
  };
}

} // namespace ndp::example
