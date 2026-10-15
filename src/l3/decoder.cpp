// Compose a pre-norm decoder block with RoPE, cached GQA, SwiGLU, and residual connections.
#include "ndp/l3/decoder.hpp"

#include <cmath>
#include <limits>

namespace ndp {
Result<Tensor> decoder_block(Session &session, const Tensor &input, const DecoderWeights &weights,
                             KVCache &cache, const Tensor &sine_table, const Tensor &cosine_table,
                             DecoderConfig config, Engine engine) {
  const auto &shape = input.desc().shape;
  if (shape.size() != 2 || !config.query_heads || !config.kv_heads || !config.head_dim ||
      config.head_dim % 2 || config.query_heads % config.kv_heads ||
      !std::isfinite(config.epsilon) || config.epsilon <= 0 ||
      config.query_heads > std::numeric_limits<uint64_t>::max() / config.head_dim) {
    return Status{ErrorCode::InvalidArgument, "Invalid decoder dimensions"};
  }
  const uint64_t model_width = config.query_heads * config.head_dim;
  const uint64_t kv_width = config.kv_heads * config.head_dim;
  const uint64_t token_count = shape[0];
  const auto &cache_description = cache.desc();
  uint64_t start_position = cache.scheduled_length();
  if (shape[1] != model_width || cache_description.kv_heads != config.kv_heads ||
      cache_description.head_dim != config.head_dim ||
      token_count > cache_description.capacity_tokens - start_position ||
      sine_table.desc() != cosine_table.desc() || sine_table.desc().shape.size() != 2 ||
      sine_table.desc().shape[1] != config.head_dim / 2 ||
      start_position > sine_table.desc().shape[0] ||
      token_count > sine_table.desc().shape[0] - start_position) {
    return Status{ErrorCode::InvalidArgument, "Decoder cache/input/RoPE table mismatch"};
  }
  auto matches = [](const Tensor &t, std::vector<uint64_t> shape) {
    return t.desc().shape == shape && t.desc().dtype == DType::F16;
  };
  if (weights.gate.tensor().desc().shape.size() != 2) {
    return Status{ErrorCode::InvalidArgument, "Invalid gate weight"};
  }
  uint64_t ffn_width = weights.gate.tensor().desc().shape[0];
  if (!matches(weights.attention_norm, {model_width}) ||
      !matches(weights.ffn_norm, {model_width}) ||
      !matches(weights.q.tensor(), {model_width, model_width}) ||
      !matches(weights.k.tensor(), {kv_width, model_width}) ||
      !matches(weights.v.tensor(), {kv_width, model_width}) ||
      !matches(weights.output.tensor(), {model_width, model_width}) ||
      !matches(weights.gate.tensor(), {ffn_width, model_width}) ||
      !matches(weights.up.tensor(), {ffn_width, model_width}) ||
      !matches(weights.down.tensor(), {model_width, ffn_width})) {
    return Status{ErrorCode::InvalidArgument, "Decoder weight shape mismatch"};
  }
  if (engine != Engine::PNM && engine != Engine::PIM) {
    return Status{ErrorCode::InvalidArgument, "Invalid projection engine"};
  }
  if (engine == Engine::PIM && token_count != 1) {
    return Status{ErrorCode::UnsupportedOperation, "PIM decoder projections require one token"};
  }
  // Ensure every operand belongs to this Session before any cache mutation.
  for (const auto *t :
       {&input, &sine_table, &cosine_table, &weights.attention_norm, &weights.ffn_norm,
        &weights.q.tensor(), &weights.k.tensor(), &weights.v.tensor(), &weights.output.tensor(),
        &weights.gate.tensor(), &weights.up.tensor(), &weights.down.tensor()}) {
    auto layout = session.query_layout(t->desc(), LayoutKind::DenseRowMajor);
    if (!layout) {
      return layout.error();
    }
    auto valid = session.materialize_layout(*t, layout.value());
    if (!valid) {
      return valid.error();
    }
  }
  try {
    OpOptions projection{engine, LayoutPolicy::RequirePrepared};
    // Normalize and project the input into query, key, and value heads.
    auto attention_norm = session.rms_norm(input, weights.attention_norm, config.epsilon);
    if (!attention_norm) {
      return attention_norm.error();
    }
    auto query_projection = session.linear(attention_norm.value(), weights.q, projection);
    if (!query_projection) {
      return query_projection.error();
    }
    auto key_projection = session.linear(attention_norm.value(), weights.k, projection);
    if (!key_projection) {
      return key_projection.error();
    }
    auto value_projection = session.linear(attention_norm.value(), weights.v, projection);
    if (!value_projection) {
      return value_projection.error();
    }
    auto queries = session.reshape(query_projection.value(),
                                   {token_count, config.query_heads, config.head_dim});
    if (!queries) {
      return queries.error();
    }
    auto keys =
        session.reshape(key_projection.value(), {token_count, config.kv_heads, config.head_dim});
    if (!keys) {
      return keys.error();
    }
    auto values =
        session.reshape(value_projection.value(), {token_count, config.kv_heads, config.head_dim});
    if (!values) {
      return values.error();
    }
    // Apply interleaved RoPE at the positions reserved for this request.
    std::vector<uint64_t> positions;
    for (uint64_t i = 0; i < token_count; ++i) {
      positions.push_back(start_position + i);
    }
    auto rotated_queries = ndp::rope(session, queries.value(), sine_table, cosine_table, positions);
    if (!rotated_queries) {
      return rotated_queries.error();
    }
    auto rotated_keys = ndp::rope(session, keys.value(), sine_table, cosine_table, positions);
    if (!rotated_keys) {
      return rotated_keys.error();
    }
    // Append K/V before attention; dependencies keep pending writes ordered.
    auto cache_update = cache.append(session, rotated_keys.value(), values.value());
    if (!cache_update) {
      return cache_update.error();
    }
    auto attention = cache.attention(session, rotated_queries.value(), start_position);
    if (!attention) {
      return attention.error();
    }
    auto merged_heads = session.reshape(attention.value(), {token_count, model_width});
    if (!merged_heads) {
      return merged_heads.error();
    }
    auto attention_projection = session.linear(merged_heads.value(), weights.output, projection);
    if (!attention_projection) {
      return attention_projection.error();
    }
    auto attention_residual = session.add(input, attention_projection.value());
    if (!attention_residual) {
      return attention_residual.error();
    }
    // Apply the SwiGLU feed-forward path and its residual connection.
    auto ffn_norm = session.rms_norm(attention_residual.value(), weights.ffn_norm, config.epsilon);
    if (!ffn_norm) {
      return ffn_norm.error();
    }
    auto gate = session.linear(ffn_norm.value(), weights.gate, projection);
    if (!gate) {
      return gate.error();
    }
    auto up = session.linear(ffn_norm.value(), weights.up, projection);
    if (!up) {
      return up.error();
    }
    auto activation = session.silu(gate.value());
    if (!activation) {
      return activation.error();
    }
    auto gated_activation = session.mul(activation.value(), up.value());
    if (!gated_activation) {
      return gated_activation.error();
    }
    auto down = session.linear(gated_activation.value(), weights.down, projection);
    if (!down) {
      return down.error();
    }
    return session.add(attention_residual.value(), down.value());
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Decoder submission allocation failed"};
  }
}
} // namespace ndp
