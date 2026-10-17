// Compose a decoder layer from Session operations for PNM prefill and PIM-projection decode.
#include "toy_decoder.hpp"

#include <iostream>

namespace {
// Check submission errors without waiting for queued execution.
template <class T> T require(ndp::Result<T> result, const char *operation) {
  if (!result) {
    throw std::runtime_error(std::string(operation) + ": " + result.error().message);
  }
  return std::move(result.value());
}
} // namespace

int main() {
  try {
    constexpr size_t device_memory_bytes = 1 << 20;
    constexpr size_t scratchpad_bytes = 256;
    constexpr uint64_t cache_capacity = 8;
    constexpr uint64_t model_width = 4;
    constexpr size_t decode_steps = 3;

    // Step 1: Open one device session.
    auto created =
        ndp::Session::create(ndp::make_functional_backend(device_memory_bytes, scratchpad_bytes));
    if (!created) {
      std::cerr << "Session creation failed: " << created.error().message << '\n';
      return 1;
    }
    auto session = std::move(created.value());

    // Step 2: Prepare model weights and position tables once.
    auto weights = ndp::example::weights(session);
    auto [sine_table, cosine_table] = ndp::example::tables(session, cache_capacity);
    const ndp::DecoderConfig config{
        .query_heads = 2,
        .kv_heads = 1,
        .head_dim = 2,
    };
    // Step 3: Allocate the request's persistent K/V storage.
    auto cache = ndp::KVCache::create(session, {cache_capacity, config.kv_heads, config.head_dim});
    if (!cache) {
      std::cerr << "KV cache creation failed: " << cache.error().message << '\n';
      return 2;
    }

    for (size_t step = 0; step <= decode_steps; ++step) {
      const bool is_prefill = step == 0;
      const uint64_t token_count = is_prefill ? 2 : 1;
      const auto engine = is_prefill ? ndp::Engine::PNM : ndp::Engine::PIM;
      // Step 4: Upload this step's hidden states.
      auto input = ndp::example::tensor(
          session, {token_count, model_width},
          ndp::example::values(token_count * model_width, static_cast<int>(step + 1)));

      // Step 5a: Normalize and project hidden states into Q, K, and V.
      const ndp::OpOptions projection{engine, ndp::LayoutPolicy::RequirePrepared};
      auto normalized = require(session.rms_norm(input, weights.attention_norm, config.epsilon),
                                "Attention RMSNorm");
      auto query =
          require(session.linear(normalized, weights.q.tensor(), projection), "Q projection");
      auto key =
          require(session.linear(normalized, weights.k.tensor(), projection), "K projection");
      auto value =
          require(session.linear(normalized, weights.v.tensor(), projection), "V projection");

      // Step 5b: Split heads and apply RoPE to Q/K at this step's absolute positions.
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

      // Step 5c: Append K/V and attend to the causal prefix, including the new tokens.
      require(cache.value().append(session, rotated_key, value_heads), "KV append");
      auto attended = require(cache.value().attention(session, rotated_query, start_position),
                              "Cached attention");

      // Step 5d: Merge heads, project, and add the attention residual.
      auto merged = require(session.reshape(attended, {token_count, model_width}), "Merge heads");
      auto projected = require(session.linear(merged, weights.output.tensor(), projection),
                               "Attention output projection");
      auto residual = require(session.add(input, projected), "Attention residual");

      // Step 5e: Apply SwiGLU FFN and add the second residual.
      auto ffn_input =
          require(session.rms_norm(residual, weights.ffn_norm, config.epsilon), "FFN RMSNorm");
      auto gate =
          require(session.linear(ffn_input, weights.gate.tensor(), projection), "Gate projection");
      auto up =
          require(session.linear(ffn_input, weights.up.tensor(), projection), "Up projection");
      auto activated = require(session.silu(gate), "SiLU");
      auto gated = require(session.mul(activated, up), "SwiGLU multiply");
      auto down =
          require(session.linear(gated, weights.down.tensor(), projection), "Down projection");
      auto output = require(session.add(residual, down), "FFN residual");

      // Step 6: Wait only where this example needs to report completion.
      auto completed = output.wait();
      if (!completed) {
        throw std::runtime_error("Decoder execution failed: " + completed.error().message);
      }

      std::cout << (is_prefill ? "prefill" : "decode")
                << ": committed tokens=" << cache.value().committed_length() << '\n';
    }

    // Step 7: Reset request state, then close the session.
    auto reset = cache.value().reset(session);
    if (!reset) {
      std::cerr << "Cache reset submission failed: " << reset.error().message << '\n';
      return 3;
    }
    auto reset_completed = reset.value().wait();
    if (!reset_completed) {
      std::cerr << "Cache reset failed: " << reset_completed.error().message << '\n';
      return 3;
    }

    auto closed = session.close();
    if (!closed) {
      std::cerr << "Session shutdown failed: " << closed.error().message << '\n';
      return 4;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
