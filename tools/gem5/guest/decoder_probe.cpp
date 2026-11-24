// Run an explicit decoder layer through guest IPC and compare it with the local emulator.
#include "../../../examples/toy_decoder.hpp"
#include "ndp/l1/gem5/backend.hpp"

#include <cerrno>
#include <iostream>
#include <sys/wait.h>

using namespace ndp;

namespace {
template <class T> T require(Result<T> result, const char *operation) {
  if (!result) {
    throw std::runtime_error(std::string(operation) + ": " + result.error().message);
  }
  return std::move(result.value());
}

void check(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void complete(Result<void> result) {
  if (!result) {
    throw std::runtime_error(result.error().message);
  }
}

// Keep layer composition visible: only projection operations switch to PIM during decode.
Tensor step(Session &session, const Tensor &input, const DecoderWeights &weights, KVCache &cache,
            const Tensor &sine_table, const Tensor &cosine_table, Engine engine) {
  const DecoderConfig config{.query_heads = 2, .kv_heads = 1, .head_dim = 2};
  const uint64_t token_count = input.desc().shape[0], model_width = 4;
  // Step 5a: Normalize and project hidden states into Q, K, and V.
  const ndp::OpOptions projection{engine, ndp::LayoutPolicy::RequirePrepared};
  auto normalized =
      require(session.rms_norm(input, weights.attention_norm, config.epsilon), "Attention RMSNorm");
  auto query = require(session.linear(normalized, weights.q.tensor(), projection), "Q projection");
  auto key = require(session.linear(normalized, weights.k.tensor(), projection), "K projection");
  auto value = require(session.linear(normalized, weights.v.tensor(), projection), "V projection");

  // Step 5b: Split heads and apply RoPE to Q/K at this step's absolute positions.
  auto query_heads = require(
      session.reshape(query, {token_count, config.query_heads, config.head_dim}), "Q heads");
  auto key_heads =
      require(session.reshape(key, {token_count, config.kv_heads, config.head_dim}), "K heads");
  auto value_heads =
      require(session.reshape(value, {token_count, config.kv_heads, config.head_dim}), "V heads");
  const uint64_t start_position = cache.scheduled_length();
  std::vector<uint64_t> positions;
  for (uint64_t token = 0; token < token_count; ++token) {
    positions.push_back(start_position + token);
  }
  auto rotated_query =
      require(ndp::rope(session, query_heads, sine_table, cosine_table, positions), "Q RoPE");
  auto rotated_key =
      require(ndp::rope(session, key_heads, sine_table, cosine_table, positions), "K RoPE");

  // Step 5c: Append K/V and attend to the causal prefix, including the new tokens.
  require(cache.append(session, rotated_key, value_heads), "KV append");
  auto attended =
      require(cache.attention(session, rotated_query, start_position), "Cached attention");

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
  auto up = require(session.linear(ffn_input, weights.up.tensor(), projection), "Up projection");
  auto activated = require(session.silu(gate), "SiLU");
  auto gated = require(session.mul(activated, up), "SwiGLU multiply");
  auto down = require(session.linear(gated, weights.down.tensor(), projection), "Down projection");
  auto output = require(session.add(residual, down), "FFN residual");

  return output;
}

using Outputs = std::vector<std::vector<std::byte>>;

Outputs run(std::unique_ptr<IBackend> backend) {
  auto session = require(Session::create(std::move(backend)), "Session");
  Outputs outputs;
  {
    auto weights = example::weights(session);
    auto [sine, cosine] = example::tables(session, 8);
    auto cache = require(KVCache::create(session, {8, 1, 2}), "KV cache");
    complete(session.synchronize());
    auto resident = session.memory_stats();
    // A, B, A requests distinguish reset/reuse from accidentally retaining old KV state.
    for (int request = 0; request < 3; ++request) {
      int seed = request == 1 ? 5 : 1;
      check(cache.scheduled_length() == 0 && cache.committed_length() == 0, "Cache not empty");
      for (int index = 0; index < 4; ++index) {
        {
          uint64_t tokens = index == 0 ? 2 : 1;
          auto input =
              example::tensor(session, {tokens, 4}, example::values(tokens * 4, seed + index));
          auto output = step(session, input, weights, cache, sine, cosine,
                             index == 0 ? Engine::PNM : Engine::PIM);
          auto host = require(session.to_host(output), "Decoder download");
          check(host.desc.shape == std::vector<uint64_t>{tokens, 4}, "Output shape");
          outputs.push_back(std::move(host.bytes));
          check(cache.committed_length() == uint64_t(index + 2), "Cache completion length");
        }
        complete(session.synchronize());
        auto current = session.memory_stats();
        check(current.live_bytes == resident.live_bytes &&
                  current.allocations == resident.allocations && current.staging_bytes == 0,
              "Temporary storage was not reclaimed");
      }
      complete(require(cache.reset(session), "Cache reset").wait());
      check(!cache.poisoned() && cache.committed_length() == 0 && cache.scheduled_length() == 0,
            "Cache reset did not commit");
    }
    for (size_t i = 0; i < 4; ++i) {
      check(outputs[i] == outputs[8 + i], "Repeated A request differs");
    }
    check(outputs[0] != outputs[4], "A/B fixture must exercise distinct inputs");
  }
  complete(session.synchronize());
  auto released = session.memory_stats();
  check(released.live_bytes == 0 && released.allocations == 0 && released.staging_bytes == 0,
        "Model/cache allocations were not reclaimed");
  complete(session.close());
  check(session.memory_stats().workspace_bytes == 0, "Workspace not released at close");
  return outputs;
}
} // namespace

int main(int argc, char **argv) {
  try {
    check(argc == 2, "Usage: ndp_decoder_probe /path/to/ndp_execution_receiver");
    const NdpSpec spec{2, 2, 512, 64, 64, 256, 128};
    auto reference = run(require(make_ndp_backend(spec), "Local backend"));
    auto actual = run(
        require(make_gem5_backend({std::filesystem::absolute(argv[1]), spec}), "Process backend"));
    check(actual == reference, "Remote/local FP16 outputs differ");
    int status;
    check(waitpid(-1, &status, WNOHANG) == -1 && errno == ECHILD, "Receiver not reaped");
    std::cout << "NDP_DECODER_EXECUTION prefill_pnm=3 decode_pim=9\n"
                 "NDP_DECODER_REFERENCE outputs=12 exact_fp16=yes\n"
                 "NDP_DECODER_REQUESTS sequence=A_B_A kv_reset=verified\n"
                 "NDP_DECODER_RECLAMATION temporaries_model_cache_workspace=verified\n"
                 "NDP_F5_PASS\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "F5 failed: " << e.what() << '\n';
    return 1;
  }
}
