// Compare prefill and incremental mixed-engine decoding against an independent scalar reference.
#include "../examples/toy_decoder.hpp"
#include "ndp/llm.hpp"
#include "test_support.hpp"

#include <atomic>
using namespace ndp;

namespace oracle {
using V = std::vector<float>;

V quantized(V x) {
  for (auto &v : x) {
    v = reference::round(v);
  }
  return x;
}

V linear(const V &x, const V &w, size_t tokens, size_t k, size_t n) {
  V y(tokens * n);
  for (size_t t = 0; t < tokens; ++t) {
    for (size_t r = 0; r < n; ++r) {
      double sum = 0;
      for (size_t d = 0; d < k; ++d) {
        sum += double(x[t * k + d]) * w[r * k + d];
      }
      y[t * n + r] = reference::round(sum);
    }
  }
  return y;
}

V norm(const V &x, const V &gamma) {
  V y(x.size());
  for (size_t t = 0; t < x.size() / 4; ++t) {
    double sum = 0;
    for (size_t d = 0; d < 4; ++d) {
      sum += double(x[t * 4 + d]) * x[t * 4 + d];
    }
    for (size_t d = 0; d < 4; ++d) {
      y[t * 4 + d] = reference::round(x[t * 4 + d] / std::sqrt(sum / 4 + 1e-5) * gamma[d]);
    }
  }
  return y;
}

V rope(V x, size_t tokens, size_t heads, size_t start) {
  for (size_t t = 0; t < tokens; ++t) {
    double sine = reference::round(std::sin(float(start + t) * 0.25f)),
           cosine = reference::round(std::cos(float(start + t) * 0.25f));
    for (size_t h = 0; h < heads; ++h) {
      size_t base = (t * heads + h) * 2;
      double a = x[base], b = x[base + 1];
      x[base] = reference::round(a * cosine - b * sine);
      x[base + 1] = reference::round(a * sine + b * cosine);
    }
  }
  return x;
}

struct Block {
  V keys, values;

  V run(V x, size_t tokens) {
    x = quantized(x);
    size_t start = keys.size() / 2;
    auto n = norm(x, {1, .875f, 1.125f, 1});
    auto q = rope(linear(n, example::values(16, 1), tokens, 4, 4), tokens, 2, start);
    auto k = rope(linear(n, example::values(8, 2), tokens, 4, 2), tokens, 1, start);
    auto v = linear(n, example::values(8, 3), tokens, 4, 2);
    keys.insert(keys.end(), k.begin(), k.end());
    values.insert(values.end(), v.begin(), v.end());
    V attended(tokens * 4);
    for (size_t t = 0; t < tokens; ++t) {
      for (size_t h = 0; h < 2; ++h) {
        std::vector<double> scores(start + t + 1);
        double maximum = -INFINITY, total = 0;
        for (size_t j = 0; j < scores.size(); ++j) {
          scores[j] = (double(q[t * 4 + h * 2]) * keys[j * 2] +
                       double(q[t * 4 + h * 2 + 1]) * keys[j * 2 + 1]) /
                      std::sqrt(2.0);
          maximum = std::max(maximum, scores[j]);
        }
        for (auto &score : scores) {
          score = std::exp(score - maximum);
          total += score;
        }
        for (size_t d = 0; d < 2; ++d) {
          double sum = 0;
          for (size_t j = 0; j < scores.size(); ++j) {
            sum += scores[j] / total * values[j * 2 + d];
          }
          attended[t * 4 + h * 2 + d] = reference::round(sum);
        }
      }
    }
    auto projection = linear(attended, example::values(16, 4), tokens, 4, 4);
    V residual(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      residual[i] = reference::round(x[i] + projection[i]);
    }
    auto ffn = norm(residual, {.875f, 1, 1, 1.125f});
    auto gate = linear(ffn, example::values(24, 5), tokens, 4, 6),
         up = linear(ffn, example::values(24, 6), tokens, 4, 6);
    for (size_t i = 0; i < gate.size(); ++i) {
      float silu = reference::round(gate[i] / (1 + std::exp(-double(gate[i]))));
      gate[i] = reference::round(silu * up[i]);
    }
    auto down = linear(gate, example::values(24, 7), tokens, 6, 4);
    for (size_t i = 0; i < down.size(); ++i) {
      down[i] = reference::round(residual[i] + down[i]);
    }
    return down;
  }
};
} // namespace oracle

struct Counts {
  std::atomic<int> packs{0}, downloads{0};
};

class Counted : public reference::ForwardBackend {
public:
  explicit Counted(std::shared_ptr<Counts> c) : c_(std::move(c)) {}

  Result<void> transform(AllocationHandle a, const LayoutDesc &ad, AllocationHandle b,
                         const LayoutDesc &bd) override {
    ++c_->packs;
    return inner->transform(a, ad, b, bd);
  }

  Result<void> download(AllocationHandle h, std::span<std::byte> b) override {
    ++c_->downloads;
    return inner->download(h, b);
  }

private:
  std::shared_ptr<Counts> c_;
};

void decoder() {
  auto counts = std::make_shared<Counts>();
  auto made = Session::create(std::make_unique<Counted>(counts));
  CHECK(made);
  auto s = std::move(made.value());
  auto weights = example::weights(s);
  auto tables = example::tables(s, 8);
  CHECK(s.synchronize());
  size_t baseline = s.memory_stats().live_bytes;
  CHECK(counts->packs == 7);
  std::vector<float> input = example::values(20, 9), mixed;
  for (int request = 0; request < 2; ++request) {
    oracle::Block reference_block;
    {
      auto cache = ndp::KVCache::create(s, {8, 1, 2});
      CHECK(cache);
      size_t offset = 0;
      for (size_t step = 0; step < 4; ++step) {
        size_t tokens = step == 0 ? 2 : 1;
        std::vector<float> slice(input.begin() + offset * 4, input.begin() + (offset + tokens) * 4);
        {
          auto x = reference::upload(s, {tokens, 4}, slice);
          auto output = decoder_block(s, x, weights, cache.value(), tables.first, tables.second,
                                      {2, 1, 2}, step == 0 ? Engine::PNM : Engine::PIM);
          CHECK(output);
          CHECK(output.value().wait());
          CHECK(counts->downloads == int(request * 4 + step));
          auto actual = reference::download(s, output.value());
          reference::close(actual, reference_block.run(slice, tokens), 0.002f, 0.003f);
          if (request == 0) {
            mixed.insert(mixed.end(), actual.begin(), actual.end());
          }
        }
        offset += tokens;
        CHECK(cache.value().committed_length() == offset);
        CHECK(counts->packs == 7);
      }
      CHECK(cache.value().committed_length() == 5);
      auto reset = cache.value().reset(s);
      CHECK(reset);
      CHECK(reset.value().wait());
      CHECK(cache.value().committed_length() == 0);
    }
    CHECK(s.synchronize());
    CHECK(s.memory_stats().live_bytes == baseline);
  }
  {
    auto cache = ndp::KVCache::create(s, {8, 1, 2});
    CHECK(cache);
    auto x = reference::upload(s, {5, 4}, input);
    auto all = decoder_block(s, x, weights, cache.value(), tables.first, tables.second, {2, 1, 2});
    CHECK(all);
    reference::close(reference::download(s, all.value()), mixed, 0.002f, 0.003f);
    CHECK(!decoder_block(s, x, weights, cache.value(), tables.first, tables.second, {2, 1, 2},
                         Engine::PIM));
  }
  CHECK(s.synchronize());
  CHECK(s.memory_stats().live_bytes == baseline);
  CHECK(s.close());
}

int main() {
  try {
    decoder();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  std::cout << "Prefill + 3 decode steps match independent reference\n";
}
