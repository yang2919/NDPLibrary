// Compare LLM primitives against independent references and validate shape and numeric edge cases.
#include "ndp/llm.hpp"
#include "test_support.hpp"

#include <algorithm>
using namespace ndp;

void primitives() {
  auto made = Session::create(make_functional_backend(65536, 0));
  CHECK(made);
  auto s = std::move(made.value());
  std::vector<float> values{-2, -1, 0, 1, 2, 3};
  auto x = reference::upload(s, {2, 3}, values), b = reference::upload(s, {3}, {1, 2, 3});
  auto add = s.add(x, b), mul = s.mul(x, b), silu = s.silu(x), norm = s.rms_norm(x, b),
       soft = s.softmax(x);
  CHECK(add && mul && silu && norm && soft);
  std::vector<float> a, m, g, n, p;
  for (size_t i = 0; i < 6; ++i) {
    a.push_back(values[i] + float(i % 3 + 1));
    m.push_back(values[i] * float(i % 3 + 1));
    g.push_back(reference::round(values[i] / (1 + std::exp(-double(values[i])))));
  }
  for (size_t row = 0; row < 2; ++row) {
    double sq = 0, total = 0, maximum = values[row * 3 + 2];
    for (size_t d = 0; d < 3; ++d) {
      sq += values[row * 3 + d] * values[row * 3 + d];
      total += std::exp(values[row * 3 + d] - maximum);
    }
    for (size_t d = 0; d < 3; ++d) {
      n.push_back(reference::round(values[row * 3 + d] / std::sqrt(sq / 3 + 1e-5) * (d + 1)));
      p.push_back(reference::round(std::exp(values[row * 3 + d] - maximum) / total));
    }
  }
  reference::close(reference::download(s, add.value()), a);
  reference::close(reference::download(s, mul.value()), m);
  reference::close(reference::download(s, silu.value()), g);
  reference::close(reference::download(s, norm.value()), n);
  reference::close(reference::download(s, soft.value()), p);
  auto bad = reference::upload(s, {2}, {1, 2});
  CHECK(!s.add(x, bad));
  CHECK(!s.rms_norm(x, b, 0));
  CHECK(!s.rms_norm(x, b, NAN));
  CHECK(!s.softmax(x, 0));
  CHECK(!s.silu(x, {Engine::PIM}));
  CHECK(!s.reshape(x, {7}));
  auto re = s.reshape(x, {3, 2});
  CHECK(re);
  reference::close(reference::download(s, re.value()), values, 0, 0);
  auto extreme = reference::upload(
      s, {3, 3}, {1000, 1001, 999, INFINITY, INFINITY, 0, -INFINITY, -INFINITY, -INFINITY});
  auto es = s.softmax(extreme);
  CHECK(es);
  auto ev = reference::download(s, es.value());
  CHECK(std::abs(ev[0] + ev[1] + ev[2] - 1) < 0.001);
  CHECK(ev[3] == 0.5f && ev[4] == 0.5f && ev[5] == 0);
  CHECK(ev[6] == 0 && ev[7] == 0 && ev[8] == 0);
  auto rx = reference::upload(s, {2, 1, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
  auto sin = reference::upload(s, {3, 2}, {0, 0, 1, 1, 0, 0}),
       cos = reference::upload(s, {3, 2}, {1, 1, 0, 0, -1, -1});
  auto rotated = ndp::rope(s, rx, sin, cos, {1, 2});
  CHECK(rotated);
  reference::close(reference::download(s, rotated.value()), {-2, 1, -4, 3, -5, -6, -7, -8}, 0, 0);
  CHECK(!ndp::rope(s, rx, sin, cos, {1}));
  CHECK(!ndp::rope(s, rx, sin, cos, {1, 3}));
  auto odd = reference::upload(s, {1, 1, 3}, {1, 2, 3});
  CHECK(!ndp::rope(s, odd, sin, cos, {0}));
  CHECK(s.memory_stats().workspace_bytes == 0);
  CHECK(s.close());
}

int main() {
  try {
    primitives();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  std::cout << "LLM primitive contracts passed\n";
}
