// Run PNM and PIM linear operations with shared prepared weights and report results and memory use.
#include "ndp/fp16.hpp"
#include "ndp/session.hpp"

#include <array>
#include <cstring>
#include <iostream>

int main() {
  constexpr size_t device_memory_bytes = 4096;
  constexpr size_t scratchpad_bytes = 64;
  auto created =
      ndp::Session::create(ndp::make_functional_backend(device_memory_bytes, scratchpad_bytes));
  if (!created) {
    std::cerr << "Session creation failed: " << created.error().message << '\n';
    return 1;
  }
  auto session = std::move(created.value());

  // W = [[1, 2, 3], [4, 5, 6]], x = [[1, 2, 3]]; xW^T = [[14, 32]].
  std::array<uint16_t, 6> host_weight{};
  for (size_t i = 0; i < host_weight.size(); ++i) {
    host_weight[i] = ndp::float_to_fp16(static_cast<float>(i + 1));
  }
  const std::array<uint16_t, 3> host_input{ndp::float_to_fp16(1), ndp::float_to_fp16(2),
                                           ndp::float_to_fp16(3)};

  auto weight = session.from_host({{{2, 3}}, std::as_bytes(std::span(host_weight))});
  auto input = session.from_host({{{1, 3}}, std::as_bytes(std::span(host_input))});
  if (!weight || !input) {
    std::cerr << "Upload failed: " << (!weight ? weight.error().message : input.error().message)
              << '\n';
    return 2;
  }

  // Budget includes 12 dense bytes and 32 packed bytes, including padding.
  const ndp::WeightPrepareOptions preparation{{ndp::Engine::PNM, ndp::Engine::PIM}, 44};
  auto prepared_weight = session.prepare_weight(weight.value(), preparation);
  if (!prepared_weight) {
    std::cerr << "Weight preparation failed: " << prepared_weight.error().message << '\n';
    return 3;
  }

  for (auto engine : {ndp::Engine::PNM, ndp::Engine::PIM}) {
    const ndp::OpOptions options{engine, ndp::LayoutPolicy::RequirePrepared};
    auto output = session.linear(input.value(), prepared_weight.value(), options);
    if (!output) {
      std::cerr << "Linear submission failed: " << output.error().message << '\n';
      return 4;
    }

    auto downloaded = session.to_host(output.value());
    if (!downloaded) {
      std::cerr << "Linear execution/download failed: " << downloaded.error().message << '\n';
      return 5;
    }
    std::array<uint16_t, 2> values{};
    std::memcpy(values.data(), downloaded.value().bytes.data(), sizeof(values));
    std::cout << (engine == ndp::Engine::PNM ? "PNM" : "PIM") << ": "
              << ndp::fp16_to_float(values[0]) << ", " << ndp::fp16_to_float(values[1]) << '\n';
  }

  std::cout << "weight bytes: " << prepared_weight.value().resident_bytes()
            << ", workspace bytes: " << session.memory_stats().workspace_bytes << '\n';
  auto closed = session.close();
  if (!closed) {
    std::cerr << "Session shutdown failed: " << closed.error().message << '\n';
    return 6;
  }
  return 0;
}
