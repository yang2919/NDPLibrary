#include "ndp/session.hpp"
#include <array>
#include <cstring>
#include <iostream>

int main() {
  auto created = ndp::Session::create(ndp::make_functional_backend(1024));
  if (!created) return 1;
  auto session = std::move(created.value());
  std::array<uint16_t, 4> input{0x3c00, 0x4000, 0x4200, 0x4400}; // FP16: 1, 2, 3, 4
  auto tensor = session.from_host({{{2, 2}, ndp::DType::F16}, std::as_bytes(std::span(input))});
  if (!tensor) return 2;
  auto output = session.to_host(tensor.value());
  if (!output) return 3;
  std::array<uint16_t, 4> values;
  std::memcpy(values.data(), output.value().bytes.data(), sizeof(values));
  for (auto value : values) std::cout << std::hex << value << ' ';
  std::cout << "\nlive bytes: " << std::dec << session.memory_stats().live_bytes << '\n';
  return session.close() ? 0 : 4;
}
