// Provide independent FP16 numerical references and backend forwarding helpers for tests.
#pragma once
#include "ndp/l2/session.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#define CHECK(x)                                                                                   \
  do {                                                                                             \
    if (!(x))                                                                                      \
      throw std::runtime_error(#x);                                                                \
  } while (false)

namespace reference {
inline double decode(uint16_t h) {
  int e = (h >> 10) & 31, f = h & 1023;
  double x = e == 0    ? std::ldexp(double(f), -24)
             : e == 31 ? (f ? NAN : INFINITY)
                       : std::ldexp(1 + f / 1024.0, e - 15);
  return h & 0x8000 ? -x : x;
}

inline uint16_t encode(double value) {
  uint16_t sign = std::signbit(value) ? 0x8000 : 0;
  double x = std::abs(value);
  if (std::isnan(x)) {
    return sign | 0x7e00;
  }
  if (x >= 65520 || std::isinf(x)) {
    return sign | 0x7c00;
  }
  if (x == 0) {
    return sign;
  }
  int e;
  std::frexp(x, &e);
  --e;
  int step = std::max(-24, e - 10);
  double scaled = std::ldexp(x, -step), base = std::floor(scaled), f = scaled - base;
  uint32_t n = uint32_t(base);
  if (f > 0.5 || (f == 0.5 && (n & 1))) {
    ++n;
  }
  if (e < -14) {
    return sign | uint16_t(n);
  }
  if (n == 2048) {
    ++e;
    n = 1024;
  }
  return sign | uint16_t(((e + 15) << 10) | (n - 1024));
}

inline float round(double x) {
  return float(decode(encode(x)));
}

inline ndp::Tensor upload(ndp::Session &s, std::vector<uint64_t> shape,
                          const std::vector<float> &values) {
  std::vector<uint16_t> bits;
  for (float x : values) {
    bits.push_back(encode(x));
  }
  auto result = s.from_host({{std::move(shape)}, std::as_bytes(std::span(bits))});
  CHECK(result);
  return result.value();
}

inline std::vector<float> download(ndp::Session &s, const ndp::Tensor &t) {
  auto result = s.to_host(t);
  CHECK(result);
  std::vector<float> values;
  for (size_t i = 0; i < result.value().bytes.size(); i += 2) {
    uint16_t h;
    std::memcpy(&h, result.value().bytes.data() + i, 2);
    values.push_back(float(decode(h)));
  }
  return values;
}

inline void close(const std::vector<float> &actual, const std::vector<float> &expected,
                  float atol = 0.002f, float rtol = 0.003f) {
  CHECK(actual.size() == expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    if (std::isnan(expected[i])) {
      CHECK(std::isnan(actual[i]));
      continue;
    }
    if (actual[i] == expected[i]) {
      continue;
    }
    if (!(std::abs(actual[i] - expected[i]) <= atol + rtol * std::abs(expected[i]))) {
      std::cerr << "Mismatch " << i << ": " << actual[i] << " vs " << expected[i] << '\n';
      CHECK(false);
    }
  }
}

class ForwardBackend : public ndp::IBackend {
public:
  std::unique_ptr<ndp::IBackend> inner = ndp::make_functional_backend(1 << 20);

  ndp::DeviceCaps capabilities() const override {
    return inner->capabilities();
  }

  ndp::Result<ndp::AllocationHandle> allocate(size_t n) override {
    return inner->allocate(n);
  }

  void release(ndp::AllocationHandle h) noexcept override {
    inner->release(h);
  }

  ndp::Result<void> upload(ndp::AllocationHandle h, std::span<const std::byte> b) override {
    return inner->upload(h, b);
  }

  ndp::Result<void> download(ndp::AllocationHandle h, std::span<std::byte> b) override {
    return inner->download(h, b);
  }

  ndp::Result<void> fill(ndp::AllocationHandle h, std::byte v) override {
    return inner->fill(h, v);
  }

  ndp::Result<void> shutdown() override {
    return inner->shutdown();
  }

  ndp::Result<ndp::LayoutDesc> query_layout(const ndp::TensorDesc &d,
                                            ndp::LayoutKind k) const override {
    return inner->query_layout(d, k);
  }

  ndp::Result<void> transform(ndp::AllocationHandle a, const ndp::LayoutDesc &ad,
                              ndp::AllocationHandle b, const ndp::LayoutDesc &bd) override {
    return inner->transform(a, ad, b, bd);
  }

  ndp::Result<ndp::KernelPlan> query_operation(const ndp::OperationSpec &o) const override {
    return inner->query_operation(o);
  }

  ndp::Result<void> reserve_workspace(size_t n) override {
    return inner->reserve_workspace(n);
  }

  ndp::Result<void> execute(const ndp::KernelPlan &p, std::span<const ndp::AllocationHandle> in,
                            ndp::AllocationHandle out) override {
    return inner->execute(p, in, out);
  }

  ndp::Result<void> copy_region(ndp::AllocationHandle a, size_t ao, ndp::AllocationHandle b,
                                size_t bo, size_t n) override {
    return inner->copy_region(a, ao, b, bo, n);
  }
};
} // namespace reference
