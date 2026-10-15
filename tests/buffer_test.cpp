// Verify generic buffer mutation, immutable snapshots, validation, and failure recovery without L3.
#include "ndp/common/fp16.hpp"
#include "ndp/l2/session.hpp"

#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#define CHECK(x)                                                                                   \
  do {                                                                                             \
    if (!(x))                                                                                      \
      throw std::runtime_error(#x);                                                                \
  } while (false)
using namespace ndp;

Tensor upload(Session &s, std::vector<uint64_t> shape, std::vector<float> values) {
  std::vector<uint16_t> bits;
  for (auto value : values) {
    bits.push_back(float_to_fp16(value));
  }
  auto result = s.from_host({{shape}, std::as_bytes(std::span(bits))});
  CHECK(result);
  return result.value();
}

void expect(Session &s, const Tensor &tensor, std::vector<float> values) {
  auto host = s.to_host(tensor);
  CHECK(host && host.value().bytes.size() == values.size() * 2);
  for (size_t i = 0; i < values.size(); ++i) {
    uint16_t bits;
    std::memcpy(&bits, host.value().bytes.data() + i * 2, 2);
    CHECK(fp16_to_float(bits) == values[i]);
  }
}

class FailingCopy final : public IBackend {
  std::unique_ptr<IBackend> inner = make_functional_backend(4096);
  int copies = 0;

public:
  DeviceCaps capabilities() const override {
    return inner->capabilities();
  }

  Result<AllocationHandle> allocate(size_t n) override {
    return inner->allocate(n);
  }

  void release(AllocationHandle h) noexcept override {
    inner->release(h);
  }

  Result<void> upload(AllocationHandle h, std::span<const std::byte> b) override {
    return inner->upload(h, b);
  }

  Result<void> download(AllocationHandle h, std::span<std::byte> b) override {
    return inner->download(h, b);
  }

  Result<void> fill(AllocationHandle h, std::byte v) override {
    return inner->fill(h, v);
  }

  Result<void> copy_region(AllocationHandle a, size_t ao, AllocationHandle b, size_t bo,
                           size_t n) override {
    if (++copies == 2) {
      return Status{ErrorCode::BackendFailure, "Injected partial batch failure"};
    }
    return inner->copy_region(a, ao, b, bo, n);
  }

  Result<void> shutdown() override {
    return inner->shutdown();
  }
};

void buffers() {
  auto made = Session::create(make_functional_backend(65536));
  CHECK(made);
  auto s = std::move(made.value());
  {
    auto buffer = s.allocate({{2, 3}});
    CHECK(buffer);
    auto alias = buffer.value();
    auto zero = s.snapshot(alias, {{2, 3}});
    CHECK(zero);
    auto x = upload(s, {4}, {1, 2, 3, 4});
    CHECK(s.copy_into({{x, alias, 1, 2, 3}}));
    auto before = s.snapshot(alias, {{2, 3}});
    CHECK(before);
    CHECK(s.copy_into({{x, alias, 0, 0, 4}}));
    auto after = s.snapshot(alias, {{2, 3}});
    CHECK(after);
    expect(s, zero.value(), {0, 0, 0, 0, 0, 0});
    expect(s, before.value(), {0, 0, 2, 3, 4, 0});
    expect(s, after.value(), {1, 2, 3, 4, 4, 0});
    auto packed = s.query_layout(before.value().desc(), LayoutKind::TestTile4x4);
    CHECK(packed);
    CHECK(s.materialize_layout(before.value(), packed.value()));
    auto invalid = s.copy_into({{x, alias, 0, 0, 1}, {x, alias, 0, 6, 1}});
    CHECK(!invalid && invalid.error().code == ErrorCode::InvalidArgument);
    CHECK(!s.copy_into({{x, alias, std::numeric_limits<size_t>::max(), 0, 1}}));
    CHECK(!s.copy_into({{x, alias, 0, 0, 0}}));
    CHECK(!s.copy_into({}));
    CHECK(!s.snapshot(alias, {{7}}));
    CHECK(!s.snapshot(alias, {{1}}, std::numeric_limits<size_t>::max()));
    CHECK(!s.allocate({{0}}));
    CHECK(!s.allocate({{1}, DType::F32}));
    auto unchanged = s.snapshot(alias, {{6}});
    CHECK(unchanged);
    expect(s, unchanged.value(), {1, 2, 3, 4, 4, 0});
    auto other = Session::create(make_functional_backend(1024));
    CHECK(other);
    CHECK(!other.value().snapshot(alias, {{1}}));
    CHECK(!other.value().clear({alias}));
    CHECK(!other.value().copy_into({{x, alias, 0, 0, 1}}));
    CHECK(s.clear({alias}));
    auto cleared = s.snapshot(alias, {{6}});
    CHECK(cleared);
    expect(s, cleared.value(), {0, 0, 0, 0, 0, 0});
    expect(s, before.value(), {0, 0, 2, 3, 4, 0});
    auto matrix = upload(s, {2, 3}, {1, 2, 3, 4, 5, 6});
    auto transpose = s.transpose(matrix);
    CHECK(transpose);
    expect(s, transpose.value(), {1, 4, 2, 5, 3, 6});
    auto gather = s.gather(matrix, {5, 0, 5, 2}, {2, 2});
    CHECK(gather);
    expect(s, gather.value(), {6, 1, 6, 3});
    auto scale = s.scale(gather.value(), -2);
    CHECK(scale);
    expect(s, scale.value(), {-12, -2, -12, -6});
    CHECK(!s.gather(matrix, {6}, {1}));
    CHECK(!s.gather(matrix, {0}, {2}));
    // Reordering is bit-preserving, including NaN payloads and signed zero.
    std::vector<uint16_t> special{0x7e35, 0x8000, 0xfc00, 0x0001};
    auto encoded = s.from_host({{{2, 2}}, std::as_bytes(std::span(special))});
    CHECK(encoded);
    auto reordered = s.gather(encoded.value(), {3, 0, 1, 2}, {2, 2});
    CHECK(reordered);
    auto transposed_bits = s.transpose(reordered.value());
    CHECK(transposed_bits);
    auto raw = s.to_host(transposed_bits.value());
    CHECK(raw);
    std::vector<uint16_t> expected_bits{0x0001, 0x8000, 0x7e35, 0xfc00};
    CHECK(std::memcmp(raw.value().bytes.data(), expected_bits.data(), 8) == 0);
    CHECK(!s.transpose(x));
    CHECK(!s.scale(x, std::numeric_limits<float>::infinity()));
    auto input = upload(s, {1, 3}, {1, 1, 1});
    auto linear = s.linear(input, matrix);
    CHECK(linear);
    expect(s, linear.value(), {6, 15});
  }
  CHECK(s.synchronize());
  CHECK(s.memory_stats().live_bytes == 0);
  CHECK(s.close());
  CHECK(!s.allocate({{1}}));
}

void failure_recovery() {
  auto made = Session::create(std::make_unique<FailingCopy>());
  CHECK(made);
  auto s = std::move(made.value());
  auto a = s.allocate({{2}}), b = s.allocate({{2}});
  CHECK(a && b);
  auto x = upload(s, {2}, {3, 4});
  auto batch = s.copy_into({{x, a.value(), 0, 0, 2}, {x, b.value(), 0, 0, 2}});
  CHECK(batch && !batch.value().wait());
  CHECK(a.value().ready_event().status() == EventStatus::Failed);
  CHECK(b.value().ready_event().status() == EventStatus::Failed);
  auto failed_read = s.snapshot(a.value(), {{2}});
  CHECK(failed_read);
  CHECK(!failed_read.value().wait());
  auto clear = s.clear({a.value(), b.value()});
  CHECK(clear && clear.value().wait());
  auto recovered = s.snapshot(a.value(), {{2}});
  CHECK(recovered);
  expect(s, recovered.value(), {0, 0});
  CHECK(s.copy_into({{x, b.value(), 0, 0, 2}}));
  auto written = s.snapshot(b.value(), {{2}});
  CHECK(written);
  expect(s, written.value(), {3, 4});
  CHECK(!s.close()); // The first execution failure remains reportable.
}

int main() {
  try {
    buffers();
    failure_recovery();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "Generic buffer contracts passed\n";
}
