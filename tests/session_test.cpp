#include "ndp/session.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>

#define CHECK(expr) do { if (!(expr)) throw std::runtime_error(#expr); } while (false)
using namespace ndp;
struct Probe { int releases = 0, shutdowns = 0, destroyed = 0; bool fail_upload = false; };
class ProbedBackend final : public IBackend {
 public:
  explicit ProbedBackend(std::shared_ptr<Probe> p) : probe(std::move(p)) {}
  ~ProbedBackend() override { ++probe->destroyed; }
  DeviceCaps capabilities() const override { return impl->capabilities(); }
  Result<AllocationHandle> allocate(size_t n) override { return impl->allocate(n); }
  void release(AllocationHandle h) noexcept override { ++probe->releases; impl->release(h); }
  Result<void> upload(AllocationHandle h, std::span<const std::byte> b) override {
    if (probe->fail_upload) return Status{ErrorCode::BackendFailure, "Injected upload failure"};
    return impl->upload(h, b);
  }
  Result<void> download(AllocationHandle h, std::span<std::byte> b) override { return impl->download(h,b); }
  Result<void> shutdown() override { ++probe->shutdowns; return impl->shutdown(); }
 private:
  std::shared_ptr<Probe> probe;
  std::unique_ptr<IBackend> impl = make_functional_backend(64);
};
Session session(size_t bytes = 64) {
  auto r = Session::create(make_functional_backend(bytes));
  CHECK(r); return std::move(r.value());
}
void roundtrip_and_ownership() {
  auto s = session(8);
  std::array<uint16_t, 4> values{0x3d00, 0xc000, 0x0000, 0x4800};
  auto expected = values;
  {
    auto t = s.from_host({{{2,2}}, std::as_bytes(std::span(values))}); CHECK(t);
    values.fill(99);
    auto alias = t.value();
    CHECK(s.memory_stats().allocations == 1);
    CHECK(s.memory_stats().live_bytes == 8);
    auto result = s.to_host(alias); CHECK(result);
    CHECK(result.value().desc.shape == std::vector<uint64_t>({2,2}));
    CHECK(std::memcmp(result.value().bytes.data(), expected.data(), sizeof(expected)) == 0);
    auto oom = s.from_host({{{4}}, std::as_bytes(std::span(values))});
    CHECK(!oom && oom.error().code == ErrorCode::OutOfMemory);
    auto other = session(); auto mismatch = other.to_host(alias);
    CHECK(!mismatch && mismatch.error().code == ErrorCode::SessionMismatch);
  }
  CHECK(s.memory_stats().live_bytes == 0);
  CHECK(s.memory_stats().peak_bytes == 8);
  CHECK(s.memory_stats().allocations == 0);
  { auto reused = s.from_host({{{4}}, std::as_bytes(std::span(values))}); CHECK(reused); CHECK(reused.value().wait()); }
  CHECK(s.synchronize()); CHECK(s.close()); CHECK(s.close());
  auto closed = s.from_host({{{4}}, std::as_bytes(std::span(values))});
  CHECK(!closed && closed.error().code == ErrorCode::SessionClosed);
}
void validation() {
  CHECK(!Session::create(nullptr));
  CHECK(!byte_size({{}, DType::F16}));
  CHECK(!byte_size({{0}, DType::F16}));
  CHECK(!byte_size({{std::numeric_limits<uint64_t>::max(), 2}, DType::F16}));
  CHECK(!byte_size({{1}, DType::F32}));
  CHECK(byte_size({{3}, DType::F16}).value() == 6);
  auto dtype = byte_size({{1}, DType::BF16});
  CHECK(!dtype && dtype.error().code == ErrorCode::UnsupportedDType);
  auto s = session();
  std::array<uint16_t, 1> value{1};
  auto bad = s.from_host({{{2}}, std::as_bytes(std::span(value))});
  CHECK(!bad && bad.error().code == ErrorCode::InvalidArgument);
  CHECK(s.memory_stats().allocations == 0);
}
void lifetime_and_failure() {
  auto probe = std::make_shared<Probe>();
  std::optional<Tensor> retained;
  std::array<uint16_t, 1> value{1};
  {
    auto created = Session::create(std::make_unique<ProbedBackend>(probe)); CHECK(created);
    auto s = std::move(created.value());
    probe->fail_upload = true;
    auto fail = s.from_host({{{1}}, std::as_bytes(std::span(value))});
    CHECK(fail);
    auto failure = fail.value().wait();
    CHECK(!failure && failure.error().code == ErrorCode::BackendFailure);
    fail = Status{ErrorCode::BackendFailure, "drop failed tensor"};
    CHECK(probe->releases == 1); CHECK(s.memory_stats().live_bytes == 0);
    probe->fail_upload = false;
    auto t = s.from_host({{{1}}, std::as_bytes(std::span(value))}); CHECK(t);
    retained = t.value();
    CHECK(t.value().wait());
    CHECK(!s.close()); CHECK(!s.close());
    CHECK(!s.to_host(*retained));
    CHECK(probe->releases == 1);
  }
  CHECK(probe->shutdowns == 1); CHECK(probe->destroyed == 0);
  CHECK(retained->desc().shape[0] == 1);
  retained.reset();
  CHECK(probe->releases == 2); CHECK(probe->destroyed == 1);
}
void move_assignment() {
  auto probe = std::make_shared<Probe>();
  auto created = Session::create(std::make_unique<ProbedBackend>(probe)); CHECK(created);
  auto s = std::move(created.value());
  s = session();
  CHECK(probe->shutdowns == 1); CHECK(probe->destroyed == 1);
  CHECK(s.synchronize()); CHECK(!created.value().synchronize());
}
int main() {
  try {
    roundtrip_and_ownership(); validation(); lifetime_and_failure(); move_assignment();
    std::cout << "All session contracts passed\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
