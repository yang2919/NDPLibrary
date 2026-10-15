// Verify packed layout mapping, padding, reuse, failures, and allocation boundaries.
#include "ndp/session.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#define CHECK(expr)                                                                                \
  do {                                                                                             \
    if (!(expr))                                                                                   \
      throw std::runtime_error(#expr);                                                             \
  } while (false)
using namespace ndp;
using namespace std::chrono_literals;

struct Control {
  std::mutex mutex;
  std::condition_variable changed;
  bool entered = false, open = false;
  std::atomic<int> transforms{0}, downloads{0};
  bool fail_first = false;

  void unblock() {
    {
      std::lock_guard lock(mutex);
      open = true;
    }
    changed.notify_all();
  }

  void wait() {
    std::unique_lock lock(mutex);
    CHECK(changed.wait_for(lock, 5s, [&] { return entered; }));
  }
};

struct Guard {
  std::shared_ptr<Control> c;

  ~Guard() {
    c->unblock();
  }
};

class Instrumented final : public IBackend {
public:
  explicit Instrumented(std::shared_ptr<Control> c) : c_(std::move(c)) {}

  DeviceCaps capabilities() const override {
    return backend_->capabilities();
  }

  Result<AllocationHandle> allocate(size_t n) override {
    return backend_->allocate(n);
  }

  void release(AllocationHandle h) noexcept override {
    backend_->release(h);
  }

  Result<void> upload(AllocationHandle h, std::span<const std::byte> b) override {
    return backend_->upload(h, b);
  }

  Result<void> download(AllocationHandle h, std::span<std::byte> b) override {
    ++c_->downloads;
    return backend_->download(h, b);
  }

  Result<void> shutdown() override {
    return backend_->shutdown();
  }

  Result<LayoutDesc> query_layout(const TensorDesc &d, LayoutKind k) const override {
    return backend_->query_layout(d, k);
  }

  Result<void> transform(AllocationHandle src, const LayoutDesc &from, AllocationHandle dst,
                         const LayoutDesc &to) override {
    int index = ++c_->transforms;
    {
      std::unique_lock lock(c_->mutex);
      c_->entered = true;
      c_->changed.notify_all();
      c_->changed.wait(lock, [&] { return c_->open; });
    }
    if (index == 1 && c_->fail_first) {
      return Status{ErrorCode::BackendFailure, "Injected transform failure"};
    }
    return backend_->transform(src, from, dst, to);
  }

private:
  std::shared_ptr<Control> c_;
  std::unique_ptr<IBackend> backend_ = make_functional_backend(4096);
};

void roundtrips() {
  auto created = Session::create(make_functional_backend(65536));
  CHECK(created);
  auto s = std::move(created.value());
  // Scalar, aligned, row/column padding, and multi-tile rectangular cases.
  for (auto [rows, cols] :
       {std::pair<uint64_t, uint64_t>{1, 1}, {4, 4}, {3, 4}, {4, 3}, {5, 9}, {9, 5}}) {
    std::vector<uint16_t> input(rows * cols);
    for (size_t i = 0; i < input.size(); ++i) {
      input[i] = static_cast<uint16_t>(i * 997); // arbitrary binary16 bits preserved
    }
    auto t = s.from_host({{{rows, cols}}, std::as_bytes(std::span(input))});
    CHECK(t);
    auto layout = s.query_layout(t.value().desc(), LayoutKind::TestTile4x4);
    CHECK(layout);
    CHECK(layout.value().physical_bytes == ((rows + 3) / 4) * ((cols + 3) / 4) * 32);
    auto e = s.materialize_layout(t.value(), layout.value());
    CHECK(e);
    auto output = s.to_host(t.value(), LayoutKind::TestTile4x4);
    CHECK(output);
    CHECK(output.value().bytes.size() == input.size() * 2);
    CHECK(std::memcmp(input.data(), output.value().bytes.data(), input.size() * 2) == 0);
    CHECK(t.value().representations().size() == 2);
    CHECK(s.memory_stats().live_bytes == input.size() * 2 + layout.value().physical_bytes);
    CHECK(s.memory_stats().allocations == 2);
    auto dense = s.to_host(t.value());
    CHECK(dense);
    CHECK(dense.value().bytes == output.value().bytes);
  }
  CHECK(s.synchronize());
  CHECK(s.memory_stats().live_bytes == 0);
}

void physical_mapping() {
  auto backend = make_functional_backend(1024);
  TensorDesc desc{{3, 5}};
  auto dense = backend->query_layout(desc, LayoutKind::DenseRowMajor).value();
  auto packed = backend->query_layout(desc, LayoutKind::TestTile4x4).value();
  auto src = backend->allocate(dense.physical_bytes).value();
  auto dst = backend->allocate(packed.physical_bytes).value();
  std::vector<uint16_t> input(15);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<uint16_t>(i + 1);
  }
  CHECK(backend->upload(src, std::as_bytes(std::span(input))));
  CHECK(backend->transform(src, dense, dst, packed));
  std::vector<uint16_t> physical(32, 0xffff);
  CHECK(backend->download(dst, std::as_writable_bytes(std::span(physical))));
  // Independent tile-order oracle: columns inside each 4x4 tile, zero padding.
  std::vector<uint16_t> expected{1, 6,  11, 0, 2, 7, 12, 0, 3, 8, 13, 0, 4, 9, 14, 0,
                                 5, 10, 15, 0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0};
  CHECK(physical == expected);
  CHECK(!backend->transform(src, dense, src, dense));
  auto forged = packed;
  forged.physical_bytes -= 2;
  CHECK(!backend->transform(src, dense, dst, forged));
  backend->release(src);
  backend->release(dst);
  CHECK(backend->shutdown());
}

void pending_dedup_and_lifetime(bool fail) {
  auto c = std::make_shared<Control>();
  c->fail_first = fail;
  auto created = Session::create(std::make_unique<Instrumented>(c));
  CHECK(created);
  auto s = std::move(created.value());
  Guard guard{c};
  std::vector<uint16_t> bits(15, 0x3c00);
  auto t = s.from_host({{{3, 5}}, std::as_bytes(std::span(bits))});
  CHECK(t);
  auto layout = s.query_layout(t.value().desc(), LayoutKind::TestTile4x4).value();
  auto first = s.materialize_layout(t.value(), layout);
  CHECK(first);
  c->wait();
  auto again = s.materialize_layout(t.value(), layout);
  CHECK(again);
  CHECK(s.memory_stats().allocations == 2);
  CHECK(c->transforms == 1);
  CHECK(again.value().status() == EventStatus::Running);
  auto download = s.to_host_async(t.value(), LayoutKind::TestTile4x4);
  CHECK(download);
  if (!fail) {
    t = Status{ErrorCode::InvalidArgument, "drop tensor before completion"};
  }
  c->unblock();
  auto out = download.value().wait();
  if (fail) {
    CHECK(!first.value().wait());
    CHECK(!again.value().wait());
    CHECK(!out && out.error().code == ErrorCode::DependencyFailed);
    CHECK(c->downloads == 0);
    auto dense = s.to_host(t.value());
    CHECK(dense); // canonical value survives failed packing
    auto retry = s.materialize_layout(t.value(), layout);
    CHECK(retry);
    CHECK(retry.value().wait());
    CHECK(c->transforms == 2);
    CHECK(s.memory_stats().allocations == 2);
    auto read = s.to_host(t.value(), LayoutKind::TestTile4x4);
    CHECK(read);
    CHECK(read.value().bytes == dense.value().bytes);
    CHECK(!s.synchronize()); // first failure remains observable
  } else {
    CHECK(out);
    CHECK(c->transforms == 2);
    CHECK(c->downloads == 1);
    CHECK(s.synchronize());
    CHECK(s.memory_stats().live_bytes == 0);
  }
}

void validation_and_capacity() {
  auto created = Session::create(make_functional_backend(64));
  CHECK(created);
  auto s = std::move(created.value());
  std::vector<uint16_t> bits(15, 1);
  auto t = s.from_host({{{3, 5}}, std::as_bytes(std::span(bits))});
  CHECK(t);
  CHECK(t.value().wait());
  auto layout = s.query_layout(t.value().desc(), LayoutKind::TestTile4x4).value();
  auto oom = s.materialize_layout(t.value(), layout);
  CHECK(!oom && oom.error().code == ErrorCode::OutOfMemory);
  CHECK(s.memory_stats().live_bytes == 30);
  CHECK(t.value().representations().size() == 1);
  auto forged = layout;
  forged.physical_bytes = 2;
  CHECK(!s.materialize_layout(t.value(), forged));
  forged = layout;
  forged.format = "other.backend";
  CHECK(!s.materialize_layout(t.value(), forged));
  forged = layout;
  forged.logical.shape = {5, 3};
  CHECK(!s.materialize_layout(t.value(), forged));
  CHECK(!s.to_host_async(t.value(), LayoutKind::TestTile4x4));
  CHECK(!s.query_layout({{3}}, LayoutKind::TestTile4x4));
  CHECK(!s.query_layout({{3, 3}, DType::F32}, LayoutKind::TestTile4x4));
  CHECK(!s.query_layout({{1, std::numeric_limits<size_t>::max() / 2}}, LayoutKind::TestTile4x4));
  auto other = Session::create(make_functional_backend(1024));
  CHECK(other);
  auto mismatch = other.value().materialize_layout(t.value(), layout);
  CHECK(!mismatch && mismatch.error().code == ErrorCode::SessionMismatch);
  CHECK(s.close());
  CHECK(!s.materialize_layout(t.value(), layout));
}

int main() {
  try {
    roundtrips();
    physical_mapping();
    pending_dedup_and_lifetime(false);
    pending_dedup_and_lifetime(true);
    validation_and_capacity();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  std::cout << "Layout contracts passed\n";
}
