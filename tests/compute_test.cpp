// Verify FP16 conversion, compute accuracy, layout preparation, and workspace reuse.
#include "ndp/common/fp16.hpp"
#include "test_support.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
using namespace ndp;
using namespace std::chrono_literals;

using reference::decode;
using reference::encode;
using reference::upload;

std::vector<uint16_t> download(Session &s, const Tensor &t) {
  auto h = s.to_host(t);
  CHECK(h);
  std::vector<uint16_t> result(h.value().bytes.size() / 2);
  std::memcpy(result.data(), h.value().bytes.data(), h.value().bytes.size());
  return result;
}

void conversion() {
  for (uint32_t i = 0; i < 65536; ++i) {
    auto h = static_cast<uint16_t>(i);
    double oracle = decode(h);
    float actual = fp16_to_float(h);
    if (std::isnan(oracle)) {
      CHECK(std::isnan(actual));
      continue;
    }
    CHECK(double(actual) == oracle);
    CHECK(std::signbit(actual) == std::signbit(oracle));
    CHECK(float_to_fp16(actual) == h);
  }
  for (float x : {0.0f, -0.0f, 65504.f, 65520.f, 1.00048828125f, 1.00146484375f,
                  std::ldexp(1.f, -25), std::ldexp(3.f, -25), -std::ldexp(3.f, -25)}) {
    CHECK(float_to_fp16(x) == encode(x));
  }
}

void numerics_and_chaining() {
  auto made = Session::create(make_functional_backend(1 << 20, 8));
  CHECK(made);
  auto s = std::move(made.value());
  // Scratchpad permits only K tiles of 2: exercise tile tails and reuse.
  for (size_t k : {1, 3, 4, 5, 65}) {
    size_t m = 3, n = 5;
    std::vector<float> a(m * k), b(k * n), w(n * k), x(k);
    for (size_t i = 0; i < a.size(); ++i) {
      a[i] = float(int(i % 11) - 5) / 8;
    }
    for (size_t i = 0; i < b.size(); ++i) {
      b[i] = float(int(i % 7) - 3) / 4;
    }
    for (size_t r = 0; r < n; ++r) {
      for (size_t c = 0; c < k; ++c) {
        w[r * k + c] = b[c * n + r];
      }
    }
    for (size_t i = 0; i < k; ++i) {
      x[i] = a[i];
    }
    auto ta = upload(s, {m, k}, a), tb = upload(s, {k, n}, b), tw = upload(s, {n, k}, w),
         tx = upload(s, {k}, x);
    auto mm = s.matmul(ta, tb);
    CHECK(mm);
    auto mv = s.matvec(tw, tx);
    CHECK(mv);
    auto pnm = s.matvec(tw, tx, {Engine::PNM});
    CHECK(pnm);
    auto matrix = download(s, mm.value()), vector = download(s, mv.value());
    CHECK(download(s, pnm.value()) == vector);
    for (size_t r = 0; r < m; ++r) {
      for (size_t c = 0; c < n; ++c) {
        double expected = 0;
        for (size_t j = 0; j < k; ++j) {
          expected += double(a[r * k + j]) * b[j * n + c];
        }
        CHECK(matrix[r * n + c] == encode(expected));
        if (r == 0) {
          CHECK(vector[c] == encode(expected));
        }
      }
    }
    auto prepared =
        s.prepare_layouts(tw, {s.query_layout(tw.desc(), LayoutKind::TestTile4x4).value()});
    CHECK(prepared);
    auto linear = s.linear(ta, prepared.value(), {Engine::PNM, LayoutPolicy::RequirePrepared});
    CHECK(linear);
    CHECK(download(s, linear.value()) == matrix);
    auto one = upload(s, {1, k}, x);
    auto pim_linear = s.linear(one, prepared.value(), {Engine::PIM, LayoutPolicy::RequirePrepared});
    CHECK(pim_linear);
    CHECK(download(s, pim_linear.value()) == vector);
    auto bias = upload(s, {n}, std::vector<float>(n, 0.5f));
    auto biased = s.linear(one, prepared.value(), {Engine::PNM}, &bias);
    CHECK(biased);
    auto actual = download(s, biased.value());
    for (size_t c = 0; c < n; ++c) {
      CHECK(actual[c] == encode(decode(vector[c]) + 0.5));
    }
  }
  // Distinguish FP32 accumulation from both FP16 and FP64 accumulation.
  auto cancellation = upload(s, {1, 3}, {2048, 1, -2048}), ones = upload(s, {3}, {1, 1, 1});
  auto sum = s.matvec(cancellation, ones);
  CHECK(sum);
  CHECK(download(s, sum.value()) == std::vector<uint16_t>{encode(1)});
  auto large = upload(s, {1, 3}, {4096, 1, -4096}), factor = upload(s, {3}, {4096, 1, 4096});
  auto rounded = s.matvec(large, factor, {Engine::PNM});
  CHECK(rounded);
  CHECK(download(s, rounded.value()) == std::vector<uint16_t>{encode(0)});
  // No host read between PNM -> PIM -> PNM; all tensors may still be pending.
  auto a = upload(s, {2, 3}, {1, 2, 3, 4, 5, 6});
  auto b = upload(s, {3, 2}, {1, 0, 0, 1, 1, 1});
  auto mm = s.matmul(a, b);
  CHECK(mm); // [[4,5],[10,11]]
  auto x = upload(s, {2}, {1, 2});
  auto mv = s.matvec(mm.value(), x);
  CHECK(mv); // [14,32]
  auto result = s.matvec(mm.value(), mv.value(), {Engine::PNM});
  CHECK(result);
  CHECK(download(s, result.value()) == std::vector<uint16_t>({encode(216), encode(492)}));
  CHECK(s.memory_stats().workspace_bytes == 8);
  CHECK(s.close());
  CHECK(s.memory_stats().workspace_bytes == 0);
}

void validation() {
  auto made = Session::create(make_functional_backend(4096));
  CHECK(made);
  auto s = std::move(made.value());
  auto w = upload(s, {2, 3}, {1, 2, 3, 4, 5, 6}), x = upload(s, {3}, {1, 2, 3});
  auto missing = s.matvec(w, x, {Engine::PIM, LayoutPolicy::RequirePrepared});
  CHECK(!missing && missing.error().code == ErrorCode::LayoutNotPrepared);
  auto packed = s.query_layout(w.desc(), LayoutKind::TestTile4x4);
  CHECK(packed);
  auto malformed = packed.value();
  ++malformed.physical_bytes;
  CHECK(!s.prepare_layouts(w, {packed.value(), malformed}));
  CHECK(w.representations().size() == 1);
  auto prepared = s.prepare_layouts(w, {packed.value()});
  CHECK(prepared);
  CHECK(w.representations().size() == 1);
  CHECK(!s.matmul(w, w));
  auto matrix = upload(s, {3, 2}, {1, 2, 3, 4, 5, 6});
  auto strict = s.matmul(w, matrix, {Engine::PIM});
  CHECK(!strict && strict.error().code == ErrorCode::UnsupportedOperation);
  auto batch = upload(s, {2, 3}, {1, 2, 3, 4, 5, 6});
  CHECK(!s.linear(batch, prepared.value(), {Engine::PIM}));
  auto one = upload(s, {1, 3}, {1, 2, 3}), bias = upload(s, {2}, {1, 2});
  CHECK(!s.linear(one, prepared.value(), {Engine::PIM}, &bias));
  auto other = Session::create(make_functional_backend(4096));
  CHECK(other);
  CHECK(!other.value().matvec(w, x));
  auto small = Session::create(make_functional_backend(4096, 2));
  CHECK(small);
  auto sw = upload(small.value(), {2, 3}, {1, 2, 3, 4, 5, 6}),
       sx = upload(small.value(), {3}, {1, 2, 3});
  auto oom = small.value().matvec(sw, sx, {Engine::PNM});
  CHECK(!oom && oom.error().code == ErrorCode::OutOfMemory);
  CHECK(small.value().matvec(sw, sx, {Engine::PIM}));
  auto minimum = Session::create(make_functional_backend(4096, 4));
  CHECK(minimum);
  auto mw = upload(minimum.value(), {2, 3}, {1, 2, 3, 4, 5, 6}),
       mx = upload(minimum.value(), {3}, {1, 2, 3});
  auto tiny = minimum.value().matvec(mw, mx, {Engine::PNM});
  CHECK(tiny);
  CHECK(download(minimum.value(), tiny.value()) == std::vector<uint16_t>({encode(14), encode(32)}));
  CHECK(minimum.value().memory_stats().workspace_bytes == 4);
}

struct Control {
  std::mutex mutex;
  std::condition_variable changed;
  bool entered = false, open = false, fail = false, fail_transform = false;
  std::atomic<int> kernels{0}, transforms{0}, workspace_reservations{0};

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
    return impl_->capabilities();
  }

  Result<AllocationHandle> allocate(size_t n) override {
    return impl_->allocate(n);
  }

  void release(AllocationHandle h) noexcept override {
    impl_->release(h);
  }

  Result<void> upload(AllocationHandle h, std::span<const std::byte> b) override {
    return impl_->upload(h, b);
  }

  Result<void> download(AllocationHandle h, std::span<std::byte> b) override {
    return impl_->download(h, b);
  }

  Result<void> shutdown() override {
    return impl_->shutdown();
  }

  Result<LayoutDesc> query_layout(const TensorDesc &d, LayoutKind k) const override {
    return impl_->query_layout(d, k);
  }

  Result<void> transform(AllocationHandle a, const LayoutDesc &ad, AllocationHandle b,
                         const LayoutDesc &bd) override {
    ++c_->transforms;
    if (c_->fail_transform) {
      return Status{ErrorCode::BackendFailure, "Injected packing failure"};
    }
    return impl_->transform(a, ad, b, bd);
  }

  Result<KernelPlan> query_operation(const OperationSpec &o) const override {
    return impl_->query_operation(o);
  }

  Result<void> reserve_workspace(size_t n) override {
    ++c_->workspace_reservations;
    return impl_->reserve_workspace(n);
  }

  Result<void> execute(const KernelPlan &p, std::span<const AllocationHandle> in,
                       AllocationHandle out) override {
    ++c_->kernels;
    {
      std::unique_lock lock(c_->mutex);
      c_->entered = true;
      c_->changed.notify_all();
      c_->changed.wait(lock, [&] { return c_->open; });
    }
    if (c_->fail) {
      return Status{ErrorCode::BackendFailure, "Injected kernel failure"};
    }
    return impl_->execute(p, in, out);
  }

private:
  std::shared_ptr<Control> c_;
  std::unique_ptr<IBackend> impl_ = make_functional_backend(4096, 16);
};

void async_and_reuse(bool fail) {
  auto c = std::make_shared<Control>();
  c->fail = fail;
  auto made = Session::create(std::make_unique<Instrumented>(c));
  CHECK(made);
  auto s = std::move(made.value());
  Guard guard{c};
  auto w = upload(s, {2, 2}, {1, 2, 3, 4}), x = upload(s, {1, 2}, {1, 2});
  auto prepared = s.prepare_layouts(w, {s.query_layout(w.desc(), LayoutKind::TestTile4x4).value()});
  CHECK(prepared);
  CHECK(c->transforms == 1);
  std::vector<Tensor> outputs;
  auto first = s.linear(x, prepared.value(), {Engine::PNM, LayoutPolicy::RequirePrepared});
  CHECK(first);
  c->wait();
  auto dependent =
      s.linear(first.value(), prepared.value(), {Engine::PNM, LayoutPolicy::RequirePrepared});
  CHECK(dependent);
  for (int i = 0; i < 4; ++i) {
    auto y = s.linear(x, prepared.value(), {Engine::PIM, LayoutPolicy::RequirePrepared});
    CHECK(y);
    outputs.push_back(y.value());
  }
  CHECK(c->workspace_reservations == 1);
  CHECK(s.memory_stats().workspace_bytes == 16);
  CHECK(c->transforms == 1);
  auto read = s.to_host_async(dependent.value());
  CHECK(read);
  first = Status{ErrorCode::InvalidArgument, "drop"};
  dependent = Status{ErrorCode::InvalidArgument, "drop"};
  outputs.clear();
  c->unblock();
  auto result = read.value().wait();
  if (fail) {
    CHECK(!result && result.error().code == ErrorCode::DependencyFailed);
    CHECK(!s.synchronize());
    CHECK(c->kernels == 5);
  } else {
    CHECK(result);
    CHECK(s.synchronize());
    CHECK(c->kernels == 6);
  }
  CHECK(c->workspace_reservations == 1);
  CHECK(c->transforms == 1);
}

void allocation_and_prepare_failure() {
  auto made = Session::create(make_functional_backend(20));
  CHECK(made);
  auto s = std::move(made.value());
  auto a = upload(s, {2, 2}, {1, 2, 3, 4}), b = upload(s, {2, 2}, {1, 2, 3, 4});
  auto oom = s.matmul(a, b);
  CHECK(!oom && oom.error().code == ErrorCode::OutOfMemory);
  CHECK(s.memory_stats().live_bytes == 16);
  auto c = std::make_shared<Control>();
  c->fail_transform = true;
  auto failing = Session::create(std::make_unique<Instrumented>(c));
  CHECK(failing);
  auto weight = upload(failing.value(), {2, 2}, {1, 2, 3, 4});
  auto prepared = failing.value().prepare_layouts(
      weight, {failing.value().query_layout(weight.desc(), LayoutKind::TestTile4x4).value()});
  CHECK(!prepared);
  CHECK(weight.representations().size() == 1);
  CHECK(failing.value().memory_stats().live_bytes == 8);
  CHECK(failing.value().memory_stats().allocations == 1);
  CHECK(failing.value().to_host(weight));
}

int main() {
  try {
    conversion();
    numerics_and_chaining();
    validation();
    allocation_and_prepare_failure();
    async_and_reuse(false);
    async_and_reuse(true);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  std::cout << "Compute contracts passed\n";
}
