// Verify KV cache ordering, capacity, partial failures, reset recovery, and causal GQA.
#include "ndp/l3/llm.hpp"
#include "test_support.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
using namespace ndp;
using namespace std::chrono_literals;

struct Control {
  std::mutex mutex;
  std::condition_variable changed;
  bool open = false, entered = false, fail_second = false;
  std::atomic<int> copies{0}, kernels{0};

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

class Delayed : public reference::ForwardBackend {
public:
  explicit Delayed(std::shared_ptr<Control> c) : c_(std::move(c)) {}

  Result<void> copy_region(AllocationHandle a, size_t ao, AllocationHandle b, size_t bo,
                           size_t n) override {
    int index = ++c_->copies;
    {
      std::unique_lock lock(c_->mutex);
      c_->entered = true;
      c_->changed.notify_all();
      c_->changed.wait(lock, [&] { return c_->open; });
    }
    if (c_->fail_second && index == 2) {
      return Status{ErrorCode::BackendFailure, "Second KV copy fails"};
    }
    return inner->copy_region(a, ao, b, bo, n);
  }

  Result<void> execute(const KernelPlan &p, std::span<const AllocationHandle> in,
                       AllocationHandle out) override {
    if (p.operation.kind == OpKind::Matmul) {
      ++c_->kernels;
    }
    return inner->execute(p, in, out);
  }

private:
  std::shared_ptr<Control> c_;
};

void scheduling_and_failure(bool fail) {
  auto c = std::make_shared<Control>();
  c->fail_second = fail;
  auto made = Session::create(std::make_unique<Delayed>(c));
  CHECK(made);
  auto s = std::move(made.value());
  Guard guard{c};
  auto cache = ndp::KVCache::create(s, {3, 1, 2});
  CHECK(cache);
  auto k = reference::upload(s, {1, 1, 2}, {1, 0}), v = reference::upload(s, {1, 1, 2}, {2, 4}),
       q = reference::upload(s, {1, 2, 2}, {1, 0, 0, 1});
  auto appended = cache.value().append(s, k, v);
  CHECK(appended);
  c->wait();
  CHECK(cache.value().scheduled_length() == 1);
  CHECK(cache.value().committed_length() == 0);
  auto attention = cache.value().attention(s, q, 0);
  CHECK(attention);
  auto next = cache.value().append(s, k, v);
  CHECK(next);
  CHECK(cache.value().scheduled_length() == 2);
  c->unblock();
  auto result = attention.value().wait();
  if (fail) {
    CHECK(!result && result.error().code == ErrorCode::DependencyFailed);
    CHECK(!next.value().wait());
    CHECK(cache.value().poisoned());
    CHECK(cache.value().committed_length() == 0);
    CHECK(c->kernels == 0);
    CHECK(!cache.value().attention(s, q, 0));
    auto reset = cache.value().reset(s);
    CHECK(reset);
    CHECK(reset.value().wait());
    CHECK(!cache.value().poisoned());
    CHECK(cache.value().scheduled_length() == 0);
    auto retry = cache.value().append(s, k, v);
    CHECK(retry);
    CHECK(retry.value().wait());
    CHECK(cache.value().committed_length() == 1);
  } else {
    CHECK(result);
    reference::close(reference::download(s, attention.value()), {2, 4, 2, 4}, 0, 0);
    CHECK(next.value().wait());
    CHECK(cache.value().committed_length() == 2);
    auto last = cache.value().append(s, k, v);
    CHECK(last);
    CHECK(!cache.value().append(s, k, v));
    CHECK(cache.value().scheduled_length() == 3);
    CHECK(last.value().wait());
    CHECK(cache.value().committed_length() == 3);
  }
  auto reset = cache.value().reset(s);
  CHECK(reset);
  CHECK(reset.value().wait());
  CHECK(cache.value().committed_length() == 0);
  CHECK(!cache.value().attention(s, q, 0));
}

void attention_reference() {
  auto made = Session::create(make_functional_backend(65536));
  CHECK(made);
  auto s = std::move(made.value());
  auto cache = ndp::KVCache::create(s, {4, 1, 2});
  CHECK(cache);
  auto k = reference::upload(s, {2, 1, 2}, {1, 0, 0, 1}),
       v = reference::upload(s, {2, 1, 2}, {2, 4, 6, 8});
  auto q = reference::upload(s, {2, 2, 2}, {1, 0, 0, 1, 1, 0, 0, 1});
  auto event = cache.value().append(s, k, v);
  CHECK(event);
  auto result = cache.value().attention(s, q, 0);
  CHECK(result);
  double p = std::exp(1 / std::sqrt(2.0)) / (1 + std::exp(1 / std::sqrt(2.0)));
  reference::close(reference::download(s, result.value()),
                   {2, 4, 2, 4, float(p * 2 + (1 - p) * 6), float(p * 4 + (1 - p) * 8),
                    float((1 - p) * 2 + p * 6), float((1 - p) * 4 + p * 8)});
  CHECK(!cache.value().attention(s, q, 1));
  auto bad = reference::upload(s, {1, 3, 3}, std::vector<float>(9, 1));
  CHECK(!cache.value().attention(s, bad, 0));
  auto other = Session::create(make_functional_backend(65536));
  CHECK(other);
  CHECK(!cache.value().reset(other.value()));
  CHECK(!cache.value().append(other.value(), k, v));
  CHECK(!ndp::KVCache::create(s, {0, 1, 2}));
  CHECK(!ndp::KVCache::create(s, {4, 1, 2, DType::F32}));
}

void lifetime() {
  auto made = Session::create(make_functional_backend(65536));
  CHECK(made);
  auto s = std::move(made.value());
  {
    auto cache = ndp::KVCache::create(s, {8, 1, 2});
    CHECK(cache);
    auto t = reference::upload(s, {1, 1, 2}, {1, 2});
    CHECK(cache.value().append(s, t, t));
    auto q = reference::upload(s, {1, 1, 2}, {0, 0});
    CHECK(cache.value().attention(s, q, 0));
  }
  CHECK(s.synchronize());
  CHECK(s.memory_stats().live_bytes == 0);
}

class AllocationFailure : public reference::ForwardBackend {
public:
  Result<AllocationHandle> allocate(size_t bytes) override {
    if (++calls_ == 2) {
      return Status{ErrorCode::OutOfMemory, "Injected V allocation failure"};
    }
    return inner->allocate(bytes);
  }

private:
  int calls_ = 0;
};

void allocation_failure() {
  auto made = Session::create(std::make_unique<AllocationFailure>());
  CHECK(made);
  auto s = std::move(made.value());
  {
    auto cache = ndp::KVCache::create(s, {8, 1, 2});
    CHECK(cache);
    CHECK(!s.synchronize());
    CHECK(cache.value().poisoned());
    auto reset = cache.value().reset(s);
    CHECK(reset);
    CHECK(!reset.value().wait());
    CHECK(cache.value().poisoned());
  }
  CHECK(!s.synchronize());
  CHECK(s.memory_stats().live_bytes == 0);
}

int main() {
  try {
    scheduling_and_failure(false);
    scheduling_and_failure(true);
    attention_reference();
    lifetime();
    allocation_failure();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  std::cout << "KV contracts passed\n";
}
