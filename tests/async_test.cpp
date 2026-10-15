// Verify FIFO execution, staging limits, dependency errors, and asynchronous buffer lifetimes.
#include "ndp/session.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
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
  std::atomic<int> uploads{0}, downloads{0}, releases{0}, shutdowns{0};

  void unblock() {
    {
      std::lock_guard lock(mutex);
      open = true;
    }
    changed.notify_all();
  }

  void await_entry() {
    std::unique_lock lock(mutex);
    CHECK(changed.wait_for(lock, 5s, [&] { return entered; }));
  }
};

struct UnblockOnExit {
  std::shared_ptr<Control> control;

  ~UnblockOnExit() {
    control->unblock();
  }
};

class DelayedBackend final : public IBackend {
public:
  DelayedBackend(std::shared_ptr<Control> c, bool fail = false, bool fail_shutdown = false)
      : control(std::move(c)), fail_(fail), fail_shutdown_(fail_shutdown) {}

  DeviceCaps capabilities() const override {
    return impl->capabilities();
  }

  Result<AllocationHandle> allocate(size_t bytes) override {
    return impl->allocate(bytes);
  }

  void release(AllocationHandle h) noexcept override {
    ++control->releases;
    impl->release(h);
  }

  Result<void> upload(AllocationHandle h, std::span<const std::byte> src) override {
    ++control->uploads;
    {
      std::unique_lock lock(control->mutex);
      control->entered = true;
      control->changed.notify_all();
      control->changed.wait(lock, [&] { return control->open; });
    }
    if (fail_) {
      return Status{ErrorCode::BackendFailure, "Injected upload failure"};
    }
    return impl->upload(h, src);
  }

  Result<void> download(AllocationHandle h, std::span<std::byte> dst) override {
    ++control->downloads;
    return impl->download(h, dst);
  }

  Result<void> shutdown() override {
    ++control->shutdowns;
    (void)impl->shutdown();
    if (fail_shutdown_) {
      return Status{ErrorCode::BackendFailure, "Injected shutdown failure"};
    }
    return {};
  }

private:
  std::shared_ptr<Control> control;
  bool fail_, fail_shutdown_;
  std::unique_ptr<IBackend> impl = make_functional_backend(1024);
};

void fifo_and_lifetime() {
  auto control = std::make_shared<Control>();
  auto created = Session::create(std::make_unique<DelayedBackend>(control));
  CHECK(created);
  auto s = std::move(created.value());
  UnblockOnExit unblock{control};
  std::array<uint16_t, 8> bits{0x3c00, 0xc000, 0x0001, 0x8000, 0x7c00, 0xfc00, 0x7e01, 0x7bff};
  auto expected = bits;
  auto a = s.from_host({{{8}}, std::as_bytes(std::span(bits))});
  CHECK(a);
  control->await_entry();
  CHECK(a.value().ready_event().status() == EventStatus::Running);
  bits.fill(0);
  auto b = s.from_host({{{8}}, std::as_bytes(std::span(bits))});
  CHECK(b);
  CHECK(b.value().ready_event().status() == EventStatus::Pending);
  auto read_a = s.to_host_async(a.value());
  CHECK(read_a);
  auto read_b = s.to_host_async(b.value());
  CHECK(read_b);
  CHECK(read_a.value().ready_event().status() == EventStatus::Pending);
  a = Status{ErrorCode::InvalidArgument, "drop"};
  b = Status{ErrorCode::InvalidArgument, "drop"};
  CHECK(control->releases == 0);
  CHECK(s.memory_stats().live_bytes == 32);
  control->unblock();
  auto result_a = read_a.value().wait();
  CHECK(result_a);
  auto result_b = read_b.value().wait();
  CHECK(result_b);
  CHECK(std::memcmp(result_a.value().bytes.data(), expected.data(), sizeof(expected)) == 0);
  CHECK(std::memcmp(result_b.value().bytes.data(), bits.data(), sizeof(bits)) == 0);
  CHECK(s.synchronize());
  CHECK(control->releases == 2);
  CHECK(control->uploads == 2);
  CHECK(control->downloads == 2);
  CHECK(s.memory_stats().live_bytes == 0);
  CHECK(s.memory_stats().staging_bytes == 0);
  CHECK(s.close());
  CHECK(control->shutdowns == 1);
}

void staging_limit() {
  auto control = std::make_shared<Control>();
  auto created = Session::create(std::make_unique<DelayedBackend>(control), {2});
  CHECK(created);
  auto s = std::move(created.value());
  UnblockOnExit unblock{control};
  std::array<uint16_t, 1> bits{0x3c00};
  auto tensor = s.from_host({{{1}}, std::as_bytes(std::span(bits))});
  CHECK(tensor);
  control->await_entry();
  auto rejected = s.from_host({{{1}}, std::as_bytes(std::span(bits))});
  CHECK(!rejected && rejected.error().code == ErrorCode::OutOfMemory);
  CHECK(!s.to_host_async(tensor.value()));
  CHECK(s.memory_stats().staging_bytes == 2);
  CHECK(s.memory_stats().allocations == 1);
  control->unblock();
  CHECK(tensor.value().wait());
  CHECK(s.memory_stats().staging_bytes == 0);
  CHECK(s.to_host(tensor.value()));
}

void dependency_failure() {
  auto control = std::make_shared<Control>();
  auto created = Session::create(std::make_unique<DelayedBackend>(control, true));
  CHECK(created);
  auto s = std::move(created.value());
  UnblockOnExit unblock{control};
  std::array<uint16_t, 1> bits{0};
  auto tensor = s.from_host({{{1}}, std::as_bytes(std::span(bits))});
  CHECK(tensor);
  control->await_entry();
  auto download = s.to_host_async(tensor.value());
  CHECK(download);
  control->unblock();
  auto failed = tensor.value().wait();
  CHECK(!failed && failed.error().code == ErrorCode::BackendFailure);
  auto dependent = download.value().wait();
  CHECK(!dependent && dependent.error().code == ErrorCode::DependencyFailed);
  CHECK(control->downloads == 0);
  CHECK(download.value().ready_event().status() == EventStatus::Failed);
  CHECK(!s.synchronize());
  CHECK(!s.close());
  CHECK(!s.close());
  CHECK(control->shutdowns == 1);
  CHECK(s.memory_stats().staging_bytes == 0);
}

void drain_and_shutdown_error() {
  auto control = std::make_shared<Control>();
  control->unblock();
  auto created = Session::create(std::make_unique<DelayedBackend>(control, false, true));
  CHECK(created);
  auto s = std::move(created.value());
  std::array<uint16_t, 1> bits{0x4000};
  auto tensor = s.from_host({{{1}}, std::as_bytes(std::span(bits))});
  CHECK(tensor);
  auto transfer = s.to_host_async(tensor.value());
  CHECK(transfer);
  CHECK(!s.close());
  CHECK(!s.close());
  CHECK(tensor.value().ready_event().status() == EventStatus::Succeeded);
  CHECK(transfer.value().wait());
  CHECK(control->shutdowns == 1);
  CHECK(!s.to_host_async(tensor.value()));
}

int main() {
  try {
    CHECK(Event{}.wait());
    fifo_and_lifetime();
    staging_limit();
    dependency_failure();
    drain_and_shutdown_error();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  std::cout << "Async FP16 contracts passed\n";
}
