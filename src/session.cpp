#include "ndp/session.hpp"
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>

namespace ndp {
Result<size_t> byte_size(const TensorDesc& desc) {
  if (desc.dtype != DType::F16)
    return Status{ErrorCode::UnsupportedDType, "This milestone supports F16 only"};
  if (desc.shape.empty()) return Status{ErrorCode::InvalidArgument, "Rank must be positive"};
  size_t bytes = sizeof(uint16_t);
  for (auto dim : desc.shape) {
    if (dim == 0 || dim > std::numeric_limits<size_t>::max() / bytes)
      return Status{ErrorCode::InvalidArgument, "Zero dimension or byte-size overflow"};
    bytes *= static_cast<size_t>(dim);
  }
  return bytes;
}
namespace detail {
struct EventState {
  mutable std::mutex mutex;
  std::condition_variable changed;
  EventStatus status = EventStatus::Pending;
  std::optional<Status> error;
};
// Resources deliberately do not reference State or its executor.
struct Resources {
  explicit Resources(std::unique_ptr<IBackend> b, SessionOptions opts)
      : backend(std::move(b)), caps(backend->capabilities()), options(opts) {}
  std::unique_ptr<IBackend> backend;
  DeviceCaps caps;
  SessionOptions options;
  std::mutex backend_mutex, stats_mutex;
  MemoryStats stats;
};
struct BufferLease {
  explicit BufferLease(std::shared_ptr<Resources> r) : resources(std::move(r)) {}
  std::shared_ptr<Resources> resources;
  std::optional<AllocationHandle> allocation;
  size_t bytes = 0;
  bool reserved = false;
  ~BufferLease() {
    if (allocation) {
      std::lock_guard lock(resources->backend_mutex);
      resources->backend->release(*allocation);
    }
    if (reserved) {
      std::lock_guard lock(resources->stats_mutex);
      resources->stats.live_bytes -= bytes;
      --resources->stats.allocations;
    }
  }
};
struct StagingLease {
  explicit StagingLease(std::shared_ptr<Resources> r) : resources(std::move(r)) {}
  std::shared_ptr<Resources> resources;
  size_t bytes = 0;
  ~StagingLease() {
    std::lock_guard lock(resources->stats_mutex);
    resources->stats.staging_bytes -= bytes;
  }
};
struct TensorRecord {
  TensorDesc desc;
  std::shared_ptr<BufferLease> buffer;
  Event ready;
};
struct Command {
  Event event;
  std::vector<Event> dependencies;
  std::function<Result<void>()> execute;
};
struct State {
  explicit State(std::unique_ptr<IBackend> backend, SessionOptions options)
      : resources(std::make_shared<Resources>(std::move(backend), options)),
        worker([this] { run(); }) {}
  std::shared_ptr<Resources> resources;
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<Command> queue;
  bool closed = false, stopping = false;
  std::optional<Status> first_error, close_error;
  std::thread worker;
  ~State() { (void)close(); }

  Event submit(std::function<Result<void>()> execute, std::vector<Event> deps = {}) {
    Command command{Event{}, std::move(deps), std::move(execute)};
    auto event = command.event;
    event.state_->status = EventStatus::Pending;
    { std::lock_guard lock(mutex); queue.push_back(std::move(command)); }
    changed.notify_one();
    return event;
  }
  void run() {
    for (;;) {
      std::optional<Command> command;
      {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return stopping || !queue.empty(); });
        if (queue.empty()) return;
        command.emplace(std::move(queue.front())); queue.pop_front();
      }
      auto event = command->event.state_;
      { std::lock_guard lock(event->mutex); event->status = EventStatus::Running; }
      Result<void> result;
      try {
        for (const auto& dependency : command->dependencies) {
          auto dependency_result = dependency.wait();
          if (!dependency_result) {
            result = Status{ErrorCode::DependencyFailed, dependency_result.error().message};
            break;
          }
        }
        if (result) result = command->execute();
      } catch (const std::bad_alloc&) {
        result = Status{ErrorCode::OutOfMemory, "Worker allocation failed"};
      } catch (const std::exception& e) {
        result = Status{ErrorCode::BackendFailure, e.what()};
      } catch (...) {
        result = Status{ErrorCode::BackendFailure, "Unknown backend exception"};
      }
      // Drop command-owned buffers/staging before publishing completion.
      command.reset();
      if (!result) {
        std::lock_guard lock(mutex);
        if (!first_error) first_error = result.error();
      }
      {
        std::lock_guard lock(event->mutex);
        if (!result) event->error = result.error();
        event->status = result ? EventStatus::Succeeded : EventStatus::Failed;
      }
      event->changed.notify_all();
    }
  }
  Result<void> close() {
    if (!closed) {
      closed = true;
      { std::lock_guard lock(mutex); stopping = true; }
      changed.notify_one();
      worker.join();
      try {
        std::lock_guard lock(resources->backend_mutex);
        auto result = resources->backend->shutdown();
        if (!result) close_error = result.error();
      } catch (const std::exception& e) {
        close_error = Status{ErrorCode::BackendFailure, e.what()};
      } catch (...) {
        close_error = Status{ErrorCode::BackendFailure, "Shutdown exception"};
      }
      if (!close_error) close_error = first_error;
    }
    if (close_error) return *close_error;
    return {};
  }
};
Result<std::shared_ptr<StagingLease>> reserve_staging(const std::shared_ptr<Resources>& resources, size_t bytes) {
  auto lease = std::make_shared<StagingLease>(resources);
  std::lock_guard lock(resources->stats_mutex);
  if (bytes > resources->options.max_staging_bytes - resources->stats.staging_bytes)
    return Status{ErrorCode::OutOfMemory, "Pending host staging budget exceeded"};
  resources->stats.staging_bytes += bytes;
  resources->stats.peak_staging_bytes = std::max(resources->stats.peak_staging_bytes, resources->stats.staging_bytes);
  lease->bytes = bytes;
  return lease;
}
}
Event::Event() : state_(std::make_shared<detail::EventState>()) {
  state_->status = EventStatus::Succeeded; // A default event is an already-complete fence.
}
EventStatus Event::status() const { std::lock_guard lock(state_->mutex); return state_->status; }
Result<void> Event::wait() const {
  std::unique_lock lock(state_->mutex);
  state_->changed.wait(lock, [&] { return state_->status == EventStatus::Succeeded || state_->status == EventStatus::Failed; });
  if (state_->error) return *state_->error;
  return {};
}
const TensorDesc& Tensor::desc() const { return record_->desc; }
Event Tensor::ready_event() const { return record_->ready; }
Result<void> Tensor::wait() const { return record_->ready.wait(); }
Result<HostTensor> HostTransfer::wait() const {
  auto result = event_.wait();
  if (!result) return result.error();
  try { return *output_; }
  catch (const std::bad_alloc&) { return Status{ErrorCode::OutOfMemory, "Host result allocation failed"}; }
}
Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&& other) noexcept {
  if (this != &other) { (void)close(); state_ = std::move(other.state_); }
  return *this;
}
Session::~Session() { (void)close(); }
Result<Session> Session::create(std::unique_ptr<IBackend> backend, SessionOptions options) {
  if (!backend) return Status{ErrorCode::InvalidArgument, "Backend is required"};
  try { return Session(std::make_shared<detail::State>(std::move(backend), options)); }
  catch (const std::bad_alloc&) { return Status{ErrorCode::OutOfMemory, "Session allocation failed"}; }
  catch (const std::exception& e) { return Status{ErrorCode::BackendFailure, e.what()}; }
}
Result<Tensor> Session::from_host(HostTensorView host) {
  if (!state_ || state_->closed) return Status{ErrorCode::SessionClosed, "Session closed"};
  auto size = byte_size(host.desc);
  if (!size) return size.error();
  if (host.bytes.size() != size.value()) return Status{ErrorCode::InvalidArgument, "Host byte count does not match shape"};
  try {
    auto resources = state_->resources;
    auto buffer = std::make_shared<detail::BufferLease>(resources);
    buffer->bytes = size.value();
    {
      std::lock_guard lock(resources->stats_mutex);
      if (buffer->bytes > resources->caps.memory_capacity_bytes - resources->stats.live_bytes)
        return Status{ErrorCode::OutOfMemory, "Device capacity exceeded"};
      resources->stats.live_bytes += buffer->bytes;
      ++resources->stats.allocations;
      resources->stats.peak_bytes = std::max(resources->stats.peak_bytes, resources->stats.live_bytes);
      buffer->reserved = true;
    }
    auto staging = detail::reserve_staging(resources, buffer->bytes);
    if (!staging) return staging.error();
    auto input = std::make_shared<std::vector<std::byte>>(host.bytes.begin(), host.bytes.end());
    auto record = std::make_shared<detail::TensorRecord>();
    record->desc = std::move(host.desc);
    record->buffer = buffer;
    record->ready = state_->submit([buffer, input, staging = staging.value()]() -> Result<void> {
      auto& resources = *buffer->resources;
      std::lock_guard lock(resources.backend_mutex);
      auto allocation = resources.backend->allocate(buffer->bytes);
      if (!allocation) return allocation.error();
      buffer->allocation = allocation.value();
      return resources.backend->upload(*buffer->allocation, *input);
    });
    return Tensor(std::move(record));
  } catch (const std::bad_alloc&) { return Status{ErrorCode::OutOfMemory, "Upload staging allocation failed"}; }
    catch (const std::length_error&) { return Status{ErrorCode::OutOfMemory, "Upload too large"}; }
}
Result<HostTransfer> Session::to_host_async(const Tensor& tensor) {
  if (!state_ || state_->closed) return Status{ErrorCode::SessionClosed, "Session closed"};
  if (!tensor.record_ || tensor.record_->buffer->resources != state_->resources)
    return Status{ErrorCode::SessionMismatch, "Tensor belongs to another session"};
  try {
    auto buffer = tensor.record_->buffer;
    auto staging = detail::reserve_staging(state_->resources, buffer->bytes);
    if (!staging) return staging.error();
    auto output = std::make_shared<HostTensor>(HostTensor{tensor.desc(), std::vector<std::byte>(buffer->bytes)});
    auto event = state_->submit([buffer, output, staging = staging.value()] {
      std::lock_guard lock(buffer->resources->backend_mutex);
      return buffer->resources->backend->download(*buffer->allocation, output->bytes);
    }, {tensor.record_->ready});
    return HostTransfer(std::move(event), std::move(output));
  } catch (const std::bad_alloc&) { return Status{ErrorCode::OutOfMemory, "Download staging allocation failed"}; }
    catch (const std::length_error&) { return Status{ErrorCode::OutOfMemory, "Download too large"}; }
}
Result<HostTensor> Session::to_host(const Tensor& tensor) {
  // Blocking convenience path does not reserve output staging until upload completes.
  if (!state_ || state_->closed) return Status{ErrorCode::SessionClosed, "Session closed"};
  if (!tensor.record_ || tensor.record_->buffer->resources != state_->resources)
    return Status{ErrorCode::SessionMismatch, "Tensor belongs to another session"};
  auto ready = tensor.wait();
  if (!ready) return ready.error();
  auto transfer = to_host_async(tensor);
  if (!transfer) return transfer.error();
  return transfer.value().wait();
}
Result<void> Session::synchronize() {
  if (!state_ || state_->closed) return Status{ErrorCode::SessionClosed, "Session closed"};
  try {
    auto fence = state_->submit([] { return Result<void>{}; });
    (void)fence.wait();
    std::lock_guard lock(state_->mutex);
    if (state_->first_error) return *state_->first_error;
    return {};
  } catch (const std::bad_alloc&) { return Status{ErrorCode::OutOfMemory, "Fence allocation failed"}; }
}
Result<void> Session::close() { return state_ ? state_->close() : Result<void>{}; }
MemoryStats Session::memory_stats() const {
  if (!state_) return {};
  std::lock_guard lock(state_->resources->stats_mutex);
  return state_->resources->stats;
}
Result<DeviceCaps> Session::capabilities() const {
  if (!state_ || state_->closed) return Status{ErrorCode::SessionClosed, "Session closed"};
  return state_->resources->caps;
}
}
