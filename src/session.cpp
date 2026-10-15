// Implement session scheduling, tensor ownership, layout preparation, and generic buffer lifetimes.
#include "ndp/session.hpp"

#include "planner.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <thread>

namespace ndp {
Result<size_t> byte_size(const TensorDesc &desc) {
  if (desc.dtype != DType::F16) {
    return Status{ErrorCode::UnsupportedDType, "This milestone supports F16 only"};
  }
  if (desc.shape.empty()) {
    return Status{ErrorCode::InvalidArgument, "Rank must be positive"};
  }
  size_t bytes = sizeof(uint16_t);
  for (auto dim : desc.shape) {
    if (dim == 0 || dim > std::numeric_limits<size_t>::max() / bytes) {
      return Status{ErrorCode::InvalidArgument, "Zero dimension or byte-size overflow"};
    }
    bytes *= static_cast<size_t>(dim);
  }
  return bytes;
}

Result<LayoutDesc> dense_layout(const TensorDesc &desc) {
  auto bytes = byte_size(desc);
  if (!bytes) {
    return bytes.error();
  }
  return LayoutDesc{LayoutKind::DenseRowMajor, desc, bytes.value(), 2, "dense.row_major.v1"};
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

struct Representation {
  LayoutDesc layout;
  std::shared_ptr<BufferLease> buffer;
  Event ready;
};

// Host-submission-thread only. Worker commands capture individual leases, not this map.
struct TensorRecord {
  TensorDesc desc;
  std::map<LayoutKind, Representation> representations;

  const Representation &dense() const {
    return representations.at(LayoutKind::DenseRowMajor);
  }
};

struct BufferState {
  TensorDesc desc;
  std::shared_ptr<BufferLease> buffer;
  Event allocated, tail;
};

struct Command {
  Event event;
  std::vector<Event> dependencies;
  std::function<Result<void>()> execute;
  std::function<void(const Result<void> &)> completed;
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

  ~State() {
    (void)close();
  }

  Event submit(std::function<Result<void>()> execute, std::vector<Event> deps = {},
               std::function<void(const Result<void> &)> completed = {}) {
    Command command{Event{}, std::move(deps), std::move(execute), std::move(completed)};
    auto event = command.event;
    event.state_->status = EventStatus::Pending;
    {
      std::lock_guard lock(mutex);
      queue.push_back(std::move(command));
    }
    changed.notify_one();
    return event;
  }

  void run() {
    for (;;) {
      std::optional<Command> command;
      {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return stopping || !queue.empty(); });
        if (queue.empty()) {
          return;
        }
        command.emplace(std::move(queue.front()));
        queue.pop_front();
      }
      auto event = command->event.state_;
      {
        std::lock_guard lock(event->mutex);
        event->status = EventStatus::Running;
      }
      Result<void> result;
      try {
        for (const auto &dependency : command->dependencies) {
          auto dependency_result = dependency.wait();
          if (!dependency_result) {
            result = Status{ErrorCode::DependencyFailed, dependency_result.error().message};
            break;
          }
        }
        if (result) {
          result = command->execute();
        }
      } catch (const std::bad_alloc &) {
        result = Status{ErrorCode::OutOfMemory, "Worker allocation failed"};
      } catch (const std::exception &e) {
        result = Status{ErrorCode::BackendFailure, e.what()};
      } catch (...) {
        result = Status{ErrorCode::BackendFailure, "Unknown backend exception"};
      }
      if (command->completed) {
        command->completed(result);
      }
      // Drop command-owned buffers/staging before publishing completion.
      command.reset();
      if (!result) {
        std::lock_guard lock(mutex);
        if (!first_error) {
          first_error = result.error();
        }
      }
      {
        std::lock_guard lock(event->mutex);
        if (!result) {
          event->error = result.error();
        }
        event->status = result ? EventStatus::Succeeded : EventStatus::Failed;
      }
      event->changed.notify_all();
    }
  }

  Result<void> close() {
    if (!closed) {
      closed = true;
      {
        std::lock_guard lock(mutex);
        stopping = true;
      }
      changed.notify_one();
      worker.join();
      try {
        std::lock_guard lock(resources->backend_mutex);
        auto result = resources->backend->shutdown();
        if (!result) {
          close_error = result.error();
        } else {
          std::lock_guard stats_lock(resources->stats_mutex);
          resources->stats.workspace_bytes = 0;
        }
      } catch (const std::exception &e) {
        close_error = Status{ErrorCode::BackendFailure, e.what()};
      } catch (...) {
        close_error = Status{ErrorCode::BackendFailure, "Shutdown exception"};
      }
      if (!close_error) {
        close_error = first_error;
      }
    }
    if (close_error) {
      return *close_error;
    }
    return {};
  }
};

Result<std::shared_ptr<BufferLease>> reserve_buffer(const std::shared_ptr<Resources> &resources,
                                                    size_t bytes) {
  auto buffer = std::make_shared<BufferLease>(resources);
  buffer->bytes = bytes;
  std::lock_guard lock(resources->stats_mutex);
  if (bytes > resources->caps.memory_capacity_bytes - resources->stats.live_bytes) {
    return Status{ErrorCode::OutOfMemory, "Device capacity exceeded (physical layout bytes)"};
  }
  resources->stats.live_bytes += bytes;
  ++resources->stats.allocations;
  resources->stats.peak_bytes = std::max(resources->stats.peak_bytes, resources->stats.live_bytes);
  buffer->reserved = true;
  return buffer;
}

Event submit_transform(State &state, const Representation &source, const LayoutDesc &target,
                       std::shared_ptr<BufferLease> destination) {
  return state.submit(
      [source, target, destination]() -> Result<void> {
        auto &resources = *destination->resources;
        std::lock_guard lock(resources.backend_mutex);
        auto allocation = resources.backend->allocate(destination->bytes);
        if (!allocation) {
          return allocation.error();
        }
        destination->allocation = allocation.value();
        return resources.backend->transform(*source.buffer->allocation, source.layout,
                                            *destination->allocation, target);
      },
      {source.ready});
}

Result<std::shared_ptr<StagingLease>> reserve_staging(const std::shared_ptr<Resources> &resources,
                                                      size_t bytes) {
  auto lease = std::make_shared<StagingLease>(resources);
  std::lock_guard lock(resources->stats_mutex);
  if (bytes > resources->options.max_staging_bytes - resources->stats.staging_bytes) {
    return Status{ErrorCode::OutOfMemory, "Pending host staging budget exceeded"};
  }
  resources->stats.staging_bytes += bytes;
  resources->stats.peak_staging_bytes =
      std::max(resources->stats.peak_staging_bytes, resources->stats.staging_bytes);
  lease->bytes = bytes;
  return lease;
}
} // namespace detail

Event::Event() : state_(std::make_shared<detail::EventState>()) {
  state_->status = EventStatus::Succeeded; // A default event is an already-complete fence.
}

EventStatus Event::status() const {
  std::lock_guard lock(state_->mutex);
  return state_->status;
}

Result<void> Event::wait() const {
  std::unique_lock lock(state_->mutex);
  state_->changed.wait(lock, [&] {
    return state_->status == EventStatus::Succeeded || state_->status == EventStatus::Failed;
  });
  if (state_->error) {
    return *state_->error;
  }
  return {};
}

const TensorDesc &Tensor::desc() const {
  return record_->desc;
}

std::vector<RepresentationInfo> Tensor::representations() const {
  std::vector<RepresentationInfo> result;
  for (const auto &[kind, representation] : record_->representations) {
    result.push_back({representation.layout, representation.ready.status()});
  }
  return result;
}

Event Tensor::ready_event() const {
  return record_->dense().ready;
}

Result<void> Tensor::wait() const {
  return record_->dense().ready.wait();
}

Result<HostTensor> HostTransfer::wait() const {
  auto result = event_.wait();
  if (!result) {
    return result.error();
  }
  try {
    return *output_;
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Host result allocation failed"};
  }
}

Session::Session(Session &&) noexcept = default;

Session &Session::operator=(Session &&other) noexcept {
  if (this != &other) {
    (void)close();
    state_ = std::move(other.state_);
  }
  return *this;
}

Session::~Session() {
  (void)close();
}

Result<Session> Session::create(std::unique_ptr<IBackend> backend, SessionOptions options) {
  if (!backend) {
    return Status{ErrorCode::InvalidArgument, "Backend is required"};
  }
  try {
    return Session(std::make_shared<detail::State>(std::move(backend), options));
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Session allocation failed"};
  } catch (const std::exception &e) {
    return Status{ErrorCode::BackendFailure, e.what()};
  }
}

Result<Tensor> Session::from_host(HostTensorView host) {
  if (!state_ || state_->closed) {
    return Status{ErrorCode::SessionClosed, "Session closed"};
  }
  auto size = byte_size(host.desc);
  if (!size) {
    return size.error();
  }
  if (host.bytes.size() != size.value()) {
    return Status{ErrorCode::InvalidArgument, "Host byte count does not match shape"};
  }
  try {
    auto resources = state_->resources;
    auto reserved = detail::reserve_buffer(resources, size.value());
    if (!reserved) {
      return reserved.error();
    }
    auto buffer = reserved.value();
    auto staging = detail::reserve_staging(resources, buffer->bytes);
    if (!staging) {
      return staging.error();
    }
    auto input = std::make_shared<std::vector<std::byte>>(host.bytes.begin(), host.bytes.end());
    auto record = std::make_shared<detail::TensorRecord>();
    record->desc = std::move(host.desc);
    // Insert before submission so allocation failures cannot leave an untracked command.
    auto layout = dense_layout(record->desc).value();
    auto [entry, inserted] = record->representations.emplace(
        LayoutKind::DenseRowMajor, detail::Representation{layout, buffer, Event{}});
    entry->second.ready =
        state_->submit([buffer, input, staging = staging.value()]() -> Result<void> {
          auto &resources = *buffer->resources;
          std::lock_guard lock(resources.backend_mutex);
          auto allocation = resources.backend->allocate(buffer->bytes);
          if (!allocation) {
            return allocation.error();
          }
          buffer->allocation = allocation.value();
          return resources.backend->upload(*buffer->allocation, *input);
        });
    return Tensor(std::move(record));
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Upload staging allocation failed"};
  } catch (const std::length_error &) {
    return Status{ErrorCode::OutOfMemory, "Upload too large"};
  }
}

Result<LayoutDesc> Session::query_layout(const TensorDesc &desc, LayoutKind kind) const {
  if (!state_ || state_->closed) {
    return Status{ErrorCode::SessionClosed, "Session closed"};
  }
  try {
    return state_->resources->backend->query_layout(desc, kind);
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Layout metadata allocation failed"};
  } catch (const std::exception &e) {
    return Status{ErrorCode::BackendFailure, e.what()};
  }
}

Result<Event> Session::materialize_layout(const Tensor &tensor, const LayoutDesc &target) {
  if (!state_ || state_->closed) {
    return Status{ErrorCode::SessionClosed, "Session closed"};
  }
  if (!tensor.record_ || tensor.record_->dense().buffer->resources != state_->resources) {
    return Status{ErrorCode::SessionMismatch, "Tensor belongs to another session"};
  }
  try {
    auto canonical = query_layout(tensor.desc(), target.kind);
    if (!canonical) {
      return canonical.error();
    }
    if (canonical.value() != target) {
      return Status{ErrorCode::InvalidArgument,
                    "Layout descriptor does not match backend/shape/dtype"};
    }
    auto &entries = tensor.record_->representations;
    auto found = entries.find(target.kind);
    if (found != entries.end()) {
      // A failed noncanonical representation can be explicitly retried.
      if (found->second.ready.status() != EventStatus::Failed ||
          target.kind == LayoutKind::DenseRowMajor) {
        return found->second.ready;
      }
      entries.erase(found);
    }
    auto buffer = detail::reserve_buffer(state_->resources, target.physical_bytes);
    if (!buffer) {
      return buffer.error();
    }
    auto [entry, inserted] =
        entries.emplace(target.kind, detail::Representation{target, buffer.value(), Event{}});
    try {
      entry->second.ready = detail::submit_transform(*state_, entries.at(LayoutKind::DenseRowMajor),
                                                     target, buffer.value());
    } catch (...) {
      entries.erase(entry);
      throw;
    }
    return entry->second.ready;
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Layout materialization allocation failed"};
  }
}

Result<HostTransfer> Session::to_host_async(const Tensor &tensor, LayoutKind source_kind) {
  if (!state_ || state_->closed) {
    return Status{ErrorCode::SessionClosed, "Session closed"};
  }
  if (!tensor.record_ || tensor.record_->dense().buffer->resources != state_->resources) {
    return Status{ErrorCode::SessionMismatch, "Tensor belongs to another session"};
  }
  try {
    auto found = tensor.record_->representations.find(source_kind);
    if (found == tensor.record_->representations.end()) {
      return Status{ErrorCode::UnsupportedLayout,
                    "Requested source representation is not materialized"};
    }
    auto source = found->second;
    const auto bytes = tensor.record_->dense().buffer->bytes;
    auto staging = detail::reserve_staging(state_->resources, bytes);
    if (!staging) {
      return staging.error();
    }
    auto output =
        std::make_shared<HostTensor>(HostTensor{tensor.desc(), std::vector<std::byte>(bytes)});
    auto buffer = source.buffer;
    const auto dense = dense_layout(tensor.desc()).value();
    if (source_kind != LayoutKind::DenseRowMajor) {
      // Unpack the selected packed representation, rather than reading the cached original.
      auto temporary = detail::reserve_buffer(state_->resources, bytes);
      if (!temporary) {
        return temporary.error();
      }
      buffer = temporary.value();
    }
    auto event = state_->submit(
        [source, buffer, dense, output, staging = staging.value()]() -> Result<void> {
          auto &resources = *buffer->resources;
          std::lock_guard lock(resources.backend_mutex);
          if (source.layout.kind != LayoutKind::DenseRowMajor) {
            auto allocation = resources.backend->allocate(buffer->bytes);
            if (!allocation) {
              return allocation.error();
            }
            buffer->allocation = allocation.value();
            auto transformed = resources.backend->transform(
                *source.buffer->allocation, source.layout, *buffer->allocation, dense);
            if (!transformed) {
              return transformed;
            }
          }
          return resources.backend->download(*buffer->allocation, output->bytes);
        },
        {source.ready});
    return HostTransfer(std::move(event), std::move(output));
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Download staging allocation failed"};
  } catch (const std::length_error &) {
    return Status{ErrorCode::OutOfMemory, "Download too large"};
  }
}

Result<HostTensor> Session::to_host(const Tensor &tensor, LayoutKind source) {
  if (!state_ || state_->closed) {
    return Status{ErrorCode::SessionClosed, "Session closed"};
  }
  if (!tensor.record_ || tensor.record_->dense().buffer->resources != state_->resources) {
    return Status{ErrorCode::SessionMismatch, "Tensor belongs to another session"};
  }
  auto found = tensor.record_->representations.find(source);
  if (found == tensor.record_->representations.end()) {
    return Status{ErrorCode::UnsupportedLayout,
                  "Requested source representation is not materialized"};
  }
  auto ready = found->second.ready.wait();
  if (!ready) {
    return ready.error();
  }
  auto transfer = to_host_async(tensor, source);
  if (!transfer) {
    return transfer.error();
  }
  return transfer.value().wait();
}

size_t PreparedWeight::resident_bytes() const {
  size_t bytes = 0;
  for (const auto &repr : tensor_.representations()) {
    bytes += repr.layout.physical_bytes;
  }
  return bytes;
}

Result<PreparedWeight> Session::prepare_weight(const Tensor &tensor, WeightPrepareOptions options) {
  if (!state_ || state_->closed) {
    return Status{ErrorCode::SessionClosed, "Session closed"};
  }
  if (!tensor.record_ || tensor.record_->dense().buffer->resources != state_->resources) {
    return Status{ErrorCode::SessionMismatch, "Weight belongs to another session"};
  }
  if (tensor.desc().shape.size() != 2 || options.engines.empty()) {
    return Status{ErrorCode::InvalidArgument, "Weight must be rank-2 with at least one engine use"};
  }
  try {
    // A private representation map shares existing immutable buffers. New layouts are
    // published only through the PreparedWeight returned on success.
    auto record = std::make_shared<detail::TensorRecord>(*tensor.record_);
    Tensor prepared(record);
    auto ready = tensor.wait();
    if (!ready) {
      return ready.error();
    }
    std::map<LayoutKind, LayoutDesc> required;
    for (auto engine : options.engines) {
      OperationSpec op{OpKind::Linear,
                       engine,
                       {{{1, tensor.desc().shape[1]}, tensor.desc().dtype}, tensor.desc()},
                       {}};
      auto plan = detail::Planner::build(*state_->resources->backend, op);
      if (!plan) {
        return plan.error();
      }
      required.emplace(plan.value().inputs[1].kind, plan.value().inputs[1]);
    }
    size_t total = 0;
    for (auto it = record->representations.begin(); it != record->representations.end();) {
      if (it->second.ready.status() == EventStatus::Failed) {
        it = record->representations.erase(it);
      } else {
        required.emplace(it->first, it->second.layout);
        ++it;
      }
    }
    for (const auto &[kind, layout] : required) {
      if (layout.physical_bytes > options.max_resident_bytes - total) {
        return Status{ErrorCode::OutOfMemory, "Prepared weight exceeds resident byte budget"};
      }
      total += layout.physical_bytes;
    }
    for (const auto &[kind, layout] : required) {
      auto event = materialize_layout(prepared, layout);
      if (!event) {
        return event.error();
      }
      auto result = event.value().wait();
      if (!result) {
        return result.error();
      }
    }
    return PreparedWeight(std::move(prepared));
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Weight preparation allocation failed"};
  } catch (const std::exception &e) {
    return Status{ErrorCode::BackendFailure, e.what()};
  }
}

Result<Tensor> Session::matmul(const Tensor &a, const Tensor &b, OpOptions options) {
  return compute(OpKind::Matmul, {a, b}, options);
}

Result<Tensor> Session::matvec(const Tensor &w, const Tensor &x, OpOptions options) {
  return compute(OpKind::Matvec, {w, x}, options);
}

Result<Tensor> Session::linear(const Tensor &x, const Tensor &w, OpOptions options,
                               const Tensor *bias) {
  if (bias) {
    return compute(OpKind::Linear, {x, w, *bias}, options);
  }
  return compute(OpKind::Linear, {x, w}, options);
}

Result<Tensor> Session::linear(const Tensor &x, const PreparedWeight &w, OpOptions options,
                               const Tensor *bias) {
  return linear(x, w.tensor(), options, bias);
}

Result<Tensor> Session::compute(OpKind kind, const std::vector<Tensor> &tensors, OpOptions options,
                                OpAttributes attributes) {
  if (!state_ || state_->closed) {
    return Status{ErrorCode::SessionClosed, "Session closed"};
  }
  if (options.layout_policy != LayoutPolicy::AllowMaterialize &&
      options.layout_policy != LayoutPolicy::RequirePrepared) {
    return Status{ErrorCode::InvalidArgument, "Invalid layout policy"};
  }
  try {
    OperationSpec op{kind, options.engine, {}, std::move(attributes)};
    for (const auto &t : tensors) {
      if (!t.record_ || t.record_->dense().buffer->resources != state_->resources) {
        return Status{ErrorCode::SessionMismatch, "Operand belongs to another session"};
      }
      op.inputs.push_back(t.desc());
    }
    auto planned = detail::Planner::build(*state_->resources->backend, op);
    if (!planned) {
      return planned.error();
    }
    auto plan = planned.value();
    // Validate all requirements before reserving output or scheduling preparation.
    for (size_t i = 0; i < tensors.size(); ++i) {
      const auto &entries = tensors[i].record_->representations;
      auto found = entries.find(plan.inputs[i].kind);
      if (options.layout_policy == LayoutPolicy::RequirePrepared &&
          (found == entries.end() || found->second.layout != plan.inputs[i] ||
           found->second.ready.status() == EventStatus::Failed)) {
        return Status{ErrorCode::LayoutNotPrepared, "Required input layout is not prepared"};
      }
    }
    auto reserved = detail::reserve_buffer(state_->resources, plan.output.physical_bytes);
    if (!reserved) {
      return reserved.error();
    }
    auto output = std::make_shared<detail::TensorRecord>();
    output->desc = plan.output.logical;
    auto [entry, inserted] = output->representations.emplace(
        LayoutKind::DenseRowMajor, detail::Representation{plan.output, reserved.value(), Event{}});
    std::vector<detail::Representation> inputs;
    std::vector<Event> dependencies;
    for (size_t i = 0; i < tensors.size(); ++i) {
      auto prepared = materialize_layout(tensors[i], plan.inputs[i]);
      if (!prepared) {
        return prepared.error();
      }
      inputs.push_back(tensors[i].record_->representations.at(plan.inputs[i].kind));
      dependencies.push_back(prepared.value());
    }
    entry->second.ready = state_->submit(
        [plan, inputs, buffer = reserved.value()]() -> Result<void> {
          auto &resources = *buffer->resources;
          std::lock_guard lock(resources.backend_mutex);
          if (plan.workspace_bytes) {
            bool needed;
            {
              std::lock_guard stats_lock(resources.stats_mutex);
              needed = resources.stats.workspace_bytes == 0;
            }
            if (needed) {
              auto workspace =
                  resources.backend->reserve_workspace(resources.caps.scratchpad_bytes);
              if (!workspace) {
                return workspace;
              }
              std::lock_guard stats_lock(resources.stats_mutex);
              resources.stats.workspace_bytes = resources.caps.scratchpad_bytes;
            }
          }
          auto allocated = resources.backend->allocate(buffer->bytes);
          if (!allocated) {
            return allocated.error();
          }
          buffer->allocation = allocated.value();
          std::vector<AllocationHandle> handles;
          for (const auto &input : inputs) {
            handles.push_back(*input.buffer->allocation);
          }
          return resources.backend->execute(plan, handles, *buffer->allocation);
        },
        dependencies);
    return Tensor(std::move(output));
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Compute planning allocation failed"};
  } catch (const std::exception &e) {
    return Status{ErrorCode::BackendFailure, e.what()};
  }
}

Result<Tensor> Session::add(const Tensor &a, const Tensor &b, OpOptions o) {
  return compute(OpKind::Add, {a, b}, o);
}

Result<Tensor> Session::mul(const Tensor &a, const Tensor &b, OpOptions o) {
  return compute(OpKind::Mul, {a, b}, o);
}

Result<Tensor> Session::rms_norm(const Tensor &x, const Tensor &gamma, float epsilon, OpOptions o) {
  OpAttributes attrs;
  attrs.epsilon = epsilon;
  return compute(OpKind::RMSNorm, {x, gamma}, o, attrs);
}

Result<Tensor> Session::softmax(const Tensor &x, int axis, OpOptions o) {
  OpAttributes attrs;
  attrs.axis = axis;
  return compute(OpKind::Softmax, {x}, o, attrs);
}

Result<Tensor> Session::silu(const Tensor &x, OpOptions o) {
  return compute(OpKind::SiLU, {x}, o);
}

Result<Tensor> Session::reshape(const Tensor &x, std::vector<uint64_t> shape) {
  OpAttributes attrs;
  attrs.shape = std::move(shape);
  return compute(OpKind::Reshape, {x}, {}, std::move(attrs));
}

Result<Tensor> Session::gather(const Tensor &x, std::vector<uint64_t> indices,
                               std::vector<uint64_t> shape) {
  OpAttributes attrs;
  attrs.indices = std::move(indices);
  attrs.shape = std::move(shape);
  return compute(OpKind::Gather, {x}, {}, std::move(attrs));
}

Result<Tensor> Session::scale(const Tensor &x, float factor) {
  OpAttributes attrs;
  attrs.factor = factor;
  return compute(OpKind::Scale, {x}, {}, attrs);
}

Result<Tensor> Session::transpose(const Tensor &x) {
  return compute(OpKind::Transpose, {x}, {});
}

const TensorDesc &DeviceBuffer::desc() const {
  return state_->desc;
}

Event DeviceBuffer::ready_event() const {
  return state_->tail;
}

Result<void> Session::validate(const Tensor &x) const {
  if (!state_ || state_->closed) {
    return Status{ErrorCode::SessionClosed, "Session closed"};
  }
  if (!x.record_ || x.record_->dense().buffer->resources != state_->resources) {
    return Status{ErrorCode::SessionMismatch, "Tensor belongs to another session"};
  }
  return {};
}

Result<void> Session::validate(const DeviceBuffer &x) const {
  if (!state_ || state_->closed) {
    return Status{ErrorCode::SessionClosed, "Session closed"};
  }
  if (!x.state_ || x.state_->buffer->resources != state_->resources) {
    return Status{ErrorCode::SessionMismatch, "Buffer belongs to another session"};
  }
  return {};
}

Result<DeviceBuffer> Session::allocate(TensorDesc desc) {
  if (!state_ || state_->closed) {
    return Status{ErrorCode::SessionClosed, "Session closed"};
  }
  auto size = byte_size(desc);
  if (!size) {
    return size.error();
  }
  try {
    auto reserved = detail::reserve_buffer(state_->resources, size.value());
    if (!reserved) {
      return reserved.error();
    }
    auto record = std::make_shared<detail::BufferState>();
    record->desc = std::move(desc);
    record->buffer = reserved.value();
    record->allocated = state_->submit([buffer = record->buffer]() -> Result<void> {
      auto &r = *buffer->resources;
      std::lock_guard lock(r.backend_mutex);
      auto allocation = r.backend->allocate(buffer->bytes);
      if (!allocation) {
        return allocation.error();
      }
      buffer->allocation = allocation.value();
      return r.backend->fill(*buffer->allocation, std::byte{0});
    });
    record->tail = record->allocated;
    return DeviceBuffer(record);
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Buffer allocation submission failed"};
  }
}

Result<Event> Session::copy_into(std::vector<TensorCopy> copies) {
  if (!state_ || state_->closed) {
    return Status{ErrorCode::SessionClosed, "Session closed"};
  }
  if (copies.empty()) {
    return Status{ErrorCode::InvalidArgument, "Empty copy batch"};
  }
  try {
    struct ResolvedCopy {
      std::shared_ptr<detail::BufferLease> source, destination;
      size_t source_offset, destination_offset, count;
    };

    std::vector<ResolvedCopy> resolved;
    resolved.reserve(copies.size());
    std::vector<Event> dependencies;
    for (const auto &copy : copies) {
      if (auto valid = validate(copy.source); !valid) {
        return valid.error();
      }
      if (auto valid = validate(copy.destination); !valid) {
        return valid.error();
      }
      const size_t source_count = byte_size(copy.source.desc()).value() / 2;
      const size_t destination_count = byte_size(copy.destination.desc()).value() / 2;
      if (!copy.count || copy.source_offset > source_count ||
          copy.count > source_count - copy.source_offset ||
          copy.destination_offset > destination_count ||
          copy.count > destination_count - copy.destination_offset) {
        return Status{ErrorCode::InvalidArgument, "Copy region out of bounds"};
      }
      resolved.push_back({copy.source.record_->dense().buffer, copy.destination.state_->buffer,
                          copy.source_offset, copy.destination_offset, copy.count});
      dependencies.push_back(copy.source.ready_event());
      dependencies.push_back(copy.destination.ready_event());
    }
    // Capture handles by value; every destination shares the batch completion event.
    auto event = state_->submit(
        [resolved = std::move(resolved)]() -> Result<void> {
          for (const auto &copy : resolved) {
            auto src = copy.source;
            auto dst = copy.destination;
            auto &r = *dst->resources;
            std::lock_guard lock(r.backend_mutex);
            auto result =
                r.backend->copy_region(*src->allocation, copy.source_offset * 2, *dst->allocation,
                                       copy.destination_offset * 2, copy.count * 2);
            if (!result) {
              return result;
            }
          }
          return {};
        },
        std::move(dependencies));
    for (auto &copy : copies) {
      copy.destination.state_->tail = event;
    }
    return event;
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Copy submission failed"};
  }
}

Result<Tensor> Session::snapshot(const DeviceBuffer &source, TensorDesc desc, size_t offset) {
  if (auto valid = validate(source); !valid) {
    return valid.error();
  }
  auto size = byte_size(desc);
  if (!size) {
    return size.error();
  }
  const size_t total = source.state_->buffer->bytes / 2;
  if (offset > total || size.value() / 2 > total - offset) {
    return Status{ErrorCode::InvalidArgument, "Snapshot region out of bounds"};
  }
  try {
    auto reserved = detail::reserve_buffer(state_->resources, size.value());
    if (!reserved) {
      return reserved.error();
    }
    auto record = std::make_shared<detail::TensorRecord>();
    record->desc = desc;
    auto [entry, inserted] = record->representations.emplace(
        LayoutKind::DenseRowMajor,
        detail::Representation{dense_layout(desc).value(), reserved.value(), Event{}});
    entry->second.ready = state_->submit(
        [src = source.state_->buffer, dst = reserved.value(), offset]() -> Result<void> {
          auto &r = *src->resources;
          std::lock_guard lock(r.backend_mutex);
          auto allocation = r.backend->allocate(dst->bytes);
          if (!allocation) {
            return allocation.error();
          }
          dst->allocation = allocation.value();
          return r.backend->copy_region(*src->allocation, offset * 2, *dst->allocation, 0,
                                        dst->bytes);
        },
        {source.ready_event()});
    return Tensor(record);
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Snapshot submission failed"};
  }
}

Result<Event> Session::clear(std::vector<DeviceBuffer> buffers) {
  if (!state_ || state_->closed) {
    return Status{ErrorCode::SessionClosed, "Session closed"};
  }
  if (buffers.empty()) {
    return Status{ErrorCode::InvalidArgument, "Empty clear batch"};
  }
  try {
    std::vector<Event> dependencies;
    for (const auto &buffer : buffers) {
      if (auto valid = validate(buffer); !valid) {
        return valid.error();
      }
      dependencies.push_back(buffer.state_->allocated);
    }
    // FIFO preserves earlier reads/writes; ignore failed writes because this overwrites all data.
    auto event = state_->submit(
        [buffers]() -> Result<void> {
          for (const auto &buffer : buffers) {
            auto lease = buffer.state_->buffer;
            auto &r = *lease->resources;
            std::lock_guard lock(r.backend_mutex);
            auto result = r.backend->fill(*lease->allocation, std::byte{0});
            if (!result) {
              return result;
            }
          }
          return {};
        },
        std::move(dependencies));
    for (auto &buffer : buffers) {
      buffer.state_->tail = event;
    }
    return event;
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Clear submission failed"};
  }
}

Result<void> Session::synchronize() {
  if (!state_ || state_->closed) {
    return Status{ErrorCode::SessionClosed, "Session closed"};
  }
  try {
    auto fence = state_->submit([] { return Result<void>{}; });
    (void)fence.wait();
    std::lock_guard lock(state_->mutex);
    if (state_->first_error) {
      return *state_->first_error;
    }
    return {};
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Fence allocation failed"};
  }
}

Result<void> Session::close() {
  return state_ ? state_->close() : Result<void>{};
}

MemoryStats Session::memory_stats() const {
  if (!state_) {
    return {};
  }
  std::lock_guard lock(state_->resources->stats_mutex);
  return state_->resources->stats;
}

Result<DeviceCaps> Session::capabilities() const {
  if (!state_ || state_->closed) {
    return Status{ErrorCode::SessionClosed, "Session closed"};
  }
  return state_->resources->caps;
}
} // namespace ndp
