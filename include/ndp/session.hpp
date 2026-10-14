#pragma once
#include "ndp/backend/backend.hpp"
#include "ndp/event.hpp"

namespace ndp {
namespace detail { struct State; struct TensorRecord; }
// F16 data is IEEE 754 binary16 stored in native-endian uint16_t words.
// No implicit float conversion is performed by upload/download.
struct HostTensorView { TensorDesc desc; std::span<const std::byte> bytes; };
struct HostTensor { TensorDesc desc; std::vector<std::byte> bytes; };
struct SessionOptions { size_t max_staging_bytes = 64 * 1024 * 1024; };
class Tensor {
 public:
  const TensorDesc& desc() const;
  Event ready_event() const;
  Result<void> wait() const;
 private:
  explicit Tensor(std::shared_ptr<detail::TensorRecord> record) : record_(std::move(record)) {}
  std::shared_ptr<detail::TensorRecord> record_;
  friend class Session;
};
class HostTransfer {
 public:
  Event ready_event() const { return event_; }
  Result<HostTensor> wait() const;
 private:
  HostTransfer(Event event, std::shared_ptr<HostTensor> output)
      : event_(std::move(event)), output_(std::move(output)) {}
  Event event_;
  std::shared_ptr<HostTensor> output_;
  friend class Session;
};
// One submitting host thread. Backend work runs on an internal FIFO worker.
class Session {
 public:
  static Result<Session> create(std::unique_ptr<IBackend>, SessionOptions = {});
  Session(Session&&) noexcept;
  Session& operator=(Session&&) noexcept;
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  ~Session();
  Result<Tensor> from_host(HostTensorView);
  Result<HostTransfer> to_host_async(const Tensor&);
  Result<HostTensor> to_host(const Tensor&);
  Result<void> synchronize();
  Result<void> close();
  MemoryStats memory_stats() const;
  Result<DeviceCaps> capabilities() const;
 private:
  explicit Session(std::shared_ptr<detail::State> state) : state_(std::move(state)) {}
  std::shared_ptr<detail::State> state_;
};
}
