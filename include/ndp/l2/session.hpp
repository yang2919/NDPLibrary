// Expose single-device sessions, tensors, mutable device buffers, prepared weights, and transfers.
#pragma once
#include "ndp/l1/backend.hpp"
#include "ndp/l2/event.hpp"

namespace ndp {
namespace detail {
struct State;
struct TensorRecord;
struct BufferState;
} // namespace detail

// F16 data is IEEE 754 binary16 stored in native-endian uint16_t words.
// No implicit float conversion is performed by upload/download.
struct HostTensorView {
  TensorDesc desc;
  std::span<const std::byte> bytes;
};

struct HostTensor {
  TensorDesc desc;
  std::vector<std::byte> bytes;
};

struct SessionOptions {
  size_t max_staging_bytes = 64 * 1024 * 1024;
};

struct RepresentationInfo {
  LayoutDesc layout;
  EventStatus status;
};

class Tensor {
public:
  const TensorDesc &desc() const;
  Event ready_event() const;
  std::vector<RepresentationInfo> representations() const;
  Result<void> wait() const;

private:
  explicit Tensor(std::shared_ptr<detail::TensorRecord> record) : record_(std::move(record)) {}

  std::shared_ptr<detail::TensorRecord> record_;
  friend class Session;
};

struct WeightPrepareOptions {
  std::vector<Engine> engines{Engine::PNM, Engine::PIM};
  size_t max_resident_bytes = static_cast<size_t>(-1);
};

class PreparedWeight {
public:
  const Tensor &tensor() const {
    return tensor_;
  }

  size_t resident_bytes() const;

private:
  explicit PreparedWeight(Tensor tensor) : tensor_(std::move(tensor)) {}

  Tensor tensor_;
  friend class Session;
};

class HostTransfer {
public:
  Event ready_event() const {
    return event_;
  }

  Result<HostTensor> wait() const;

private:
  HostTransfer(Event event, std::shared_ptr<HostTensor> output)
      : event_(std::move(event)), output_(std::move(output)) {}

  Event event_;
  std::shared_ptr<HostTensor> output_;
  friend class Session;
};

// Mutable dense storage. Reading creates an independent immutable Tensor snapshot.
class DeviceBuffer {
public:
  const TensorDesc &desc() const;
  Event ready_event() const;

private:
  explicit DeviceBuffer(std::shared_ptr<detail::BufferState> state) : state_(std::move(state)) {}

  std::shared_ptr<detail::BufferState> state_;
  friend class Session;
};

// Offsets and count are FP16 elements in flattened dense order, not bytes.
struct TensorCopy {
  Tensor source;
  DeviceBuffer destination;
  size_t source_offset, destination_offset, count;
};

// One submitting host thread. Backend work runs on an internal FIFO worker.
class Session {
public:
  static Result<Session> create(std::unique_ptr<IBackend>, SessionOptions = {});
  Session(Session &&) noexcept;
  Session &operator=(Session &&) noexcept;
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  ~Session();
  Result<Tensor> from_host(HostTensorView);
  Result<LayoutDesc> query_layout(const TensorDesc &, LayoutKind) const;
  Result<Event> materialize_layout(const Tensor &, const LayoutDesc &);
  // source selects the representation to read; packed sources are unpacked in-device.
  Result<HostTransfer> to_host_async(const Tensor &, LayoutKind source = LayoutKind::DenseRowMajor);
  Result<HostTensor> to_host(const Tensor &, LayoutKind source = LayoutKind::DenseRowMajor);
  Result<PreparedWeight> prepare_weight(const Tensor &, WeightPrepareOptions = {});
  Result<Tensor> matmul(const Tensor &a, const Tensor &b, OpOptions = {});
  Result<Tensor> matvec(const Tensor &w, const Tensor &x, OpOptions = {Engine::PIM});
  Result<Tensor> linear(const Tensor &x, const Tensor &w, OpOptions = {},
                        const Tensor *bias = nullptr);
  Result<Tensor> linear(const Tensor &x, const PreparedWeight &w, OpOptions = {},
                        const Tensor *bias = nullptr);
  Result<Tensor> add(const Tensor &, const Tensor &, OpOptions = {});
  Result<Tensor> mul(const Tensor &, const Tensor &, OpOptions = {});
  Result<Tensor> rms_norm(const Tensor &, const Tensor &gamma, float epsilon = 1e-5f,
                          OpOptions = {});
  Result<Tensor> softmax(const Tensor &, int axis = -1, OpOptions = {});
  Result<Tensor> silu(const Tensor &, OpOptions = {});
  Result<Tensor> reshape(const Tensor &, std::vector<uint64_t> shape);
  Result<Tensor> gather(const Tensor &, std::vector<uint64_t> indices, std::vector<uint64_t> shape);
  Result<Tensor> scale(const Tensor &, float factor);
  Result<Tensor> transpose(const Tensor &);
  Result<DeviceBuffer> allocate(TensorDesc);
  Result<Event> copy_into(std::vector<TensorCopy>);
  Result<Tensor> snapshot(const DeviceBuffer &, TensorDesc, size_t offset = 0);
  // Full overwrite; can recover storage after a failed write, but not failed allocation.
  Result<Event> clear(std::vector<DeviceBuffer>);
  Result<void> validate(const Tensor &) const;
  Result<void> validate(const DeviceBuffer &) const;
  Result<void> synchronize();
  Result<void> close();
  MemoryStats memory_stats() const;
  Result<DeviceCaps> capabilities() const;

private:
  explicit Session(std::shared_ptr<detail::State> state) : state_(std::move(state)) {}

  Result<Tensor> compute(OpKind, const std::vector<Tensor> &, OpOptions, OpAttributes = {});
  std::shared_ptr<detail::State> state_;
};
} // namespace ndp
