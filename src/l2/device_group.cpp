// Stage immutable activations between independent backends through bounded host memory.
#include "ndp/l2/device_group.hpp"
#include <algorithm>
#include <cstring>
#include "ndp/common/fp16.hpp"

namespace ndp {
Result<DeviceGroup> DeviceGroup::create(std::vector<std::shared_ptr<Session>> devices) {
  if (devices.empty()) return Status{ErrorCode::InvalidArgument, "Device group is empty"};
  for (auto it = devices.begin(); it != devices.end(); ++it) {
    if (!*it || std::find(devices.begin(), it, *it) != it)
      return Status{ErrorCode::InvalidArgument, "Expected distinct device sessions"};
    if (auto caps = (*it)->capabilities(); !caps) return caps.error();
  }
  return DeviceGroup(std::move(devices));
}

Result<std::shared_ptr<Session>> DeviceGroup::device(DeviceId id) const {
  if (id >= devices_.size()) return Status{ErrorCode::InvalidArgument, "Invalid device ordinal"};
  return devices_[id];
}

Result<Tensor> DeviceGroup::copy_to(const Tensor &tensor, DeviceId destination) const {
  auto target = device(destination);
  if (!target) return target.error();
  if (auto caps = target.value()->capabilities(); !caps) return caps.error();
  for (const auto &source : devices_) {
    if (!source->owns(tensor)) continue;
    if (auto valid = source->validate(tensor); !valid) return valid.error();
    if (auto ready = tensor.wait(); !ready) return ready.error();
    if (source == target.value()) return tensor;
    auto transfer = source->to_host_async(tensor);
    if (!transfer) return transfer.error();
    if (auto ready = transfer.value().ready_event().wait(); !ready) return ready.error();
    // Reuse the completed download as the destination's staging buffer.
    const auto &host = transfer.value().output_;
    std::shared_ptr<std::vector<std::byte>> bytes(host, &host->bytes);
    auto copied = target.value()->upload_dense({host->desc, host->bytes}, std::move(bytes));
    if (!copied) return copied.error();
    if (auto ready = copied.value().wait(); !ready) return ready.error();
    return copied;
  }
  return Status{ErrorCode::SessionMismatch, "Source tensor is outside the device group"};
}
Result<std::vector<Tensor>> DeviceGroup::broadcast(const Tensor &tensor) const {
  std::shared_ptr<Session> source;
  for (const auto &session : devices_) {
    if (auto caps = session->capabilities(); !caps) return caps.error();
    if (session->owns(tensor)) source = session;
  }
  if (!source) return Status{ErrorCode::SessionMismatch, "Source tensor is outside the device group"};
  if (auto valid = source->validate(tensor); !valid) return valid.error();
  if (auto ready = tensor.wait(); !ready) return ready.error();
  try {
    if (devices_.size() == 1) return std::vector<Tensor>{tensor};
    auto transfer = source->to_host_async(tensor);
    if (!transfer) return transfer.error();
    if (auto ready = transfer.value().ready_event().wait(); !ready) return ready.error();
    const auto &host = transfer.value().output_;
    std::shared_ptr<std::vector<std::byte>> bytes(host, &host->bytes);
    std::vector<Tensor> result;
    result.reserve(devices_.size());
    for (const auto &target : devices_) {
      if (target == source) {
        result.push_back(tensor);
        continue;
      }
      auto replica = target->upload_dense({host->desc, host->bytes}, bytes);
      if (!replica) return replica.error();
      if (auto ready = replica.value().wait(); !ready) return ready.error();
      result.push_back(std::move(replica.value()));
    }
    return result;
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Broadcast staging allocation failed"};
  }
}
Result<std::vector<Tensor>> DeviceGroup::all_gather_rows(const std::vector<Tensor> &shards) const {
  if (shards.size() != devices_.size())
    return Status{ErrorCode::InvalidArgument, "Expected one output shard per device"};
  size_t width = 0;
  for (size_t i = 0; i < shards.size(); ++i) {
    if (auto valid = devices_[i]->validate(shards[i]); !valid) return valid.error();
    const auto &shape = shards[i].desc().shape;
    if (shards[i].desc().dtype != DType::F16 || shape.size() != 2 || !shape[0] || shape[0] != shards[0].desc().shape[0] || shape[1] > SIZE_MAX / 2 - width)
      return Status{ErrorCode::InvalidArgument, "Expected FP16 matrix shards with matching row counts"};
    width += shape[1];
  }
  try {
    if (shards.size() == 1) {
      if (auto ready = shards.front().wait(); !ready) return ready.error();
      return shards;
    }
    const TensorDesc desc{{shards[0].desc().shape[0], width}};
    auto &root = *devices_.front();
    auto buffer = root.allocate(desc);
    if (!buffer) return buffer.error();
    size_t offset = 0;
    for (const auto &shard : shards) {
      auto local = copy_to(shard, 0);
      if (!local) return local.error();
      size_t count = shard.desc().shape[1];
      std::vector<TensorCopy> rows;
      for (size_t row = 0; row < desc.shape[0]; ++row)
        rows.push_back({local.value(), buffer.value(), row*count, row*width+offset, count});
      auto copied = root.copy_into(std::move(rows));
      if (!copied) return copied.error();
      offset += count;
    }
    auto full = root.snapshot(buffer.value(), desc);
    if (!full) return full.error();
    if (auto ready = full.value().wait(); !ready) return ready.error();
    return broadcast(full.value());
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "All-gather allocation failed"};
  }
}
Result<Tensor> DeviceGroup::reduce_partials(const std::vector<Tensor> &partials, DeviceId destination) const {
  auto target = device(destination);
  if (!target) return target.error();
  if (partials.size() != devices_.size())
    return Status{ErrorCode::InvalidArgument, "Expected one partial per device"};
  for (size_t i = 0; i < partials.size(); ++i) {
    if (auto valid = devices_[i]->validate(partials[i]); !valid) return valid.error();
    if (partials[i].desc().dtype != DType::F32 || partials[i].desc() != partials[0].desc())
      return Status{ErrorCode::InvalidArgument, "Expected matching FP32 partials"};
  }
  try {
    // Reuse one downloaded buffer as the FP32 accumulator.
    auto total = devices_[0]->to_host(partials[0]);
    if (!total) return total.error();
    for (size_t device = 1; device < partials.size(); ++device) {
      auto next = devices_[device]->to_host(partials[device]);
      if (!next) return next.error();
      for (size_t offset = 0; offset < total.value().bytes.size(); offset += 4) {
        float a, b;
        std::memcpy(&a, total.value().bytes.data()+offset, 4);
        std::memcpy(&b, next.value().bytes.data()+offset, 4);
        a += b;
        std::memcpy(total.value().bytes.data()+offset, &a, 4);
      }
    }
    auto &host = total.value();
    size_t count = host.bytes.size()/4;
    for (size_t i = 0; i < count; ++i) {
      float value;
      std::memcpy(&value, host.bytes.data()+4*i, 4);
      uint16_t half = float_to_fp16(value);
      std::memcpy(host.bytes.data()+2*i, &half, 2);
    }
    host.bytes.resize(count*2);
    host.desc.dtype = DType::F16;
    auto result = target.value()->from_host({host.desc, host.bytes});
    if (!result) return result.error();
    if (auto ready = result.value().wait(); !ready) return ready.error();
    return result;
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Partial reduction staging allocation failed"};
  }
}
} // namespace ndp
