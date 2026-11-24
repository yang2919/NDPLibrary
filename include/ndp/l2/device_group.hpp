// Address independent single-device sessions by group-local ordinal.
#pragma once
#include "ndp/l2/session.hpp"

namespace ndp {
using DeviceId = size_t;

class DeviceGroup {
public:
  static Result<DeviceGroup> create(std::vector<std::shared_ptr<Session>> devices);
  size_t size() const { return devices_.size(); }
  Result<std::shared_ptr<Session>> device(DeviceId id) const;
  // Blocking dense handoff. A local tensor on the destination is reused.
  Result<Tensor> copy_to(const Tensor &tensor, DeviceId destination) const;

  // Download once and reuse host staging for every other device. Source is reused.
  Result<std::vector<Tensor>> broadcast(const Tensor &tensor) const;

  // One [rows, local_width] shard per ordinal. Return full matrices on every device.
  Result<std::vector<Tensor>> all_gather_rows(const std::vector<Tensor> &shards) const;

  // Sum matching FP32 partials in ordinal order, then round once to FP16.
  Result<Tensor> reduce_partials(const std::vector<Tensor> &partials, DeviceId destination) const;

private:
  explicit DeviceGroup(std::vector<std::shared_ptr<Session>> devices) : devices_(std::move(devices)) {}
  std::vector<std::shared_ptr<Session>> devices_;
};
} // namespace ndp
