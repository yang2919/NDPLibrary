// Prepare reusable model weights through generic L2 layout APIs.
#pragma once
#include "ndp/l2/session.hpp"

namespace ndp {
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
  friend Result<PreparedWeight> prepare_weight(Session &, const Tensor &, WeightPrepareOptions);
};

Result<PreparedWeight> prepare_weight(Session &, const Tensor &, WeightPrepareOptions = {});
} // namespace ndp
