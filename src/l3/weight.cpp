// Select projection weight layouts and enforce the model's preparation budget in L3.
#include "ndp/l3/weight.hpp"

#include <map>

namespace ndp {
size_t PreparedWeight::resident_bytes() const {
  size_t bytes = 0;
  for (const auto &representation : tensor_.representations()) {
    bytes += representation.layout.physical_bytes;
  }
  return bytes;
}

Result<PreparedWeight> prepare_weight(Session &session, const Tensor &tensor,
                                      WeightPrepareOptions options) {
  if (auto valid = session.validate(tensor); !valid) {
    return valid.error();
  }
  if (tensor.desc().shape.size() != 2 || options.engines.empty()) {
    return Status{ErrorCode::InvalidArgument, "Weight must be rank-2 with at least one engine use"};
  }
  try {
    auto ready = tensor.wait();
    if (!ready) {
      return ready.error();
    }
    std::map<LayoutKind, LayoutDesc> required;
    for (auto engine : options.engines) {
      OperationSpec operation{OpKind::Linear,
                              engine,
                              {{{1, tensor.desc().shape[1]}, tensor.desc().dtype}, tensor.desc()},
                              {}};
      auto plan = session.query_operation(operation);
      if (!plan) {
        return plan.error();
      }
      required.emplace(plan.value().inputs[1].kind, plan.value().inputs[1]);
    }
    for (const auto &representation : tensor.representations()) {
      if (representation.status != EventStatus::Failed) {
        required.emplace(representation.layout.kind, representation.layout);
      }
    }
    size_t bytes = 0;
    std::vector<LayoutDesc> layouts;
    for (const auto &[kind, layout] : required) {
      if (layout.physical_bytes > options.max_resident_bytes - bytes) {
        return Status{ErrorCode::OutOfMemory, "Prepared weight exceeds resident byte budget"};
      }
      bytes += layout.physical_bytes;
      layouts.push_back(layout);
    }
    auto prepared = session.prepare_layouts(tensor, layouts);
    if (!prepared) {
      return prepared.error();
    }
    return PreparedWeight(std::move(prepared.value()));
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Weight preparation failed"};
  }
}
} // namespace ndp
