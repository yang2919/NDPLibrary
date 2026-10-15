// Validate backend kernel plans against operation semantics and available device resources.
#pragma once
#include "ndp/backend/backend.hpp"

namespace ndp::detail {
// Pure planning: validates semantics and the backend contract; allocates no buffers.
class Planner {
public:
  static Result<KernelPlan> build(const IBackend &backend, const OperationSpec &operation) {
    auto inferred = infer_output(operation);
    if (!inferred) {
      return inferred.error();
    }
    auto queried = backend.query_operation(operation);
    if (!queried) {
      return queried.error();
    }
    const auto &plan = queried.value();
    if (plan.operation != operation || plan.inputs.size() != operation.inputs.size() ||
        plan.output.logical != inferred.value() || plan.output.kind != LayoutKind::DenseRowMajor ||
        plan.accumulator != DType::F32 || plan.output != dense_layout(inferred.value()).value()) {
      return Status{ErrorCode::BackendFailure, "Backend returned inconsistent kernel metadata"};
    }
    for (size_t i = 0; i < plan.inputs.size(); ++i) {
      auto layout = backend.query_layout(operation.inputs[i], plan.inputs[i].kind);
      if (!layout) {
        return layout.error();
      }
      if (layout.value() != plan.inputs[i]) {
        return Status{ErrorCode::BackendFailure, "Invalid input layout requirement"};
      }
    }
    if (plan.workspace_bytes > backend.capabilities().scratchpad_bytes) {
      return Status{ErrorCode::OutOfMemory, "Kernel workspace exceeds available scratchpad"};
    }
    return queried;
  }
};
} // namespace ndp::detail
