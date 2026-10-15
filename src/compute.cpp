// Validate operation shapes and attributes and infer logical output tensor descriptions.
#include "ndp/compute.hpp"

#include <cmath>

namespace ndp {
Result<TensorDesc> infer_output(const OperationSpec &op) {
  if (op.engine != Engine::PNM && op.engine != Engine::PIM) {
    return Status{ErrorCode::InvalidArgument, "Invalid engine"};
  }
  const size_t count = op.inputs.size();
  const bool unary = op.kind == OpKind::Softmax || op.kind == OpKind::SiLU ||
                     op.kind == OpKind::Reshape || op.kind == OpKind::Gather ||
                     op.kind == OpKind::Scale || op.kind == OpKind::Transpose;
  if ((unary && count != 1) ||
      (!unary && count != 2 && !(op.kind == OpKind::Linear && count == 3))) {
    return Status{ErrorCode::InvalidArgument, "Invalid operand count"};
  }
  for (const auto &input : op.inputs) {
    auto bytes = byte_size(input);
    if (!bytes) {
      return bytes.error();
    }
  }
  const auto &a = op.inputs[0].shape;
  const auto &b = op.inputs[count > 1 ? 1 : 0].shape;
  TensorDesc output;
  if (op.kind == OpKind::Matmul) {
    if (a.size() != 2 || b.size() != 2 || a[1] != b[0]) {
      return Status{ErrorCode::InvalidArgument, "matmul requires [M,K] x [K,N]"};
    }
    output.shape = {a[0], b[1]};
  } else if (op.kind == OpKind::Matvec) {
    if (a.size() != 2 || b.size() != 1 || a[1] != b[0]) {
      return Status{ErrorCode::InvalidArgument, "matvec requires [N,K] x [K]"};
    }
    output.shape = {a[0]};
  } else if (op.kind == OpKind::Linear) {
    if (a.size() != 2 || b.size() != 2 || a[1] != b[1]) {
      return Status{ErrorCode::InvalidArgument, "linear requires x[M,K], W[N,K]"};
    }
    if (op.inputs.size() == 3 && op.inputs[2].shape != std::vector<uint64_t>{b[0]}) {
      return Status{ErrorCode::InvalidArgument, "linear bias must be [N]"};
    }
    output.shape = {a[0], b[0]};
  } else if (op.kind == OpKind::Add || op.kind == OpKind::Mul) {
    if (b != a && b != std::vector<uint64_t>{a.back()}) {
      return Status{ErrorCode::InvalidArgument,
                    "Only equal shape or last-axis vector broadcast supported"};
    }
    output = op.inputs[0];
  } else if (op.kind == OpKind::RMSNorm) {
    if (b != std::vector<uint64_t>{a.back()} || !std::isfinite(op.attributes.epsilon) ||
        op.attributes.epsilon <= 0) {
      return Status{ErrorCode::InvalidArgument, "Invalid RMSNorm gamma or epsilon"};
    }
    output = op.inputs[0];
  } else if (op.kind == OpKind::Softmax || op.kind == OpKind::SiLU) {
    if (op.kind == OpKind::Softmax && op.attributes.axis != -1 &&
        op.attributes.axis != int(a.size()) - 1) {
      return Status{ErrorCode::InvalidArgument, "Softmax supports last axis only"};
    }
    output = op.inputs[0];
  } else if (op.kind == OpKind::Reshape) {
    output = {op.attributes.shape, op.inputs[0].dtype};
    auto bytes = byte_size(output);
    if (!bytes) {
      return bytes.error();
    }
    if (bytes.value() != byte_size(op.inputs[0]).value()) {
      return Status{ErrorCode::InvalidArgument, "Reshape must preserve element count"};
    }
  } else if (op.kind == OpKind::Gather) {
    output = {op.attributes.shape, op.inputs[0].dtype};
    auto bytes = byte_size(output);
    if (!bytes) {
      return bytes.error();
    }
    if (op.attributes.indices.size() != bytes.value() / 2) {
      return Status{ErrorCode::InvalidArgument, "Gather shape/count mismatch"};
    }
    const auto limit = byte_size(op.inputs[0]).value() / 2;
    for (auto index : op.attributes.indices) {
      if (index >= limit) {
        return Status{ErrorCode::InvalidArgument, "Gather index out of range"};
      }
    }
  } else if (op.kind == OpKind::Scale) {
    if (!std::isfinite(op.attributes.factor)) {
      return Status{ErrorCode::InvalidArgument, "Scale must be finite"};
    }
    output = op.inputs[0];
  } else if (op.kind == OpKind::Transpose) {
    if (a.size() != 2) {
      return Status{ErrorCode::InvalidArgument, "Transpose requires a matrix"};
    }
    output = {{a[1], a[0]}, op.inputs[0].dtype};
  } else {
    return Status{ErrorCode::UnsupportedOperation, "Unknown operation"};
  }
  auto bytes = byte_size(output);
  if (!bytes) {
    return bytes.error();
  }
  return output;
}
} // namespace ndp
