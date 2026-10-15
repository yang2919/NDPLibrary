// Describe logical operations, execution options, attributes, and backend kernel requirements.
#pragma once
#include "ndp/common/layout.hpp"

namespace ndp {
enum class OpKind {
  Matmul,
  Matvec,
  Linear,
  Add,
  Mul,
  RMSNorm,
  Softmax,
  SiLU,
  Reshape,
  Gather,
  Scale,
  Transpose
};

struct OpAttributes {
  float epsilon = 1e-5f;
  int axis = -1;
  std::vector<uint64_t> indices;
  float factor = 1;
  std::vector<uint64_t> shape;
  bool operator==(const OpAttributes &) const = default;
};
enum class LayoutPolicy { AllowMaterialize, RequirePrepared };

struct OpOptions {
  Engine engine = Engine::PNM;
  LayoutPolicy layout_policy = LayoutPolicy::AllowMaterialize;
};

struct OperationSpec {
  OpKind kind;
  Engine engine;
  std::vector<TensorDesc> inputs; // matmul: A,B; matvec: W,x; linear: x,W[,bias]
  OpAttributes attributes;
  bool operator==(const OperationSpec &) const = default;
};

Result<TensorDesc> infer_output(const OperationSpec &);

struct KernelPlan {
  OperationSpec operation;
  std::string kernel_id;
  std::vector<LayoutDesc> inputs;
  LayoutDesc output;
  size_t workspace_bytes = 0;
  size_t tile_k = 0;
  DType accumulator = DType::F32;
  bool operator==(const KernelPlan &) const = default;
};
} // namespace ndp
