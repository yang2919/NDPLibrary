// Declare host kernels for generic tensor operations.
#pragma once
#include "ndp/common/compute.hpp"

#include <span>

namespace ndp::detail {
bool is_tensor_op(OpKind kind);
Result<void> execute_tensor_op(const KernelPlan &, const std::vector<std::span<const std::byte>> &,
                               std::span<std::byte> output);
} // namespace ndp::detail
