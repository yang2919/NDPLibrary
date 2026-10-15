// Describe physical tensor layouts and the functional test packing format.
#pragma once
#include "ndp/common/types.hpp"

namespace ndp {
// TestTile4x4 is a functional test format, NOT an AiM/FPGA memory mapping.
enum class LayoutKind { DenseRowMajor, TestTile4x4 };

struct LayoutDesc {
  LayoutKind kind = LayoutKind::DenseRowMajor;
  TensorDesc logical;
  size_t physical_bytes = 0;
  size_t alignment = 2;
  std::string format; // backend format/version identifier
  bool operator==(const LayoutDesc &) const = default;
};

Result<LayoutDesc> dense_layout(const TensorDesc &);
} // namespace ndp
