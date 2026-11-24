// Validate tensor byte sizes and describe canonical dense storage.
#include "ndp/common/layout.hpp"

#include <limits>

namespace ndp {
Result<size_t> byte_size(const TensorDesc &desc) {
  if (desc.dtype != DType::F16 && desc.dtype != DType::F32) {
    return Status{ErrorCode::UnsupportedDType, "Expected F16 or F32 storage"};
  }
  if (desc.shape.empty()) {
    return Status{ErrorCode::InvalidArgument, "Rank must be positive"};
  }
  size_t bytes = desc.dtype == DType::F32 ? sizeof(float) : sizeof(uint16_t);
  for (auto dim : desc.shape) {
    if (dim == 0 || dim > std::numeric_limits<size_t>::max() / bytes) {
      return Status{ErrorCode::InvalidArgument, "Zero dimension or byte-size overflow"};
    }
    bytes *= static_cast<size_t>(dim);
  }
  return bytes;
}

Result<LayoutDesc> dense_layout(const TensorDesc &desc) {
  auto bytes = byte_size(desc);
  if (!bytes) {
    return bytes.error();
  }
  return LayoutDesc{LayoutKind::DenseRowMajor, desc, bytes.value(), desc.dtype == DType::F32 ? size_t(4) : size_t(2), "dense.row_major.v1"};
}

} // namespace ndp
