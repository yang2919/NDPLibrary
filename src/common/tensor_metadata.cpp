// Validate tensor byte sizes and describe canonical dense storage.
#include "ndp/common/layout.hpp"

#include <limits>

namespace ndp {
Result<size_t> byte_size(const TensorDesc &desc) {
  if (desc.dtype != DType::F16) {
    return Status{ErrorCode::UnsupportedDType, "This milestone supports F16 only"};
  }
  if (desc.shape.empty()) {
    return Status{ErrorCode::InvalidArgument, "Rank must be positive"};
  }
  size_t bytes = sizeof(uint16_t);
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
  return LayoutDesc{LayoutKind::DenseRowMajor, desc, bytes.value(), 2, "dense.row_major.v1"};
}

} // namespace ndp
