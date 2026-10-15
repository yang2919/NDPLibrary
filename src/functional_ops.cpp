// Evaluate generic elementwise, normalization, and data movement operations on the host.
#include "functional_ops.hpp"

#include "ndp/fp16.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace ndp::detail {
bool is_tensor_op(OpKind k) {
  return k != OpKind::Matmul && k != OpKind::Matvec && k != OpKind::Linear;
}

Result<void> execute_tensor_op(const KernelPlan &plan,
                               const std::vector<std::span<const std::byte>> &in,
                               std::span<std::byte> out) {
  auto read = [&](size_t operand, size_t i) {
    uint16_t h;
    std::memcpy(&h, in[operand].data() + i * 2, 2);
    return fp16_to_float(h);
  };
  auto write = [&](size_t i, float value) {
    auto h = float_to_fp16(value);
    std::memcpy(out.data() + i * 2, &h, 2);
  };
  const auto &op = plan.operation;
  const auto &shape = op.inputs[0].shape;
  const size_t count = out.size() / 2, width = shape.back();
  if (op.kind == OpKind::Reshape) {
    std::copy(in[0].begin(), in[0].end(), out.begin());
    return {};
  }
  if (op.kind == OpKind::Add || op.kind == OpKind::Mul) {
    size_t other = in[1].size() / 2;
    for (size_t i = 0; i < count; ++i) {
      write(i, op.kind == OpKind::Add ? read(0, i) + read(1, i % other)
                                      : read(0, i) * read(1, i % other));
    }
  } else if (op.kind == OpKind::SiLU) {
    for (size_t i = 0; i < count; ++i) {
      float x = read(0, i);
      float y = x >= 0 ? x / (1 + std::exp(-x)) : x * std::exp(x) / (1 + std::exp(x));
      if (x == -std::numeric_limits<float>::infinity()) {
        y = -0.0f;
      }
      write(i, y);
    }
  } else if (op.kind == OpKind::RMSNorm) {
    for (size_t base = 0; base < count; base += width) {
      float sum = 0;
      for (size_t d = 0; d < width; ++d) {
        float x = read(0, base + d);
        sum = std::fma(x, x, sum);
      }
      float inv = 1 / std::sqrt(sum / float(width) + op.attributes.epsilon);
      for (size_t d = 0; d < width; ++d) {
        write(base + d, read(0, base + d) * inv * read(1, d));
      }
    }
  } else if (op.kind == OpKind::Softmax) {
    for (size_t base = 0; base < count; base += width) {
      float maximum = -INFINITY, total = 0;
      bool nan = false;
      size_t positive_inf = 0;
      for (size_t d = 0; d < width; ++d) {
        float x = read(0, base + d);
        nan |= std::isnan(x);
        positive_inf += x == INFINITY;
        maximum = std::max(maximum, x);
      }
      if (!nan && !positive_inf && maximum != -INFINITY) {
        for (size_t d = 0; d < width; ++d) {
          total += std::exp(read(0, base + d) - maximum);
        }
      }
      for (size_t d = 0; d < width; ++d) {
        float x = read(0, base + d);
        float y = nan                    ? NAN
                  : positive_inf         ? (x == INFINITY ? 1.0f / float(positive_inf) : 0)
                  : maximum == -INFINITY ? 0
                                         : std::exp(x - maximum) / total;
        write(base + d, y);
      }
    }
  } else if (op.kind == OpKind::Gather) {
    for (size_t i = 0; i < count; ++i) {
      std::memcpy(out.data() + i * 2, in[0].data() + op.attributes.indices[i] * 2, 2);
    }
  } else if (op.kind == OpKind::Scale) {
    for (size_t i = 0; i < count; ++i) {
      write(i, read(0, i) * op.attributes.factor);
    }
  } else if (op.kind == OpKind::Transpose) {
    for (size_t row = 0; row < shape[0]; ++row) {
      for (size_t col = 0; col < shape[1]; ++col) {
        std::memcpy(out.data() + (col * shape[0] + row) * 2,
                    in[0].data() + (row * shape[1] + col) * 2, 2);
      }
    }
  } else {
    return Status{ErrorCode::UnsupportedOperation, "Unknown elementwise/data kernel"};
  }
  return {};
}
} // namespace ndp::detail
