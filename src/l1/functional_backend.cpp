// Provide an in-memory backend for transfer, layout, workspace, and compute correctness tests.
#include "functional_ops.hpp"
#include "ndp/l1/backend.hpp"
#include "ndp/common/fp16.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <new>

namespace ndp {
namespace {
class FunctionalBackend final : public IBackend {
public:
  explicit FunctionalBackend(size_t capacity, size_t scratchpad)
      : capacity_(capacity), scratchpad_(scratchpad) {}

  DeviceCaps capabilities() const override {
    return {"functional", capacity_, scratchpad_};
  }

  Result<AllocationHandle> allocate(size_t bytes) override {
    if (closed_) {
      return Status{ErrorCode::SessionClosed, "Backend closed"};
    }
    if (bytes == 0) {
      return Status{ErrorCode::InvalidArgument, "Zero-byte allocation"};
    }
    if (bytes > capacity_ - used_) {
      return Status{ErrorCode::OutOfMemory, "Device capacity exceeded"};
    }
    try {
      buffers_.emplace(next_, std::vector<std::byte>(bytes));
    } catch (const std::bad_alloc &) {
      return Status{ErrorCode::OutOfMemory, "Host backing allocation failed"};
    } catch (const std::length_error &) {
      return Status{ErrorCode::OutOfMemory, "Backing allocation too large"};
    }
    used_ += bytes;
    return next_++;
  }

  void release(AllocationHandle handle) noexcept override {
    auto it = buffers_.find(handle);
    if (it != buffers_.end()) {
      used_ -= it->second.size();
      buffers_.erase(it);
    }
  }

  Result<void> upload(AllocationHandle handle, std::span<const std::byte> src) override {
    if (auto s = validate(handle, src.size()); !s) {
      return s;
    }
    std::copy(src.begin(), src.end(), buffers_.at(handle).begin());
    return {};
  }

  Result<void> download(AllocationHandle handle, std::span<std::byte> dst) override {
    if (auto s = validate(handle, dst.size()); !s) {
      return s;
    }
    std::copy(buffers_.at(handle).begin(), buffers_.at(handle).end(), dst.begin());
    return {};
  }

  Result<LayoutDesc> query_layout(const TensorDesc &desc, LayoutKind kind) const override {
    auto dense = dense_layout(desc);
    if (!dense) {
      return dense.error();
    }
    if (kind == LayoutKind::DenseRowMajor) {
      return dense;
    }
    if (kind != LayoutKind::TestTile4x4 || desc.shape.size() != 2) {
      return Status{ErrorCode::UnsupportedLayout, "TestTile4x4 requires rank-2 F16"};
    }
    auto rows = desc.shape[0], cols = desc.shape[1];
    const auto limit = std::numeric_limits<size_t>::max();
    if (rows > limit - 3 || cols > limit - 3) {
      return Status{ErrorCode::InvalidArgument, "Tile rounding overflow"};
    }
    size_t tile_rows = (rows + 3) / 4, tile_cols = (cols + 3) / 4;
    if (tile_cols > limit / 32 || tile_rows > limit / (tile_cols * 32)) {
      return Status{ErrorCode::InvalidArgument, "Packed byte-size overflow"};
    }
    return LayoutDesc{kind, desc, tile_rows * tile_cols * 32, 2, "functional.tile4x4.v1"};
  }

  Result<void> transform(AllocationHandle src, const LayoutDesc &from, AllocationHandle dst,
                         const LayoutDesc &to) override {
    auto expected_src = query_layout(from.logical, from.kind);
    auto expected_dst = query_layout(to.logical, to.kind);
    if (!expected_src) {
      return expected_src.error();
    }
    if (!expected_dst) {
      return expected_dst.error();
    }
    if (from != expected_src.value() || to != expected_dst.value() || from.logical != to.logical ||
        src == dst) {
      return Status{ErrorCode::InvalidArgument, "Invalid transform descriptors or alias"};
    }
    if (auto valid = validate(src, from.physical_bytes); !valid) {
      return valid;
    }
    if (auto valid = validate(dst, to.physical_bytes); !valid) {
      return valid;
    }
    const auto &input = buffers_.at(src);
    auto &output = buffers_.at(dst);
    std::fill(output.begin(), output.end(), std::byte{0});
    if (from.kind == LayoutKind::DenseRowMajor && to.kind == LayoutKind::DenseRowMajor) {
      std::copy(input.begin(), input.end(), output.begin());
      return {};
    }
    size_t rows = from.logical.shape[0], cols = from.logical.shape[1];
    auto index = [cols](LayoutKind kind, size_t r, size_t c) {
      if (kind == LayoutKind::DenseRowMajor) {
        return r * cols + c;
      }
      return ((r / 4) * ((cols + 3) / 4) + c / 4) * 16 + (c % 4) * 4 + r % 4;
    };
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < cols; ++c) {
        const auto in = index(from.kind, r, c) * 2, out = index(to.kind, r, c) * 2;
        output[out] = input[in];
        output[out + 1] = input[in + 1];
      }
    }
    return {};
  }

  Result<KernelPlan> query_operation(const OperationSpec &op) const override {
    auto output = infer_output(op);
    if (!output) {
      return output.error();
    }
    if (detail::is_tensor_op(op.kind)) {
      if (op.engine != Engine::PNM) {
        return Status{ErrorCode::UnsupportedOperation, "Tensor primitives require PNM"};
      }
      KernelPlan plan;
      plan.operation = op;
      plan.kernel_id = "functional.tensor.v1";
      plan.output = dense_layout(output.value()).value();
      for (const auto &input : op.inputs) {
        plan.inputs.push_back(dense_layout(input).value());
      }
      return plan;
    }
    if (op.engine == Engine::PIM &&
        (op.kind == OpKind::Matmul ||
         (op.kind == OpKind::Linear && (op.inputs[0].shape[0] != 1 || op.inputs.size() == 3)))) {
      return Status{ErrorCode::UnsupportedOperation,
                    "PIM supports matvec and bias-free M=1 linear only"};
    }
    KernelPlan plan;
    plan.operation = op;
    plan.kernel_id = op.engine == Engine::PNM ? "functional.pnm.v1" : "functional.pim.v1";
    plan.output = dense_layout(output.value()).value();
    for (size_t i = 0; i < op.inputs.size(); ++i) {
      const bool weight = op.kind == OpKind::Matvec ? i == 0 : i == 1;
      auto kind =
          op.engine == Engine::PIM && weight ? LayoutKind::TestTile4x4 : LayoutKind::DenseRowMajor;
      auto layout = query_layout(op.inputs[i], kind);
      if (!layout) {
        return layout.error();
      }
      plan.inputs.push_back(layout.value());
    }
    if (op.engine == Engine::PNM) {
      if (scratchpad_ < 4) {
        return Status{ErrorCode::OutOfMemory, "PNM needs at least 4 bytes of scratchpad"};
      }
      const size_t k = op.inputs[0].shape[1];
      plan.tile_k = std::min({size_t{64}, k, scratchpad_ / 4});
      plan.workspace_bytes = plan.tile_k * 4;
    }
    return plan;
  }

  Result<void> reserve_workspace(size_t bytes) override {
    if (closed_) {
      return Status{ErrorCode::SessionClosed, "Backend closed"};
    }
    if (bytes > scratchpad_) {
      return Status{ErrorCode::OutOfMemory, "Scratchpad capacity exceeded"};
    }
    if (workspace_.size() >= bytes) {
      return {};
    }
    try {
      workspace_.resize(bytes);
    } catch (const std::bad_alloc &) {
      return Status{ErrorCode::OutOfMemory, "Workspace allocation failed"};
    } catch (const std::length_error &) {
      return Status{ErrorCode::OutOfMemory, "Workspace too large"};
    }
    return {};
  }

  Result<void> execute(const KernelPlan &plan, std::span<const AllocationHandle> inputs,
                       AllocationHandle output) override {
    auto expected = query_operation(plan.operation);
    if (!expected) {
      return expected.error();
    }
    if (expected.value() != plan || inputs.size() != plan.inputs.size()) {
      return Status{ErrorCode::InvalidArgument, "Invalid kernel plan"};
    }
    if (workspace_.size() < plan.workspace_bytes) {
      return Status{ErrorCode::OutOfMemory, "Workspace is not reserved"};
    }
    if (auto valid = validate(output, plan.output.physical_bytes); !valid) {
      return valid;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i] == output) {
        return Status{ErrorCode::InvalidArgument, "Output must not alias input"};
      }
      if (auto valid = validate(inputs[i], plan.inputs[i].physical_bytes); !valid) {
        return valid;
      }
    }
    if (detail::is_tensor_op(plan.operation.kind)) {
      std::vector<std::span<const std::byte>> views;
      for (auto handle : inputs) {
        views.emplace_back(buffers_.at(handle));
      }
      return detail::execute_tensor_op(plan, views, buffers_.at(output));
    }
    auto load = [&](size_t operand, size_t row, size_t col) {
      const auto &layout = plan.inputs[operand];
      size_t index = col;
      if (layout.logical.shape.size() == 2) {
        const size_t cols = layout.logical.shape[1];
        index = layout.kind == LayoutKind::DenseRowMajor
                    ? row * cols + col
                    : ((row / 4) * ((cols + 3) / 4) + col / 4) * 16 + (col % 4) * 4 + row % 4;
      }
      uint16_t value;
      std::memcpy(&value, buffers_.at(inputs[operand]).data() + index * 2, 2);
      return value;
    };
    const auto &op = plan.operation;
    size_t m = 1, n = 0, k = op.inputs[0].shape[1];
    if (op.kind == OpKind::Matvec) {
      n = op.inputs[0].shape[0];
    } else {
      m = op.inputs[0].shape[0];
      n = op.kind == OpKind::Matmul ? op.inputs[1].shape[1] : op.inputs[1].shape[0];
    }
    for (size_t row = 0; row < m; ++row) {
      for (size_t col = 0; col < n; ++col) {
        float sum = 0;
        const size_t tile = op.engine == Engine::PNM ? plan.tile_k : k;
        for (size_t start = 0; start < k;) {
          const size_t count = std::min(tile, k - start);
          for (size_t j = 0; j < count; ++j) {
            const size_t inner = start + j;
            uint16_t left = op.kind == OpKind::Matvec ? load(0, col, inner) : load(0, row, inner);
            uint16_t right = op.kind == OpKind::Matvec   ? load(1, 0, inner)
                             : op.kind == OpKind::Matmul ? load(1, inner, col)
                                                         : load(1, col, inner);
            if (op.engine == Engine::PNM) {
              std::memcpy(workspace_.data() + j * 2, &left, 2);
              std::memcpy(workspace_.data() + plan.tile_k * 2 + j * 2, &right, 2);
            } else {
              sum = std::fma(fp16_to_float(left), fp16_to_float(right), sum);
            }
          }
          if (op.engine == Engine::PNM) {
            for (size_t j = 0; j < count; ++j) {
              uint16_t left, right;
              std::memcpy(&left, workspace_.data() + j * 2, 2);
              std::memcpy(&right, workspace_.data() + plan.tile_k * 2 + j * 2, 2);
              sum = std::fma(fp16_to_float(left), fp16_to_float(right), sum);
            }
          }
          start += count;
        }
        if (op.inputs.size() == 3) {
          sum += fp16_to_float(load(2, 0, col));
        }
        uint16_t value = float_to_fp16(sum);
        std::memcpy(buffers_.at(output).data() + (row * n + col) * 2, &value, 2);
      }
    }
    return {};
  }

  Result<void> copy_region(AllocationHandle src, size_t src_offset, AllocationHandle dst,
                           size_t dst_offset, size_t bytes) override {
    if (closed_) {
      return Status{ErrorCode::SessionClosed, "Backend closed"};
    }
    auto a = buffers_.find(src), b = buffers_.find(dst);
    if (a == buffers_.end() || b == buffers_.end() || src_offset > a->second.size() ||
        dst_offset > b->second.size() || bytes > a->second.size() - src_offset ||
        bytes > b->second.size() - dst_offset) {
      return Status{ErrorCode::InvalidArgument, "Region copy outside allocation"};
    }
    std::memmove(b->second.data() + dst_offset, a->second.data() + src_offset, bytes);
    return {};
  }

  Result<void> fill(AllocationHandle handle, std::byte value) override {
    if (closed_) {
      return Status{ErrorCode::SessionClosed, "Backend closed"};
    }
    if (!buffers_.contains(handle)) {
      return Status{ErrorCode::InvalidArgument, "Invalid allocation"};
    }
    std::fill(buffers_.at(handle).begin(), buffers_.at(handle).end(), value);
    return {};
  }

  Result<void> shutdown() override {
    closed_ = true;
    std::vector<std::byte>().swap(workspace_);
    return {};
  }

private:
  Result<void> validate(AllocationHandle handle, size_t size) const {
    if (closed_) {
      return Status{ErrorCode::SessionClosed, "Backend closed"};
    }
    auto it = buffers_.find(handle);
    if (it == buffers_.end() || it->second.size() != size) {
      return Status{ErrorCode::InvalidArgument, "Invalid allocation or byte count"};
    }
    return {};
  }

  size_t capacity_, scratchpad_, used_ = 0;
  std::vector<std::byte> workspace_;
  AllocationHandle next_ = 1;
  bool closed_ = false;
  std::map<AllocationHandle, std::vector<std::byte>> buffers_;
};
} // namespace

std::unique_ptr<IBackend> make_functional_backend(size_t capacity, size_t scratchpad) {
  return std::make_unique<FunctionalBackend>(capacity, scratchpad);
}
} // namespace ndp
