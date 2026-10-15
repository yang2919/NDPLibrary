// Define the blocking device backend contract for memory, layouts, execution, and completion.
#pragma once
#include "ndp/common/compute.hpp"

#include <memory>
#include <span>

namespace ndp {
using AllocationHandle = uint64_t;

struct DeviceCaps {
  std::string name;
  size_t memory_capacity_bytes;
  size_t scratchpad_bytes = 0;
};

// Blocking transport adapter, invoked by the S2 FIFO worker.
// Native asynchronous hardware must wait for quiescence inside this adapter.
// Every method returns (including errors/exceptions) only after device accesses
// have stopped. An uncertain completion must not return through this interface.
// shutdown stops new I/O; release must remain valid for surviving tensor handles.
class IBackend {
public:
  virtual ~IBackend() = default;
  virtual DeviceCaps capabilities() const = 0;
  virtual Result<AllocationHandle> allocate(size_t bytes) = 0;
  virtual void release(AllocationHandle) noexcept = 0;
  virtual Result<void> upload(AllocationHandle, std::span<const std::byte>) = 0;
  virtual Result<void> download(AllocationHandle, std::span<std::byte>) = 0;

  // Metadata-only, thread-safe queries: may run while a transfer is executing.
  virtual Result<LayoutDesc> query_layout(const TensorDesc &desc, LayoutKind kind) const {
    if (kind == LayoutKind::DenseRowMajor) {
      return dense_layout(desc);
    }
    return Status{ErrorCode::UnsupportedLayout, "Backend does not support this layout"};
  }

  virtual Result<void> transform(AllocationHandle, const LayoutDesc &, AllocationHandle,
                                 const LayoutDesc &) {
    return Status{ErrorCode::UnsupportedLayout, "Backend has no layout transform"};
  }

  virtual Result<KernelPlan> query_operation(const OperationSpec &) const {
    return Status{ErrorCode::UnsupportedOperation, "Backend has no compute kernels"};
  }

  // Separate from DRAM allocations. One persistent arena per session/backend.
  virtual Result<void> reserve_workspace(size_t bytes) {
    if (!bytes) {
      return {};
    }
    return Status{ErrorCode::UnsupportedOperation, "Backend has no workspace"};
  }

  virtual Result<void> execute(const KernelPlan &, std::span<const AllocationHandle>,
                               AllocationHandle) {
    return Status{ErrorCode::UnsupportedOperation, "Backend has no compute kernels"};
  }

  virtual Result<void> copy_region(AllocationHandle, size_t, AllocationHandle, size_t, size_t) {
    return Status{ErrorCode::UnsupportedOperation, "Backend has no region copy"};
  }

  // Fill an entire allocation with a byte value; used to initialize/reset generic buffers.
  virtual Result<void> fill(AllocationHandle, std::byte) {
    return Status{ErrorCode::UnsupportedOperation, "Backend has no buffer fill"};
  }

  virtual Result<void> shutdown() = 0;
};

std::unique_ptr<IBackend> make_functional_backend(size_t capacity_bytes,
                                                  size_t scratchpad_bytes = 4096);
} // namespace ndp
