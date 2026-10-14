#pragma once
#include "ndp/types.hpp"
#include <memory>
#include <span>

namespace ndp {
using AllocationHandle = uint64_t;
struct DeviceCaps { std::string name; size_t memory_capacity_bytes; };
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
  virtual Result<void> shutdown() = 0;
};
std::unique_ptr<IBackend> make_functional_backend(size_t capacity_bytes);
} // namespace ndp
