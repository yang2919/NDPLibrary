#include "ndp/backend/backend.hpp"
#include <algorithm>
#include <map>
#include <new>

namespace ndp {
namespace {
class FunctionalBackend final : public IBackend {
 public:
  explicit FunctionalBackend(size_t capacity) : capacity_(capacity) {}
  DeviceCaps capabilities() const override { return {"functional", capacity_}; }
  Result<AllocationHandle> allocate(size_t bytes) override {
    if (closed_) return Status{ErrorCode::SessionClosed, "Backend closed"};
    if (bytes == 0) return Status{ErrorCode::InvalidArgument, "Zero-byte allocation"};
    if (bytes > capacity_ - used_) return Status{ErrorCode::OutOfMemory, "Device capacity exceeded"};
    try {
      buffers_.emplace(next_, std::vector<std::byte>(bytes));
    } catch (const std::bad_alloc&) {
      return Status{ErrorCode::OutOfMemory, "Host backing allocation failed"};
    } catch (const std::length_error&) {
      return Status{ErrorCode::OutOfMemory, "Backing allocation too large"};
    }
    used_ += bytes;
    return next_++;
  }
  void release(AllocationHandle handle) noexcept override {
    auto it = buffers_.find(handle);
    if (it != buffers_.end()) { used_ -= it->second.size(); buffers_.erase(it); }
  }
  Result<void> upload(AllocationHandle handle, std::span<const std::byte> src) override {
    if (auto s = validate(handle, src.size()); !s) return s;
    std::copy(src.begin(), src.end(), buffers_.at(handle).begin());
    return {};
  }
  Result<void> download(AllocationHandle handle, std::span<std::byte> dst) override {
    if (auto s = validate(handle, dst.size()); !s) return s;
    std::copy(buffers_.at(handle).begin(), buffers_.at(handle).end(), dst.begin());
    return {};
  }
  Result<void> shutdown() override { closed_ = true; return {}; }
 private:
  Result<void> validate(AllocationHandle handle, size_t size) const {
    if (closed_) return Status{ErrorCode::SessionClosed, "Backend closed"};
    auto it = buffers_.find(handle);
    if (it == buffers_.end() || it->second.size() != size)
      return Status{ErrorCode::InvalidArgument, "Invalid allocation or byte count"};
    return {};
  }
  size_t capacity_, used_ = 0;
  AllocationHandle next_ = 1;
  bool closed_ = false;
  std::map<AllocationHandle, std::vector<std::byte>> buffers_;
};
}
std::unique_ptr<IBackend> make_functional_backend(size_t capacity) {
  return std::make_unique<FunctionalBackend>(capacity);
}
}
