// Inject a delayed worker failure only in explicitly enabled adapter test builds.
#include "../test_support.hpp"

#include <chrono>
#include <thread>

namespace ndp::python {
namespace {
class FailingBackend final : public reference::ForwardBackend {
public:
  Result<void> upload(AllocationHandle, std::span<const std::byte>) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    return Status{ErrorCode::BackendFailure, "Injected upload failure"};
  }
};
} // namespace

std::unique_ptr<IBackend> make_failure_backend() {
  return std::make_unique<FailingBackend>();
}
} // namespace ndp::python
