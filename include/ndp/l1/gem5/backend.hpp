// Run the existing NDP planner through an owned guest software receiver.
#pragma once
#include "ndp/l1/ndp/backend.hpp"

#include <chrono>

namespace ndp {
struct Gem5BackendOptions {
  std::filesystem::path receiver_executable;
  NdpSpec spec;
  std::chrono::milliseconds timeout{30000};
};

// Runs inside Linux FS (or natively); does not launch gem5 or model device timing.
Result<std::unique_ptr<IBackend>> make_gem5_backend(const Gem5BackendOptions &);
} // namespace ndp
