// Probe command delivery over a Unix stream without claiming device execution.
#pragma once
#include "../ndp/codec.hpp"
#include "ndp/l1/ndp/backend.hpp"

#include <chrono>

namespace ndp::detail::metadata {
enum class Kind : uint16_t {
  Request = 1,
  Accepted = 2,
  Rejected = 3,
  ExecuteRequest = 4,
  ExecuteComplete = 5,
  ExecuteError = 6
};

struct Frame {
  Kind kind;
  uint64_t epoch, sequence;
  std::vector<std::byte> payload;
};

struct IoOptions {
  std::chrono::milliseconds timeout{5000};
  size_t chunk_bytes = 4096;
};

// Descriptors are borrowed. Each frame has one deadline covering all partial I/O.
Result<void> send_frame(int fd, const Frame &, IoOptions = {});
Result<Frame> receive_frame(int fd, IoOptions = {});
Result<void> validate_metadata(std::span<const command::Command>, const NdpSpec &);
Result<void> probe(int fd, uint64_t epoch, uint64_t sequence, std::span<const std::byte> batch,
                   IoOptions = {});
// Serve exactly count requests on one connection. Rejected requests are never executed.
Result<void> serve(int fd, uint64_t epoch, size_t count, const NdpSpec &, IoOptions = {});
} // namespace ndp::detail::metadata
