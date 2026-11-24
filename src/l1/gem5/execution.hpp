// Execute framed requests against one persistent emulator; used by the owned Session transport.
#pragma once
#include "metadata.hpp"

namespace ndp::detail::execution {
// The socket is borrowed. A failed exchange faults the client; never retry an
// uncertain request. The process owner must terminate/reap a failed receiver.
class Client {
public:
  static Result<std::unique_ptr<Client>> connect(int fd, uint64_t epoch, const NdpSpec &,
                                                 metadata::IoOptions = {});
  Result<void> upload(command::Address, std::span<const std::byte>);
  Result<void> download(command::Address, std::span<std::byte>);
  Result<void> submit(std::span<const command::Command>);
  Result<void> close();

  bool faulted() const noexcept {
    return faulted_;
  }

private:
  Client(int fd, uint64_t epoch, metadata::IoOptions options)
      : fd_(fd), epoch_(epoch), options_(options) {}

  Result<std::vector<std::byte>> exchange(std::vector<std::byte>, size_t expected_bytes);
  int fd_;
  uint64_t epoch_, sequence_ = 1;
  metadata::IoOptions options_;
  bool faulted_ = false, closed_ = false;
};

// Owns one emulator for the connection. A close acknowledgement follows teardown.
Result<void> serve(int fd, uint64_t epoch, const NdpSpec &, metadata::IoOptions = {},
                   bool wait_for_request = false);
} // namespace ndp::detail::execution
