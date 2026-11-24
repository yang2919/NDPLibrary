// Separate synchronous command delivery from allocation and command planning.
#pragma once
#include "command.hpp"
#include "ndp/l1/ndp/backend.hpp"

#include <memory>
#include <span>

namespace ndp::detail::command {
class ICommandTransport {
public:
  virtual ~ICommandTransport() = default;
  virtual Result<void> upload(Address, std::span<const std::byte>) = 0;
  virtual Result<void> download(Address, std::span<std::byte>) = 0;
  // Success means execution and writes have completed, not merely receipt.
  // On any return, caller buffers are no longer accessed. Calls are serialized
  // by the owner; transports do not introduce a worker or a scheduling queue.
  virtual Result<void> submit(std::span<const Command>) = 0;
  // Idempotent; quiesce work and release device state before returning.
  virtual Result<void> shutdown() = 0;
};

// The caller must validate geometry before constructing the shared planner.
Result<std::unique_ptr<IBackend>> make_command_backend(const NdpSpec &,
                                                       std::unique_ptr<ICommandTransport>);

Result<std::unique_ptr<ICommandTransport>> make_emulator_transport(const NdpSpec &);
} // namespace ndp::detail::command
