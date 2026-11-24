// Adapt the existing synchronous emulator without changing its numeric behavior.
#include "transport.hpp"

#include "emulator.hpp"

#include <new>

namespace ndp::detail::command {
namespace {
Status closed() {
  return {ErrorCode::SessionClosed, "Command transport closed"};
}

class EmulatorTransport final : public ICommandTransport {
public:
  explicit EmulatorTransport(std::unique_ptr<Emulator> device) : device_(std::move(device)) {}

  Result<void> upload(Address address, std::span<const std::byte> data) override {
    return device_ ? device_->upload(address, data) : Result<void>(closed());
  }

  Result<void> download(Address address, std::span<std::byte> data) override {
    return device_ ? device_->download(address, data) : Result<void>(closed());
  }

  Result<void> submit(std::span<const Command> commands) override {
    return device_ ? device_->submit(commands) : Result<void>(closed());
  }

  Result<void> shutdown() override {
    device_.reset();
    return {};
  }

private:
  std::unique_ptr<Emulator> device_;
};
} // namespace

Result<std::unique_ptr<ICommandTransport>> make_emulator_transport(const NdpSpec &spec) {
  auto device = Emulator::create(spec);
  if (!device) {
    return device.error();
  }
  try {
    return std::unique_ptr<ICommandTransport>(new EmulatorTransport(std::move(device.value())));
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Command transport allocation failed"};
  }
}
} // namespace ndp::detail::command
