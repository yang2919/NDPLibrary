// Execute semantic commands against shared memory, GPR, GB, and MAC state.
#pragma once
#include "command.hpp"
#include "ndp/l1/ndp/backend.hpp"
#include "program.hpp"

#include <memory>
#include <span>

namespace ndp::detail::command {
// A functional mapping, not vendor physical addressing: [row][channel][bank][byte].
using Profile = NdpSpec;

struct LaunchState {
  std::shared_ptr<const LoadedProgram> program;
  uint64_t pc, sp, ra, a0, a1, instruction_limit;
};

class Emulator {
public:
  static Result<std::unique_ptr<Emulator>> create(const Profile &);
  Result<void> upload(Address, std::span<const std::byte>);
  Result<void> download(Address, std::span<std::byte>) const;
  Result<void> submit(std::span<const Command>);
  // Synchronous control operations on the same single-owner device instance.
  Result<uint64_t> load_program(std::span<const std::byte>, const ProgramRequirements &);
  Result<void> unload_program(uint64_t);
  Result<LaunchState> prepare_launch(const LaunchPnmProgram &) const;
  Result<uint64_t> bank_address(size_t channel, size_t bank, size_t row, size_t byte_offset) const;

private:
  explicit Emulator(const Profile &, size_t shared_bytes);
  Result<void> validate(const Command &) const;
  Result<void> execute(const Command &);
  bool range(Address, size_t bytes) const;
  bool mask(uint64_t, size_t width) const;
  std::vector<std::byte> *memory(Space);
  const std::vector<std::byte> *memory(Space) const;
  Result<void> validate_pnm(const Command &) const;
  Result<void> execute_pnm(const Command &);
  Profile profile_;
  std::vector<std::byte> shared_, gpr_, scratchpad_, gb_;
  std::vector<bool> gb_initialized_, mac_initialized_;
  std::vector<float> mac_;
  ProgramStore programs_;
};
} // namespace ndp::detail::command
