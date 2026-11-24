// Validate launch operands and construct guest register state without executing instructions.
#include "emulator.hpp"

namespace ndp::detail::command {
Result<uint64_t> Emulator::load_program(std::span<const std::byte> image,
                                        const ProgramRequirements &requirements) {
  return programs_.load(image, requirements);
}

Result<void> Emulator::unload_program(uint64_t handle) {
  return programs_.unload(handle);
}

Result<LaunchState> Emulator::prepare_launch(const LaunchPnmProgram &cmd) const {
  auto program = programs_.get(cmd.program);
  if (!program) {
    return Status{ErrorCode::InvalidArgument, "Unknown PNM program handle"};
  }
  constexpr uint64_t window = 0x40000000;
  if (shared_.size() > window || gpr_.size() > window || scratchpad_.size() > window) {
    return Status{ErrorCode::UnsupportedOperation, "PNM data exceeds initial guest windows"};
  }
  if ((cmd.arguments.space != Space::Shared && cmd.arguments.space != Space::Gpr) ||
      cmd.arguments.offset % 8 || cmd.argument_bytes < 8 || cmd.argument_bytes > UINT32_MAX ||
      !range(cmd.arguments, cmd.argument_bytes) || cmd.stack.space != Space::Scratchpad ||
      cmd.stack.offset % 16 || cmd.stack_bytes % 16 || !range(cmd.stack, cmd.stack_bytes) ||
      !cmd.instruction_limit) {
    return Status{ErrorCode::InvalidArgument, "Invalid PNM arguments, stack, or execution limit"};
  }
  const auto &bytes = *memory(cmd.arguments.space);
  auto read32 = [&](size_t offset) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) {
      value |= uint32_t(std::to_integer<unsigned>(bytes[offset + i])) << (8 * i);
    }
    return value;
  };
  if (read32(cmd.arguments.offset) != 1 || read32(cmd.arguments.offset + 4) != cmd.argument_bytes) {
    return Status{ErrorCode::InvalidArgument, "Invalid argument ABI version or block size"};
  }
  uint64_t base = cmd.arguments.space == Space::Shared ? 0x40000000 : 0x80000000;
  return LaunchState{program,
                     program->entry,
                     0xc0000000 + cmd.stack.offset + cmd.stack_bytes,
                     0x1ffff000,
                     base + cmd.arguments.offset,
                     cmd.argument_bytes,
                     cmd.instruction_limit};
}
} // namespace ndp::detail::command
