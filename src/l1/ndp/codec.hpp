// Encode existing command batches as bounded, versioned little-endian records.
#pragma once
#include "command.hpp"

#include <span>

namespace ndp::detail::command::wire {
// These are wire IDs, independent of C++ variant ordering and enum values.
enum class Opcode : uint16_t {
  Transfer = 1,
  InitializeMac = 2,
  // Opcode 3 belonged to the v1 GB broadcast.
  MacAllBanks = 4,
  ReadMac = 5,
  LaunchPnmProgram = 6,
  PnmCompute = 7,
  PnmRelayout = 8,
  CopyBytes = 9,
  FillBytes = 10
};
inline constexpr uint16_t version = 2;
inline constexpr uint16_t fp16_profile = 1;
inline constexpr size_t header_bytes = 16;

struct Limits {
  size_t max_bytes = 8 * 1024 * 1024;
  size_t max_commands = 65536;
};

// Decoding validates structure only. Geometry, ranges, kernel semantics and
// program existence remain emulator/receiver checks. No execution occurs here.
Result<std::vector<std::byte>> encode(std::span<const Command>, Limits = {});
Result<std::vector<Command>> decode(std::span<const std::byte>, Limits = {});
} // namespace ndp::detail::command::wire
