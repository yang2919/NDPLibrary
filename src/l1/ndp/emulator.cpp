// Interpret ordered commands without reconstructing tensors or choosing tile schedules.
#include "emulator.hpp"

#include "ndp/common/fp16.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ndp::detail::command {
namespace {
Status invalid(const char *message) {
  return {ErrorCode::InvalidArgument, message};
}

float read_half(const std::vector<std::byte> &data, size_t offset) {
  auto bits = uint16_t(std::to_integer<unsigned>(data[offset]) |
                       (std::to_integer<unsigned>(data[offset + 1]) << 8));
  return fp16_to_float(bits);
}

void write_half(std::vector<std::byte> &data, size_t offset, float value) {
  auto bits = float_to_fp16(value);
  data[offset] = std::byte(bits & 255);
  data[offset + 1] = std::byte(bits >> 8);
}

// Volatile stores establish FP32 rounding points even on extended-precision hosts.
float add(float a, float b) {
  volatile float value = a + b;
  return value;
}

float multiply(float a, float b) {
  volatile float value = a * b;
  return value;
}

size_t selected(uint64_t mask) {
  return std::popcount(mask);
}
} // namespace

Emulator::Emulator(const Profile &p, size_t bytes)
    : profile_(p), shared_(bytes), gpr_(p.gpr_bytes), scratchpad_(p.scratchpad_bytes),
      gb_(p.channels * p.gb_bytes), gb_initialized_(gb_.size(), false),
      mac_initialized_(p.channels * p.banks, false), mac_(p.channels * p.banks) {}

Result<std::unique_ptr<Emulator>> Emulator::create(const Profile &p) {
  if (!p.channels || p.channels > 64 || !p.banks || p.banks > 64 || !p.rows || !p.row_bytes ||
      p.row_bytes % 32 || !p.gb_bytes || p.gb_bytes % 32 || !p.gpr_bytes || p.gpr_bytes % 2 ||
      !p.scratchpad_bytes) {
    return invalid("Invalid NDP memory geometry");
  }
  size_t bytes = p.channels;
  for (size_t factor : {p.banks, p.rows, p.row_bytes}) {
    if (factor > std::numeric_limits<size_t>::max() / bytes) {
      return invalid("Shared memory size overflow");
    }
    bytes *= factor;
  }
  if (p.gb_bytes > std::numeric_limits<size_t>::max() / p.channels) {
    return invalid("Global buffer size overflow");
  }
  try {
    return std::unique_ptr<Emulator>(new Emulator(p, bytes));
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Device state allocation failed"};
  } catch (const std::length_error &) {
    return Status{ErrorCode::OutOfMemory, "Device state exceeds container limits"};
  }
}

const std::vector<std::byte> *Emulator::memory(Space s) const {
  switch (s) {
  case Space::GlobalBuffer:
    return nullptr; // Only Transfer can write the channel-local GB.
  case Space::Shared:
    return &shared_;
  case Space::Gpr:
    return &gpr_;
  case Space::Scratchpad:
    return &scratchpad_;
  }
  return nullptr;
}

std::vector<std::byte> *Emulator::memory(Space s) {
  return const_cast<std::vector<std::byte> *>(std::as_const(*this).memory(s));
}

bool Emulator::range(Address a, size_t bytes) const {
  const auto *m = memory(a.space);
  return m && bytes && a.offset <= m->size() && bytes <= m->size() - a.offset;
}

bool Emulator::mask(uint64_t bits, size_t width) const {
  return bits && (width == 64 || (bits >> width) == 0);
}

Result<void> Emulator::upload(Address a, std::span<const std::byte> bytes) {
  if (!range(a, bytes.size())) {
    return invalid("Upload range invalid");
  }
  std::memcpy(memory(a.space)->data() + a.offset, bytes.data(), bytes.size());
  return {};
}

Result<void> Emulator::download(Address a, std::span<std::byte> bytes) const {
  if (!range(a, bytes.size())) {
    return invalid("Download range invalid");
  }
  std::memcpy(bytes.data(), memory(a.space)->data() + a.offset, bytes.size());
  return {};
}

Result<uint64_t> Emulator::bank_address(size_t c, size_t b, size_t r, size_t offset) const {
  if (c >= profile_.channels || b >= profile_.banks || r >= profile_.rows ||
      offset >= profile_.row_bytes) {
    return invalid("Bank coordinate out of range");
  }
  return uint64_t(((r * profile_.channels + c) * profile_.banks + b) * profile_.row_bytes + offset);
}

Result<void> Emulator::validate(const Command &command) const {
  return std::visit(
      [&](const auto &cmd) -> Result<void> {
        using T = std::decay_t<decltype(cmd)>;
        if constexpr (std::is_same_v<T, Transfer>) {
          if (cmd.destination.space == Space::GlobalBuffer) {
            const auto offset = cmd.destination.offset;
            if ((cmd.source.space != Space::Gpr && cmd.source.space != Space::Shared) ||
                !range(cmd.source, cmd.bytes) || cmd.source.offset % 32 || offset % 32 ||
                cmd.bytes % 32 || offset / profile_.gb_bytes >= profile_.channels ||
                cmd.bytes > profile_.gb_bytes - offset % profile_.gb_bytes) {
              return invalid("Invalid single-channel GB transfer");
            }
            return {};
          }
          bool route =
              (cmd.source.space == Space::Shared && (cmd.destination.space == Space::Gpr ||
                                                     cmd.destination.space == Space::Scratchpad)) ||
              (cmd.destination.space == Space::Shared &&
               (cmd.source.space == Space::Gpr || cmd.source.space == Space::Scratchpad));
          if (!route || !range(cmd.source, cmd.bytes) || !range(cmd.destination, cmd.bytes)) {
            return invalid("Unsupported transfer route or invalid range");
          }
        } else if constexpr (std::is_same_v<T, InitializeMac>) {
          if (!mask(cmd.channels, profile_.channels) || !mask(cmd.banks, profile_.banks) ||
              cmd.gpr_offset % 2 ||
              !range({Space::Gpr, cmd.gpr_offset},
                     selected(cmd.channels) * selected(cmd.banks) * 2)) {
            return invalid("Invalid MAC initialization operands");
          }
        } else if constexpr (std::is_same_v<T, MacAllBanks>) {
          size_t columns = profile_.row_bytes / 32;
          if (!mask(cmd.channels, profile_.channels) || cmd.row >= profile_.rows || !cmd.bursts ||
              cmd.column >= columns || cmd.bursts > columns - cmd.column ||
              cmd.column >= profile_.gb_bytes / 32 ||
              cmd.bursts > profile_.gb_bytes / 32 - cmd.column) {
            return invalid("MAC exceeds row or GB bounds");
          }
        } else if constexpr (std::is_same_v<T, ReadMac>) {
          if (!mask(cmd.channels, profile_.channels) || cmd.gpr_offset % 2 ||
              !range({Space::Gpr, cmd.gpr_offset}, selected(cmd.channels) * profile_.banks * 2)) {
            return invalid("Invalid MAC result destination");
          }
        } else if constexpr (std::is_same_v<T, LaunchPnmProgram>) {
          auto launch = prepare_launch(cmd);
          if (!launch) {
            return launch.error();
          }
          return Status{ErrorCode::UnsupportedOperation, "PNM binary executor is not connected"};
        } else {
          return validate_pnm(command);
        }
        return {};
      },
      command);
}

Result<void> Emulator::execute(const Command &command) {
  return std::visit(
      [&](const auto &cmd) -> Result<void> {
        using T = std::decay_t<decltype(cmd)>;
        if constexpr (std::is_same_v<T, Transfer>) {
          if (cmd.destination.space == Space::GlobalBuffer) {
            const size_t offset = cmd.destination.offset;
            std::memcpy(gb_.data() + offset, memory(cmd.source.space)->data() + cmd.source.offset,
                        cmd.bytes);
            std::fill_n(gb_initialized_.begin() + offset, cmd.bytes, true);
            return {};
          }
          std::memcpy(memory(cmd.destination.space)->data() + cmd.destination.offset,
                      memory(cmd.source.space)->data() + cmd.source.offset, cmd.bytes);
        } else if constexpr (std::is_same_v<T, InitializeMac>) {
          size_t source = cmd.gpr_offset;
          for (size_t c = 0; c < profile_.channels; ++c) {
            if (cmd.channels & (uint64_t{1} << c)) {
              for (size_t b = 0; b < profile_.banks; ++b) {
                if (cmd.banks & (uint64_t{1} << b)) {
                  size_t index = c * profile_.banks + b;
                  mac_[index] = read_half(gpr_, source);
                  mac_initialized_[index] = true;
                  source += 2;
                }
              }
            }
          }
        } else if constexpr (std::is_same_v<T, ReadMac> || std::is_same_v<T, MacAllBanks>) {
          // Validate all selected state before modifying any bank or result bytes.
          for (size_t c = 0; c < profile_.channels; ++c) {
            if (cmd.channels & (uint64_t{1} << c)) {
              for (size_t b = 0; b < profile_.banks; ++b) {
                if (!mac_initialized_[c * profile_.banks + b]) {
                  return Status{ErrorCode::InvalidState, "MAC register has not been initialized"};
                }
              }
              if constexpr (std::is_same_v<T, MacAllBanks>) {
                size_t offset = c * profile_.gb_bytes + cmd.column * 32;
                for (size_t i = 0; i < cmd.bursts * 32; ++i) {
                  if (!gb_initialized_[offset + i]) {
                    return Status{ErrorCode::InvalidState,
                                  "Global buffer range has not been loaded"};
                  }
                }
              }
            }
          }
          size_t destination = 0;
          if constexpr (std::is_same_v<T, ReadMac>) {
            destination = cmd.gpr_offset;
          }
          for (size_t c = 0; c < profile_.channels; ++c) {
            if (cmd.channels & (uint64_t{1} << c)) {
              for (size_t b = 0; b < profile_.banks; ++b) {
                size_t index = c * profile_.banks + b;
                if constexpr (std::is_same_v<T, ReadMac>) {
                  write_half(gpr_, destination, mac_[index]);
                  destination += 2;
                } else {
                  for (size_t burst = cmd.column; burst < cmd.column + cmd.bursts; ++burst) {
                    size_t weight = bank_address(c, b, cmd.row, burst * 32).value();
                    size_t vector = c * profile_.gb_bytes + burst * 32;
                    std::array<float, 16> tree;
                    for (size_t lane = 0; lane < 16; ++lane) {
                      tree[lane] = multiply(read_half(shared_, weight + lane * 2),
                                            read_half(gb_, vector + lane * 2));
                    }
                    for (size_t width = 16; width > 1; width /= 2) {
                      for (size_t i = 0; i < width / 2; ++i) {
                        tree[i] = add(tree[2 * i], tree[2 * i + 1]);
                      }
                    }
                    mac_[index] = add(mac_[index], tree[0]);
                  }
                }
              }
            }
          }
        } else {
          return execute_pnm(command);
        }
        return {};
      },
      command);
}

Result<void> Emulator::submit(std::span<const Command> commands) {
  // Do not silently use a caller's directed rounding mode for the numeric profile.
  if (std::fegetround() != FE_TONEAREST) {
    return Status{ErrorCode::InvalidState, "NDP FP16 profile requires round-to-nearest"};
  }
  for (const auto &cmd : commands) {
    auto valid = validate(cmd);
    if (!valid) {
      return valid;
    }
  }
  for (const auto &cmd : commands) {
    auto result = execute(cmd);
    if (!result) {
      return result;
    }
  }
  return {};
}
} // namespace ndp::detail::command
