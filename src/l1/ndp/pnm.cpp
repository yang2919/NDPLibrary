// Execute functional PNM commands directly on device storage, without tensor metadata or binaries.
#include "emulator.hpp"
#include "ndp/common/fp16.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

namespace ndp::detail::command {
namespace {
bool product(size_t a, size_t b, size_t &out) {
  if (!a || !b || b > SIZE_MAX / a) {
    return false;
  }
  out = a * b;
  return true;
}

bool overlaps(Address a, size_t na, Address b, size_t nb) {
  return a.space == b.space && a.offset < b.offset + nb && b.offset < a.offset + na;
}

Status invalid() {
  return {ErrorCode::InvalidArgument, "Invalid PNM command operands"};
}
} // namespace

Result<void> Emulator::validate_pnm(const Command &command) const {
  auto operand = [&](Operand o) {
    return o.elements <= SIZE_MAX / 2 && o.address.offset % 2 == 0 &&
           range(o.address, o.elements * 2);
  };
  return std::visit(
      [&](const auto &cmd) -> Result<void> {
        using T = std::decay_t<decltype(cmd)>;
        if constexpr (std::is_same_v<T, CopyBytes>) {
          if (!range(cmd.source, cmd.bytes) || !range(cmd.destination, cmd.bytes)) {
            return invalid();
          }
        } else if constexpr (std::is_same_v<T, FillBytes>) {
          if (!range(cmd.destination, cmd.bytes)) {
            return invalid();
          }
        } else if constexpr (std::is_same_v<T, PnmRelayout>) {
          size_t dense, packed, stripe, tiles, groups;
          if (!product(cmd.rows, cmd.columns, dense) || dense > SIZE_MAX / 2 ||
              !cmd.tile_elements || cmd.row_bytes % 32 || cmd.tile_elements > cmd.row_bytes / 2 ||
              !product(cmd.banks, cmd.row_bytes, stripe)) {
            return invalid();
          }
          tiles = (cmd.columns - 1) / cmd.tile_elements + 1;
          groups = (cmd.rows - 1) / cmd.banks + 1;
          if (!product(tiles, groups, packed) || !product(packed, stripe, packed)) {
            return invalid();
          }
          size_t source = cmd.pack ? dense * 2 : packed, dest = cmd.pack ? packed : dense * 2;
          if (cmd.source.offset % 2 || cmd.destination.offset % 2 || !range(cmd.source, source) ||
              !range(cmd.destination, dest) ||
              overlaps(cmd.source, source, cmd.destination, dest)) {
            return invalid();
          }
        } else if constexpr (std::is_same_v<T, PnmCompute>) {
          if (!operand(cmd.output) || !std::isfinite(cmd.factor) || !std::isfinite(cmd.epsilon) ||
              cmd.epsilon <= 0) {
            return invalid();
          }
          for (auto in : cmd.inputs) {
            if (!operand(in) || overlaps(in.address, in.elements * 2, cmd.output.address,
                                         cmd.output.elements * 2)) {
              return invalid();
            }
          }
          size_t count;
          if (!product(cmd.rows, cmd.width, count) || count != cmd.output.elements) {
            return invalid();
          }
          auto inputs = cmd.inputs.size();
          switch (cmd.kernel) {
          case PnmKernel::Matmul: {
            size_t a, b;
            if ((inputs != 2 && inputs != 3) || !product(cmd.rows, cmd.inner, a) ||
                !product(cmd.inner, cmd.width, b) || cmd.inputs[0].elements != a ||
                cmd.inputs[1].elements != b ||
                (inputs == 3 && cmd.inputs[2].elements != cmd.width) || scratchpad_.size() < 4) {
              return invalid();
            }
            // The first four scratchpad bytes are private operand staging slots.
            if (cmd.output.address.space == Space::Scratchpad) {
              return invalid();
            }
            for (auto input : cmd.inputs) {
              if (input.address.space == Space::Scratchpad) {
                return invalid();
              }
            }
            break;
          }
          case PnmKernel::Add:
          case PnmKernel::Mul:
            if (inputs != 2 || cmd.inputs[0].elements != count ||
                (cmd.inputs[1].elements != count && cmd.inputs[1].elements != cmd.width)) {
              return invalid();
            }
            break;
          case PnmKernel::RMSNorm:
            if (inputs != 2 || cmd.inputs[0].elements != count ||
                cmd.inputs[1].elements != cmd.width) {
              return invalid();
            }
            break;
          case PnmKernel::Gather:
            if (inputs != 1 || cmd.indices.size() != count) {
              return invalid();
            }
            for (auto i : cmd.indices) {
              if (i >= cmd.inputs[0].elements) {
                return invalid();
              }
            }
            break;
          case PnmKernel::Copy:
          case PnmKernel::Scale:
          case PnmKernel::SiLU:
          case PnmKernel::Softmax:
          case PnmKernel::Transpose:
            if (inputs != 1 || cmd.inputs[0].elements != count) {
              return invalid();
            }
            break;
          default:
            return invalid();
          }
        } else {
          return Status{ErrorCode::UnsupportedOperation, "No PNM binary executor"};
        }
        return {};
      },
      command);
}

Result<void> Emulator::execute_pnm(const Command &command) {
  auto ptr = [&](Address a) { return memory(a.space)->data() + a.offset; };
  auto read = [&](Address a, size_t i) {
    auto *p = ptr(a) + i * 2;
    return fp16_to_float(
        uint16_t(std::to_integer<unsigned>(p[0]) | (std::to_integer<unsigned>(p[1]) << 8)));
  };
  auto write = [&](Address a, size_t i, float value) {
    auto h = float_to_fp16(value);
    auto *p = ptr(a) + i * 2;
    p[0] = std::byte(h & 255);
    p[1] = std::byte(h >> 8);
  };
  return std::visit(
      [&](const auto &cmd) -> Result<void> {
        using T = std::decay_t<decltype(cmd)>;
        if constexpr (std::is_same_v<T, CopyBytes>) {
          std::memmove(ptr(cmd.destination), ptr(cmd.source), cmd.bytes);
        } else if constexpr (std::is_same_v<T, FillBytes>) {
          std::fill_n(ptr(cmd.destination), cmd.bytes, cmd.value);
        } else if constexpr (std::is_same_v<T, PnmRelayout>) {
          size_t tiles = (cmd.columns - 1) / cmd.tile_elements + 1,
                 groups = (cmd.rows - 1) / cmd.banks + 1;
          if (cmd.pack) {
            std::fill_n(ptr(cmd.destination), groups * tiles * cmd.banks * cmd.row_bytes,
                        std::byte{0});
          }
          for (size_t r = 0; r < cmd.rows; ++r) {
            for (size_t c = 0; c < cmd.columns; ++c) {
              size_t dense = (r * cmd.columns + c) * 2;
              size_t packed =
                  ((r / cmd.banks) * tiles + c / cmd.tile_elements) * cmd.banks * cmd.row_bytes +
                  (r % cmd.banks) * cmd.row_bytes + (c % cmd.tile_elements) * 2;
              std::memcpy(ptr(cmd.destination) + (cmd.pack ? packed : dense),
                          ptr(cmd.source) + (cmd.pack ? dense : packed), 2);
            }
          }
        } else if constexpr (std::is_same_v<T, PnmCompute>) {
          auto get = [&](size_t operand, size_t i) { return read(cmd.inputs[operand].address, i); };
          auto put = [&](size_t i, float value) { write(cmd.output.address, i, value); };
          size_t count = cmd.output.elements, width = cmd.width;
          switch (cmd.kernel) {
          case PnmKernel::Matmul:
            for (size_t r = 0; r < cmd.rows; ++r) {
              for (size_t c = 0; c < width; ++c) {
                float sum = 0;
                for (size_t k = 0; k < cmd.inner; ++k) {
                  // Two FP16 operand slots provide a bounded scratchpad model.
                  std::memcpy(scratchpad_.data(),
                              ptr(cmd.inputs[0].address) + (r * cmd.inner + k) * 2, 2);
                  std::memcpy(scratchpad_.data() + 2,
                              ptr(cmd.inputs[1].address) +
                                  (cmd.transpose_b ? c * cmd.inner + k : k * width + c) * 2,
                              2);
                  sum = std::fma(read({Space::Scratchpad, 0}, 0), read({Space::Scratchpad, 0}, 1),
                                 sum);
                }
                sum *= cmd.factor;
                if (cmd.inputs.size() == 3) {
                  sum += get(2, c);
                }
                put(r * width + c, sum);
              }
            }
            break;
          case PnmKernel::Add:
          case PnmKernel::Mul:
            for (size_t i = 0; i < count; ++i) {
              put(i, cmd.kernel == PnmKernel::Add ? get(0, i) + get(1, i % cmd.inputs[1].elements)
                                                  : get(0, i) * get(1, i % cmd.inputs[1].elements));
            }
            break;
          case PnmKernel::Copy:
            std::memcpy(ptr(cmd.output.address), ptr(cmd.inputs[0].address), count * 2);
            break;
          case PnmKernel::Gather:
            for (size_t i = 0; i < count; ++i) {
              std::memcpy(ptr(cmd.output.address) + i * 2,
                          ptr(cmd.inputs[0].address) + cmd.indices[i] * 2, 2);
            }
            break;
          case PnmKernel::Transpose:
            for (size_t r = 0; r < cmd.rows; ++r) {
              for (size_t c = 0; c < width; ++c) {
                std::memcpy(ptr(cmd.output.address) + (c * cmd.rows + r) * 2,
                            ptr(cmd.inputs[0].address) + (r * width + c) * 2, 2);
              }
            }
            break;
          case PnmKernel::Scale:
            for (size_t i = 0; i < count; ++i) {
              put(i, get(0, i) * cmd.factor);
            }
            break;
          case PnmKernel::SiLU:
            for (size_t i = 0; i < count; ++i) {
              float x = get(0, i);
              put(i, x == -INFINITY ? -0.f
                     : x >= 0       ? x / (1 + std::exp(-x))
                                    : x * std::exp(x) / (1 + std::exp(x)));
            }
            break;
          case PnmKernel::RMSNorm:
            for (size_t base = 0; base < count; base += width) {
              float sum = 0;
              for (size_t d = 0; d < width; ++d) {
                sum = std::fma(get(0, base + d), get(0, base + d), sum);
              }
              float inv = 1 / std::sqrt(sum / float(width) + cmd.epsilon);
              for (size_t d = 0; d < width; ++d) {
                put(base + d, get(0, base + d) * inv * get(1, d));
              }
            }
            break;
          case PnmKernel::Softmax:
            for (size_t base = 0; base < count; base += width) {
              float maximum = -INFINITY, total = 0;
              size_t inf = 0;
              bool nan = false;
              for (size_t d = 0; d < width; ++d) {
                float x = get(0, base + d);
                nan |= std::isnan(x);
                inf += x == INFINITY;
                maximum = std::max(maximum, x);
              }
              if (!nan && !inf && maximum != -INFINITY) {
                for (size_t d = 0; d < width; ++d) {
                  total += std::exp(get(0, base + d) - maximum);
                }
              }
              for (size_t d = 0; d < width; ++d) {
                float x = get(0, base + d);
                put(base + d, nan                    ? NAN
                              : inf                  ? (x == INFINITY ? 1.f / inf : 0)
                              : maximum == -INFINITY ? 0
                                                     : std::exp(x - maximum) / total);
              }
            }
            break;
          }
        } else {
          return Status{ErrorCode::UnsupportedOperation, "No PNM binary executor"};
        }
        return {};
      },
      command);
}
} // namespace ndp::detail::command
