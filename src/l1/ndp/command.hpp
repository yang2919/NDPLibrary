// Define private semantic commands for the single-device AiM-derived extension.
#pragma once
#include "ndp/common/types.hpp"

#include <variant>

namespace ndp::detail::command {
enum class Space { Shared, Gpr, Scratchpad, GlobalBuffer };

struct Address {
  Space space;
  uint64_t offset = 0;
};

// All sizes are bytes except explicitly named burst counts and row/column indices.
struct Transfer {
  Address source, destination;
  size_t bytes;
};

struct InitializeMac {
  uint64_t channels, banks;
  size_t gpr_offset;
};

struct MacAllBanks {
  uint64_t channels;
  size_t row, column, bursts;
};

struct ReadMac {
  uint64_t channels;
  size_t gpr_offset;
};

// Arguments begin with little-endian uint32 ABI version (1) and total byte size.
struct LaunchPnmProgram {
  uint64_t program;
  Address arguments;
  size_t argument_bytes;
  Address stack;
  size_t stack_bytes;
  uint64_t instruction_limit;
};

enum class PnmKernel { Matmul, Add, Mul, RMSNorm, Softmax, SiLU, Copy, Gather, Scale, Transpose };

struct Operand {
  Address address;
  size_t elements;
};

struct PnmCompute {
  PnmKernel kernel;
  std::vector<Operand> inputs;
  Operand output;
  size_t rows = 1, width = 1, inner = 1;
  bool transpose_b = false;
  float factor = 1, epsilon = 1e-5f;
  std::vector<uint64_t> indices;
};

struct PnmRelayout {
  Address source, destination;
  size_t rows, columns, banks, tile_elements, row_bytes;
  bool pack;
};

struct CopyBytes {
  Address source, destination;
  size_t bytes;
};

struct FillBytes {
  Address destination;
  size_t bytes;
  std::byte value;
};

using Command = std::variant<Transfer, InitializeMac, MacAllBanks, ReadMac, LaunchPnmProgram,
                             PnmCompute, PnmRelayout, CopyBytes, FillBytes>;
} // namespace ndp::detail::command
