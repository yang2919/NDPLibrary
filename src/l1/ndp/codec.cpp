// Serialize scalar fields explicitly; never expose native object representations.
#include "codec.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace ndp::detail::command::wire {
namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(sizeof(size_t) <= sizeof(uint64_t));
constexpr uint32_t magic = 0x3150444e; // ASCII NDP1

struct Invalid {
  const char *message;
};

void require(bool valid, const char *message) {
  if (!valid) {
    throw Invalid{message};
  }
}

uint8_t space_id(Space space) {
  switch (space) {
  case Space::Shared:
    return 1;
  case Space::Gpr:
    return 2;
  case Space::Scratchpad:
    return 3;
  case Space::GlobalBuffer:
    return 4;
  }
  throw Invalid{"Unknown address space"};
}

Space space_value(uint8_t id) {
  switch (id) {
  case 1:
    return Space::Shared;
  case 2:
    return Space::Gpr;
  case 3:
    return Space::Scratchpad;
  case 4:
    return Space::GlobalBuffer;
  default:
    throw Invalid{"Unknown wire address space"};
  }
}

uint16_t kernel_id(PnmKernel kernel) {
  switch (kernel) {
  case PnmKernel::Matmul:
    return 1;
  case PnmKernel::Add:
    return 2;
  case PnmKernel::Mul:
    return 3;
  case PnmKernel::RMSNorm:
    return 4;
  case PnmKernel::Softmax:
    return 5;
  case PnmKernel::SiLU:
    return 6;
  case PnmKernel::Copy:
    return 7;
  case PnmKernel::Gather:
    return 8;
  case PnmKernel::Scale:
    return 9;
  case PnmKernel::Transpose:
    return 10;
  }
  throw Invalid{"Unknown PNM kernel"};
}

PnmKernel kernel_value(uint16_t id) {
  switch (id) {
  case 1:
    return PnmKernel::Matmul;
  case 2:
    return PnmKernel::Add;
  case 3:
    return PnmKernel::Mul;
  case 4:
    return PnmKernel::RMSNorm;
  case 5:
    return PnmKernel::Softmax;
  case 6:
    return PnmKernel::SiLU;
  case 7:
    return PnmKernel::Copy;
  case 8:
    return PnmKernel::Gather;
  case 9:
    return PnmKernel::Scale;
  case 10:
    return PnmKernel::Transpose;
  default:
    throw Invalid{"Unknown wire PNM kernel"};
  }
}

class Writer {
public:
  explicit Writer(size_t limit) : limit_(limit) {}

  void integer(uint64_t value, size_t width) {
    require(width <= limit_ - bytes.size(), "Encoded batch exceeds byte limit");
    for (size_t i = 0; i < width; ++i) {
      bytes.push_back(std::byte((value >> (8 * i)) & 255));
    }
  }

  void patch32(size_t offset, size_t value) {
    require(value <= UINT32_MAX, "Wire length overflow");
    for (size_t i = 0; i < 4; ++i) {
      bytes[offset + i] = std::byte((value >> (8 * i)) & 255);
    }
  }

  void u64(uint64_t value) {
    integer(value, 8);
  }

  void size(size_t value) {
    u64(value);
  }

  void boolean(bool value) {
    integer(value ? 1 : 0, 1);
  }

  void floating(float value) {
    integer(std::bit_cast<uint32_t>(value), 4);
  }

  void byte(std::byte value) {
    integer(std::to_integer<uint8_t>(value), 1);
  }

  void address(Address value) {
    integer(space_id(value.space), 1);
    u64(value.offset);
  }

  void operand(const Operand &value) {
    address(value.address);
    size(value.elements);
  }

  void kernel(PnmKernel value) {
    integer(kernel_id(value), 2);
  }

  void operands(const std::vector<Operand> &values) {
    count(values.size());
    for (const auto &value : values) {
      operand(value);
    }
  }

  void indices(const std::vector<uint64_t> &values) {
    count(values.size());
    for (auto value : values) {
      u64(value);
    }
  }

  void count(size_t value) {
    require(value <= UINT32_MAX, "Wire count overflow");
    integer(value, 4);
  }

  std::vector<std::byte> bytes;

private:
  size_t limit_;
};

class Reader {
public:
  explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

  uint64_t integer(size_t width) {
    require(width <= remaining(), "Truncated wire record");
    uint64_t result = 0;
    for (size_t i = 0; i < width; ++i) {
      result |= uint64_t(std::to_integer<uint8_t>(bytes_[position_++])) << (8 * i);
    }
    return result;
  }

  size_t remaining() const {
    return bytes_.size() - position_;
  }

  std::span<const std::byte> take(size_t bytes) {
    require(bytes <= remaining(), "Truncated command payload");
    auto result = bytes_.subspan(position_, bytes);
    position_ += bytes;
    return result;
  }

  void u64(uint64_t &value) {
    value = integer(8);
  }

  void size(size_t &value) {
    auto encoded = integer(8);
    require(encoded <= std::numeric_limits<size_t>::max(), "Size not representable on receiver");
    value = static_cast<size_t>(encoded);
  }

  void boolean(bool &value) {
    auto encoded = integer(1);
    require(encoded <= 1, "Invalid wire boolean");
    value = encoded != 0;
  }

  void floating(float &value) {
    value = std::bit_cast<float>(uint32_t(integer(4)));
  }

  void byte(std::byte &value) {
    value = std::byte(integer(1));
  }

  void address(Address &value) {
    value.space = space_value(integer(1));
    u64(value.offset);
  }

  void operand(Operand &value) {
    address(value.address);
    size(value.elements);
  }

  void kernel(PnmKernel &value) {
    value = kernel_value(integer(2));
  }

  void operands(std::vector<Operand> &values) {
    auto count = integer(4);
    require(count <= remaining() / 17, "Invalid operand count");
    values.resize(static_cast<size_t>(count));
    for (auto &value : values) {
      operand(value);
    }
  }

  void indices(std::vector<uint64_t> &values) {
    auto count = integer(4);
    require(count <= remaining() / 8, "Invalid index count");
    values.resize(static_cast<size_t>(count));
    for (auto &value : values) {
      u64(value);
    }
  }

private:
  std::span<const std::byte> bytes_;
  size_t position_ = 0;
};

// One field schema is shared by both directions; IDs are assigned independently
// of the in-memory variant. Adding a command requires an explicit schema and ID.
template <class Archive, class T> void fields(Archive &a, T &c) {
  using C = std::remove_cvref_t<T>;
  if constexpr (std::is_same_v<C, Transfer> || std::is_same_v<C, CopyBytes>) {
    a.address(c.source);
    a.address(c.destination);
    a.size(c.bytes);
  } else if constexpr (std::is_same_v<C, InitializeMac>) {
    a.u64(c.channels);
    a.u64(c.banks);
    a.size(c.gpr_offset);
  } else if constexpr (std::is_same_v<C, MacAllBanks>) {
    a.u64(c.channels);
    a.size(c.row);
    a.size(c.column);
    a.size(c.bursts);
  } else if constexpr (std::is_same_v<C, ReadMac>) {
    a.u64(c.channels);
    a.size(c.gpr_offset);
  } else if constexpr (std::is_same_v<C, LaunchPnmProgram>) {
    a.u64(c.program);
    a.address(c.arguments);
    a.size(c.argument_bytes);
    a.address(c.stack);
    a.size(c.stack_bytes);
    a.u64(c.instruction_limit);
  } else if constexpr (std::is_same_v<C, PnmCompute>) {
    a.kernel(c.kernel);
    a.operands(c.inputs);
    a.operand(c.output);
    a.size(c.rows);
    a.size(c.width);
    a.size(c.inner);
    a.boolean(c.transpose_b);
    a.floating(c.factor);
    a.floating(c.epsilon);
    a.indices(c.indices);
  } else if constexpr (std::is_same_v<C, PnmRelayout>) {
    a.address(c.source);
    a.address(c.destination);
    a.size(c.rows);
    a.size(c.columns);
    a.size(c.banks);
    a.size(c.tile_elements);
    a.size(c.row_bytes);
    a.boolean(c.pack);
  } else if constexpr (std::is_same_v<C, FillBytes>) {
    a.address(c.destination);
    a.size(c.bytes);
    a.byte(c.value);
  } else {
    static_assert(!sizeof(C), "Missing wire field schema");
  }
}

template <class T> constexpr Opcode opcode() {
  if constexpr (std::is_same_v<T, Transfer>) {
    return Opcode::Transfer;
  } else if constexpr (std::is_same_v<T, InitializeMac>) {
    return Opcode::InitializeMac;
  } else if constexpr (std::is_same_v<T, MacAllBanks>) {
    return Opcode::MacAllBanks;
  } else if constexpr (std::is_same_v<T, ReadMac>) {
    return Opcode::ReadMac;
  } else if constexpr (std::is_same_v<T, LaunchPnmProgram>) {
    return Opcode::LaunchPnmProgram;
  } else if constexpr (std::is_same_v<T, PnmCompute>) {
    return Opcode::PnmCompute;
  } else if constexpr (std::is_same_v<T, PnmRelayout>) {
    return Opcode::PnmRelayout;
  } else if constexpr (std::is_same_v<T, CopyBytes>) {
    return Opcode::CopyBytes;
  } else if constexpr (std::is_same_v<T, FillBytes>) {
    return Opcode::FillBytes;
  } else {
    static_assert(!sizeof(T), "Missing wire opcode");
  }
}

template <class T> Command read_command(Reader &r) {
  T command{};
  fields(r, command);
  return command;
}

Command read_command(Opcode op, Reader &r) {
  switch (op) {
  case Opcode::Transfer:
    return read_command<Transfer>(r);
  case Opcode::InitializeMac:
    return read_command<InitializeMac>(r);
  case Opcode::MacAllBanks:
    return read_command<MacAllBanks>(r);
  case Opcode::ReadMac:
    return read_command<ReadMac>(r);
  case Opcode::LaunchPnmProgram:
    return read_command<LaunchPnmProgram>(r);
  case Opcode::PnmCompute:
    return read_command<PnmCompute>(r);
  case Opcode::PnmRelayout:
    return read_command<PnmRelayout>(r);
  case Opcode::CopyBytes:
    return read_command<CopyBytes>(r);
  case Opcode::FillBytes:
    return read_command<FillBytes>(r);
  }
  throw Invalid{"Unknown wire opcode"};
}
} // namespace

Result<std::vector<std::byte>> encode(std::span<const Command> commands, Limits limits) {
  try {
    require(commands.size() <= limits.max_commands, "Command count exceeds limit");
    Writer writer(std::min(limits.max_bytes, size_t(UINT32_MAX)));
    writer.integer(magic, 4);
    writer.integer(version, 2);
    writer.integer(fp16_profile, 2);
    writer.integer(0, 4);
    writer.count(commands.size());
    for (const auto &command : commands) {
      require(!command.valueless_by_exception(), "Valueless command");
      std::visit(
          [&](const auto &c) {
            writer.integer(static_cast<uint16_t>(opcode<std::remove_cvref_t<decltype(c)>>()), 2);
            writer.integer(0, 2);
            size_t length_offset = writer.bytes.size();
            writer.integer(0, 4);
            size_t begin = writer.bytes.size();
            fields(writer, c);
            writer.patch32(length_offset, writer.bytes.size() - begin);
          },
          command);
    }
    writer.patch32(8, writer.bytes.size());
    return std::move(writer.bytes);
  } catch (const Invalid &e) {
    return Status{ErrorCode::InvalidArgument, e.message};
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Wire encoding allocation failed"};
  } catch (const std::length_error &) {
    return Status{ErrorCode::OutOfMemory, "Wire encoding exceeds container limits"};
  }
}

Result<std::vector<Command>> decode(std::span<const std::byte> bytes, Limits limits) {
  try {
    require(bytes.size() <= limits.max_bytes, "Batch exceeds byte limit");
    Reader reader(bytes);
    require(reader.integer(4) == magic, "Invalid batch magic");
    require(reader.integer(2) == version, "Unsupported wire version");
    require(reader.integer(2) == fp16_profile, "Unsupported numeric profile");
    require(reader.integer(4) == bytes.size(), "Invalid batch length");
    auto count = reader.integer(4);
    require(count <= limits.max_commands && count <= reader.remaining() / 8,
            "Invalid command count");
    std::vector<Command> commands;
    commands.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
      auto op = static_cast<Opcode>(reader.integer(2));
      require(reader.integer(2) == 0, "Nonzero reserved record flags");
      auto length = reader.integer(4);
      Reader payload(reader.take(static_cast<size_t>(length)));
      commands.push_back(read_command(op, payload));
      require(payload.remaining() == 0, "Trailing command payload");
    }
    require(reader.remaining() == 0, "Trailing batch bytes");
    return commands;
  } catch (const Invalid &e) {
    return Status{ErrorCode::InvalidArgument, e.message};
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Wire decoding allocation failed"};
  } catch (const std::length_error &) {
    return Status{ErrorCode::OutOfMemory, "Wire decoding exceeds container limits"};
  }
}
} // namespace ndp::detail::command::wire
