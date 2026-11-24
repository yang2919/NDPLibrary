// Frame, validate and acknowledge metadata; no emulator is instantiated here.
#include "metadata.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <limits>
#include <new>
#include <poll.h>
#include <sys/socket.h>
#include <type_traits>

namespace ndp::detail::metadata {
namespace {
constexpr size_t header_size = 32;
constexpr size_t max_payload = 8 * 1024 * 1024;

Status invalid(const char *s) {
  return {ErrorCode::InvalidArgument, s};
}

Status io_error(const char *s) {
  return {ErrorCode::BackendFailure, s};
}

void put(std::span<std::byte> bytes, size_t at, uint64_t value, size_t width) {
  for (size_t i = 0; i < width; ++i) {
    bytes[at + i] = std::byte((value >> (8 * i)) & 255);
  }
}

uint64_t get(std::span<const std::byte> bytes, size_t at, size_t width) {
  uint64_t value = 0;
  for (size_t i = 0; i < width; ++i) {
    value |= uint64_t(std::to_integer<unsigned>(bytes[at + i])) << (8 * i);
  }
  return value;
}

uint32_t crc(std::span<const std::byte> bytes) {
  uint32_t value = ~uint32_t{0};
  for (auto b : bytes) {
    value ^= std::to_integer<unsigned>(b);
    for (unsigned i = 0; i < 8; ++i) {
      value = (value >> 1) ^ (0xedb88320u & (0u - (value & 1)));
    }
  }
  return ~value;
}

using Clock = std::chrono::steady_clock;

Result<void> transfer(int fd, std::span<std::byte> bytes, bool write, Clock::time_point deadline,
                      size_t chunk) {
#ifdef SO_NOSIGPIPE
  if (write) {
    int enabled = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled)) != 0) {
      return io_error("Cannot disable SIGPIPE");
    }
  }
#endif
  while (!bytes.empty()) {
    auto left =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
    if (left <= 0) {
      return io_error("Metadata I/O timeout");
    }
    pollfd event{fd, short(write ? POLLOUT : POLLIN), 0};
    int ready = poll(&event, 1, int(std::min<int64_t>(left, INT32_MAX)));
    if (ready < 0 && errno == EINTR) {
      continue;
    }
    if (ready < 0) {
      return io_error("Metadata poll failed");
    }
    if (ready == 0) {
      continue;
    }
    size_t n = std::min(bytes.size(), chunk);
    int flags = MSG_DONTWAIT;
#ifdef MSG_NOSIGNAL
    if (write) {
      flags |= MSG_NOSIGNAL;
    }
#endif
    auto done = write ? send(fd, bytes.data(), n, flags) : recv(fd, bytes.data(), n, flags);
    if (done < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
      continue;
    }
    if (done <= 0) {
      return io_error("Metadata peer closed or I/O failed");
    }
    bytes = bytes.subspan(static_cast<size_t>(done));
  }
  return {};
}

bool options_valid(IoOptions o) {
  return o.timeout.count() > 0 && o.timeout.count() <= INT32_MAX && o.chunk_bytes > 0;
}

bool multiply(uint64_t a, uint64_t b, uint64_t &out) {
  if (a && b > UINT64_MAX / a) {
    return false;
  }
  out = a * b;
  return true;
}
} // namespace

Result<void> send_frame(int fd, const Frame &frame, IoOptions options) {
  if (!options_valid(options) || frame.payload.size() > max_payload ||
      (frame.kind != Kind::Request && frame.kind != Kind::Accepted &&
       frame.kind != Kind::Rejected && frame.kind != Kind::ExecuteRequest &&
       frame.kind != Kind::ExecuteComplete && frame.kind != Kind::ExecuteError)) {
    return invalid("Invalid outgoing metadata frame/options");
  }
  std::array<std::byte, header_size> header{};
  put(header, 0, 0x3150494e, 4); // NIP1
  put(header, 4, 1, 2);
  put(header, 6, uint16_t(frame.kind), 2);
  put(header, 8, frame.epoch, 8);
  put(header, 16, frame.sequence, 8);
  put(header, 24, frame.payload.size(), 4);
  put(header, 28, crc(frame.payload), 4);
  auto deadline = Clock::now() + options.timeout;
  auto result = transfer(fd, header, true, deadline, options.chunk_bytes);
  if (!result) {
    return result;
  }
  // send() does not modify the payload; transfer shares read/write loop mechanics.
  return transfer(fd, {const_cast<std::byte *>(frame.payload.data()), frame.payload.size()}, true,
                  deadline, options.chunk_bytes);
}

Result<Frame> receive_frame(int fd, IoOptions options) {
  if (!options_valid(options)) {
    return invalid("Invalid metadata I/O options");
  }
  std::array<std::byte, header_size> header{};
  auto deadline = Clock::now() + options.timeout;
  auto read = transfer(fd, header, false, deadline, options.chunk_bytes);
  if (!read) {
    return read.error();
  }
  auto kind = get(header, 6, 2), size = get(header, 24, 4);
  if (get(header, 0, 4) != 0x3150494e || get(header, 4, 2) != 1 || kind < 1 || kind > 6 ||
      size > max_payload) {
    return invalid("Invalid metadata frame header");
  }
  try {
    Frame frame{Kind(kind), get(header, 8, 8), get(header, 16, 8), std::vector<std::byte>(size)};
    read = transfer(fd, frame.payload, false, deadline, options.chunk_bytes);
    if (!read) {
      return read.error();
    }
    if (crc(frame.payload) != get(header, 28, 4)) {
      return invalid("Metadata checksum mismatch");
    }
    return frame;
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Metadata frame allocation failed"};
  }
}

Result<void> validate_metadata(std::span<const command::Command> commands, const NdpSpec &p) {
  using namespace command;
  uint64_t shared = 0, stripes = 0;
  if (!p.channels || p.channels > 64 || !p.banks || p.banks > 64 || !p.rows || !p.row_bytes ||
      p.row_bytes % 32 || !p.gb_bytes || p.gb_bytes % 32 || !p.gpr_bytes || p.gpr_bytes % 2 ||
      !p.scratchpad_bytes || !multiply(p.channels, p.banks, stripes) ||
      !multiply(stripes, p.row_bytes, stripes) || !multiply(stripes, p.rows, shared)) {
    return invalid("Invalid metadata geometry");
  }
  auto range = [&](Address a, uint64_t bytes) {
    uint64_t capacity = a.space == Space::Shared       ? shared
                        : a.space == Space::Gpr        ? p.gpr_bytes
                        : a.space == Space::Scratchpad ? p.scratchpad_bytes
                                                       : 0;
    return capacity && a.offset <= capacity && bytes <= capacity - a.offset;
  };
  auto operand = [&](const Operand &o) {
    uint64_t bytes;
    return multiply(o.elements, 2, bytes) && o.address.offset % 2 == 0 && range(o.address, bytes);
  };
  auto mask = [](uint64_t m, size_t n) { return m && (n == 64 || (m >> n) == 0); };
  for (const auto &command : commands) {
    bool valid = std::visit(
        [&](const auto &c) {
          using C = std::decay_t<decltype(c)>;
          if constexpr (std::is_same_v<C, Transfer> || std::is_same_v<C, CopyBytes>) {
            if constexpr (std::is_same_v<C, Transfer>) {
              if (c.destination.space == Space::GlobalBuffer) {
                const auto offset = c.destination.offset;
                return (c.source.space == Space::Gpr || c.source.space == Space::Shared) &&
                       range(c.source, c.bytes) && c.bytes && c.source.offset % 32 == 0 &&
                       offset % 32 == 0 && c.bytes % 32 == 0 && offset / p.gb_bytes < p.channels &&
                       c.bytes <= p.gb_bytes - offset % p.gb_bytes;
              }
            }
            return range(c.source, c.bytes) && range(c.destination, c.bytes);
          } else if constexpr (std::is_same_v<C, FillBytes>) {
            return range(c.destination, c.bytes);
          } else if constexpr (std::is_same_v<C, InitializeMac>) {
            return mask(c.channels, p.channels) && mask(c.banks, p.banks) &&
                   c.gpr_offset % 2 == 0 &&
                   range({Space::Gpr, c.gpr_offset},
                         std::popcount(c.channels) * std::popcount(c.banks) * 2);
          } else if constexpr (std::is_same_v<C, MacAllBanks>) {
            return mask(c.channels, p.channels) && c.row < p.rows && c.column < p.row_bytes / 32 &&
                   c.bursts && c.bursts <= p.row_bytes / 32 - c.column &&
                   c.column < p.gb_bytes / 32 && c.bursts <= p.gb_bytes / 32 - c.column;
          } else if constexpr (std::is_same_v<C, ReadMac>) {
            return mask(c.channels, p.channels) && c.gpr_offset % 2 == 0 &&
                   range({Space::Gpr, c.gpr_offset}, std::popcount(c.channels) * p.banks * 2);
          } else if constexpr (std::is_same_v<C, LaunchPnmProgram>) {
            return range(c.arguments, c.argument_bytes) && range(c.stack, c.stack_bytes);
          } else if constexpr (std::is_same_v<C, PnmCompute>) {
            return operand(c.output) && std::all_of(c.inputs.begin(), c.inputs.end(), operand);
          } else if constexpr (std::is_same_v<C, PnmRelayout>) {
            uint64_t elements, dense, tiles, groups, packed;
            if (!c.banks || !c.tile_elements || !c.row_bytes ||
                !multiply(c.rows, c.columns, elements) || !multiply(elements, 2, dense)) {
              return false;
            }
            tiles = c.columns / c.tile_elements + (c.columns % c.tile_elements != 0);
            groups = c.rows / c.banks + (c.rows % c.banks != 0);
            if (!multiply(groups, tiles, packed) || !multiply(packed, c.banks, packed) ||
                !multiply(packed, c.row_bytes, packed)) {
              return false;
            }
            return range(c.source, c.pack ? dense : packed) &&
                   range(c.destination, c.pack ? packed : dense);
          }
        },
        command);
    if (!valid) {
      return invalid("Command metadata address/mask range rejected");
    }
  }
  return {};
}

Result<void> probe(int fd, uint64_t epoch, uint64_t sequence, std::span<const std::byte> batch,
                   IoOptions options) {
  if (batch.size() > max_payload) {
    return invalid("Metadata batch too large");
  }
  Frame frame{Kind::Request, epoch, sequence, {batch.begin(), batch.end()}};
  auto sent = send_frame(fd, frame, options);
  if (!sent) {
    return sent;
  }
  auto response = receive_frame(fd, options);
  if (!response) {
    return response.error();
  }
  const auto &r = response.value();
  if (r.epoch != epoch || r.sequence != sequence || r.kind != Kind::Accepted ||
      r.payload != frame.payload) {
    return invalid("Metadata rejected or acknowledgement mismatch");
  }
  return {};
}

Result<void> serve(int fd, uint64_t epoch, size_t count, const NdpSpec &spec, IoOptions options) {
  auto geometry = validate_metadata({}, spec);
  if (!geometry || !epoch) {
    return invalid("Invalid receiver configuration");
  }
  uint64_t expected = 1;
  for (size_t i = 0; i < count; ++i) {
    auto request = receive_frame(fd, options);
    if (!request) {
      return request.error();
    }
    auto frame = std::move(request.value());
    if (frame.kind != Kind::Request) {
      return invalid("Receiver expected request");
    }
    bool accepted = false;
    if (frame.epoch == epoch && frame.sequence == expected && expected != UINT64_MAX) {
      ++expected; // A well-framed request consumes its sequence even if metadata is rejected.
      auto commands = command::wire::decode(frame.payload);
      accepted = commands && bool(validate_metadata(commands.value(), spec));
      if (accepted) {
        auto encoded = command::wire::encode(commands.value());
        accepted = encoded && encoded.value() == frame.payload;
        if (accepted) {
          frame.payload = std::move(encoded.value());
        }
      }
    }
    frame.kind = accepted ? Kind::Accepted : Kind::Rejected;
    if (!accepted) {
      frame.payload.clear();
    }
    auto sent = send_frame(fd, frame, options);
    if (!sent) {
      return sent;
    }
  }
  return {};
}
} // namespace ndp::detail::metadata
