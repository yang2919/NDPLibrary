// Keep execution acknowledgements distinct from F2 metadata acceptance.
#include "execution.hpp"

#include "../ndp/transport.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <limits>
#include <new>
#include <poll.h>

namespace ndp::detail::execution {
namespace {
using metadata::Frame;
using metadata::Kind;
constexpr size_t max_payload = 8 * 1024 * 1024;

enum Op : unsigned { Hello = 1, Upload = 2, Download = 3, Submit = 4, Close = 5 };

Status invalid(const char *s) {
  return {ErrorCode::InvalidArgument, s};
}

void append(std::vector<std::byte> &out, uint64_t v, size_t width) {
  for (size_t i = 0; i < width; ++i) {
    out.push_back(std::byte((v >> (i * 8)) & 255));
  }
}

uint64_t get(std::span<const std::byte> bytes, size_t at, size_t width) {
  uint64_t v = 0;
  for (size_t i = 0; i < width; ++i) {
    v |= uint64_t(std::to_integer<unsigned>(bytes[at + i])) << (i * 8);
  }
  return v;
}

std::vector<std::byte> hello(const NdpSpec &p) {
  std::vector<std::byte> out{std::byte{Hello}};
  append(out, command::wire::version, 2);
  append(out, command::wire::fp16_profile, 2);
  // Mapping ID 1: [row][channel][bank][byte], as in the existing emulator.
  append(out, 1, 2);
  for (auto n :
       {p.channels, p.banks, p.rows, p.row_bytes, p.gb_bytes, p.gpr_bytes, p.scratchpad_bytes}) {
    append(out, n, 8);
  }
  return out;
}

Result<std::vector<std::byte>> addressed(Op op, command::Address a, size_t bytes) {
  unsigned space = 0;
  switch (a.space) {
  case command::Space::Shared:
    space = 1;
    break;
  case command::Space::Gpr:
    space = 2;
    break;
  case command::Space::Scratchpad:
    space = 3;
    break;
  default:
    return invalid("Invalid transfer address space");
  }
  std::vector<std::byte> out{std::byte(op), std::byte(space)};
  append(out, a.offset, 8);
  append(out, bytes, 8);
  return out;
}

uint16_t error_id(ErrorCode c) {
  switch (c) {
  case ErrorCode::InvalidArgument:
    return 1;
  case ErrorCode::UnsupportedOperation:
    return 2;
  case ErrorCode::OutOfMemory:
    return 3;
  default:
    return 4;
  }
}

ErrorCode error_code(uint16_t id) {
  switch (id) {
  case 1:
    return ErrorCode::InvalidArgument;
  case 2:
    return ErrorCode::UnsupportedOperation;
  case 3:
    return ErrorCode::OutOfMemory;
  default:
    return ErrorCode::BackendFailure;
  }
}

Frame failure(const Frame &request, const Status &error) {
  Frame result{Kind::ExecuteError, request.epoch, request.sequence, {}};
  result.payload.push_back(request.payload.empty() ? std::byte{0} : request.payload[0]);
  append(result.payload, error_id(error.code), 2);
  for (size_t i = 0; i < std::min(size_t{256}, error.message.size()); ++i) {
    result.payload.push_back(std::byte(static_cast<unsigned char>(error.message[i])));
  }
  return result;
}

Result<std::vector<std::byte>> dispatch(const Frame &request, command::ICommandTransport &device) {
  auto bytes = std::span<const std::byte>(request.payload);
  if (bytes.empty()) {
    return invalid("Missing execution operation");
  }
  auto op = std::to_integer<unsigned>(bytes[0]);
  if (op == Upload || op == Download) {
    if (bytes.size() < 18) {
      return invalid("Truncated transfer descriptor");
    }
    auto space = std::to_integer<unsigned>(bytes[1]);
    if (space < 1 || space > 3) {
      return invalid("Unknown transfer address space");
    }
    command::Address address{space == 1   ? command::Space::Shared
                             : space == 2 ? command::Space::Gpr
                                          : command::Space::Scratchpad,
                             get(bytes, 2, 8)};
    auto length = get(bytes, 10, 8);
    if (length > max_payload - (op == Upload ? 18 : 1)) {
      return invalid("Transfer exceeds frame limit");
    }
    if (op == Upload) {
      if (length != bytes.size() - 18) {
        return invalid("Upload length mismatch");
      }
      auto result = device.upload(address, bytes.subspan(18));
      if (!result) {
        return result.error();
      }
      return std::vector<std::byte>{std::byte{Upload}};
    }
    if (bytes.size() != 18) {
      return invalid("Trailing download bytes");
    }
    std::vector<std::byte> result(size_t(length) + 1);
    result[0] = std::byte{Download};
    auto read = device.download(address, std::span(result).subspan(1));
    if (!read) {
      return read.error();
    }
    return result;
  }
  if (op == Submit) {
    auto commands = command::wire::decode(bytes.subspan(1));
    if (!commands) {
      return commands.error();
    }
    auto result = device.submit(commands.value());
    if (!result) {
      return result.error();
    }
    return std::vector<std::byte>{std::byte{Submit}};
  }
  if (op == Close && bytes.size() == 1) {
    auto result = device.shutdown();
    if (!result) {
      return result.error();
    }
    return std::vector<std::byte>{std::byte{Close}};
  }
  return invalid("Unknown operation or invalid close payload");
}
} // namespace

Result<std::vector<std::byte>> Client::exchange(std::vector<std::byte> payload,
                                                size_t expected_bytes) {
  if (closed_ || faulted_) {
    return Status{ErrorCode::InvalidState, "Execution client is closed or faulted"};
  }
  if (payload.empty() || payload.size() > max_payload || sequence_ == UINT64_MAX) {
    return invalid("Invalid outgoing execution request");
  }
  auto op = payload[0];
  Frame request{Kind::ExecuteRequest, epoch_, sequence_++, std::move(payload)};
  // Until a complete matching response is validated, failure means uncertain completion.
  faulted_ = true;
  auto sent = metadata::send_frame(fd_, request, options_);
  if (!sent) {
    return sent.error();
  }
  auto response = metadata::receive_frame(fd_, options_);
  if (!response) {
    return response.error();
  }
  const auto &r = response.value();
  if (r.epoch != request.epoch || r.sequence != request.sequence || r.payload.empty() ||
      r.payload[0] != op) {
    return invalid("Execution response identity mismatch");
  }
  if (r.kind == Kind::ExecuteError) {
    if (r.payload.size() < 3 || r.payload.size() > 259 || get(r.payload, 1, 2) < 1 ||
        get(r.payload, 1, 2) > 4) {
      return invalid("Invalid execution error response");
    }
    faulted_ = false; // Receiver acknowledged a completed failed operation.
    std::string message;
    for (size_t i = 3; i < r.payload.size(); ++i) {
      message.push_back(char(std::to_integer<unsigned char>(r.payload[i])));
    }
    return Status{error_code(get(r.payload, 1, 2)), std::move(message)};
  }
  if (r.kind != Kind::ExecuteComplete || r.payload.size() != expected_bytes) {
    return invalid("Invalid execution completion response");
  }
  faulted_ = false;
  return std::move(response.value().payload);
}

Result<std::unique_ptr<Client>> Client::connect(int fd, uint64_t epoch, const NdpSpec &spec,
                                                metadata::IoOptions options) {
  auto valid = metadata::validate_metadata({}, spec);
  if (!valid || !epoch) {
    return invalid("Invalid execution client configuration");
  }
  auto client = std::unique_ptr<Client>(new Client(fd, epoch, options));
  auto expected = hello(spec);
  auto response = client->exchange(expected, expected.size());
  if (!response) {
    return response.error();
  }
  if (response.value() != expected) {
    return invalid("Receiver profile/geometry mismatch");
  }
  return client;
}

Result<void> Client::upload(command::Address address, std::span<const std::byte> bytes) {
  if (bytes.size() > max_payload - 18) {
    return invalid("Upload exceeds frame limit");
  }
  auto payload = addressed(Upload, address, bytes.size());
  if (!payload) {
    return payload.error();
  }
  payload.value().insert(payload.value().end(), bytes.begin(), bytes.end());
  auto result = exchange(std::move(payload.value()), 1);
  return result ? Result<void>{} : Result<void>{result.error()};
}

Result<void> Client::download(command::Address address, std::span<std::byte> bytes) {
  if (bytes.size() > max_payload - 1) {
    return invalid("Download exceeds frame limit");
  }
  auto payload = addressed(Download, address, bytes.size());
  if (!payload) {
    return payload.error();
  }
  auto result = exchange(std::move(payload.value()), bytes.size() + 1);
  if (!result) {
    return result.error();
  }
  std::copy(result.value().begin() + 1, result.value().end(), bytes.begin());
  return {};
}

Result<void> Client::submit(std::span<const command::Command> commands) {
  auto payload = command::wire::encode(commands, {max_payload - 1, 65536});
  if (!payload) {
    return payload.error();
  }
  payload.value().insert(payload.value().begin(), std::byte{Submit});
  auto result = exchange(std::move(payload.value()), 1);
  return result ? Result<void>{} : Result<void>{result.error()};
}

Result<void> Client::close() {
  if (closed_) {
    return {};
  }
  auto result = exchange({std::byte{Close}}, 1);
  if (!result) {
    return result.error();
  }
  closed_ = true;
  return {};
}

Result<void> serve(int fd, uint64_t epoch, const NdpSpec &spec, metadata::IoOptions options,
                   bool wait_for_request) {
  auto valid = metadata::validate_metadata({}, spec);
  if (!valid || !epoch) {
    return invalid("Invalid execution receiver configuration");
  }
  std::unique_ptr<command::ICommandTransport> device;
  uint64_t sequence = 1;
  for (;;) {
    if (wait_for_request) {
      pollfd socket{fd, POLLIN, 0};
      int ready;
      do {
        ready = poll(&socket, 1, -1);
      } while (ready < 0 && errno == EINTR);
      if (ready < 0) {
        return Status{ErrorCode::BackendFailure, "Receiver idle poll failed"};
      }
    }
    auto read = metadata::receive_frame(fd, options);
    if (!read) {
      return read.error();
    }
    auto request = std::move(read.value());
    if (request.kind != Kind::ExecuteRequest || request.epoch != epoch ||
        request.sequence != sequence || sequence == UINT64_MAX) {
      auto error = invalid("Execution request identity mismatch");
      metadata::send_frame(fd, failure(request, error), options);
      return error;
    }
    ++sequence;
    Result<std::vector<std::byte>> result = invalid("Invalid execution request");
    bool close = false;
    try {
      if (!device) {
        if (request.payload != hello(spec)) {
          auto error = invalid("Hello profile/geometry mismatch");
          metadata::send_frame(fd, failure(request, error), options);
          return error;
        }
        auto created = command::make_emulator_transport(spec);
        if (!created) {
          metadata::send_frame(fd, failure(request, created.error()), options);
          return created.error();
        }
        device = std::move(created.value());
        result = request.payload;
      } else {
        result = dispatch(request, *device);
        close = result && request.payload[0] == std::byte{Close};
      }
    } catch (const std::bad_alloc &) {
      result = Status{ErrorCode::OutOfMemory, "Execution receiver allocation failed"};
    } catch (const std::exception &) {
      result = Status{ErrorCode::BackendFailure, "Execution receiver failed"};
    }
    auto sent = metadata::send_frame(
        fd,
        result ? Frame{Kind::ExecuteComplete, epoch, request.sequence, std::move(result.value())}
               : failure(request, result.error()),
        options);
    if (!sent) {
      return sent;
    }
    if (close) {
      return {};
    }
  }
}
} // namespace ndp::detail::execution
