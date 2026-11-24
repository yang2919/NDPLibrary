// Parse ELF64 little-endian fields explicitly; never map file structures onto host pointers.
#include "program.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>

namespace ndp::detail::command {
namespace {
Status invalid(const char *message) {
  return {ErrorCode::InvalidArgument, message};
}

Status unsupported(const char *message) {
  return {ErrorCode::UnsupportedOperation, message};
}

bool fits(uint64_t offset, uint64_t bytes, uint64_t size) {
  return offset <= size && bytes <= size - offset;
}

uint64_t field(std::span<const std::byte> data, size_t offset, size_t count) {
  uint64_t value = 0;
  for (size_t i = 0; i < count; ++i) {
    value |= uint64_t(std::to_integer<unsigned>(data[offset + i])) << (8 * i);
  }
  return value;
}

Result<LoadedProgram> parse(std::span<const std::byte> data, const ProgramRequirements &req,
                            const ProgramProfile &profile) {
  if (!profile.bytes || profile.base > UINT64_MAX - profile.bytes || req.xlen != 64 ||
      req.vector_major != 1 || req.vector_minor != 0 || req.minimum_vlen_bits < 128 ||
      (req.minimum_vlen_bits & (req.minimum_vlen_bits - 1)) ||
      req.minimum_vlen_bits > profile.vlen_bits) {
    return unsupported("Unsupported PNM address or RV64/RVV requirements");
  }
  for (const auto &extension : req.extensions) {
    if (std::find(profile.extensions.begin(), profile.extensions.end(), extension) ==
        profile.extensions.end()) {
      return unsupported("Required RISC-V extension unavailable");
    }
  }
  for (const auto *required : {"i", "v"}) {
    if (std::find(req.extensions.begin(), req.extensions.end(), required) == req.extensions.end()) {
      return invalid("Manifest must declare I and V extensions");
    }
  }
  if (data.size() < 64 || field(data, 0, 4) != 0x464c457f || field(data, 4, 1) != 2 ||
      field(data, 5, 1) != 1 || field(data, 6, 1) != 1 || field(data, 16, 2) != 2 ||
      field(data, 18, 2) != 243 || field(data, 20, 4) != 1 || field(data, 52, 2) != 64) {
    return invalid("Expected static ELF64 little-endian RISC-V ET_EXEC");
  }
  auto flags = field(data, 48, 4);
  // RVC and hard-float ABI flags require matching manifest extensions. Reject RVE/TSO/unknown
  // flags.
  if (flags & ~uint64_t{7}) {
    return unsupported("Unsupported RISC-V ELF flags");
  }
  auto has = [&](const char *ext) {
    return std::find(req.extensions.begin(), req.extensions.end(), ext) != req.extensions.end();
  };
  if (((flags & 1) && !has("c")) || ((flags & 6) == 2 && !has("f")) ||
      ((flags & 6) == 4 && !has("d")) || (flags & 6) == 6) {
    return unsupported("ELF ABI flags exceed manifest requirements");
  }
  uint64_t phoff = field(data, 32, 8), count = field(data, 56, 2);
  if (!count || count == 65535 || field(data, 54, 2) != 56 || phoff < 64 ||
      !fits(phoff, count * 56, data.size())) {
    return invalid("Invalid ELF program header table");
  }
  // Section metadata is optional, but reject relocations/dynamic sections if present.
  uint64_t shoff = field(data, 40, 8), sections = field(data, 60, 2);
  if (shoff || sections) {
    if (!sections || field(data, 58, 2) != 64 || shoff < 64 ||
        !fits(shoff, sections * 64, data.size())) {
      return invalid("Invalid or extended ELF section table");
    }
    for (size_t i = 0; i < sections; ++i) {
      size_t at = shoff + i * 64;
      auto type = field(data, at + 4, 4), offset = field(data, at + 24, 8),
           size = field(data, at + 32, 8);
      if (type == 4 || type == 9 || type == 6 || type == 19) {
        return unsupported("Relocations and dynamic sections are not supported");
      }
      if (type != 8 && !fits(offset, size, data.size())) {
        return invalid("ELF section outside image");
      }
    }
  }
  LoadedProgram program{field(data, 24, 8), 0, req, {}};

  struct Segment {
    uint64_t offset, address, files, memory, flags;
  };

  std::vector<Segment> segments;
  bool entry_found = false;
  for (size_t i = 0; i < count; ++i) {
    size_t at = phoff + i * 56;
    auto type = field(data, at, 4);
    if (type == 0) {
      continue;
    }
    auto bits = field(data, at + 4, 4), offset = field(data, at + 8, 8),
         address = field(data, at + 16, 8);
    auto files = field(data, at + 32, 8), memory = field(data, at + 40, 8),
         alignment = field(data, at + 48, 8);
    if (!fits(offset, files, data.size())) {
      return invalid("ELF segment outside image");
    }
    if (type == 2 || type == 3 || type == 7) {
      return unsupported("Dynamic linking, interpreters and TLS unsupported");
    }
    if (type != 1) {
      if (type == 4 || type == 6 || type == 0x70000003 || type == 0x6474e551 ||
          type == 0x6474e552) {
        continue;
      }
      return unsupported("Unsupported ELF program header type");
    }
    if (files > memory || address < profile.base ||
        !fits(address - profile.base, memory, profile.bytes) ||
        (alignment > 1 &&
         ((alignment & (alignment - 1)) || address % alignment != offset % alignment)) ||
        (bits & ~uint64_t{7}) || !(bits & 4) || ((bits & 3) == 3)) {
      return invalid("Invalid ELF load range, alignment or permissions");
    }
    if (!memory) {
      continue;
    }
    for (const auto &segment : segments) {
      if (address < segment.address + segment.memory && segment.address < address + memory) {
        return invalid("Overlapping ELF load segments");
      }
    }
    if (!segments.empty() && address < segments.back().address) {
      return invalid("Unsorted ELF load segments");
    }
    if (memory > profile.total_loaded_bytes - program.memory_bytes) {
      return Status{ErrorCode::OutOfMemory, "Program exceeds memory budget"};
    }
    program.memory_bytes += memory;
    if ((bits & 1) && program.entry >= address && program.entry - address < files) {
      entry_found = true;
    }
    segments.push_back({offset, address, files, memory, bits});
  }
  if (!entry_found || program.entry % (has("c") ? 2 : 4)) {
    return invalid("Entry is not aligned executable file data");
  }
  for (const auto &segment : segments) {
    ProgramSegment loaded{segment.address, uint32_t(segment.flags),
                          std::vector<std::byte>(segment.memory)};
    std::copy_n(data.begin() + segment.offset, segment.files, loaded.bytes.begin());
    program.segments.push_back(std::move(loaded));
  }
  return program;
}
} // namespace

Result<uint64_t> ProgramStore::load(std::span<const std::byte> image,
                                    const ProgramRequirements &req) {
  try {
    auto parsed = parse(image, req, profile_);
    if (!parsed) {
      return parsed.error();
    }
    if (parsed.value().memory_bytes > profile_.total_loaded_bytes - used_ || next_ == UINT64_MAX) {
      return Status{ErrorCode::OutOfMemory, "Loaded program capacity exhausted"};
    }
    // ET_EXEC addresses are fixed: simultaneously resident programs cannot overlap.
    for (const auto &[id, existing] : programs_) {
      (void)id;
      for (const auto &a : existing->segments) {
        for (const auto &b : parsed.value().segments) {
          if (a.address < b.address + b.bytes.size() && b.address < a.address + a.bytes.size()) {
            return invalid("Program load ranges overlap an existing program");
          }
        }
      }
    }
    auto program = std::make_shared<const LoadedProgram>(std::move(parsed.value()));
    programs_.emplace(next_, program);
    used_ += program->memory_bytes;
    return next_++;
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Program allocation failed"};
  } catch (const std::length_error &) {
    return Status{ErrorCode::OutOfMemory, "Program allocation too large"};
  }
}

Result<void> ProgramStore::unload(uint64_t handle) {
  auto it = programs_.find(handle);
  if (it == programs_.end()) {
    return invalid("Unknown program handle");
  }
  if (it->second.use_count() != 1) {
    return Status{ErrorCode::InvalidState, "Program is retained by a launch"};
  }
  used_ -= it->second->memory_bytes;
  programs_.erase(it);
  return {};
}

std::shared_ptr<const LoadedProgram> ProgramStore::get(uint64_t handle) const {
  auto it = programs_.find(handle);
  return it == programs_.end() ? nullptr : it->second;
}
} // namespace ndp::detail::command
