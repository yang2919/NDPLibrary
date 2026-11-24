// Validate static RISC-V ELF images and retain immutable program segments for future execution.
#pragma once
#include "ndp/common/types.hpp"

#include <map>
#include <memory>
#include <span>

namespace ndp::detail::command {
struct ProgramRequirements {
  unsigned xlen = 64;
  unsigned vector_major = 1, vector_minor = 0;
  unsigned minimum_vlen_bits = 128;
  // Explicit manifest; instruction/attribute discovery is deferred to the executor.
  std::vector<std::string> extensions = {"i", "m", "a", "f", "d", "v"};
};

struct ProgramProfile {
  uint64_t base = 0x10000000;
  size_t bytes = 16 * 1024 * 1024;
  size_t total_loaded_bytes = 32 * 1024 * 1024;
  unsigned vlen_bits = 128;
  std::vector<std::string> extensions = {"i", "m", "a", "f", "d", "v"};
};

struct ProgramSegment {
  uint64_t address;
  uint32_t flags;
  std::vector<std::byte> bytes;
};

struct LoadedProgram {
  uint64_t entry;
  size_t memory_bytes;
  ProgramRequirements requirements;
  std::vector<ProgramSegment> segments;
};

class ProgramStore {
public:
  explicit ProgramStore(ProgramProfile profile = {}) : profile_(std::move(profile)) {}

  Result<uint64_t> load(std::span<const std::byte>, const ProgramRequirements &);
  Result<void> unload(uint64_t);
  std::shared_ptr<const LoadedProgram> get(uint64_t) const;

  const ProgramProfile &profile() const {
    return profile_;
  }

private:
  ProgramProfile profile_;
  std::map<uint64_t, std::shared_ptr<const LoadedProgram>> programs_;
  uint64_t next_ = 1;
  size_t used_ = 0;
};
} // namespace ndp::detail::command
