// Construct the command-driven FP16 emulator with functional PNM kernels and PIM GEMV.
#pragma once
#include "ndp/l1/backend.hpp"

#include <filesystem>

namespace ndp {
struct NdpSpec {
  size_t channels = 2;
  size_t banks = 16;
  size_t rows = 128;
  size_t row_bytes = 2048;
  size_t gb_bytes = 2048;
  size_t gpr_bytes = 512 * 1024;
  size_t scratchpad_bytes = 64 * 1024;
};

Result<NdpSpec> load_ndp_spec(const std::filesystem::path &);

Result<std::unique_ptr<IBackend>> make_ndp_backend(const NdpSpec &spec = {});
} // namespace ndp
