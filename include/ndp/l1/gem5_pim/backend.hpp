// Connect a Session to an owned FP16 libpim worker process.
#pragma once
#include "ndp/l1/backend.hpp"

#include <filesystem>

namespace ndp {
Result<std::unique_ptr<IBackend>> make_gem5_pim_backend(const std::filesystem::path &worker);
// Capture each GEMV launch on a v3 worker and compute its result on the v2 reference worker.
Result<std::unique_ptr<IBackend>> make_gem5_pim_capture_backend(
    const std::filesystem::path &reference_worker, const std::filesystem::path &capture_worker,
    const std::filesystem::path &gemv_elf, const std::filesystem::path &pointwise_elf = {},
    const std::filesystem::path &attention_elf = {});
// Duplicate a connected stream. Its worker and handles belong exclusively to this session.
Result<std::unique_ptr<IBackend>> connect_gem5_pim_backend(int socket_fd);
}
