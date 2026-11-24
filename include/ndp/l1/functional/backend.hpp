// Construct the CPU reference backend for runtime correctness tests.
#pragma once
#include "ndp/l1/backend.hpp"

namespace ndp {
std::unique_ptr<IBackend> make_functional_backend(size_t capacity_bytes,
                                                  size_t scratchpad_bytes = 4096);
} // namespace ndp
