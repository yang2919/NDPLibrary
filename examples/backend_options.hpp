// Select the reference backend or load an emulator specification at program startup.
#pragma once
#include "ndp/l1/functional/backend.hpp"
#include "ndp/l1/ndp/backend.hpp"

#include <string_view>

namespace ndp::example {
inline Result<std::unique_ptr<IBackend>> backend_from_args(int argc, char **argv, size_t capacity,
                                                           size_t scratchpad) {
  if (argc == 1) {
    return make_functional_backend(capacity, scratchpad);
  }
  if (argc == 2 && std::string_view(argv[1]) == "--ndp") {
    return make_ndp_backend();
  }
  if (argc == 3 && std::string_view(argv[1]) == "--ndp-config") {
    auto spec = load_ndp_spec(std::filesystem::path(argv[2]));
    if (!spec) {
      return spec.error();
    }
    return make_ndp_backend(spec.value());
  }
  return Status{ErrorCode::InvalidArgument,
                "Usage: executable [--ndp | --ndp-config path]"};
}
} // namespace ndp::example
