// Serve one inherited Unix socket until Close or a connection failure.
#include "l1/gem5/execution.hpp"

#include <charconv>
#include <iostream>
#include <string_view>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc != 12 || std::string_view(argv[1]) != "--serve") {
    return 2;
  }
  uint64_t values[10]{};
  for (int i = 0; i < 10; ++i) {
    std::string_view text(argv[i + 2]);
    auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), values[i]);
    if (error != std::errc{} || end != text.data() + text.size()) {
      return 2;
    }
  }
  if (values[0] != 3 || !values[1] || !values[2] || values[2] > 3600000) {
    return 2;
  }
  ndp::NdpSpec spec{values[3], values[4], values[5], values[6], values[7], values[8], values[9]};
  auto result = ndp::detail::execution::serve(3, values[1], spec,
                                              {std::chrono::milliseconds(values[2])}, true);
  ::close(3);
  if (!result) {
    std::cerr << result.error().message << '\n';
  }
  return result ? 0 : 1;
}
