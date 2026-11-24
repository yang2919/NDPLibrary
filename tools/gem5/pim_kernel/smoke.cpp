// Run the real worker on embedded protocol bytes and check every reply.
#include <sstream>
#include "fixture.hpp"

#define main worker_main
#include "worker.cpp"
#undef main

// Successful replies match exactly. An expected error requires status 1 and a
// nonempty diagnostic, whose wording can differ between libc++ and libstdc++.
bool replies_match(const std::string &actual, const std::string &golden) {
  size_t a = 0, g = 0;
  auto frame = [](const std::string &data, size_t &at, std::string &body) {
    if (data.size() - at < 4) return false;
    uint32_t length;
    std::memcpy(&length, data.data() + at, 4);
    at += 4;
    if (!length || length > data.size() - at) return false;
    body = data.substr(at, length);
    at += length;
    return true;
  };
  while (g < golden.size()) {
    std::string got, want;
    if (!frame(actual, a, got) || !frame(golden, g, want)) return false;
    if (want[0] == 1) {
      if (got[0] != 1 || got.size() < 2) return false;
    } else if (got != want) {
      return false;
    }
  }
  return a == actual.size();
}

int main() {
  std::istringstream input(std::string(reinterpret_cast<const char *>(requests), sizeof(requests)));
  std::ostringstream output;
  auto *old_input = std::cin.rdbuf(input.rdbuf());
  auto *old_output = std::cout.rdbuf(output.rdbuf());
  const int status = worker_main();
  std::cin.rdbuf(old_input);
  std::cout.rdbuf(old_output);
  const std::string golden(reinterpret_cast<const char *>(expected), sizeof(expected));
  if (status != 0 || !replies_match(output.str(), golden)) {
    std::cerr << "NDP_PIM_SMOKE_FAIL status=" << status
              << " reply_bytes=" << output.str().size() << '\n';
    return 1;
  }
  std::cout << fixture_marker << '\n';
  return 0;
}
