// Read the versioned command-backend geometry without interpreting legacy emulation profiles.
#include "ndp/l1/ndp/backend.hpp"

#include <charconv>
#include <fstream>
#include <map>
#include <set>

namespace ndp {
Result<NdpSpec> load_ndp_spec(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file) {
    return Status{ErrorCode::InvalidArgument, "Cannot open NDP configuration"};
  }
  NdpSpec spec;
  size_t version = 0;
  std::map<std::string, size_t *> fields{
      {"version", &version},          {"channels", &spec.channels},
      {"banks", &spec.banks},         {"rows", &spec.rows},
      {"row_bytes", &spec.row_bytes}, {"gb_bytes", &spec.gb_bytes},
      {"gpr_bytes", &spec.gpr_bytes}, {"scratchpad_bytes", &spec.scratchpad_bytes}};
  auto trim = [](std::string s) {
    auto a = s.find_first_not_of(" \t\r");
    return a == std::string::npos ? std::string{}
                                  : s.substr(a, s.find_last_not_of(" \t\r") - a + 1);
  };
  std::set<std::string> seen;
  std::string line;
  size_t number = 0;
  while (std::getline(file, line)) {
    ++number;
    line = trim(line.substr(0, line.find('#')));
    if (line.empty()) {
      continue;
    }
    auto eq = line.find('=');
    auto key = trim(line.substr(0, eq));
    auto value = eq == std::string::npos ? std::string{} : trim(line.substr(eq + 1));
    size_t n = 0;
    auto parsed = std::from_chars(value.data(), value.data() + value.size(), n);
    if (!fields.contains(key) || !seen.insert(key).second || value.empty() ||
        parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
      return Status{ErrorCode::InvalidArgument,
                    "Invalid NDP configuration line " + std::to_string(number)};
    }
    *fields.at(key) = n;
  }
  if (file.bad() || seen.size() != fields.size() || version != 1) {
    return Status{ErrorCode::InvalidArgument, "Incomplete or unsupported NDP configuration"};
  }
  // Construction validates geometry and resource limits without hiding allocation failures.
  return spec;
}
} // namespace ndp
