// Verify that a statically linked C++ program executes inside guest Linux.
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string_view>
#include <vector>
#include <sys/utsname.h>

int main() {
    utsname system{};
    if (uname(&system) != 0) return 1;
    if (std::string_view(system.sysname) != "Linux" ||
        std::string_view(system.machine) != "x86_64") return 3;
    const std::vector<std::uint32_t> values{1, 2, 3, 4};
    if (std::accumulate(values.begin(), values.end(), 0u) != 10) return 2;
    std::cout << "guest=" << system.sysname << " arch=" << system.machine << '\n';
    std::cout << "NDP_FS_SMOKE_PASS" << std::endl;
}
