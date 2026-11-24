// Compare persistent remote PNM/PIM execution with the existing local emulator.
#include "l1/gem5/execution.hpp"
#include "l1/ndp/transport.hpp"
#include "ndp/common/fp16.hpp"

#include <cerrno>
#include <cstdio>
#include <signal.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace ndp;
using namespace ndp::detail;
using namespace ndp::detail::command;

namespace {
void check(bool value, const char *message) {
  if (!value) {
    throw std::runtime_error(message);
  }
}

std::vector<std::byte> halves(std::initializer_list<float> values) {
  std::vector<std::byte> bytes;
  for (auto v : values) {
    auto h = float_to_fp16(v);
    bytes.push_back(std::byte(h & 255));
    bytes.push_back(std::byte(h >> 8));
  }
  return bytes;
}

template <class Device> std::vector<std::byte> exercise(Device &d) {
  auto put = [&](Address a, const std::vector<std::byte> &v) {
    check(bool(d.upload(a, v)), "Upload failed");
  };
  auto run = [&](std::initializer_list<Command> commands) {
    check(bool(d.submit(std::span<const Command>(commands.begin(), commands.size()))),
          "Submit failed");
  };
  std::vector<std::byte> evidence;
  auto read = [&](Address a, const std::vector<std::byte> &expected) {
    std::vector<std::byte> bytes(expected.size());
    check(bool(d.download(a, bytes)), "Download failed");
    check(bytes == expected, "Independent numeric/state oracle mismatch");
    evidence.insert(evidence.end(), bytes.begin(), bytes.end());
  };
  std::vector<std::byte> ones, twos;
  for (int i = 0; i < 16; ++i) {
    auto a = halves({1}), b = halves({2});
    ones.insert(ones.end(), a.begin(), a.end());
    twos.insert(twos.end(), b.begin(), b.end());
  }
  put({Space::Shared, 64}, ones);
  put({Space::Gpr, 0}, twos);
  put({Space::Gpr, 64}, halves({0}));
  put({Space::Shared, 256}, halves({1}));
  run({InitializeMac{1, 1, 64}, Transfer{{Space::Gpr, 0}, {Space::GlobalBuffer, 0}, 32},
       MacAllBanks{1, 1, 0, 1}, ReadMac{1, 96}});
  read({Space::Gpr, 96}, halves({32}));
  run({MacAllBanks{1, 1, 0, 1}, ReadMac{1, 96}});
  read({Space::Gpr, 96}, halves({64})); // MAC state survives separate requests.
  run({PnmCompute{PnmKernel::Add,
                  {{{Space::Gpr, 96}, 1}, {{Space::Shared, 256}, 1}},
                  {{Space::Scratchpad, 0}, 1},
                  1,
                  1}});
  read({Space::Scratchpad, 0}, halves({65}));
  run({CopyBytes{{Space::Scratchpad, 0}, {Space::Shared, 300}, 2},
       FillBytes{{Space::Shared, 302}, 2, std::byte{0}}});
  read({Space::Shared, 300}, halves({65, 0}));
  put({Space::Shared, 512}, halves({1, 2, 3, 4}));
  put({Space::Shared, 544}, halves({5, 6, 7, 8}));
  run({PnmCompute{PnmKernel::Matmul,
                  {{{Space::Shared, 512}, 4}, {{Space::Shared, 544}, 4}},
                  {{Space::Shared, 576}, 4},
                  2,
                  2,
                  2}});
  read({Space::Shared, 576}, halves({19, 22, 43, 50}));
  // Layout conversion is commanded explicitly, using the same stored bytes.
  run({PnmRelayout{{Space::Shared, 512}, {Space::Shared, 768}, 2, 2, 1, 16, 64, true},
       PnmRelayout{{Space::Shared, 768}, {Space::Shared, 1024}, 2, 2, 1, 16, 64, false}});
  read({Space::Shared, 1024}, halves({1, 2, 3, 4}));
  const std::vector<Command> bad{FillBytes{{Space::Shared, 576}, 8, std::byte{0}},
                                 CopyBytes{{Space::Shared, UINT64_MAX}, {Space::Gpr, 0}, 2}};
  auto rejected = d.submit(bad);
  check(!rejected && rejected.error().code == ErrorCode::InvalidArgument,
        "Invalid batch not rejected");
  read({Space::Shared, 576}, halves({19, 22, 43, 50})); // Preflight prevented the fill.
  std::vector<std::byte> unchanged(4, std::byte{0x5a});
  check(!d.download({Space::Shared, UINT64_MAX}, unchanged), "Invalid download accepted");
  check(unchanged == std::vector<std::byte>(4, std::byte{0x5a}),
        "Failed download changed caller buffer");
  // A completed failure does not destroy valid device state.
  run({FillBytes{{Space::Shared, 302}, 2, std::byte{0}}});
  read({Space::Shared, 300}, halves({65, 0}));
  return evidence;
}
} // namespace

int main() {
  const NdpSpec spec{1, 1, 64, 64, 64, 256, 256};
  constexpr uint64_t epoch = 0x334455667788;
  metadata::IoOptions options{std::chrono::milliseconds(30000), 17};
  int sockets[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
    std::perror("socketpair");
    return 1;
  }
  pid_t child = fork();
  if (child < 0) {
    std::perror("fork");
    close(sockets[0]);
    close(sockets[1]);
    return 1;
  }
  if (child == 0) {
    close(sockets[0]);
    auto result = execution::serve(sockets[1], epoch, spec, options);
    if (!result) {
      std::fprintf(stderr, "Execution receiver: %s\n", result.error().message.c_str());
    }
    close(sockets[1]);
    _exit(result ? 0 : 1);
  }
  close(sockets[1]);
  try {
    auto connected = execution::Client::connect(sockets[0], epoch, spec, options);
    check(bool(connected), "Execution handshake failed");
    auto remote = exercise(*connected.value());
    auto local = make_emulator_transport(spec);
    check(bool(local), "Cannot create local reference");
    auto reference = exercise(*local.value());
    check(remote == reference, "Local/remote mismatch");
    check(bool(connected.value()->close()), "Close failed");
    check(bool(connected.value()->close()), "Repeated close failed");
    check(!connected.value()->submit({}), "Closed client accepted submit");
    close(sockets[0]);
    sockets[0] = -1;
    int status = 0;
    pid_t waited;
    do {
      waited = waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    bool reaped = waited == child;
    if (reaped || (waited < 0 && errno == ECHILD)) {
      child = -1;
    }
    check(reaped && WIFEXITED(status) && WEXITSTATUS(status) == 0, "Receiver failed to exit");
    std::puts("NDP_EXECUTION_COMPLETE pnm=pim=verified");
    std::puts("NDP_PERSISTENT_STATE mac_memory=verified");
    std::puts("NDP_REFERENCE_MATCH exact_fp16=yes");
    std::puts("NDP_EXECUTION_ERRORS preflight_download_recovery=verified");
    std::puts("NDP_F3_PASS");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "F3 failed: %s\n", e.what());
    if (sockets[0] >= 0) {
      close(sockets[0]);
    }
    if (child > 0) {
      kill(child, SIGKILL);
      while (waitpid(child, nullptr, 0) < 0 && errno == EINTR) {
      }
    }
    return 1;
  }
}
