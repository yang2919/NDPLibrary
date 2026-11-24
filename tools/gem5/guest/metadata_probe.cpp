// Run separate sender/receiver processes and verify metadata-only IPC delivery.
#include "l1/gem5/metadata.hpp"

#include <cerrno>
#include <cstdio>
#include <signal.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace ndp;
using namespace ndp::detail;

namespace {
void check(bool good, const char *message) {
  if (!good) {
    throw std::runtime_error(message);
  }
}

std::vector<std::byte> batch(bool large) {
  using namespace command;
  std::vector<Command> commands{
      Transfer{{Space::Shared, 64}, {Space::Gpr, 0}, 32},
      InitializeMac{1, 1, 64},
      Transfer{{Space::Gpr, 0}, {Space::GlobalBuffer, 0}, 32},
      MacAllBanks{1, 1, 0, 1},
      ReadMac{1, 96},
      PnmCompute{PnmKernel::Add,
                 {{{Space::Gpr, 96}, 1}, {{Space::Shared, 128}, 1}},
                 {{Space::Scratchpad, 0}, 1},
                 1,
                 1},
      PnmRelayout{{Space::Shared, 128}, {Space::Shared, 256}, 1, 16, 1, 16, 32, true},
      CopyBytes{{Space::Scratchpad, 0}, {Space::Shared, 160}, 2},
      FillBytes{{Space::Shared, 192}, 32, std::byte{0x5a}}};
  if (large) {
    for (size_t i = 0; i < 300; ++i) {
      commands.push_back(FillBytes{{Space::Shared, 192}, 32, std::byte(i & 255)});
    }
  }
  auto encoded = wire::encode(commands);
  check(bool(encoded), "Cannot encode fixture");
  return std::move(encoded.value());
}

void rejected(int fd, uint64_t epoch, uint64_t sequence, const std::vector<std::byte> &bytes,
              metadata::IoOptions options) {
  check(bool(metadata::send_frame(fd, {metadata::Kind::Request, epoch, sequence, bytes}, options)),
        "Cannot send rejection case");
  auto result = metadata::receive_frame(fd, options);
  check(bool(result), "Missing rejection response");
  const auto &r = result.value();
  check(r.kind == metadata::Kind::Rejected && r.epoch == epoch && r.sequence == sequence &&
            r.payload.empty(),
        "Unexpected rejection response");
}
} // namespace

int main() {
  int sockets[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
    std::perror("socketpair");
    return 1;
  }
  constexpr uint64_t epoch = 0x1020304050607080;
  const NdpSpec spec{1, 1, 32, 32, 32, 256, 256};
  // Small chunks deliberately exercise framing across many send/recv operations.
  const metadata::IoOptions options{std::chrono::milliseconds(30000), 17};
  pid_t child = fork();
  if (child < 0) {
    std::perror("fork");
    close(sockets[0]);
    close(sockets[1]);
    return 1;
  }
  if (child == 0) {
    close(sockets[0]);
    auto result = metadata::serve(sockets[1], epoch, 8, spec, options);
    if (!result) {
      std::fprintf(stderr, "Receiver: %s\n", result.error().message.c_str());
    }
    close(sockets[1]);
    _exit(result ? 0 : 1);
  }
  close(sockets[1]);
  try {
    auto small = batch(false), large = batch(true);
    check(bool(metadata::probe(sockets[0], epoch, 1, small, options)),
          "Small metadata probe failed");
    check(bool(metadata::probe(sockets[0], epoch, 2, large, options)),
          "Large metadata probe failed");
    rejected(sockets[0], epoch, 2, small, options);     // Duplicate sequence.
    rejected(sockets[0], epoch, 9, small, options);     // Out-of-order sequence.
    rejected(sockets[0], epoch + 1, 3, small, options); // Stale/wrong epoch.
    auto invalid_opcode = small;
    invalid_opcode[16] = std::byte{0xff};
    rejected(sockets[0], epoch, 3, invalid_opcode, options);
    command::Command invalid_range =
        command::CopyBytes{{command::Space::Shared, UINT64_MAX}, {command::Space::Gpr, 0}, 2};
    auto invalid_bytes =
        command::wire::encode(std::span<const command::Command>(&invalid_range, 1));
    check(bool(invalid_bytes), "Cannot encode invalid-range fixture");
    rejected(sockets[0], epoch, 4, invalid_bytes.value(), options);
    check(bool(metadata::probe(sockets[0], epoch, 5, small, options)),
          "Receiver failed after rejections");
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
    check(reaped && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "Receiver did not exit successfully");
    std::puts("NDP_METADATA_ACCEPTED batches=3 exact_bytes=yes");
    std::puts("NDP_METADATA_REJECTIONS duplicate_order_epoch_opcode_range=5");
    std::puts("NDP_METADATA_ONLY execution=not_performed");
    std::puts("NDP_F2_PASS");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "F2 failed: %s\n", e.what());
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
