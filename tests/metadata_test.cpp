// Reject malformed framing and exercise deadline, disconnect and range boundaries.
#include "l1/gem5/metadata.hpp"
#include "test_support.hpp"

#include <sys/socket.h>
#include <unistd.h>
using namespace ndp;
using namespace ndp::detail;

namespace {
struct Pair {
  int fd[2];

  Pair() {
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
  }

  ~Pair() {
    close(fd[0]);
    close(fd[1]);
  }
};

void raw_failure(std::vector<std::byte> bytes) {
  Pair p;
  CHECK(write(p.fd[0], bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size()));
  CHECK(shutdown(p.fd[0], SHUT_WR) == 0);
  CHECK(!metadata::receive_frame(p.fd[1]));
}

void framing() {
  Pair p;
  metadata::Frame request{metadata::Kind::Request, 1, 1, {std::byte{0x42}}};
  CHECK(metadata::send_frame(p.fd[0], request));
  std::vector<std::byte> bytes(33);
  CHECK(recv(p.fd[1], bytes.data(), bytes.size(), MSG_WAITALL) == 33);
  for (size_t n : {0u, 1u, 31u, 32u}) {
    raw_failure({bytes.begin(), bytes.begin() + n});
  }
  for (size_t at : {0u, 4u, 6u, 27u, 28u, 32u}) {
    auto bad = bytes;
    bad[at] = std::byte{0xff};
    raw_failure(bad);
  }
  auto result = metadata::receive_frame(p.fd[1], {std::chrono::milliseconds(20), 1});
  CHECK(!result && result.error().code == ErrorCode::BackendFailure);
  CHECK(!metadata::receive_frame(p.fd[1], {std::chrono::milliseconds(20), 0}));
  CHECK(shutdown(p.fd[1], SHUT_RDWR) == 0);
  CHECK(!metadata::send_frame(p.fd[0], request)); // Must not terminate via SIGPIPE.
  Pair blocked;
  int buffer = 1024;
  CHECK(setsockopt(blocked.fd[0], SOL_SOCKET, SO_SNDBUF, &buffer, sizeof(buffer)) == 0);
  request.payload.resize(128 * 1024);
  CHECK(!metadata::send_frame(blocked.fd[0], request, {std::chrono::milliseconds(20), 4096}));
}

void acknowledgements() {
  auto bytes = command::wire::encode({});
  CHECK(bytes);
  for (int fault = 0; fault < 4; ++fault) {
    Pair p;
    metadata::Frame reply{metadata::Kind::Accepted, 7, 1, bytes.value()};
    if (fault == 0) {
      reply.epoch = 8;
    }
    if (fault == 1) {
      reply.sequence = 2;
    }
    if (fault == 2) {
      reply.kind = metadata::Kind::Request;
    }
    if (fault == 3) {
      reply.payload[0] = std::byte{0};
    }
    CHECK(metadata::send_frame(p.fd[1], reply));
    CHECK(!metadata::probe(p.fd[0], 7, 1, bytes.value()));
  }
}

void ranges() {
  using namespace command;
  const NdpSpec p{2, 2, 4, 64, 32, 128, 64};
  auto valid = [&](Command command) {
    return metadata::validate_metadata(std::span<const Command>(&command, 1), p);
  };
  CHECK(valid(InitializeMac{1, 1, 126})); // Selected banks only.
  CHECK(!valid(InitializeMac{1, 1, 128}));
  CHECK(valid(ReadMac{1, 124}));
  CHECK(!valid(ReadMac{1, 126}));
  CHECK(!valid(MacAllBanks{1, 0, 1, 1})); // Row permits this column but GB does not.
  CHECK(!valid(MacAllBanks{4, 0, 0, 1}));
  CHECK(!valid(FillBytes{{Space::Shared, UINT64_MAX}, 1, std::byte{0}}));
  CHECK(!valid(PnmCompute{PnmKernel::Copy, {}, {{Space::Shared, 0}, SIZE_MAX}}));
  CHECK(!valid(PnmRelayout{{Space::Shared, 0}, {Space::Shared, 0}, SIZE_MAX, 2, 1, 16, 32, true}));
}
} // namespace

int main() {
  framing();
  acknowledgements();
  ranges();
  std::cout << "Metadata framing and range tests passed\n";
}
