// Test execution-mode negotiation, malformed requests and faulted-client behavior.
#include "l1/gem5/execution.hpp"
#include "test_support.hpp"

#include <sys/socket.h>
#include <thread>
#include <unistd.h>
using namespace ndp;
using namespace ndp::detail;

namespace {
const NdpSpec spec{1, 1, 8, 64, 64, 256, 128};
const metadata::IoOptions options{std::chrono::milliseconds(200), 17};

struct Pair {
  int fd[2];

  Pair() {
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
  }

  ~Pair() {
    ::close(fd[0]);
    ::close(fd[1]);
  }
};

std::vector<std::byte> hello() {
  std::vector<std::byte> bytes{std::byte{1}, std::byte{2}, std::byte{0}, std::byte{1},
                               std::byte{0}, std::byte{1}, std::byte{0}};
  for (uint64_t n : {1, 1, 8, 64, 64, 256, 128}) {
    for (unsigned i = 0; i < 8; ++i) {
      bytes.push_back(std::byte((n >> (i * 8)) & 255));
    }
  }
  return bytes;
}

void mismatch() {
  Pair pair;
  bool rejected = false;
  std::jthread receiver([&] {
    auto result = execution::serve(pair.fd[1], 1, spec, options);
    rejected = !result;
  });
  auto other = spec;
  ++other.rows;
  CHECK(!execution::Client::connect(pair.fd[0], 1, other, options));
  receiver.join();
  CHECK(rejected);
}

void malformed_requests() {
  Pair pair;
  bool ok = false;
  std::jthread receiver([&] { ok = bool(execution::serve(pair.fd[1], 1, spec, options)); });
  uint64_t sequence = 0;
  auto request = [&](std::vector<std::byte> payload, metadata::Kind expected) {
    ++sequence;
    CHECK(metadata::send_frame(
        pair.fd[0], {metadata::Kind::ExecuteRequest, 1, sequence, std::move(payload)}, options));
    auto response = metadata::receive_frame(pair.fd[0], options);
    CHECK(response && response.value().kind == expected && response.value().sequence == sequence);
  };
  request(hello(), metadata::Kind::ExecuteComplete);
  for (unsigned op : {0u, 1u, 2u, 3u, 4u, 6u}) {
    request({std::byte(op)}, metadata::Kind::ExecuteError);
  }
  auto too_large = std::vector<std::byte>(18, std::byte{0});
  too_large[0] = std::byte{3};
  too_large[1] = std::byte{1};
  for (size_t i = 10; i < 18; ++i) {
    too_large[i] = std::byte{0xff};
  }
  request(too_large, metadata::Kind::ExecuteError);
  too_large[0] = std::byte{2};
  request(too_large, metadata::Kind::ExecuteError);
  too_large[1] = std::byte{99};
  request(too_large, metadata::Kind::ExecuteError);
  request({std::byte{5}, std::byte{0}}, metadata::Kind::ExecuteError);
  request({std::byte{5}}, metadata::Kind::ExecuteComplete);
  receiver.join();
  CHECK(ok);
}

void identity_rejection() {
  for (unsigned fault = 0; fault < 2; ++fault) {
    Pair pair;
    bool rejected = false;
    std::jthread receiver([&] { rejected = !execution::serve(pair.fd[1], 1, spec, options); });
    CHECK(metadata::send_frame(
        pair.fd[0],
        {metadata::Kind::ExecuteRequest, fault == 0 ? 2u : 1u, fault == 1 ? 2u : 1u, hello()},
        options));
    auto response = metadata::receive_frame(pair.fd[0], options);
    CHECK(response && response.value().kind == metadata::Kind::ExecuteError);
    receiver.join();
    CHECK(rejected);
  }
}

void faulted_clients() {
  for (unsigned fault = 0; fault < 4; ++fault) {
    Pair pair;
    // Preload replies so the client can be tested without running any receiver.
    CHECK(metadata::send_frame(pair.fd[1], {metadata::Kind::ExecuteComplete, 1, 1, hello()},
                               options));
    auto client = execution::Client::connect(pair.fd[0], 1, spec, options);
    CHECK(client);
    if (fault != 3) {
      metadata::Frame response{metadata::Kind::ExecuteComplete, 1, 2, {std::byte{4}}};
      if (fault == 0) {
        response.kind = metadata::Kind::Accepted; // F2 ack is not execution.
      }
      if (fault == 1) {
        response.sequence = 3;
      }
      if (fault == 2) {
        response.payload.push_back(std::byte{0});
      }
      CHECK(metadata::send_frame(pair.fd[1], response, options));
    }
    CHECK(!client.value()->submit({})); // Fourth case times out.
    auto retry = client.value()->submit({});
    CHECK(!retry && retry.error().code == ErrorCode::InvalidState);
    CHECK(!client.value()->close());
  }
}
} // namespace

int main() {
  mismatch();
  malformed_requests();
  identity_rejection();
  faulted_clients();
  std::cout << "Execution protocol tests passed\n";
}
