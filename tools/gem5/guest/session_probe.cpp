// Exercise Session dependencies, resident buffers, and owned receiver fault cleanup.
#include "l1/gem5/metadata.hpp"
#include "ndp/common/fp16.hpp"
#include "ndp/l1/gem5/backend.hpp"
#include "ndp/l2/session.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

using namespace ndp;
#define CHECK(x)                                                                                   \
  do {                                                                                             \
    if (!(x))                                                                                      \
      throw std::runtime_error(#x);                                                                \
  } while (false)

Tensor upload(Session &s, std::vector<uint64_t> shape, std::vector<float> values) {
  std::vector<uint16_t> bits;
  for (float v : values) {
    bits.push_back(float_to_fp16(v));
  }
  auto t = s.from_host({{shape}, std::as_bytes(std::span(bits))});
  CHECK(t);
  return t.value(); // Session owns a staging copy after this function returns.
}

void expect(Session &s, const Tensor &t, std::vector<float> expected) {
  auto host = s.to_host(t);
  CHECK(host && host.value().bytes.size() == expected.size() * 2);
  for (size_t i = 0; i < expected.size(); ++i) {
    uint16_t bits;
    std::memcpy(&bits, host.value().bytes.data() + i * 2, 2);
    CHECK(fp16_to_float(bits) == expected[i]);
  }
}

void reaped() {
  int status;
  auto child = waitpid(-1, &status, WNOHANG);
  if (child != -1 || errno != ECHILD) {
    throw std::runtime_error("Unreaped child: result=" + std::to_string(child) +
                             " errno=" + std::to_string(errno));
  }
}

// Fault injection belongs to this probe only, never to the production receiver.
int faulty_receiver() {
  using namespace detail::metadata;
  auto hello = receive_frame(3);
  if (!hello) {
    return 2;
  }
  auto reply = hello.value();
  reply.kind = Kind::ExecuteComplete;
  if (!send_frame(3, reply)) {
    return 2;
  }
  auto request = receive_frame(3);
  if (!request) {
    return 2;
  }
  std::string mode = std::getenv("NDP_F4_TEST_FAULT");
  if (mode == "disconnect") {
    return 0;
  }
  if (mode == "wrong_ack") {
    reply = request.value();
    reply.kind = Kind::Accepted;
    if (!send_frame(3, reply)) {
      return 2;
    }
  }
  for (;;) {
    pause(); // Parent must terminate and reap before completing the failed event.
  }
}

void healthy(const std::filesystem::path &receiver) {
  Gem5BackendOptions options{receiver, {2, 2, 96, 64, 64, 256, 128}};
  auto backend = make_gem5_backend(options);
  CHECK(backend);
  auto created = Session::create(std::move(backend.value()));
  CHECK(created);
  auto &s = created.value();
  auto w = upload(s, {2, 2}, {1, 2, 3, 4});
  auto x = upload(s, {2}, {5, 6});
  auto pim = s.matvec(w, x, {Engine::PIM});
  CHECK(pim);
  auto pnm = s.add(pim.value(), x, {Engine::PNM});
  CHECK(pnm);
  auto buffer = s.allocate({{4}});
  CHECK(buffer);
  CHECK(s.copy_into({{pnm.value(), buffer.value(), 0, 1, 2}}));
  auto before = s.snapshot(buffer.value(), {{4}});
  CHECK(before);
  CHECK(s.clear({buffer.value()}));
  auto after = s.snapshot(buffer.value(), {{4}});
  CHECK(after);
  expect(s, before.value(), {0, 22, 45, 0});
  expect(s, after.value(), {0, 0, 0, 0});
  auto again = s.matvec(w, x, {Engine::PIM});
  CHECK(again);
  expect(s, again.value(), {17, 39});
  std::cout << "NDP_SESSION_DEPENDENCIES pnm_pim=verified\n"
               "NDP_RESIDENT_BUFFERS copy_fill_snapshot=verified\n";
  // Close drains queued work, including work whose input handle is no longer held.
  auto pending = s.scale(upload(s, {2}, {2, 3}), 2);
  CHECK(pending);
  CHECK(s.close());
  CHECK(pending.value().wait());
  CHECK(s.close());
  CHECK(!s.scale(x, 2));
  CHECK(s.memory_stats().staging_bytes == 0);
  reaped(); // Tensor and DeviceBuffer handles intentionally still exist here.
  std::cout << "NDP_SESSION_CLOSE drain_live_handles_reaped=verified\n";
}

void faults(const std::filesystem::path &self) {
  for (auto mode : {"timeout", "disconnect", "wrong_ack"}) {
    CHECK(setenv("NDP_F4_TEST_FAULT", mode, 1) == 0);
    Gem5BackendOptions options{self, {1, 1, 64, 64, 64, 256, 128}, std::chrono::milliseconds(500)};
    auto backend = make_gem5_backend(options);
    CHECK(backend);
    CHECK(unsetenv("NDP_F4_TEST_FAULT") == 0);
    auto created = Session::create(std::move(backend.value()));
    CHECK(created);
    auto &s = created.value();
    auto input = upload(s, {2}, {1, 2});
    auto dependent = s.scale(input, 2);
    CHECK(dependent);
    CHECK(!input.wait());
    auto result = dependent.value().wait();
    CHECK(!result && result.error().code == ErrorCode::DependencyFailed);
    reaped(); // Check before close: failed operation itself must quiesce the child.
    auto later = upload(s, {1}, {3});
    CHECK(!later.wait()); // No reconnection or replay after uncertain completion.
    CHECK(!s.close());
    CHECK(!s.close());
    CHECK(s.memory_stats().staging_bytes == 0);
    reaped();
  }
  CHECK(!make_gem5_backend({"/nonexistent/ndp-receiver"}));
  reaped();
  std::cout << "NDP_SESSION_FAULTS timeout_disconnect_protocol=verified\n";
}

void destruction_and_idle(const std::filesystem::path &receiver) {
  Gem5BackendOptions options{
      receiver, {1, 1, 64, 64, 64, 256, 128}, std::chrono::milliseconds(500)};
  std::optional<Tensor> surviving;
  {
    auto backend = make_gem5_backend(options);
    CHECK(backend);
    auto created = Session::create(std::move(backend.value()));
    CHECK(created);
    surviving = upload(created.value(), {1}, {7});
    CHECK(surviving->wait());
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    expect(created.value(), *surviving, {7});
  }
  reaped();
  CHECK(surviving->wait());
  std::cout << "NDP_SESSION_LIFETIME idle_destructor=verified\n";
}

int main(int argc, char **argv) {
  if (argc > 1 && std::string_view(argv[1]) == "--serve") {
    return faulty_receiver();
  }
  try {
    CHECK(argc == 2);
    healthy(std::filesystem::absolute(argv[1]));
    faults(std::filesystem::absolute(argv[0]));
    destruction_and_idle(std::filesystem::absolute(argv[1]));
    std::cout << "NDP_F4_PASS\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "F4 failed: " << e.what() << '\n';
    return 1;
  }
}
