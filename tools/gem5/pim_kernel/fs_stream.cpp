// Adapt the worker's framed streams to gem5 readfile/writefile at guest rendezvous.
#include <gem5/m5ops.h>
#include <sstream>
#define main worker_main
#include "worker.cpp"
#undef main

namespace {
std::ostringstream outgoing;

void publish_reply() {
  auto data = outgoing.str();
  if (!data.empty() && m5_write_file(data.data(), data.size(), 0, "ndp-reply.bin") != data.size())
    throw std::runtime_error("short gem5 writefile");
  outgoing.str("");
}

class Requests : public std::streambuf {
  Bytes frame;
  bool first_request = true;
  int_type underflow() override {
    if (first_request) {
      // pim_init has finished. Switch to Timing before accepting any host work.
      m5_hypercall(10);
      first_request = false;
    }
    publish_reply();
    m5_hypercall(42);
    uint32_t size = 0;
    auto count = m5_read_file(&size, sizeof(size), 0);
    if (count == 0) return traits_type::eof();  // Peer disconnected between frames.
    if (count != 4 || !size || size > max_frame) throw std::runtime_error("invalid FS frame");
    frame.resize(4 + size);
    std::memcpy(frame.data(), &size, 4);
    if (m5_read_file(frame.data() + 4, size, 4) != size) throw std::runtime_error("short FS frame");
    setg(frame.data(), frame.data(), frame.data() + frame.size());
    return traits_type::to_int_type(*gptr());
  }
};
}

int main() {
  Requests incoming;
  auto *input = std::cin.rdbuf(&incoming);
  auto *output = std::cout.rdbuf(outgoing.rdbuf());
  int status = run_worker([] {
    // All synchronous commands and session-owned buffers are finished.
    publish_reply();
    m5_hypercall(44);
  });
  std::cin.rdbuf(input);
  std::cout.rdbuf(output);
  if (status) return status;
  m5_hypercall(43);  // Session destroyed and pim_finalize returned.
  std::cout << "NDP_FS_WORKER_CLOSED" << std::endl;
  return 0;
}
