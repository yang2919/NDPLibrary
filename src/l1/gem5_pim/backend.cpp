// Bind logical Session tensors to resident libpim handles over worker protocol v2.
#include "ndp/l1/gem5_pim/backend.hpp"
#include "ndp/l1/gem5_pim/dense_kernel.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <bit>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
extern char **environ;

namespace ndp {
namespace {
using Bytes = std::vector<uint8_t>;
constexpr size_t frame_limit = 16 * 1024 * 1024;
constexpr size_t max_elements = (frame_limit - 5) / 4; // worker's FP32 staging limit
static_assert(std::endian::native == std::endian::little);

Status invalid(const char *reason) {
  return {ErrorCode::InvalidArgument, reason};
}

void append(Bytes &b, uint32_t n) {
  for (int i = 0; i < 4; ++i) {
    b.push_back(uint8_t(n >> (8 * i)));
  }
}

uint32_t word(const uint8_t *p) {
  uint32_t n;
  std::memcpy(&n, p, 4);
  return n;
}

Bytes request(char op, std::initializer_list<uint32_t> args) {
  Bytes b{static_cast<uint8_t>(op)};
  for (auto v : args) {
    append(b, v);
  }
  return b;
}

// A broken stream invalidates every handle in this session. Local workers
// are also stopped and reaped by their owning backend.
class Worker {
  int fd = -1;
  pid_t pid = -1;

  void io(bool writing, uint8_t *p, size_t n) {
    while (n) {
#ifdef MSG_NOSIGNAL
      constexpr int flags = MSG_NOSIGNAL;
#else
      constexpr int flags = 0;
#endif
      ssize_t done = writing ? send(fd, p, n, flags) : recv(fd, p, n, 0);
      if (done < 0 && errno == EINTR) {
        continue;
      }
      if (done <= 0) {
        throw std::runtime_error("libpim worker disconnected or timed out");
      }
      p += done;
      n -= done;
    }
  }

public:
  bool connected() const { return fd >= 0; }

  ~Worker() {
    stop();
  }

  void stop() noexcept {
    if (fd >= 0) {
      close(fd);
      fd = -1;
    }
    if (pid > 0) {
      kill(pid, SIGKILL);
      while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
      }
      pid = -1;
    }
  }

  Result<void> finish(bool eof = false) {
    if (eof && fd >= 0) ::shutdown(fd, SHUT_WR);
    auto result = eof ? Result<Bytes>(Bytes{}) : call({'Q'});
    if (!result) {
      stop();
      return result.error();
    }
    if (pid < 0) { stop(); return {}; }
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < end) {
      int status;
      const auto done = waitpid(pid, &status, WNOHANG);
      if (done == pid) {
        pid = -1;
        stop();
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
          return {};
        }
        return Status{ErrorCode::BackendFailure, "worker cleanup failed"};
      }
      if (done < 0 && errno != EINTR) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    stop();
    return Status{ErrorCode::BackendFailure, "worker did not exit cleanly"};
  }

  Result<void> start(const std::filesystem::path &path, bool capture = false) {
    int pair[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair)) {
      return Status{ErrorCode::BackendFailure, strerror(errno)};
    }
    fd = pair[0];
    posix_spawn_file_actions_t actions;
    int err = posix_spawn_file_actions_init(&actions);
    if (!err) {
      timeval timeout{10, 0};
      if (fcntl(fd, F_SETFD, FD_CLOEXEC) || fcntl(pair[1], F_SETFD, FD_CLOEXEC) ||
          setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) ||
          setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout))) {
        err = errno;
      }
#ifdef SO_NOSIGPIPE
      int yes = 1;
      if (!err && setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes))) {
        err = errno;
      }
#endif
      if (!err) {
        err = posix_spawn_file_actions_adddup2(&actions, pair[1], STDIN_FILENO);
      }
      if (!err) {
        err = posix_spawn_file_actions_adddup2(&actions, pair[1], STDOUT_FILENO);
      }
      char *argv[] = {const_cast<char *>(path.c_str()), nullptr};
      pid_t child;
      if (!err) {
        err = posix_spawn(&child, path.c_str(), &actions, nullptr, argv, environ);
        if (!err) {
          pid = child;
        }
      }
      posix_spawn_file_actions_destroy(&actions);
    }
    close(pair[1]);
    if (err) {
      return Status{ErrorCode::BackendFailure, strerror(err)};
    }
    if (capture) {
      auto reply = call({'H', 3}, 2);
      if (!reply) return reply.error();
      if (reply.value() != Bytes{3, 0}) return invalid("expected v3 capture worker");
      return {};
    }
    return handshake();
  }

  Result<void> attach(int socket_fd) {
    int flags = fcntl(socket_fd, F_GETFL);
    if (flags < 0) return Status{ErrorCode::BackendFailure, strerror(errno)};
    if (flags & O_NONBLOCK) return Status{ErrorCode::BackendFailure, "worker socket must be blocking"};
    fd = fcntl(socket_fd, F_DUPFD_CLOEXEC, 3);
    if (fd < 0) return Status{ErrorCode::BackendFailure, strerror(errno)};
    timeval timeout{300, 0}; // Wall time includes simulated Linux execution.
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) ||
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)))
      return Status{ErrorCode::BackendFailure, strerror(errno)};
#ifdef SO_NOSIGPIPE
    int enabled = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled)))
      return Status{ErrorCode::BackendFailure, strerror(errno)};
#endif
    return handshake();
  }

  Result<void> handshake() {
    auto profile = call({'H', 2}, 16); // LPDDR6/FP16/FP32
    if (!profile) {
      return profile.error();
    }
    if (std::string(profile.value().begin(), profile.value().end()) != "LPDDR6/FP16/FP32") {
      return Status{ErrorCode::BackendFailure, "unexpected libpim profile"};
    }
    return {};
  }

  Result<Bytes> call(Bytes data, size_t expected = 0) {
    if (fd < 0) {
      return Status{ErrorCode::SessionClosed, "worker closed"};
    }
    try {
      uint32_t size = data.size();
      io(true, reinterpret_cast<uint8_t *>(&size), 4);
      io(true, data.data(), data.size());
      io(false, reinterpret_cast<uint8_t *>(&size), 4);
      if (!size || size > frame_limit) {
        throw std::runtime_error("invalid worker frame");
      }
      Bytes reply(size);
      io(false, reply.data(), size);
      if (reply[0] == 1) {
        return Status{ErrorCode::BackendFailure, std::string(reply.begin() + 1, reply.end())};
      }
      if (reply[0] || size != expected + 1) {
        throw std::runtime_error("invalid worker reply");
      }
      reply.erase(reply.begin());
      return reply;
    } catch (const std::exception &e) {
      stop();
      return Status{ErrorCode::BackendFailure, e.what()};
    }
  }
};

void append64(Bytes &data, uint64_t value) {
  append(data, uint32_t(value));
  append(data, uint32_t(value >> 32));
}

uint64_t read64(const uint8_t *data) {
  return word(data) | (uint64_t(word(data + 4)) << 32);
}

using gem5_pim::DenseOp;
using gem5_pim::DenseShape;

bool attention_op(OpKind op) {
  return op == OpKind::Matmul || op == OpKind::Linear || op == OpKind::Transpose ||
         op == OpKind::RMSNorm || op == OpKind::Add || op == OpKind::Softmax || op == OpKind::Reshape || op == OpKind::Gather;
}

DenseShape dense_shape(const KernelPlan &plan) {
  const auto &op = plan.operation;
  auto &a = op.inputs[0].shape;
  if (op.kind == OpKind::RMSNorm || op.kind == OpKind::Add)
    return {op.kind == OpKind::Add ? DenseOp::Add : DenseOp::RMSNorm,
            uint32_t(plan.output.physical_bytes/2/a.back()), uint32_t(a.back()), 0,
            op.kind == OpKind::Add ? 1.f : op.attributes.epsilon};
  if (op.kind == OpKind::Transpose)
    return {DenseOp::Transpose, uint32_t(a[0]), uint32_t(a[1]), 0, 1};
  if (op.kind == OpKind::Softmax)
    return {DenseOp::Softmax, uint32_t(plan.output.physical_bytes/2/a.back()), uint32_t(a.back()), 0, 1};
  return {op.kind == OpKind::Linear ? DenseOp::Linear : DenseOp::Matmul,
          uint32_t(a[0]), uint32_t(plan.output.logical.shape.back()), uint32_t(a[1]),
          op.kind == OpKind::Linear ? 1.f : op.attributes.factor};
}

void append_shape(Bytes &bytes, DenseShape shape) {
  for (auto field : {uint32_t(shape.op), shape.m, shape.n, shape.k, std::bit_cast<uint32_t>(shape.factor)})
    append(bytes, field);
}

// One v3 worker owns the ELF and its shape-specific manifest for this Session.
class ProgramCapture {
  Worker worker;
  Bytes gemv_image, pointwise_image, attention_image, arguments;
  const Bytes *active_image = nullptr;
  std::map<uint64_t, uint64_t> addresses;
  uint64_t program = 0;
  uint32_t active_slots = 0;

  Result<void> discard(char opcode, uint64_t id) {
    Bytes msg{uint8_t(opcode)};
    append64(msg, id);
    auto reply = worker.call(std::move(msg));
    if (!reply) return reply.error();
    return {};
  }

public:
  static Result<void> read_image(const std::filesystem::path &elf, Bytes &image) {
    std::ifstream file(elf, std::ios::binary | std::ios::ate);
    auto size = file.tellg();
    if (size < 64 || size > std::streamoff(frame_limit - 157))
      return invalid("missing, truncated or oversized program ELF");
    image.resize(size_t(size));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char *>(image.data()), size)) return invalid("cannot read program ELF");
    return {};
  }

  bool supports_pointwise() const { return !pointwise_image.empty(); }

  bool supports_attention() const { return !attention_image.empty(); }

  Result<void> open(const std::filesystem::path &path, const std::filesystem::path &elf,
                    const std::filesystem::path &pointwise, const std::filesystem::path &attention) {
    if (auto result = read_image(elf, gemv_image); !result) return result;
    if (!pointwise.empty()) {
      if (auto result = read_image(pointwise, pointwise_image); !result) return result;
    }
    if (!attention.empty()) {
      if (auto result = read_image(attention, attention_image); !result) return result;
    }
    return worker.start(path, true);
  }

  Result<void> submit(bool pim, uint32_t first, uint32_t second, const std::array<uint64_t *, 3> &handles) {
    const auto &image = pim ? gemv_image : pointwise_image;

    Bytes args;
    for (uint32_t v : {1u, 40u, first, second}) append(args, v);
    args.resize(40);
    // N3 metadata fixture reserves one contiguous row-slice descriptor.
    const std::array<uint64_t, 3> sizes{pim ? 16ull : 2ull * first,
                                        2ull * (pim ? second : first), 2ull * first};
    return launch_image(image, args, 16, handles, sizes);
  }

  Result<void> submit_dense(DenseShape shape, const std::array<uint64_t *, 4> &handles,
                            const std::array<uint64_t, 4> &sizes) {
    Bytes args;
    append(args, 1);
    append(args, 64);
    append_shape(args, shape);
    args.resize(64);
    return launch_image(attention_image, args, 32, handles, sizes);
  }

  Result<void> launch_image(const Bytes &image, const Bytes &args, uint32_t patch,
                            std::span<uint64_t *const> handles, std::span<const uint64_t> sizes) {
    bool pim = &image == &gemv_image;
    auto slots = uint32_t(std::count_if(handles.begin(), handles.end(), [](auto h) { return h != nullptr; }));
    auto arg_size = uint32_t(args.size());
    if (arguments != args || active_image != &image || slots != active_slots) {
      if (program) {
        auto released = discard('U', program);
        if (!released) return released;
        program = 0;
        arguments.clear();
      }
      Bytes load = request('P', {pim ? 2u : 1u, arg_size, slots});
      load.insert(load.end(), args.begin(), args.end());
      for (unsigned i = 0; i < handles.size(); ++i) {
        if (!handles[i]) continue;
        append(load, patch + 8*i);
        append(load, i+1 == handles.size() ? 2 : 1);
        append(load, pim && i == 0 ? 4 : 2);
        append64(load, sizes[i]);
      }
      load.insert(load.end(), image.begin(), image.end());
      auto reply = worker.call(std::move(load), 8);
      if (!reply) return reply.error();
      program = read64(reply.value().data());
      arguments = args;
      active_image = &image;
      active_slots = slots;
    }
    Bytes launch{'J'};
    append64(launch, program);
    append(launch, arg_size);
    append(launch, slots);
    launch.insert(launch.end(), args.begin(), args.end());
    for (unsigned i = 0; i < handles.size(); ++i) {
      if (!handles[i]) continue;
      if (!*handles[i]) {
        Bytes alloc{'A'};
        append64(alloc, sizes[i]);
        append(alloc, pim && i == 0 ? 1 : 3); // A dense output may become the next input.
        auto reply = worker.call(std::move(alloc), 8);
        if (!reply) return reply.error();
        *handles[i] = read64(reply.value().data());
      }
      append(launch, patch + 8*i);
      append(launch, i+1 == handles.size() ? 2 : 1);
      append64(launch, *handles[i]);
      append64(launch, 0);
      append64(launch, sizes[i]);
    }
    auto receipt = worker.call(std::move(launch), 8+arg_size);
    if (!receipt) return receipt.error();
    auto data = receipt.value().data();
    if (read64(data) != read64(image.data() + 24) || !std::equal(args.begin(), args.begin() + patch, data + 8))
      return invalid("invalid program capture receipt");
    for (unsigned i = 0; i < handles.size(); ++i) {
      auto address = read64(data + 8 + patch + i * 8);
      if (handles[i] ? (!address || address % (pim && i == 0 ? 4 : 2)) : address != 0) return invalid("invalid relocated program pointer");
      if (handles[i] && addresses.emplace(*handles[i], address).first->second != address)
        return invalid("live buffer changed PNM address between launches");
    }
    return {};
  }

  void release(uint64_t id) noexcept {
    if (!id) return;
    addresses.erase(id);
    try {
      if (!discard('F', id)) worker.stop();
    } catch (...) { worker.stop(); }
  }

  Result<void> close() {
    if (program) {
      auto status = discard('U', program);
      if (!status) return status;
    }
    return worker.finish(true);
  }
};

class Backend final : public IBackend {
  struct Buffer {
    size_t bytes;
    bool fp32 = false;
    uint32_t remote = 0;
    uint64_t capture_id = 0;
  };

  Worker worker;
  std::unique_ptr<ProgramCapture> capture;
  std::map<AllocationHandle, Buffer> buffers;
  AllocationHandle next = 1;
  size_t banks = 0, row_bytes = 0, capacity = 0, used = 0;
  bool closed = false;
  bool dense_worker = false, decoder_worker = false, partial_worker = false;

  Result<void> run(Bytes data) {
    auto r = worker.call(std::move(data));
    if (!r) {
      return r.error();
    }
    return {};
  }

  Result<void> bind(Buffer &b, char role, uint32_t rows, uint32_t cols) {
    if (b.remote) {
      return invalid("destination is already bound");
    }
    Bytes msg{'C', static_cast<uint8_t>(role)};
    append(msg, rows);
    append(msg, cols);
    auto id = worker.call(std::move(msg), 4);
    if (!id) {
      return id.error();
    }
    b.remote = word(id.value().data());
    return {};
  }

public:
  Result<void> open(const std::filesystem::path &path, int fd = -1) {
    if (auto r = fd < 0 ? worker.start(path) : worker.attach(fd); !r) {
      return r;
    }
    auto r = worker.call({'I'}, 12);
    if (!r) {
      return r.error();
    }
    auto p = r.value().data();
    banks = word(p);
    row_bytes = word(p + 4);
    size_t rows = word(p + 8);
    if (!banks || row_bytes < 2 || row_bytes % 2 || !rows || banks > SIZE_MAX / row_bytes ||
        rows > SIZE_MAX / (banks * row_bytes)) {
      return invalid("invalid worker geometry");
    }
    capacity = banks * row_bytes * rows;
    // Older v2 workers reject this optional opcode but keep the stream open.
    auto features = worker.call({'V'}, 1);
    if (features) {
      dense_worker = features.value()[0] & 1;
      decoder_worker = features.value()[0] & 2;
      partial_worker = features.value()[0] & 4;
    }
    else if (!worker.connected()) return features.error();
    return {};
  }

  Result<void> enable_capture(const std::filesystem::path &worker_path,
                              const std::filesystem::path &elf, const std::filesystem::path &pointwise, const std::filesystem::path &attention) {
    capture = std::make_unique<ProgramCapture>();
    return capture->open(worker_path, elf, pointwise, attention);
  }

  bool supports_storage(DType dtype) const override {
    return dtype == DType::F16 || (dtype == DType::F32 && partial_worker && !capture);
  }

  DeviceCaps capabilities() const override {
    return {"gem5_pim", capacity, 0};
  }

  Result<AllocationHandle> allocate(size_t bytes) override {
    if (closed) {
      return Status{ErrorCode::SessionClosed, "worker closed"};
    }
    if (!bytes || bytes % 2) {
      return invalid("invalid allocation size");
    }
    if (bytes > capacity - used || !next) {
      return Status{ErrorCode::OutOfMemory, "allocation limit"};
    }
    buffers.emplace(next, Buffer{bytes});
    used += bytes;
    return next++;
  }

  void release(AllocationHandle h) noexcept override {
    auto i = buffers.find(h);
    if (i == buffers.end()) {
      return;
    }
    if (i->second.remote && !closed) {
      try {
        if (!run(request('R', {i->second.remote}))) {
          worker.stop();
        }
      } catch (...) {
        worker.stop();
      }
    }
    if (capture && !closed) capture->release(i->second.capture_id);
    used -= i->second.bytes;
    buffers.erase(i);
  }

  Result<void> upload(AllocationHandle h, std::span<const std::byte> data) override {
    auto i = buffers.find(h);
    if (i == buffers.end() || data.size() != i->second.bytes || data.size() / 2 > max_elements) {
      return invalid("invalid FP16 upload");
    }
    auto &b = i->second;
    if (!b.remote) {
      auto r = bind(b, 'X', 1, data.size() / 2);
      if (!r) {
        return r;
      }
    }
    Bytes msg = request('U', {b.remote});
    auto ptr = reinterpret_cast<const uint8_t *>(data.data());
    msg.insert(msg.end(), ptr, ptr + data.size());
    return run(std::move(msg));
  }

  Result<void> upload_typed(AllocationHandle h, std::span<const std::byte> data, DType dtype) override {
    if (dtype == DType::F16) return upload(h, data);
    auto it = buffers.find(h);
    if (dtype != DType::F32 || !partial_worker || capture)
      return Status{ErrorCode::UnsupportedDType, "FP32 storage requires the native partial worker"};
    if (it == buffers.end() || data.size() != it->second.bytes || data.size()%4 ||
        data.size()/4 > max_elements || it->second.remote)
      return invalid("invalid FP32 upload");
    auto &b = it->second;
    auto bound = bind(b, 'P', 1, data.size()/4);
    if (!bound) return bound;
    b.fp32 = true;
    auto msg = request('U', {b.remote});
    auto ptr = reinterpret_cast<const uint8_t *>(data.data());
    msg.insert(msg.end(), ptr, ptr+data.size());
    return run(std::move(msg));
  }

  Result<void> download(AllocationHandle h, std::span<std::byte> data) override {
    auto i = buffers.find(h);
    if (i == buffers.end() || !i->second.remote || data.size() != i->second.bytes) {
      return invalid("invalid FP16 download");
    }
    auto r = worker.call(request('D', {i->second.remote}), data.size());
    if (!r) {
      return r.error();
    }
    std::memcpy(data.data(), r.value().data(), data.size());
    return {};
  }

  Result<void> ensure_dense(Buffer &buffer) {
    if (buffer.bytes/2 > max_elements) return invalid("dense buffer exceeds worker limit");
    return buffer.remote ? Result<void>{} : bind(buffer, 'X', 1, buffer.bytes/2);
  }

  Result<void> fill(AllocationHandle h, std::byte value) override {
    auto it = buffers.find(h);
    if (it == buffers.end()) return invalid("invalid fill handle");
    if (!dense_worker) return Status{ErrorCode::UnsupportedOperation, "worker lacks dense memory operations"};
    if (!it->second.remote) {
      if (it->second.bytes/2 > max_elements) return invalid("dense buffer exceeds worker limit");
      if (auto status = bind(it->second, 'B', 1, it->second.bytes/2); !status) return status;
    }
    return run(request('F', {it->second.remote, uint32_t(value)}));
  }

  Result<void> copy_region(AllocationHandle src, size_t from, AllocationHandle dst,
                           size_t to, size_t bytes) override {
    if (!dense_worker) return Status{ErrorCode::UnsupportedOperation, "worker lacks dense memory operations"};
    auto a = buffers.find(src), b = buffers.find(dst);
    if (a == buffers.end() || b == buffers.end() || !a->second.remote || (from|to|bytes)%2 ||
        from > a->second.bytes || bytes > a->second.bytes-from ||
        to > b->second.bytes || bytes > b->second.bytes-to) return invalid("invalid dense copy range");
    if (auto status = ensure_dense(b->second); !status) return status;
    return run(request('B', {b->second.remote, a->second.remote, uint32_t(from/2),
                             uint32_t(to/2), uint32_t(bytes/2)}));
  }

  Result<LayoutDesc> query_layout(const TensorDesc &d, LayoutKind k) const override {
    auto l = dense_layout(d);
    if (!l || k == LayoutKind::DenseRowMajor) {
      return l;
    }
    if (k != LayoutKind::PimBanked || d.shape.size() != 2) {
      return Status{ErrorCode::UnsupportedLayout, "weight layout requires a matrix"};
    }
    if (l.value().physical_bytes / 2 > max_elements) {
      return invalid("matrix exceeds worker limit");
    }
    const size_t groups = 1 + (d.shape[0] - 1) / banks,
                 tiles = 1 + (d.shape[1] - 1) / (row_bytes / 2);
    const size_t stripe = banks * row_bytes;
    if (tiles > capacity / stripe / groups) {
      return Status{ErrorCode::OutOfMemory, "weight exceeds row capacity"};
    }
    return LayoutDesc{k, d, groups * tiles * stripe, 2, "libpim.lpddr6.fp16.matrix_ab"};
  }

  Result<void> transform(AllocationHandle src, const LayoutDesc &from, AllocationHandle dst,
                         const LayoutDesc &to) override {
    auto a = buffers.find(src), b = buffers.find(dst);
    auto f = query_layout(from.logical, from.kind), t = query_layout(to.logical, to.kind);
    if (!f || !t || f.value() != from || t.value() != to || src == dst ||
        from.logical != to.logical || a == buffers.end() || b == buffers.end() ||
        !a->second.remote || a->second.bytes != from.physical_bytes ||
        b->second.bytes != to.physical_bytes) {
      return invalid("invalid layout conversion");
    }
    bool weight = to.kind == LayoutKind::PimBanked;
    auto r = bind(b->second, weight ? 'W' : 'X', weight ? to.logical.shape[0] : 1,
                  weight ? to.logical.shape[1] : byte_size(to.logical).value() / 2);
    if (!r) {
      return r;
    }
    return run(request('T', {b->second.remote, a->second.remote}));
  }

  Result<KernelPlan> query_operation(const OperationSpec &op) const override {
    if (op.attributes.output_dtype == DType::F32 && (!partial_worker || capture))
      return Status{ErrorCode::UnsupportedOperation, "FP32 partials require native worker execution"};
    auto output = infer_output(op);
    if (!output) {
      return output.error();
    }
    if ((op.kind == OpKind::RMSNorm || op.kind == OpKind::Add) &&
        (!decoder_worker || (op.kind == OpKind::Add && op.inputs[0] != op.inputs[1])))
      return Status{ErrorCode::UnsupportedOperation, "decoder requires RMSNorm or equal-shape add"};
    if (op.engine == Engine::PNM && dense_worker && attention_op(op.kind) && (!capture || capture->supports_attention())) {
      KernelPlan plan;
      plan.operation = op;
      plan.kernel_id = "pnm.dense.fp16";
      plan.output = dense_layout(output.value()).value();
      for (const auto &input : op.inputs) plan.inputs.push_back(dense_layout(input).value());
      if (plan.output.physical_bytes/2 > max_elements ||
          std::any_of(plan.inputs.begin(), plan.inputs.end(), [](const auto &l) {
            return l.physical_bytes/2 > max_elements;
          }) || (op.kind == OpKind::Gather && op.attributes.indices.size() > (frame_limit-9)/4))
        return invalid("dense operation exceeds worker limit");
      return plan;
    }
    if (op.engine == Engine::PNM && (capture ? capture->supports_pointwise() : decoder_worker) &&
        (op.kind == OpKind::SiLU || (op.kind == OpKind::Mul && op.inputs[0] == op.inputs[1]))) {
      KernelPlan plan;
      plan.operation = op;
      plan.kernel_id = op.kind == OpKind::SiLU ? "pnm.silu.fp16" : "pnm.mul.fp16";
      plan.output = dense_layout(output.value()).value();
      if (plan.output.physical_bytes / 2 > max_elements) return invalid("pointwise extent exceeds worker limit");
      plan.inputs.assign(op.inputs.size(), plan.output);
      return plan;
    }
    bool linear = op.kind == OpKind::Linear;
    if (op.engine != Engine::PIM ||
        (op.kind != OpKind::Matvec &&
         !(linear && op.inputs.size() == 2 && op.inputs[0].shape[0] == 1))) {
      return Status{ErrorCode::UnsupportedOperation,
                    "unsupported worker operation, shape or engine"};
    }
    KernelPlan plan;
    plan.operation = op;
    plan.kernel_id = "libpim.gemv.fp16";
    const size_t weight_index = linear ? 1 : 0;
    for (size_t i = 0; i < op.inputs.size(); ++i) {
      auto l = query_layout(op.inputs[i],
                            i == weight_index ? LayoutKind::PimBanked : LayoutKind::DenseRowMajor);
      if (!l) {
        return l.error();
      }
      plan.inputs.push_back(l.value());
    }
    plan.output = dense_layout(output.value()).value();
    return plan;
  }

  Result<void> execute(const KernelPlan &plan, std::span<const AllocationHandle> inputs,
                       AllocationHandle dst) override {
    auto expected = query_operation(plan.operation);
    if (!expected || plan != expected.value() || inputs.size() != plan.inputs.size() ||
        std::find(inputs.begin(), inputs.end(), dst) != inputs.end()) {
      return invalid("invalid worker kernel plan");
    }
    auto b = buffers.find(dst);
    if (b == buffers.end() || b->second.bytes != plan.output.physical_bytes) {
      return invalid("invalid output");
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto it = buffers.find(inputs[i]);
      if (it == buffers.end() || !it->second.remote ||
          it->second.bytes != plan.inputs[i].physical_bytes) {
        return invalid("invalid worker kernel input");
      }
    }
    if (plan.operation.engine == Engine::PNM && attention_op(plan.operation.kind)) {
      auto &output = b->second;
      if (plan.output.logical.dtype == DType::F32) {
        if (auto status = bind(output, 'P', 1, output.bytes/4); !status) return status;
        output.fp32 = true;
      } else if (auto status = ensure_dense(output); !status) return status;
      auto &left = buffers.at(inputs[0]);
      if (plan.operation.kind == OpKind::Reshape)
        return run(request('T', {output.remote, left.remote}));
      if (plan.operation.kind == OpKind::Gather) {
        auto msg = request('K', {output.remote, left.remote});
        for (auto index : plan.operation.attributes.indices) append(msg, uint32_t(index));
        return run(std::move(msg));
      }
      auto shape = dense_shape(plan);
      std::array<uint64_t *, 4> handles{nullptr, nullptr, nullptr, &output.capture_id};
      std::array<uint64_t, 4> sizes{0, 0, 0, output.bytes};
      auto msg = request('A', {output.remote});
      for (size_t i = 0; i < 3; ++i) {
        uint32_t remote = 0;
        if (i < inputs.size()) {
          auto &input = buffers.at(inputs[i]);
          handles[i] = &input.capture_id;
          sizes[i] = input.bytes;
          remote = input.remote;
        }
        append(msg, remote);
      }
      if (capture) {
        if (auto status = capture->submit_dense(shape, handles, sizes); !status) return status;
      }
      append_shape(msg, shape);
      return run(std::move(msg));
    }
    if (plan.operation.engine == Engine::PNM) {
      const bool multiply = plan.operation.kind == OpKind::Mul;
      const uint32_t count = b->second.bytes / 2;
      auto &left = buffers.at(inputs[0]);
      if (capture) {
        auto status = capture->submit(false, count, multiply ? 2 : 1,
            {&left.capture_id, multiply ? &buffers.at(inputs[1]).capture_id : nullptr, &b->second.capture_id});
        if (!status) return status;
      }
      auto status = bind(b->second, 'X', 1, count);
      if (!status) return status;
      auto msg = request(multiply ? 'M' : 'S', {b->second.remote, left.remote});
      if (multiply) append(msg, buffers.at(inputs[1]).remote);
      return run(std::move(msg));
    }
    size_t w = plan.operation.kind == OpKind::Linear ? 1 : 0;
    if (capture) {
      const auto &shape = plan.inputs[w].logical.shape;
      auto status = capture->submit(true, shape[0], shape[1], {&buffers.at(inputs[w]).capture_id,
                                    &buffers.at(inputs[1-w]).capture_id, &b->second.capture_id});
      if (!status) return status;
    }
    bool partial = plan.output.logical.dtype == DType::F32;
    auto r = bind(b->second, partial ? 'P' : 'Y', 1, b->second.bytes / (partial ? 4 : 2));
    b->second.fp32 = partial;
    if (!r) {
      return r;
    }
    return run(request(
        'G', {b->second.remote, buffers.at(inputs[w]).remote, buffers.at(inputs[1 - w]).remote}));
  }

  Result<void> shutdown() override {
    if (closed) {
      return {};
    }
    closed = true;
    auto result = worker.finish();
    if (capture) {
      auto status = capture->close();
      if (!status) return status;
    }
    return result;
  }
};
} // namespace

Result<std::unique_ptr<IBackend>> make_gem5_pim_backend(const std::filesystem::path &path) {
  auto backend = std::make_unique<Backend>();
  auto r = backend->open(path);
  if (!r) {
    return r.error();
  }
  return std::unique_ptr<IBackend>(std::move(backend));
}
Result<std::unique_ptr<IBackend>> make_gem5_pim_capture_backend(
    const std::filesystem::path &reference_worker, const std::filesystem::path &capture_worker,
    const std::filesystem::path &gemv_elf, const std::filesystem::path &pointwise_elf,
    const std::filesystem::path &attention_elf) {
  auto backend = std::make_unique<Backend>();
  auto status = backend->open(reference_worker);
  if (!status) return status.error();
  status = backend->enable_capture(capture_worker, gemv_elf, pointwise_elf, attention_elf);
  if (!status) return status.error();
  return std::unique_ptr<IBackend>(std::move(backend));
}
Result<std::unique_ptr<IBackend>> connect_gem5_pim_backend(int fd) {
  if (fd < 0) return invalid("invalid worker socket");
  auto backend = std::make_unique<Backend>();
  auto r = backend->open({}, fd);
  if (!r) return r.error();
  return std::unique_ptr<IBackend>(std::move(backend));
}
} // namespace ndp
