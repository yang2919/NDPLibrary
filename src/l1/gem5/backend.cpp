// Own receiver lifetime so uncertain completion cannot outlive a Session operation.
#include "ndp/l1/gem5/backend.hpp"

#include "../ndp/transport.hpp"
#include "execution.hpp"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
extern char **environ;

namespace ndp {
namespace {
using namespace detail;

Status failure(const char *message) {
  return {ErrorCode::BackendFailure, message};
}

class ProcessTransport final : public command::ICommandTransport {
public:
  int fd = -1;
  pid_t pid = -1;
  std::unique_ptr<execution::Client> client;
  std::chrono::milliseconds timeout;

  explicit ProcessTransport(std::chrono::milliseconds t) : timeout(t) {}

  ~ProcessTransport() override {
    stop();
  }

  // Exclusive child ownership: reap before publishing an uncertain completion.
  void stop() noexcept {
    if (pid > 0) {
      int status;
      pid_t result;
      do {
        result = waitpid(pid, &status, WNOHANG);
      } while (result < 0 && errno == EINTR);
      if (result == 0) {
        kill(pid, SIGKILL);
        do {
          result = waitpid(pid, &status, 0);
        } while (result < 0 && errno == EINTR);
      }
      pid = -1;
    }
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
  }

  template <class F> Result<void> invoke(F operation) {
    if (fd < 0) {
      return Status{ErrorCode::SessionClosed, "Receiver closed or faulted"};
    }
    try {
      auto result = operation();
      if (client->faulted()) {
        stop();
      }
      return result;
    } catch (...) {
      stop();
      throw;
    }
  }

  Result<void> upload(command::Address a, std::span<const std::byte> b) override {
    return invoke([&] { return client->upload(a, b); });
  }

  Result<void> download(command::Address a, std::span<std::byte> b) override {
    return invoke([&] { return client->download(a, b); });
  }

  Result<void> submit(std::span<const command::Command> c) override {
    return invoke([&] { return client->submit(c); });
  }

  Result<void> shutdown() override {
    if (fd < 0) {
      return {};
    }
    auto result = invoke([&] { return client->close(); });
    if (pid > 0 && result) {
      auto deadline = std::chrono::steady_clock::now() + timeout;
      int status;
      while (true) {
        auto reaped = waitpid(pid, &status, WNOHANG);
        if (reaped == pid) {
          pid = -1;
          if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            result = failure("Receiver exited unsuccessfully");
          }
          break;
        }
        if (reaped < 0 && errno != EINTR) {
          pid = -1;
          result = failure("Receiver could not be reaped");
          break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
          result = failure("Receiver exit timed out");
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    stop();
    return result;
  }
};
} // namespace

Result<std::unique_ptr<IBackend>> make_gem5_backend(const Gem5BackendOptions &o) {
  if (o.receiver_executable.empty() || o.timeout.count() <= 0 || o.timeout.count() > 3600000) {
    return Status{ErrorCode::InvalidArgument, "Receiver path and timeout (1..3600000 ms) required"};
  }
  if (access(o.receiver_executable.c_str(), X_OK) != 0) {
    return failure("Receiver executable is missing or not executable");
  }
  auto valid = metadata::validate_metadata({}, o.spec);
  if (!valid) {
    return valid.error();
  }
  try {
    auto transport = std::make_unique<ProcessTransport>(o.timeout);
    static std::atomic<uint64_t> epochs{1};
    auto epoch = epochs.fetch_add(1);
    std::vector<std::string> args{o.receiver_executable.string(), "--serve", "3",
                                  std::to_string(epoch), std::to_string(o.timeout.count())};
    for (auto n : {o.spec.channels, o.spec.banks, o.spec.rows, o.spec.row_bytes, o.spec.gb_bytes,
                   o.spec.gpr_bytes, o.spec.scratchpad_bytes}) {
      args.push_back(std::to_string(n));
    }
    std::vector<char *> argv;
    for (auto &arg : args) {
      argv.push_back(arg.data());
    }
    argv.push_back(nullptr);
    int sockets[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets)) {
      return failure("socketpair failed");
    }
    transport->fd = sockets[0];
    // Both original descriptors must be closed on exec; dup2 preserves only fd 3.
    if (fcntl(sockets[0], F_SETFD, FD_CLOEXEC) || fcntl(sockets[1], F_SETFD, FD_CLOEXEC)) {
      ::close(sockets[1]);
      return failure("Setting close-on-exec failed");
    }
    posix_spawn_file_actions_t actions;
    int error = posix_spawn_file_actions_init(&actions);
    if (error) {
      ::close(sockets[1]);
      return failure("Spawn actions failed");
    }
    error = posix_spawn_file_actions_adddup2(&actions, sockets[1], 3);
    if (!error && sockets[0] != 3) {
      error = posix_spawn_file_actions_addclose(&actions, sockets[0]);
    }
    if (!error && sockets[1] != 3) {
      error = posix_spawn_file_actions_addclose(&actions, sockets[1]);
    }
    if (!error) {
      error = posix_spawn(&transport->pid, argv[0], &actions, nullptr, argv.data(), environ);
    }
    posix_spawn_file_actions_destroy(&actions);
    ::close(sockets[1]);
    if (error) {
      transport->pid = -1;
      return failure("Could not spawn receiver executable");
    }
    auto client = execution::Client::connect(transport->fd, epoch, o.spec, {o.timeout});
    if (!client) {
      return client.error();
    }
    transport->client = std::move(client.value());
    return command::make_command_backend(o.spec, std::move(transport));
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Process backend allocation failed"};
  }
}
} // namespace ndp
