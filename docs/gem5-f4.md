# F4: Session integration with an owned receiver

F3 FS verification has been reported complete by the user. F4 connects the
existing L2 Session to the persistent software emulator through a new L1
transport. The user has reported F4 FS verification complete. FS reruns remain user-operated; see
[the separate F4 verification procedure](gem5-f4-user-verification.md).

## Execution path and ownership

```text
Application (inside the guest)
  Session: FIFO worker, dependencies, staging and buffer leases
    Existing NDP backend: allocations, layouts, plans and command lowering
      ProcessTransport: owns socket, receiver PID and execution client
        Unix socket: F3 requests and execution completions
          Receiver process: persistent PNM/PIM emulator
```

The host machine launches stock gem5. Both application and receiver run inside
its Linux guest. The same binaries can also run natively. This is software
execution through ordinary guest IPC, not an NDP MMIO device or hardware timing
model. No gem5 C++ changes are required.

`make_gem5_backend()` reuses the NDP backend's existing planner and allocation
map through the private `make_command_backend()` factory. The receiver receives
already lowered commands, addresses and payloads; it does not reconstruct Tensor
metadata or KernelPlans. No new L2 scheduler or worker thread was introduced.

## Creating a Session

Link your CMake target with `ndp_gem5` and build `ndp_execution_receiver`.
The factory is available on Unix platforms:

```cpp
#include "ndp/l1/gem5/backend.hpp"
#include "ndp/l2/session.hpp"
#include <stdexcept>

ndp::Gem5BackendOptions options;
options.receiver_executable = "/absolute/path/to/ndp_execution_receiver";
auto spec = ndp::load_ndp_spec("configs/ndp-small.ini");
if (!spec) throw std::runtime_error(spec.error().message);
options.spec = spec.value();
options.timeout = std::chrono::milliseconds(30000);

auto backend = ndp::make_gem5_backend(options);
if (!backend) throw std::runtime_error(backend.error().message);
auto created = ndp::Session::create(std::move(backend.value()));
if (!created) throw std::runtime_error(created.error().message);
auto session = std::move(created.value());
// Use from_host(), matvec(..., {Engine::PIM}), add(), copy_into(), etc.
// Check asynchronous Tensor/Event completion and the final close result.
auto closed = session.close();
if (!closed) throw std::runtime_error(closed.error().message);
```

| Option | Meaning |
| --- | --- |
| `receiver_executable` | Executable path passed to `posix_spawn`; no PATH lookup. Use an absolute path. |
| `spec` | Existing `NdpSpec`; passed to the receiver and checked in the F3 geometry/profile handshake. |
| `timeout` | Positive milliseconds, at most 3,600,000. Bounds each frame send/receive and graceful child-exit wait. Default 30 seconds. |

The timeout is not a whole-Session deadline. Idle time between requests does not
expire the receiver. Once a request begins, existing framing deadlines apply.
The process backend owns its child exclusively: applications must not reap that
child independently or configure automatic child reaping.

## Completion, errors and memory lifetime

| Situation | Behavior |
| --- | --- |
| Successful request | Backend returns only after the receiver's execution completion. Session then publishes its Event. |
| Dependent operation | Existing Session dependency handling waits for its inputs; failed inputs produce `DependencyFailed`. |
| Resident data | Receiver memory persists across requests. Reusing weights or buffers does not download them. |
| Copy, fill and snapshot | Existing lowering is reused. Snapshot creates independent immutable storage. |
| Acknowledged operation error | Error is propagated; the connection remains usable because completion is known. |
| Timeout, disconnect or invalid completion | Transport becomes unusable, terminates and reaps its receiver before returning failure. No reconnect or replay. Later transport operations fail. |
| Explicit close | Session drains its queue, sends Close, waits for receiver teardown/exit, and reaps it. Repeated close preserves the Session result. |
| Destructor or incomplete construction | Owned process and socket are cleaned up. Tensor/DeviceBuffer handles may outlive Session close; releasing them requires only host allocation bookkeeping. |

Session records the first failed operation and reports it from synchronize/close.
It can still accept API submissions after a transport failure, but dependent work
fails and independent device requests cannot reach the terminated receiver. Pure
host metadata operations need not fail. Recovery requires a new backend/Session
and re-uploading data; remote state is discarded.

Forced cleanup uses SIGKILL followed by waitpid. Reaping takes precedence over
returning quickly: a timeout alone is not proof that device work has stopped.
The receiver has no descendants. This contract assumes normal OS process ownership.

## Runnable example and validation

[`session_probe.cpp`](../tools/gem5/guest/session_probe.cpp) explicitly queues:

1. FP16 weight and vector uploads.
2. PIM GEMV followed by a dependent PNM addition.
3. Buffer copy, immutable snapshot, clear and another snapshot.
4. Another GEMV using the same resident inputs.
5. Pending work followed by close while Tensor/DeviceBuffer handles still exist.

The probe also injects timeout, disconnect and wrong-completion failures through
its own test receiver mode. It checks dependency failure, no replay, child reaping
before failed Events become observable, repeated close, destructor cleanup and
idle Sessions. The production receiver has no fault-injection switches.

```sh
cmake -S . -B build
cmake --build build -j 4
ctest --test-dir build --output-on-failure
# Focused native run:
./build/ndp_session_probe ./build/ndp_execution_receiver
```

Native validation on 2026-11-06 passed all 24 CTest cases in both the regular
and AddressSanitizer/UndefinedBehaviorSanitizer builds. Shell syntax and Python
compilation checks passed for the FS runner. Both guest source lists also built
and passed the F4 probe using the native compiler without CMake. These results do not establish
Linux cross-compilation or F4 FS success.

F4 retains the F3 8 MiB frame limit: upload data is at most 8 MiB minus 18 bytes,
download data at most 8 MiB minus one byte, and encoded command batches at most
8 MiB minus one byte. Larger requests are rejected; automatic transfer chunking
is not implemented. FP16 numerical behavior and functional PNM kernels remain
unchanged. The small decoder integration fixture is documented in [F5](gem5-f5.md).
