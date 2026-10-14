# NDP Library

Single-device L2/L3 library. The device management object is called **Session**.

Implemented: S0–S2, **FP16 (IEEE 754 binary16)** tensors, bounded FunctionalBackend,
FIFO asynchronous transfers, Event dependencies, shared immutable Tensor storage,
allocation/staging accounting, and draining Session shutdown. FunctionalBackend
executes on the host; it is not a hardware simulator or FPGA backend.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
./build/host_roundtrip
```

See [the example](examples/host_roundtrip.cpp) and
[implementation plan](docs/single-device-implementation-plan.md).

## FP16 and asynchronous use

```cpp
std::array<uint16_t, 2> bits{0x3c00, 0x4000}; // binary16 values 1, 2
// Check each Result before accessing value(); abbreviated here.
auto tensor = session.from_host({{{2}}, std::as_bytes(std::span(bits))});
// Input is copied to owned staging before return. `bits` may now be changed.
auto transfer = session.to_host_async(tensor.value());
auto result = transfer.value().wait(); // waits for upload, then download
```

The default and currently supported dtype is `DType::F16`, two bytes per element.
Host I/O accepts native-endian binary16 bit patterns, not float32 arrays or numeric
uint16 values. No implicit conversion is performed. F32/BF16 requests return
UnsupportedDType. Arithmetic and conversion kernels are not implemented yet.

## Ownership, execution and errors

- One submitting host thread per Session; a private worker executes backend calls
  in FIFO order. Tensor wait/Event status synchronize with that worker.
- `from_host` reserves device bytes and copies input to bounded staging before
  returning. Native allocation and upload run asynchronously. Submission validation
  and capacity errors are immediate; backend execution errors are observed via
  `Tensor::wait`, `Event::wait`, dependent transfers, synchronize, or close.
- `to_host_async` returns a HostTransfer. Its wait returns a separate owned HostTensor
  copy, only after success. `to_host` is the blocking convenience method.
- Tensor copies share immutable storage. Commands retain BufferLease references,
  so dropping all user handles does not release in-flight buffers.
- Dependent downloads skip backend execution if their upload failed. Independent
  operations continue. synchronize reports the first execution error persistently.
- `close` stops submissions, drains queued work, joins the worker, and shuts down
  the backend. Repeated close returns the same error. Destruction closes automatically;
  explicitly close to inspect errors. Buffers retained by surviving Tensor handles
  are released when those handles disappear. Backend release remains valid after shutdown.
- The blocking IBackend adapter must return (including errors or exceptions) only
  after accesses have stopped. A native async hardware adapter must establish that
  guarantee before returning; uncertain completion/quarantine needs a future interface.
- Session mismatch is rejected. Moved-from Session I/O is rejected; moved-from Tensor
  and Event handles are only assignable/destructible. A default Event is complete.
- Memory stats include device reservations before backend allocation, and device bytes
  retained by failed or surviving tensors. Staging stats count pending transfers only;
  completed user-held HostTransfers/HostTensors are excluded. There is no allocation pool.
- SessionOptions::max_staging_bytes defaults to 64 MiB. Budget exhaustion returns
  OutOfMemory rather than blocking submission. Wait for outstanding transfers or increase
  the budget; a single transfer must fit. No hidden chunking is performed.

S3 adds layout representations; S4 adds PNM/PIM compute and PreparedWeight. The
current backend interface is not a stable external ABI.
