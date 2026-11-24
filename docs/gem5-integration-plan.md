# Stock gem5 full-system integration

This F0–F5 plan replaces the earlier G0–G6 SE/device-model proposal. The entire
plan runs on unmodified gem5 in full-system (FS) mode. No NDP SimObject, custom
ISA, MMIO region, doorbell, DMA engine, interrupt, or guest driver is required.

## Implementation and validation ownership

Effective 2026-11-03, the assistant implements changes and validates them only in
native environments. The user owns all gem5 FS builds, resource downloads, runs,
and acceptance checks. The assistant must not launch, poll, or automatically retry
FS jobs unless the user explicitly changes this arrangement.

For each milestone, the assistant delivers:

- The implementation and relevant native test results, with platform limitations.
- Exact user-run preparation, build and FS execution commands.
- Expected output, numerical tolerances where applicable, and pass/fail criteria.
- Log locations and the smallest set of evidence needed to diagnose a failure.

Validate codecs, IPC, emulator arithmetic and lifecycle behavior natively before
handing them over for guest execution. Native Linux processes, when needed for
Linux-specific behavior, are distinct from simulated guest execution. A macOS
native result must not be presented as Linux or FS validation.

Track **implementation/native validation** and **user FS validation** separately.
Native success alone does not complete an FS acceptance criterion in the milestone
table. Mark unexecuted guest checks as **awaiting user FS validation**. Analyze
user-provided results and reproduce failures natively where possible; hand revised
commands back to the user when another FS run is necessary.

The earlier F0 result remains historical evidence, not evidence for later changes.
F1–F5 retain their existing technical scope and acceptance criteria.

## Execution boundary

```text
Mac Docker Linux container
  stock gem5 process
    simulated x86 machine running Linux
      application → L3 → L2 Session → existing NDP command builder
        → guest IPC sender → Unix domain socket → guest receiver
                                                  → command emulator
```

Both sender and receiver execute inside the simulated guest. The receiver uses
ordinary guest memory for emulated bank memory, GPR, Global Buffer and scratchpad.
These are software representations, not new hardware address spaces in gem5.
Existing command address-space tags and offsets retain their meaning within that
representation. Guest virtual/physical addresses and host pointers are not NDP
addresses.

L2/L3, allocation ownership, geometry, layout planning and the command builder
remain the compatibility baseline. Extract a private command transport from the
existing L1 backend; do not reconstruct Tensor metadata or add a second planner.
The receiver validates and executes already prepared commands.

## What the stock target proves

The metadata probe verifies exact serialization, framing, ordering, address tags,
lengths and receiver interpretation across guest IPC. Receipt is not successful
execution and must never mark an output Tensor ready.

The executing receiver later verifies functional results and persistent memory
using the existing emulator. gem5 simulates the CPU instructions, Linux syscalls
and memory traffic implementing this software. A socket message is not one gem5
Packet, and successful IPC does not establish delivery to an NDP controller.
Neither mode measures PNM/PIM hardware latency, bandwidth or coherence. A future
hardware fork needs a separately designed adapter and validation effort.

## Milestones

| Stage | Implementation | Completion criterion |
|---|---|---|
| F0 | Pinned stock gem5, Linux guest, C++ smoke program and Docker runner | Actual FS boot and guest success markers; reproducible command and logs |
| F1 | Private transport boundary and versioned command codec | Existing backend tests pass; round-trip and malformed-record tests pass |
| F2 | Guest sender/receiver over Unix domain sockets, metadata-only probe | Exact bytes/fields/order verified; fragmented reads/writes and rejection tested |
| F3 | Persistent receiver using the existing command emulator | Upload/execute/download and PNM/PIM results match native reference |
| F4 | Session lifetimes, dependencies and error handling | Resident buffers, copy/fill, close and fault behavior preserve blocking completion |
| F5 | Guest decoder example and integration report | PNM prefill/PIM projections, repeated requests, KV reset and reclamation pass |

F0 uses AtomicSimpleCPU to reduce boot cost on the ARM Mac, where x86 KVM cannot
be used. This is an FS functional baseline, not timing-packet validation. Later
transport tests may use TimingSimpleCPU when memory-traffic observation is useful;
receiver-side byte validation remains necessary in either CPU mode.

F1's codec must encode explicit fixed-width fields, lengths, opcodes and existing
address tags. Never serialize C++ variants, padding, pointers or native `size_t`.
Version and numerical-profile checks must reject mismatches rather than silently
reinterpret the FP16 extension model.

F2 begins with one outstanding batch and bounded messages. Its report must say
`metadata accepted`, not `execution complete`. Test truncated/oversized messages,
unknown opcodes, stale sequence numbers and invalid address ranges.

F3 keeps one receiver memory image across requests. Completion follows all memory
effects. Preserve the emulator's preflight behavior and numerical profile; do not
add a second compute implementation to the receiver.

F4 must handle uncertain completion explicitly. A timeout alone does not prove
that a command stopped accessing a buffer. Drain or stop the owned receiver before
reclaiming buffers, fault the session when completion is unknown, and never blindly
replay a possibly executed batch after reconnecting.

## F0 implementation

F0 passed on 2026-11-02: stock gem5 in Docker booted Linux with the image's
existing script as init and executed the guest C++ smoke program. F2–F5 FS verification has also been reported complete by the user; F0 itself does not validate NDP transport integration.

See [Stock FS quick start](gem5-fs.md). The tooling is separate from the regular
CMake build and does not change the library or require gem5 for native tests.
Large resources, binaries and run logs are kept under ignored `build-gem5-fs/`.

## F1 implementation

F1 implementation and native validation are complete (2026-11-03). The NDP
backend now calls a private synchronous `ICommandTransport`; its initial adapter
owns the existing emulator. The bounded version-1 codec supports all ten existing
command types with explicit wire IDs and fields. See [the protocol and native test
commands](command-transport.md).

F1 adds no IPC or guest runner. FS verification of transport begins with the
user-run F2 sender/receiver; no new FS result is claimed for F1.

## F2 implementation and user validation

F2 implementation and native tests passed on 2026-11-05. A two-process Unix
socket probe checks F1 command bytes, framing, identity, ranges and rejections.
It is metadata-only and does not implement an executing backend. **The user has reported F2 FS verification complete.** See [F2 commands, expected output and failure reporting](gem5-f2.md).

## F3 implementation and user validation

F3 now provides upload/submit/download and Close over a persistent emulator
receiver, with explicit execution completions and a geometry/profile handshake.
Native regression and sanitizer tests passed. **The user has reported F3 FS verification complete.**
See [the implementation](gem5-f3.md) and [user verification steps](gem5-f3-user-verification.md).
Session/process lifetime integration is implemented in F4.

## F4 implementation and user validation

F4 adds an owned process transport and `make_gem5_backend()` while reusing the
existing NDP planner and Session worker. Receiver termination/reaping precedes
uncertain-operation failure publication. Native tests cover resident buffers,
dependencies, close, destructor cleanup and injected transport failures.
**The user has reported F4 FS verification complete.** See [F4 implementation](gem5-f4.md)
and [user verification steps](gem5-f4-user-verification.md).

## F5 implementation and user validation

F5 adds an explicit decoder layer probe with three requests (A, B, A), each
containing a PNM prefill and three PIM-projection decode steps. It compares all
12 remote outputs with the local command emulator and checks KV reset, temporary
and resident allocation reclamation, workspace release and receiver reaping.
**The user has reported both F4 and F5 FS verification complete.**
See [the integration report](gem5-f5.md) and [user verification](gem5-f5-user-verification.md).

## After F0–F5

The stock-FS stages are complete, including user-reported F4 and F5 acceptance.
The next target-specific work requires inspecting the PNM/PIM-enabled gem5 fork.
See [the extension handoff](gem5-extension-handoff.md) for prerequisites and proposed
steps. This does not reintroduce the excluded custom-device-model implementation.

## Upstream references

- [Pinned gem5 source](https://github.com/gem5/gem5/tree/b9da2bfe1e21f6efa2730edd4e80e58c5d965458)
- [X86 FS tutorial](https://www.gem5.org/documentation/gem5-stdlib/x86-full-system-tutorial)
- [Original Ubuntu image scripts](https://github.com/gem5/gem5-resources/tree/v22.1.0.0/src/x86-ubuntu)
- [Memory-system Requests and Packets](https://www.gem5.org/documentation/general_docs/memory_system/)
