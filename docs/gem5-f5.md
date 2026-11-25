# F5: guest decoder integration

F5 runs a small decoder layer through the public Session API and the F4 process
backend. The receiver executes the existing PNM/PIM emulator. Both application
and receiver run inside stock gem5's Linux guest when using the FS runner.
No gem5 source changes or NDP device model are introduced.

F3, F4 and F5 FS verification have been reported complete by the user.
Implementation/native validation and user-reported FS acceptance are tracked
separately. The assistant has not independently inspected the FS logs. See [F5 user verification](stock-gem5-verification.md).

## Example and execution path

[`decoder_probe.cpp`](../tools/gem5/guest/decoder_probe.cpp) implements the layer
explicitly in `step()`, rather than calling the prebuilt `decoder_block()`:

| Layer operation | API | Execution |
| --- | --- | --- |
| Attention normalization | `Session::rms_norm` | PNM |
| Q/K/V projections | `Session::linear` | PNM for prefill, PIM for decode |
| Head reshape and RoPE | `Session::reshape`, `rope` | Metadata / PNM operations |
| Append K/V and causal GQA | `KVCache::append`, `KVCache::attention` | Generic Session buffer and PNM operations |
| Output projection | `Session::linear` | PNM for prefill, PIM for decode |
| Residual and FFN normalization | `Session::add`, `Session::rms_norm` | PNM |
| Gate/up/down projections | `Session::linear` | PNM for prefill, PIM for decode |
| SwiGLU and final residual | `Session::silu`, `mul`, `add` | PNM |

The fixture has model width 4, two query heads, one KV head, head dimension 2,
FFN width 6, and KV capacity 8. These dimensions keep guest execution small;
this is a functional integration fixture, not a performance benchmark.

`example::weights()` prepares dense and PIM layouts once. Projection calls use
`RequirePrepared`; the same weights and cache allocations are reused across
requests. PIM projection results are consumed by subsequent PNM operations through
the existing layout and command contracts. There is no host readback between
layer operations. The probe downloads each final layer output for verification.

## Checks

`run()` first executes the local command emulator and then the process backend
with the same geometry: 2 channels, 2 banks, 512 rows, 64-byte rows/GB,
256-byte GPR, and 128-byte scratchpad. Each backend executes three requests:

- A: two-token PNM prefill, followed by three single-token PIM projection steps.
- B: the same schedule with different deterministic input values.
- A again: the original inputs after another KV reset.

For each backend, all four outputs of the repeated A request must exactly match
the first A. The B prefill must differ so that the fixture cannot accidentally
check identical inputs. Scheduled and committed cache lengths must return to zero
after reset; committed length advances from 2 to 5 within each request.

All 12 process-backend output byte arrays must exactly match the corresponding
local FP16 output arrays. Both paths share numerical kernels and layer composition;
this checks transport/integration equivalence, not an independent mathematical
oracle. Existing numerical/decoder contract tests remain responsible for kernel
and layer correctness. The input schedule represents decoder invocations, not
an autoregressive text-generation/tokenizer implementation.

Memory checks compare live bytes and live allocation counts against the resident
weights/tables/cache baseline after every step. Staging must be zero. After all
model and cache handles leave scope, live allocations and bytes must be zero;
workspace must be zero after Session close. The owned receiver must be reaped.
Peak memory is intentionally not expected to return to zero.

## Native run

```sh
cmake -S . -B build
cmake --build build -j 4
ctest --test-dir build --output-on-failure
# Decoder integration only:
./build/ndp_decoder_probe ./build/ndp_execution_receiver
```

The `ndp_decoder_probe` target links `ndp_gem5` and `ndp_llm`. Production L1/L2/L3
APIs and numerical behavior did not need changes for F5. The FS runner uses the
same source code and injects two static x86-64 binaries into the guest.

Native validation on 2026-11-06 passed all 25 CTest cases in both regular and
ASan/UBSan builds. Both FS source lists also compiled directly with the native
compiler and passed the F5 probe. Shell syntax and Python compilation checks
passed. Linux cross-compilation and gem5 FS execution were not run by the assistant.

## Integration report and limits

| Stage | Delivered behavior | FS acceptance |
| --- | --- | --- |
| F0 | Stock Linux boot and C++ smoke program | Previously verified |
| F1 | Command transport contract and wire codec | Native contract checks; no separate FS run |
| F2 | Metadata framing, byte identity and rejection | User reported complete |
| F3 | Persistent emulator execution and state | User reported complete |
| F4 | Session dependencies, process lifetime and fault cleanup | User reported complete |
| F5 | Explicit decoder, repeated requests, KV reset and reclamation | User reported complete |

The resulting path validates guest software IPC, command delivery and functional
execution. It does not validate NDP MMIO/doorbells, a custom gem5 ISA/device,
hardware timing, physical bank mapping or coherence. PNM remains functional
emulation, not a RISC-V ELF executor. The F3 frame-size limits remain unchanged.
A future PNM/PIM gem5 integration needs a separate device-facing transport contract
and validation; pure FS PASS markers must not be used as evidence for it.
