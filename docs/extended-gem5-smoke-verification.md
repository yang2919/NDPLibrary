# Extended gem5 PIM: one-command Docker verification

This uses the same host-script pattern as the earlier stock FS verification:
start Docker Desktop, run one command on the Mac, and read the final result.
There is no interactive container setup, second terminal, or manual guest file
transfer. Run from the NDP repository root.

## Verified two-device Linear

The user confirmed that the two-device Linear test passed in SE Atomic and
Timing. It checks output shards and host reduction of FP32 partials.
See the [two-endpoint client procedure](gem5-se-external-client.md).

## Stateful decoder block

```sh
STAGE=SE SUITE=decoder_state CPU=atomic ./tools/gem5/pim_kernel/run.sh
```

After Atomic passes:

```sh
STAGE=SE SUITE=decoder_state CPU=timing SE_TIMEOUT=600 ./tools/gem5/pim_kernel/run.sh
```

Expected fixture marker alongside the SE verdict and PIM trace:

```text
NDP_PIM_DECODER_STATE_PASS schedules=3 tokens=7 checks=33
```

This is a deterministic tiny decoder block, not a checkpoint/model benchmark:
width 2, one attention head, FFN width 2, capacity 3 tokens, fixed test weights,
RMSNorm epsilon 0.5 and attention scale 0.5. RoPE uses exact rotations at
0, pi/2 and pi to isolate state and indexing behavior. Arithmetic rounds to FP16
at the same operation boundaries as the worker.

The graph includes attention RMSNorm, Q/K/V projections, RoPE, KV append, causal
attention (scores, softmax and weighted values), output projection/residual,
FFN RMSNorm, gate/up projections, SiLU/multiply, down projection and residual.

Three schedules reuse the same weights and cache allocations:

1. Two-token prefill, followed by one-token decode.
2. Reset caches and execute the same three tokens individually.
3. Reset caches again and execute only the first token.

Two-row Q/K/V projections use the worker CPU's GEMM/Linear path. Single-row
projections use PIM GEMV. Attention and pointwise arithmetic use the worker CPU;
FFN/output projections in this fixture run per token. This does not claim a
fully batched prefill FFN implementation or separate PNM-core execution.

Both tokens' K/V entries are appended before attention for a prefill chunk. Each
query then copies only its causal prefix out of the mutable cache. This tests
future-token exclusion even though future K/V values are already stored. Cache
snapshots include unused slots, which must remain zero after reset.

The 33 exact result checks cover attention scores, probabilities, block outputs
and full K/V snapshots. The independent Python reference also requires the three
schedules' corresponding outputs to agree. Downloading snapshots is observational:
no downloaded cache or intermediate is re-uploaded to continue computation.
Temporary handles are released between schedules while weights and cache handles
remain live. All 33 checks passed natively. The user reported SE Timing acceptance with
WrGb 688, AbMac 688 and RdReg 86 completions.

This fixture exercises generic worker APIs composed into a decoder graph. It is
not yet an end-to-end test of the production PyTorch decoder adapter, multi-head
GQA, cache capacity errors, parallel requests or multi-device behavior.

## Decoder arithmetic

```sh
STAGE=SE SUITE=decoder CPU=atomic ./tools/gem5/pim_kernel/run.sh
```

After Atomic passes:

```sh
STAGE=SE SUITE=decoder CPU=timing SE_TIMEOUT=600 ./tools/gem5/pim_kernel/run.sh
```

Expected marker alongside the SE verdict and PIM trace:

```text
NDP_PIM_DECODER_OPS_PASS checks=11 pim_gemv=1
```

| Check | Shape / behavior | Actual execution in this worker |
|---|---|---|
| GEMM | 2x3 times 3x2, scaled by 0.5 | Worker CPU |
| Linear | Two rows, transposed weight convention, bias and scaling | Worker CPU |
| FP32 partial | Preserve `1 + 2^-11` without an FP16 output conversion | Worker CPU; FP32 download |
| Transpose | 2x3 rectangular matrix | Worker CPU |
| Softmax | One unequal row and one uniform row | Worker CPU |
| RMSNorm | Two rows, nonzero epsilon and per-column weights | Worker CPU |
| Add | Resident GEMM plus normalization result | Worker CPU |
| SiLU | Negative, zero and positive inputs | Worker CPU |
| Multiply | Resident rounded SiLU output times gate values | Worker CPU |
| RoPE | Split-half rotation composed from gather, multiply and add | Worker CPU; no dedicated RoPE instruction |
| Final GEMV | Consume resident RoPE output with a 1x4 banked weight | PIM custom instructions/controller |

Expected values are computed independently in Python using simple equations and
explicit FP16 boundaries. This small fixture compares all successful reply bytes
exactly, including the FP32 partial. It is not a broad transcendental-error study;
any target-only mismatch should be localized before changing numeric tolerances.
The RoPE reference uses FP16 cosine/sine and rounds each multiply before addition,
matching the composition exposed by the current API.

The final GEMV provides PIM trace evidence for the mixed CPU/PIM path. Its trace
must not be interpreted as evidence that softmax, RMSNorm or GEMM ran in PIM.
Nor does this suite validate a separate PNM RISC-V core. It checks the existing
Linux/SE worker implementation, its operand representations and output handling.
Although intermediate results are downloaded for inspection, they remain resident
and are reused without re-upload in the linked operations.

All 11 checks passed natively. The user reported SE Timing acceptance with
WrGb 16, AbMac 16 and RdReg 1 completion.
This suite does not yet execute a full decoder block, KV cache update, causal
attention, multiple tokens or host reduction across devices.

## Resident chains and layout copies and layout copies

```sh
STAGE=SE SUITE=chain CPU=atomic ./tools/gem5/pim_kernel/run.sh
```

After Atomic passes, run the same fixture on Timing:

```sh
STAGE=SE SUITE=chain CPU=timing SE_TIMEOUT=600 ./tools/gem5/pim_kernel/run.sh
```

Expected fixture marker, in addition to the SE pass and PIM trace verdict:

```text
NDP_PIM_CHAIN_PASS cases=3 launches=18 intermediate_downloads=0
```

The three chains have shapes `[3] -> [H] -> [2]` with `H=3,257,1025`.
They cover a small rounding-sensitive case, a bank-group-of-rows tail and a
column-tile tail under the current profile. Two inputs reuse the same weights
and output handles. For each input:

1. First GEMV produces a logical FP16 intermediate.
2. Second GEMV consumes that resident result directly.
3. A separate path copies the intermediate to an input representation, through
   mutable storage and back (clearing the destination before restoration), then
   runs the second GEMV again.
4. Only the two final outputs are downloaded and compared to an independent dense
   reference that rounds to FP16 at the intermediate and final boundaries.

Before the chains, the first weight is copied dense -> banked -> dense -> banked.
Its dense source is cleared before the reverse copy and released before compute.
The suite therefore exercises independent representation lifetime, round-trip
conversion, repeated writes and direct versus copied input equivalence.

This validates logical FP16 representation handling in the current worker.
`T` and `B` copies use worker-side CPU staging through libpim; `Y` also keeps a
rounded logical input representation. No intermediate `D` request goes to the
external host, but this is not proof of zero-copy device data movement or a
separate PNM core. Physical guard regions and asynchronous races are not covered.

Native chain execution passed. The user reported SE Timing chain acceptance:
3 cases, 18 GEMVs, no intermediate downloads; WrGb 5,568, AbMac 5,568 and
RdReg 2,594 completions.

## Previous user verification: SE Timing

FS smoke, SE Atomic boundary and SE Atomic lifecycle have passed. The next check
runs the existing boundary and lifecycle fixtures on the Timing CPU, in order:

```sh
./tools/gem5/pim_kernel/run-timing.sh
```

The runner forces `STAGE=SE` and `CPU=timing`, reuses build caches, and stops on
the first failed build, timeout, numerical comparison or trace check. It does not
boot FS or automatically retry. Each suite has its own 600-second simulation
limit; compilation is outside that limit. To change the limit explicitly:

```sh
SE_TIMEOUT=1200 ./tools/gem5/pim_kernel/run-timing.sh
```

Expected final result after both suites:

```text
NDP_PIM_SE_TIMING_PASS suites=boundary,lifecycle
```

Individual results must include `NDP_PIM_SE_PASS cpu=timing`, the boundary or
lifecycle fixture marker, and nonzero completed `WrGb`, `AbMac` and `RdReg`
controller requests. This checks exact success replies, expected error handling,
and actual Timing completion traces. It is not a throughput benchmark, an
exhaustive ordering proof, or validation of concurrent requests.

Inspect the batch summary from the Mac:

```sh
batch=$(cat build/extended-gem5/latest-timing.txt)
cat "$batch/result.txt"
cat "$batch/runs.tsv"
```

`runs.tsv` points to each suite's independent run directory, containing its
`result.txt`, `driver.log`, `simulation.log`, `sim/program.out`, `sim/pim.trace`,
`sim/stats.txt` and configuration. Do not run other verification scripts
concurrently: they share build caches and the latest-run pointer. On failure,
report the failed suite and its logs. You may rerun only that suite using the
individual commands below; it does not turn the old batch failure into a pass.

The assistant prepared and syntax-checked this runner without executing SE or FS.

## Individual SE boundary verification

SE and FS simulations are run by the user. The assistant performs native tests,
static checks and prepares the target build/run procedure; it does not launch
simulations unless explicitly requested.

For the new tiling/padding and reuse cases, run on the Mac:

```sh
STAGE=SE SUITE=boundary CPU=atomic ./tools/gem5/pim_kernel/run.sh
```

Expected final output:

```text
NDP_PIM_SE_PASS cpu=atomic: exact replies and PIM trace
NDP_PIM_BOUNDARY_PASS cases=8 launches=24
Atomic PIM accesses: ...
```

For timing-path verification of the same cases:

```sh
STAGE=SE SUITE=boundary CPU=timing SE_TIMEOUT=600 ./tools/gem5/pim_kernel/run.sh
```

Atomic verifies functional execution of custom instructions and PIM accesses;
Timing additionally checks completed controller requests. Neither replaces FS
kernel/permission/cache-environment acceptance. No SE runtime is promised by these
initial timeout settings.

The dimensions are generated from the selected layout. With the 16-channel FP16
profile, the eight `(M,K)` cases are `(2,3)`, `(255,17)`, `(256,17)`, `(257,17)`,
`(3,1023)`, `(3,1024)`, `(3,1025)`, and `(257,1025)`. They cover the 256-bank-lane
and 1024-element column tile boundaries. Each performs three GEMVs: two inputs
with resident weights, then overwritten weights and a third input. Expected
values come from an independent dense dot product with exactly representable
binary fractions. All reply bytes must match. This tests logical tail/padding
results, not physical guard-byte integrity or every layout policy.

The native suite passed all 8 cases / 24 launches. The user also reported SE
Atomic acceptance with 15,936 WrGb, 15,936 AbMac and 3,108 RdReg accesses.
The user reported SE Timing acceptance as well.
An assistant-started SE job was stopped at the user's request during preparation;
it is not an accepted or failed simulator test. A subsequent run resumes SCons
preparation incrementally; an interrupted link may have to be repeated.

## SE lifecycle and error recovery verification

Run the new suite on the Mac:

```sh
STAGE=SE SUITE=lifecycle CPU=atomic ./tools/gem5/pim_kernel/run.sh
```

Expected:

```text
NDP_PIM_SE_PASS cpu=atomic: exact replies and PIM trace
NDP_PIM_LIFECYCLE_PASS errors=11 launches=4
```

The suite checks zero-sized allocation, unknown/released handles, unreadable
outputs, uninitialized inputs, output/input aliasing, a short upload payload,
incompatible dimensions, and double release. It checks that a failed GEMV makes
its output unreadable, a rejected upload preserves the old weights, and a new
allocation cannot revive an old handle. Four successful GEMVs in the same session
prove recovery after these rejected requests.

Successful reply bytes are compared exactly. Expected error frames must contain
status 1 and a nonempty diagnostic; diagnostic wording is not compared because
standard-library exceptions differ between Mac and Linux. The fixture's
`expected.bin` uses empty error bodies as checker placeholders, so byte-for-byte
`cmp` is not an appropriate checker for this suite. The embedded executable
performs the framed comparison automatically.

These are complete protocol frames containing invalid requests. Truncated stream
handling, disconnect, out-of-memory and concurrent sessions are not covered by
this suite. Native and user-run SE Atomic validation passed; SE Timing also passed according to the user. To check
Timing behavior, use `CPU=timing SE_TIMEOUT=600` with the same command.

## FS smoke verification

```sh
STAGE=FS BOOT_TIMEOUT=7200 ./tools/gem5/pim_kernel/run.sh
```

The script automatically:

1. Builds missing cross-compiler and validation Docker images.
2. Generates the 16-channel LPDDR6 FP16/FP32 profile and builds the worker/smoke ELF.
3. Builds the supplied `gem5-ParallelPIM-main` incrementally.
4. Downloads and caches the versioned guest resources.
5. Builds the supplier's patched kernel and disk if not already prepared.
6. Boots FS, runs the embedded two-GEMV test, and checks its replies and controller trace.

The FS runner uses the supplier's `--boot no-systemd` path: it boots the real
Linux kernel and starts the guest test without initializing the full systemd
service set. It uses Atomic through guest-script transfer and executable
restoration, then switches to Timing at hypercall 4 immediately before running
the test (`--switch-at work`). This avoids timing-simulating the base64/file setup.
This remains full-system verification, not SE. Expected progress includes
`First exit: kernel booted`, then `Second exit: started after_boot.sh` and the
CPU-switch message and `PIM-FS work begin` immediately before the test begins.
Program startup and libpim memory mapping still run on Timing; this change does
not claim a kernel-only ROI or a measured completion time.

Expected final result:

```text
NDP_PIM_FS_PASS cpu=timing: exact replies and PIM trace
Completed controller requests: ...
```

The checker requires the exact numerical marker, `PIM-FS exit 0`, and completed
`WrGb`, `AbMac` and `RdReg` requests in the PIM trace. Booting successfully alone
is not a pass. The matrix is `[[1,2,3],[-1,0,2]]`; two inputs reuse that weight and
produce `[6,1]` and `[-1,-4]`. The checker does not prescribe an exact dynamic
instruction count or claim separate PNM-core execution.

Initial setup builds gem5 and a Linux kernel and can be substantial. Later runs
reuse Docker images, compiled gem5 objects, downloads and the patched kernel/disk.
The boot timeout covers simulation, not image downloads or builds. There are no
automatic simulation retries.

## Optional: prepare first, then run

To separate lengthy preparation from verification:

```sh
STAGE=PREPARE ./tools/gem5/pim_kernel/run.sh
STAGE=FS BOOT_TIMEOUT=7200 ./tools/gem5/pim_kernel/run.sh
```

`PREPARE` builds artifacts and prepares FS resources but does not simulate or boot.
Its result is `NDP_PIM_PREPARE_PASS`, which is not an execution-verification pass.

For a smaller SE diagnostic before FS:

```sh
STAGE=SE ./tools/gem5/pim_kernel/run.sh
```

SE prepares the worker and simulator but skips kernel/disk preparation. It runs
the same NDP FP16 test with a default 180-second simulation timeout. Expected:
`NDP_PIM_SE_PASS`. This convenience path does not run the separate supplier BF16
baseline; use that test when isolating a supplier failure. SE does not replace FS.

## Inspect results

Every invocation prints its unique results directory and updates `latest-run.txt`.
From the Mac:

```sh
run_dir=$(cat build/extended-gem5/latest-run.txt)
cat "$run_dir/result.txt"
```

| File | Contents |
|---|---|
| `driver.log` | Docker setup, worker/gem5/kernel build output and run status |
| `simulation.log` | Simulator console output, when a simulation was reached |
| `result.txt` | Final pass or failure status |
| `sim/board.platform.terminal` | FS guest serial output |
| `sim/program.out` | SE program output |
| `sim/pim.trace` | Actual PIM controller operations |
| `sim/config.ini`, `sim/stats.txt` | Simulated configuration and statistics |
| `docker-image.json`, `artifacts.sha256`, `*.manifest.json` | Build/image identities |

On failure, send `result.txt` and the relevant portion of `driver.log`; include
`simulation.log` and guest output if boot was reached. A missing pass marker is
reported as failure rather than guessed from the simulator's exit status.

## Configuration and reuse

| Setting | Default | Use |
|---|---|---|
| `STAGE` | `SE` | `PREPARE`, `SE` or `FS` |
| `SUITE` | `smoke` | `smoke`, `boundary`, `lifecycle`, `chain`, `decoder` or `decoder_state` |
| `CPU` | `timing` | SE: `atomic` or `timing`; FS: `timing` only |
| `JOBS` | `2` | Number of Docker CPUs/build jobs; must not exceed CPUs assigned to Docker Desktop |
| `BOOT_TIMEOUT` | `3600` | FS simulation wall-clock limit in seconds |
| `SE_TIMEOUT` | `180` | SE simulation wall-clock limit in seconds |
| `REBUILD` | `0` | Set `1` to rerun both Docker image builds after Dockerfile changes |
| `REBUILD_KERNEL` | `0` | Set `1` to force patched kernel/disk regeneration |

Examples:

```sh
JOBS=1 STAGE=FS BOOT_TIMEOUT=7200 ./tools/gem5/pim_kernel/run.sh
REBUILD=1 STAGE=PREPARE ./tools/gem5/pim_kernel/run.sh
```

The runner resolves the repository path automatically. Mac paths are bind-mounted
at `/workspace`. Linux resources and kernel files persist in the Docker volume
`ndp-pim-linux-data`; outputs persist under `build/extended-gem5` on the Mac.
The simulator executable and incremental build cache are under
`build/RISCV_RAMULATOR2`. The runner passes an absolute target to SCons because
gem5 resolves relative build targets against the invocation directory even with
`scons -C`. Existing builds at this location are reused.
No privileged container or guest filesystem mount is required. Do not run two
instances simultaneously against these shared build and kernel caches.

Kernel reuse checks the stock resources, supplier build/patch/config files,
bridge source files and compiler version. Changes trigger regeneration. The old
stock x86 FS kernel and simulator are not reused: this target needs RISC-V and
user cache-block operations. The orchestration pattern is reused, not the old
machine image.

## Troubleshooting

- Docker unavailable: start Docker Desktop, check `docker info`, then rerun.
- Build killed or exit 137: inspect Docker memory/disk capacity; try `JOBS=1`.
- Exit 124: simulation exceeded its time limit; inspect progress before increasing it.
- Download failure: fix connectivity and rerun; completed builds/resources are retained.
- Illegal instruction or cache error: retain the serial/trace output; do not disable checks.
- Repeated SC-failure warnings before `Second exit`: inspect guest serial progress.
  SC is RISC-V Store-Conditional, not a PIM command. The inspected RISC-V handler
  increments the warning counter without resetting it on success, so the word
  "consecutive" alone is not evidence of a permanent stall. Earlier full-systemd
  runs spent this phase starting services; the runner now uses `no-systemd`.
- Trace check fails despite correct values: retain `sim/pim.trace`; numerical success
  alone does not prove the required timing/controller path.

## What has been checked

The user reported successful FS smoke verification with 32 WrGb, 32 AbMac and
4 RdReg completions. That result covers the original two-launch smoke fixture.
The new boundary suite passed native execution (8 dimensions, 24 launches).
Runner syntax and Atomic/Timing result-checker behavior were checked locally.
The user reported boundary SE Atomic acceptance (15,936 WrGb, 15,936 AbMac,
3,108 RdReg). The user reported boundary SE Timing acceptance. The lifecycle suite passed
natively and in user-run SE Atomic with 11 expected errors and 4 successful
launches (WrGb 64, AbMac 64, RdReg 8). The user also reported combined SE Timing acceptance for both suites. Simulation execution remains the user's step.

## Next integration gate

For an external Python session connected to a persistent simulated worker, use
[the SE external-client procedure](gem5-se-external-client.md). That endpoint
requires separate user acceptance; the completed fixture tests do not cover it.
