# Stock gem5 FS baseline (F0)

This harness boots a real Linux kernel in unmodified gem5 and runs a statically
linked C++ smoke program inside that guest. It does not yet connect the NDP library.
To run the subsequent guest checks, follow [F2–F5 verification](stock-gem5-verification.md).

For the F2 metadata probe, use [the F2 procedure](gem5-f2.md) with `STAGE=F2`.
The default invocation below remains F0. The user has reported F2–F5 FS validation
complete. Stage-specific procedures remain available for reproducible reruns; see
[the integration report](gem5-f5.md).

## Who runs what

From 2026-11-03 onward, the assistant maintains the code and these instructions,
and performs native checks only. The user prepares and runs gem5 FS and verifies
the results. FS jobs are not started, monitored or retried by the assistant unless
explicitly requested under a changed arrangement.

Native tests can validate library behavior and script syntax, but cannot prove
that Linux boots or that a guest program executes. Report these checks separately
from user-confirmed FS results. The historical F0 result below is retained for
reference; it does not validate future changes.

## Historical verified result (2026-11-02)

F0 passed on this ARM64 Mac using Docker. The final runner exited with code 0;
`config.ini` records `full_system=true`. Evidence is retained locally in
`build-gem5-fs/run-biBl7n6y/`, including `result.txt`, the manifest and serial log:

```text
guest=Linux arch=x86_64
NDP_FS_SMOKE_PASS
NDP_FS_SCRIPT_COMPLETE
```

An earlier systemd run reached the Ubuntu multi-user target and automatic root
login, but did not complete the smoke test. It reported failures in optional
services and segmentation faults during login initialization. The verified F0
baseline instead runs the image's init script directly from Linux and uses a
line-wrapped heredoc for binary transfer. It does not certify the health of all
Ubuntu services. Neither run adds or changes a gem5 device model.

## Baseline

| Component | Selection |
|---|---|
| Build environment | Docker Linux, native host architecture; Ubuntu 22.04 |
| Simulator | gem5 v24.1.0.3, commit `b9da2bfe1e21f6efa2730edd4e80e58c5d965458` |
| Simulated machine | X86Board, one AtomicSimpleCPU, 3 GiB DDR4, classic caches |
| Guest | Official Ubuntu 18.04 disk, Linux 5.4.49 kernel, v22-1 resources |
| Guest compiler | Ubuntu's `x86_64-linux-gnu-g++`, C++17, static linking |
| Device extensions | None |

The ARM64 Mac builds and runs gem5 natively in Docker; gem5 simulates the x86
instructions. This does not require privileged Docker or `/dev/kvm`. Atomic CPU
statistics are not NDP performance measurements.

The older guest pair was selected because its official artifact URLs and MD5
checksums were accessible while the Ubuntu 24.04 resource lookup was unavailable.
`tools/gem5/fs/resources.json` freezes those URLs and verifies uncompressed bytes.
The simulator source is pinned; Ubuntu package repositories can change. Each run
records the actual compiler version and artifact SHA-256 values for comparison.

## User-run preparation and execution

Start Docker Desktop and open a terminal at the repository root. Check that the
Docker daemon is available:

```sh
docker info
```

Then run the harness and record its exit status (the assignment must immediately
follow the run command):

```sh
./tools/gem5/fs/run.sh
fs_status=$?
printf 'FS runner exit status: %s\n' "$fs_status"
```

The first run builds gem5 with two parallel compiler jobs, downloads about 715 MB
of compressed resources, expands the disk image, cross-compiles the C++ binary,
and boots guest Linux. Allow several GB of free disk space and at least 8 GB of
Docker memory. Compilation and software-simulated boot can take a long time.
Later runs reuse the existing Docker image and verified resources. Set
`REBUILD=1` to explicitly rebuild the image after changing its Dockerfile.

```sh
JOBS=2 BOOT_TIMEOUT=7200 ./tools/gem5/fs/run.sh
```

`BOOT_TIMEOUT` bounds the simulator run in wall-clock seconds (default 3600),
excluding the image build and resource preparation; forced termination follows
after a further 30 seconds if needed. Timeout, nonzero simulator
exit, or missing guest markers fails the command. No automatic replay occurs.

## User acceptance checks

A successful run prints the following terminal message and exits with status 0:

```text
F0 PASS: Linux guest executed the C++ smoke program.
FS runner exit status: 0
```

Locate this run's logs from the repository root:

```sh
run_name=$(basename "$(cat build-gem5-fs/latest-run.txt)")
run_dir="build-gem5-fs/$run_name"
cat "$run_dir/result.txt"
grep '^full_system=' "$run_dir/config.ini"
tail -n 20 "$run_dir/board.pc.com_1.device"
```

Confirm all of the following:

- The runner exited with status 0 and `result.txt` contains `F0 PASS`.
- `config.ini` contains `full_system=true`.
- The serial log contains `guest=Linux arch=x86_64`, followed by complete lines
  `NDP_FS_SMOKE_PASS` and `NDP_FS_SCRIPT_COMPLETE`.
- The run directory and manifest belong to this invocation, not an older run.

A kernel boot banner alone is not a pass. Missing markers, timeout or a nonzero
exit status means the run has not passed. Do not infer success from an old
`result.txt`: failures during build, download or cross-compilation can occur
before `latest-run.txt` is updated.

## Failure triage and reporting

| Symptom | Evidence to check or share |
|---|---|
| Docker is unavailable | `docker info` error; start Docker Desktop before running again |
| Build, download or cross-link failure | Terminal error and exit status; a new run directory may not exist |
| Timeout (usually status 124, or 137 after forced termination) | Last 40 lines of `console.log` and serial log; absence of success markers |
| Simulator exits or guest crashes | Terminal exit status and final error lines from both logs |
| Simulator exits without `F0 PASS` | Serial markers and `result.txt`; simulator exit alone is insufficient |

For a failure after a run directory was created:

```sh
tail -n 40 "$run_dir/console.log"
tail -n 40 "$run_dir/board.pc.com_1.device"
```

Send the run command, exit status, run directory name and relevant final log
lines. For success, the three guest lines above and `F0 PASS` are sufficient.
Include `manifest.txt` if versions or artifacts differ. Full build logs, disk
images and repeated progress updates are unnecessary unless specifically needed
for diagnosis. A longer `BOOT_TIMEOUT` may be appropriate when the guest is still
making progress; changing the limit and rerunning is a user decision.

## Native checks performed by the assistant

For edits to this harness, shell syntax and Python parsing can be checked without
Docker, downloads or gem5 execution:

```sh
bash -n tools/gem5/fs/run.sh tools/gem5/fs/run-container.sh
python3 -m py_compile tools/gem5/fs/boot.py tools/gem5/fs/fetch.py
```

Python parsing does not import or validate gem5 APIs. `smoke.cpp` deliberately
requires Linux/x86-64 at runtime, so running it on the ARM Mac is not an equivalent
acceptance test. For library changes, run the relevant native build/tests and
report any guest-only checks as awaiting user verification.

## Files and evidence

| File | Purpose |
|---|---|
| `tools/gem5/fs/Dockerfile` | Build pinned stock gem5 and guest cross-compiler |
| `tools/gem5/fs/resources.json` | Official kernel/disk URLs and checksums |
| `tools/gem5/fs/fetch.py` | Download, expand and verify resources |
| `tools/gem5/fs/boot.py` | Configure the stock FS board and inject a readfile script |
| `tools/gem5/fs/smoke.cpp` | Check Linux identity and basic C++ execution |
| `tools/gem5/fs/run-container.sh` | Run, record artifacts, and verify guest markers |

Each invocation creates a fresh `build-gem5-fs/run-*` directory. Its path is written
to `build-gem5-fs/latest-run.txt` using the container's `/output` mount prefix.

- `manifest.txt`: source commit, simulator/compiler versions and artifact hashes.
- `console.log`: simulator output and diagnostics.
- `board.pc.com_1.device`: guest serial console, including the smoke output.
- `config.ini`, `config.json`, `stats.txt`: gem5-generated configuration/statistics.
- `result.txt`: created only after both `NDP_FS_SMOKE_PASS` and
  `NDP_FS_SCRIPT_COMPLETE` appear as complete serial-console lines.

On ARM64 Jammy, the packaged cross `libm.a` linker script points to native
absolute paths. The runner creates a corrected script under the output directory
which refers to the installed cross archives; it does not modify the toolchain.

The binary is base64-encoded as a line-wrapped heredoc in a host readfile script.
The kernel runs the official image's existing `/root/gem5_init.sh` directly as
init (`rw init=/root/gem5_init.sh IGNORE_M5=1`). This script retrieves the payload
through `m5 readfile`, writes the binary inside guest `/tmp`, and executes it.
This remains full-system simulation: Linux handles processes, syscalls and disk
I/O. It skips systemd services, which are unnecessary for the F0 smoke test. The disk image is used through gem5's normal
copy-on-write disk wrapper; the downloaded base image is not edited.

F0 proves guest boot and guest C++ execution. Command transport, NDP metadata
validation and PNM/PIM numerical tests belong to F1–F5.
