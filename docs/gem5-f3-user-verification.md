# F3 user verification

F2 has been reported verified. F3 checks actual software-emulated PNM/PIM execution
and memory/register persistence across IPC requests. It does not validate hardware
latency, a real RISC-V executor, Session integration or a complete decoder.

## 1. Prepare

Start Docker Desktop, keep the Mac awake, and use the repository root:

```sh
cd /Users/yang/Documents/GitHub/NDPLibrary
docker info
```

Use the existing F0/F2 Docker environment and resources. No manual guest build is
needed. Run only one FS job at a time: F0/F2/F3 share generated files. If resources
or the image are missing, the runner downloads/builds them before simulation.

## 2. Run

Run in the same terminal and capture the status immediately:

```sh
previous_run=$(cat build-gem5-fs/latest-run.txt 2>/dev/null || true)
STAGE=F3 BOOT_TIMEOUT=3600 ./tools/gem5/fs/run.sh
fs_status=$?
printf 'FS runner exit status: %s\n' "$fs_status"
```

The runner compiles the probe and emulator sources into a stripped static x86-64
binary, then executes it under the existing Linux FS configuration. Numerical
build flags disable fast-math and implicit FMA contraction, matching the emulator
contract. No gem5 source modifications are needed.

The timeout covers simulator execution, not preparation/compilation, with up to
30 additional seconds for forced termination. Quiet boot output is not itself a
failure. Increase `BOOT_TIMEOUT` only after the preceding run ends if more time
is required; timeout does not count as a pass.

## 3. Check the current run

```sh
current_run=$(cat build-gem5-fs/latest-run.txt 2>/dev/null || true)
printf 'Previous run: %s\nCurrent run: %s\n' "$previous_run" "$current_run"
run_name=$(basename "$current_run")
run_dir="build-gem5-fs/$run_name"
```

Require a new directory starting with `run-F3-`. If it is unchanged or absent,
preparation failed before simulation: use the terminal error, not old logs.

```sh
cat "$run_dir/result.txt"
grep '^full_system=' "$run_dir/config.ini"
tail -n 20 "$run_dir/board.pc.com_1.device"
```

## 4. Confirm success

Require exit status `0`, `full_system=true`, and this `result.txt` message:

```text
F3 PASS: Persistent guest emulator execution verified.
```

The serial log must contain all six lines:

```text
NDP_EXECUTION_COMPLETE pnm=pim=verified
NDP_PERSISTENT_STATE mac_memory=verified
NDP_REFERENCE_MATCH exact_fp16=yes
NDP_EXECUTION_ERRORS preflight_download_recovery=verified
NDP_F3_PASS
NDP_FS_SCRIPT_COMPLETE
```

The error-recovery marker is expected: the fixture intentionally submits an invalid
batch/download and checks that state remains usable. The receiver must exit
successfully after Close. A boot banner or an F2 metadata PASS is not sufficient.

## 5. Report

For success, send:

```text
F3 FS: PASS
Run directory: run-F3-...
Exit status: 0
full_system=true
```

Include the six guest lines and the `F3 PASS` message above.

For failure after a new directory was created:

```sh
tail -n 40 "$run_dir/console.log"
tail -n 40 "$run_dir/board.pc.com_1.device"
```

Send the command, exit status, run directory and these final log lines. If the
cross-compiler or resource preparation failed before a new directory appeared,
send the terminal error instead. Missing `result.txt` or success markers means
the run has not passed. Do not substitute results from a previous stage/run.
Full build logs and images are unnecessary initially. The assistant diagnoses
natively; any subsequent FS run remains user-operated.
