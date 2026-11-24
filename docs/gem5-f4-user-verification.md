# F4 user verification

**Status: the user has reported F4 FS verification complete.** The procedure is
retained for reruns; the assistant has not independently inspected the FS logs.

F3 FS verification is complete according to the user. F4 validates the public
Session path, persistent buffers and receiver lifecycle in stock gem5 Linux FS.
The assistant validates natively; the following FS run is performed by the user.

## Run

Start Docker Desktop and run from the repository root. Use the existing image
and resources. Run only one FS job at a time because stages share output files.

```sh
cd /Users/yang/Documents/GitHub/NDPLibrary
previous_run=$(cat build-gem5-fs/latest-run.txt 2>/dev/null || true)
STAGE=F4 BOOT_TIMEOUT=3600 ./tools/gem5/fs/run.sh
fs_status=$?
printf 'FS runner exit status: %s\n' "$fs_status"
```

The runner compiles two static x86-64 programs: a Session probe and its receiver.
Both are installed in the guest. The probe spawns the receiver, executes the
checks, and exits. No gem5 source changes or manual guest installation are needed.
Missing images/resources may trigger preparation before simulation.

`BOOT_TIMEOUT` bounds simulation, not compilation/preparation, and allows up to
30 extra seconds for forced termination. Internal fault tests deliberately wait
for a 500 ms guest timeout; idle validation waits 600 ms of guest time. These
are simulated guest times, not host wall-clock estimates. Quiet output alone
is not failure. Increase the outer timeout only after the preceding run ends.

## Check the current run

```sh
current_run=$(cat build-gem5-fs/latest-run.txt 2>/dev/null || true)
printf 'Previous run: %s\nCurrent run: %s\n' "$previous_run" "$current_run"
run_name=$(basename "$current_run")
run_dir="build-gem5-fs/$run_name"
cat "$run_dir/result.txt"
grep '^full_system=' "$run_dir/config.ini"
tail -n 20 "$run_dir/board.pc.com_1.device"
```

Require a **new** directory beginning with `run-F4-`, exit status `0`, and
`full_system=true`. If the directory is unchanged or absent, preparation failed;
use the terminal error instead of old logs.

`result.txt` must contain:

```text
F4 PASS: Session lifecycle and receiver fault cleanup verified.
```

The serial log must contain all seven lines:

```text
NDP_SESSION_DEPENDENCIES pnm_pim=verified
NDP_RESIDENT_BUFFERS copy_fill_snapshot=verified
NDP_SESSION_CLOSE drain_live_handles_reaped=verified
NDP_SESSION_FAULTS timeout_disconnect_protocol=verified
NDP_SESSION_LIFETIME idle_destructor=verified
NDP_F4_PASS
NDP_FS_SCRIPT_COMPLETE
```

The fault marker is expected: failures are injected intentionally and must be
handled without leaving a receiver process alive. A boot banner or an F3 PASS
is insufficient. This validates software execution and lifetime semantics,
not accelerator timing or a full LLM decoder.

## Report

On success, send the run directory, exit status, `full_system=true`, and the
seven serial markers plus the `F4 PASS` result. On failure, send the command,
exit status and these final lines:

```sh
tail -n 40 "$run_dir/console.log"
tail -n 40 "$run_dir/board.pc.com_1.device"
```

If compilation/preparation failed before a new directory appeared, send the
terminal error. Missing result/markers is not a pass. Full resource images and
complete build logs are unnecessary initially.
