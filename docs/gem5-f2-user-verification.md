# F2 user verification

This guide contains only the steps needed to run and verify F2 in gem5 full-system
(FS) mode. F1 needs no separate FS check. F2 checks metadata delivery between two
guest processes; it does not execute PNM/PIM operations.

## 1. Prepare

- Start Docker Desktop and wait until its engine is ready.
- Allocate at least 8 GB of Docker memory and ensure sufficient free disk space
  for the gem5 build and guest image. Existing F0 artifacts are reused.
- Keep the Mac awake during the run.
- Run one FS job at a time; F0 and F2 share generated files.

Open a terminal and enter the repository directory:

```sh
cd /Users/yang/Documents/GitHub/NDPLibrary
docker info
```

If `docker info` reports an error, resolve it before continuing. No manual gem5
installation or guest compilation is needed: the runner performs those steps.
Missing images/resources require downloads and a first build, which can take
considerably longer than subsequent runs.

## 2. Run F2

Run these lines in the same terminal. Capture the exit status immediately after
the run command:

```sh
previous_run=$(cat build-gem5-fs/latest-run.txt 2>/dev/null || true)
STAGE=F2 BOOT_TIMEOUT=3600 ./tools/gem5/fs/run.sh
fs_status=$?
printf 'FS runner exit status: %s\n' "$fs_status"
```

`BOOT_TIMEOUT=3600` limits the simulator run to one hour; image preparation and
compilation are outside this limit. The runner may take another 30 seconds to
force termination. Do not restart merely because boot output is quiet.

If a longer run is needed, repeat this step with `BOOT_TIMEOUT=7200` after the
previous job has ended. A timeout is not a pass.

## 3. Locate this run's results

In the same terminal:

```sh
current_run=$(cat build-gem5-fs/latest-run.txt 2>/dev/null || true)
printf 'Previous run: %s\nCurrent run: %s\n' "$previous_run" "$current_run"
run_name=$(basename "$current_run")
run_dir="build-gem5-fs/$run_name"
printf 'Local results: %s\n' "$run_dir"
```

The current run must differ from the previous run and its name must start with
`run-F2-`. If no new run exists, preparation or compilation failed: use the
terminal error, not an old result file. `latest-run.txt` uses the container's
`/output` prefix; the commands above map it to the local directory.

Once you have confirmed a new F2 directory:

```sh
cat "$run_dir/result.txt"
grep '^full_system=' "$run_dir/config.ini"
tail -n 20 "$run_dir/board.pc.com_1.device"
```

## 4. Confirm success

All of the following are required:

- The runner's exit status is `0`.
- `config.ini` contains `full_system=true`.
- `result.txt` contains:

```text
F2 PASS: Guest metadata IPC verified; no device execution.
```

- The guest serial log contains all five lines:

```text
NDP_METADATA_ACCEPTED batches=3 exact_bytes=yes
NDP_METADATA_REJECTIONS duplicate_order_epoch_opcode_range=5
NDP_METADATA_ONLY execution=not_performed
NDP_F2_PASS
NDP_FS_SCRIPT_COMPLETE
```

The five rejections are intentional checks of duplicate/out-of-order sequences,
wrong epoch, unknown opcode and invalid address range. They are expected on a
successful run. A boot banner, an F0 PASS or a native test PASS does not establish
F2 FS success.

## 5. Report the result

For success, send:

```text
F2 FS: PASS
Run directory: run-F2-...
Exit status: 0
full_system=true
```

Also include the five guest lines and the `F2 PASS` line above.

For a failure after a new run directory was created, collect:

```sh
tail -n 40 "$run_dir/console.log"
tail -n 40 "$run_dir/board.pc.com_1.device"
```

Send the run command, exit status, directory name and these final log lines. If
preparation/compilation failed before a new directory was created, send the
terminal error instead. Missing logs or `result.txt` are useful failure evidence;
do not substitute files from a previous run. Include `manifest.txt` only if asked
or if the build environment/artifacts differ. Full build logs and disk images
are not needed initially.

| Result | Interpretation |
|---|---|
| Docker error | Execution environment is not ready |
| Build/download/compiler error | Guest verification has not started |
| Timeout, typically status 124 (or 137 after forced termination) | Verification did not finish |
| `F2 failed:` or receiver error | IPC or metadata validation failed; report final logs |
| Missing expected markers or nonzero status | Not a pass, even if Linux booted |

The assistant will analyze the supplied evidence and provide corrections using
native checks. Any subsequent FS execution remains a user-run step.
