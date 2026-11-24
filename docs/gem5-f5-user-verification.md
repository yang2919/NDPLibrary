# F5 user verification

**Status: the user has reported F5 FS verification complete.** The procedure below
is retained for reproducible reruns; the assistant has not independently inspected
the FS logs.

F3 FS verification is complete according to the user. F5 validates a decoder layer, repeated requests, KV reset and memory reclamation
in stock gem5 Linux FS. F4 FS verification has also been reported complete by the user.
The assistant validates natively; the following FS run is performed by the user.

## Run

Start Docker Desktop and run from the repository root. Use the existing image
and resources. Run only one FS job at a time because stages share output files.

```sh
cd /Users/yang/Documents/GitHub/NDPLibrary
previous_run=$(cat build-gem5-fs/latest-run.txt 2>/dev/null || true)
STAGE=F5 BOOT_TIMEOUT=3600 ./tools/gem5/fs/run.sh
fs_status=$?
printf 'FS runner exit status: %s\n' "$fs_status"
```

The runner compiles two static x86-64 programs: a decoder probe and its receiver.
Both are installed in the guest. The probe spawns the receiver, executes the
checks, and exits. No gem5 source changes or manual guest installation are needed.
Missing images/resources may trigger preparation before simulation.

`BOOT_TIMEOUT` bounds simulation, not compilation/preparation, and allows up to
30 extra seconds for forced termination. The probe runs a local reference and the process backend, each with three small
requests. Quiet output alone is not failure. Increase the outer timeout only
after the preceding run ends. This stage does not repeat F4's fault injection.

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

Require a **new** directory beginning with `run-F5-`, exit status `0`, and
`full_system=true`. If the directory is unchanged or absent, preparation failed;
use the terminal error instead of old logs.

`result.txt` must contain:

```text
F5 PASS: Guest decoder, KV reset and memory reclamation verified.
```

The serial log must contain all six lines:

```text
NDP_DECODER_EXECUTION prefill_pnm=3 decode_pim=9
NDP_DECODER_REFERENCE outputs=12 exact_fp16=yes
NDP_DECODER_REQUESTS sequence=A_B_A kv_reset=verified
NDP_DECODER_RECLAMATION temporaries_model_cache_workspace=verified
NDP_F5_PASS
NDP_FS_SCRIPT_COMPLETE
```

The counts describe the process-backend run: three prefills and nine decode
steps, with 12 final outputs compared byte-for-byte against the local emulator.
A boot banner or an earlier stage's PASS is insufficient. This validates a small
software-emulated decoder layer, not accelerator timing or a full LLM service.

## Report

On success, send the run directory, exit status, `full_system=true`, and the
six serial markers plus the `F5 PASS` result. On failure, send the command,
exit status and these final lines:

```sh
tail -n 40 "$run_dir/console.log"
tail -n 40 "$run_dir/board.pc.com_1.device"
```

If compilation/preparation failed before a new directory appeared, send the
terminal error. Missing result/markers is not a pass. Full resource images and
complete build logs are unnecessary initially.
