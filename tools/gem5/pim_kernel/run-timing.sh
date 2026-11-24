#!/usr/bin/env bash
# Run the user-owned SE Timing acceptance suites sequentially, stopping on failure.
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
output="$root/build/extended-gem5"
mkdir -p "$output"
batch=$(mktemp -d "$output/timing-XXXXXXXX")
printf '%s\n' "$batch" > "$output/latest-timing.txt"
printf 'suite\tstatus\tresult_directory\n' > "$batch/runs.tsv"
printf 'Timing results: %s\n' "$batch"
for suite in boundary lifecycle; do
    printf '\nRunning SE Timing: %s\n' "$suite"
    if STAGE=SE CPU=timing SUITE="$suite" SE_TIMEOUT="${SE_TIMEOUT:-600}" \
        "$root/tools/gem5/pim_kernel/run.sh"; then
        run_dir=$(cat "$output/latest-run.txt")
        if ! grep -q '^NDP_PIM_SE_PASS cpu=timing:' "$run_dir/result.txt"; then
            printf '%s\tFAIL\t%s\n' "$suite" "$run_dir" >> "$batch/runs.tsv"
            echo "FAIL: missing Timing verdict for $suite" > "$batch/result.txt"
            cat "$batch/result.txt"
            exit 1
        fi
        printf '%s\tPASS\t%s\n' "$suite" "$run_dir" >> "$batch/runs.tsv"
    else
        status=$?
        run_dir=$(cat "$output/latest-run.txt")
        printf '%s\tFAIL(%s)\t%s\n' "$suite" "$status" "$run_dir" >> "$batch/runs.tsv"
        printf 'FAIL: %s exited with %s; no subsequent suite was started.\n' \
            "$suite" "$status" > "$batch/result.txt"
        cat "$batch/result.txt"
        exit "$status"
    fi
done
echo 'NDP_PIM_SE_TIMING_PASS suites=boundary,lifecycle' > "$batch/result.txt"
cat "$batch/result.txt"
