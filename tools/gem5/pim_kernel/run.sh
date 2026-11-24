#!/usr/bin/env bash
# Prepare and verify the extended RISC-V PIM system from the host.
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
stage=${STAGE:-SE}
suite=${SUITE:-smoke}
cpu=${CPU:-timing}
case "$suite" in smoke|boundary|lifecycle|chain|decoder|decoder_state) ;; *) echo "SUITE must be smoke, boundary, lifecycle, chain, decoder or decoder_state" >&2; exit 2 ;; esac
case "$cpu" in timing|atomic) ;; *) echo "CPU must be timing or atomic" >&2; exit 2 ;; esac
if [[ "$stage" != SE && "$stage" != STREAM && "$cpu" != timing ]]; then echo "CPU=atomic is supported only for SE" >&2; exit 2; fi
case "$stage" in PREPARE|SE|FS|STREAM|FSSTREAM|QWEN) ;; *) echo 'STAGE must be PREPARE, SE, FS, STREAM, FSSTREAM or QWEN' >&2; exit 2 ;; esac
default_boot_timeout=3600
if [[ "$stage" == FSSTREAM ]]; then default_boot_timeout=10800; fi
jobs=${JOBS:-2}
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo 'JOBS must be positive' >&2; exit 2; }
output="$root/build/extended-gem5"
mkdir -p "$output"
run_dir=$(mktemp -d "$output/run-$stage-XXXXXXXX")
printf '%s\n' "$run_dir" > "$output/latest-run.txt"
exec > >(tee "$run_dir/driver.log") 2>&1
trap 'rc=$?; printf "FAIL exit=%s; see driver.log\n" "$rc" > "$run_dir/result.txt"; exit "$rc"' ERR
printf 'Results: %s\n' "$run_dir"
docker info >/dev/null
for kind in cross validation; do
    image="ndp-pim-$kind:24.04"
    file="$root/tools/gem5/pim_kernel/Dockerfile"
    [[ "$kind" == validation ]] && file="$file.validation"
    if [[ "${REBUILD:-0}" == 1 ]] || ! docker image inspect "$image" >/dev/null 2>&1; then
        docker build -t "$image" -f "$file" "$root/tools/gem5/pim_kernel"
    fi
done
docker image inspect ndp-pim-validation:24.04 > "$run_dir/docker-image.json"
docker volume create ndp-pim-linux-data >/dev/null
set --
if [[ "$stage" == STREAM || "$stage" == FSSTREAM ]]; then
    set -- -p "127.0.0.1:${PORT:-45240}:45240"
fi
# Affinity bounds nproc in the supplier's kernel build as well as SCons jobs.
docker run --rm --init "$@" --cpuset-cpus="${CPUSET:-0-$((jobs - 1))}" \
    -v "$root:/workspace" -v ndp-pim-linux-data:/var/lib/ndp-pim \
    -e "MODEL=${MODEL:-moe}" -e "CONTEXT=${CONTEXT:-16}" -e "ROI_TIMEOUT=${ROI_TIMEOUT:-0}" \
    -e GEM5_RESOURCE_DIR=/var/lib/ndp-pim/resources \
    -e "NDP_HOST_PORT=${PORT:-45240}" -e "FS_CASE=${FS_CASE:-gemv}" -e "STAGE=$stage" -e "JOBS=$jobs" -e "SUITE=$suite" -e "CPU=$cpu" \
    -e "BOOT_TIMEOUT=${BOOT_TIMEOUT:-$default_boot_timeout}" -e "SE_TIMEOUT=${SE_TIMEOUT:-180}" \
    -e "CONNECT_TIMEOUT=${CONNECT_TIMEOUT:-0}" \
    -e "REBUILD_KERNEL=${REBUILD_KERNEL:-0}" \
    -e "RUN_DIR=/workspace/build/extended-gem5/$(basename "$run_dir")" \
    ndp-pim-validation:24.04 bash /workspace/tools/gem5/pim_kernel/run-container.sh
cat "$run_dir/result.txt"
