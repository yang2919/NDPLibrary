#!/usr/bin/env bash
# Build and run the stock gem5 FS baseline in Docker, then check guest output.
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
image=ndp-gem5-f0:24.1.0.3
stage=${STAGE:-F0}
case "$stage" in F0|F2|F3|F4|F5|K4) ;; *) echo "STAGE must be F0, F2, F3, F4, F5 or K4" >&2; exit 2 ;; esac
output="$root/build-gem5-fs"
mkdir -p "$output"
if [[ "${REBUILD:-0}" == 1 ]] || ! docker image inspect "$image" >/dev/null 2>&1; then
    docker build --build-arg "JOBS=${JOBS:-2}" -t "$image" "$root/tools/gem5/fs"
fi
set --
if [[ "$stage" == K4 ]]; then
    set -- -p "127.0.0.1:${K4_PORT:-45240}:45240"
fi
docker run --rm --init "$@" \
    -v "$root:/workspace:ro" -v "$output:/output" \
    -e "BOOT_TIMEOUT=${BOOT_TIMEOUT:-3600}" -e "STAGE=$stage" "$image" \
    bash /workspace/tools/gem5/fs/run-container.sh
