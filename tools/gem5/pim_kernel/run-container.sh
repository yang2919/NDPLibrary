#!/usr/bin/env bash
# Build prerequisites, retain caches, and check an SE or FS PIM smoke run.
set -Eeuo pipefail
cd /workspace
: "${RUN_DIR:?Set RUN_DIR to a unique output directory}"
trap 'rc=$?; printf "FAIL exit=%s; see driver.log and simulation.log\n" "$rc" > "$RUN_DIR/result.txt"; exit "$rc"' ERR
scripts=/workspace/tools/gem5/pim_kernel
fork=/workspace/gem5-ParallelPIM-main
out=/workspace/build/extended-gem5
suite=${SUITE:-smoke}
cpu=${CPU:-timing}
binary="$out/riscv/pim-$suite-fp16"
# gem5 resolves build targets relative to SCons's launch directory, even with -C.
# Use the absolute workspace target, including the existing cached build.
gem5=/workspace/build/RISCV_RAMULATOR2/gem5.opt

make -C "$scripts" profile
if [[ "$STAGE" == QWEN ]]; then
    case "$MODEL" in moe|dense) ;; *) echo 'MODEL must be moe or dense' >&2; exit 2 ;; esac
    [[ "$CONTEXT" =~ ^[1-9][0-9]{0,3}$ && "$CONTEXT" -le 4096 ]]
    tp=${TP:-1}
    case "$MODEL:$tp" in dense:8|*:1|*:2|*:4) ;; *) echo "unsupported TP for $MODEL" >&2; exit 2 ;; esac
    binary="$RUN_DIR/qwen-roi"
    make -C "$scripts" riscv-fp16 WORKER_SOURCE=qwen_roi.cpp RISCV_WORKER="$binary" \
        EXTRA_SOURCE="$fork/util/m5/src/abi/riscv/m5op.S"
    scons -C "$fork" "$gem5" -j"${JOBS:-2}"
    cp "$out/lpddr6_fp16.h" "$RUN_DIR/"
    timeout --kill-after=30s "$ROI_TIMEOUT" "$gem5" --outdir="$RUN_DIR/sim" \
        "$fork/configs/example/gem5_library/riscv-pim.py" --mode se --cpu timing --cores 1 \
        --memory lpddr6 --channels 16 --policy RoRaBaChCo --num-fmt fp16 --acc-fmt fp32 \
        --layout "$RUN_DIR/lpddr6_fp16.h" --binary "$binary" --args "$MODEL" "$CONTEXT" "$tp" \
        2>&1 | tee "$RUN_DIR/simulation.log"
    python3 "$scripts/qwen_stats.py" "$RUN_DIR" "$MODEL" "$CONTEXT" "$tp" | tee "$RUN_DIR/result.txt"
    exit 0
fi
make -C "$scripts" riscv-fp16
make -C "$scripts" riscv-smoke SUITE="$suite"
scons -C "$fork" "$gem5" -j"${JOBS:-2}"
test -x "$gem5"
cp "$out/lpddr6_fp16.h" "$RUN_DIR/"
cp "$out/fixture/$suite/cases.json" "$RUN_DIR/cases.json"
cp "$out/riscv/"*.manifest.json "$out/riscv/"*.inspection.json "$RUN_DIR/"
sha256sum "$gem5" "$binary" "$out/lpddr6_fp16.h" > "$RUN_DIR/artifacts.sha256"

if [[ "$STAGE" == FSSTREAM || "$STAGE" == FS || "$STAGE" == PREPARE ]]; then
    # Fetch resources without instantiating or running a simulated machine.
    cat > "$RUN_DIR/fetch.py" <<'PY'
from gem5.resources.resource import obtain_resource
for name, version in [("riscv-linux-6.8.12-kernel", "1.0.0"),
                      ("riscv-ubuntu-24.04-img", "2.0.0")]:
    print(obtain_resource(name, resource_version=version).get_local_path())
PY
    "$gem5" --outdir="$RUN_DIR/resource-fetch" "$RUN_DIR/fetch.py"
    export WORK=/var/lib/ndp-pim/kernel
    export STOCK_KERNEL="$GEM5_RESOURCE_DIR/riscv-linux-6.8.12-kernel-1.0.0"
    export STOCK_IMAGE="$GEM5_RESOURCE_DIR/riscv-ubuntu-24.04-img-2.0.0"
    mkdir -p "$WORK"
    # Invalidate only for kernel build inputs, not for every NDP source edit.
    sha256sum "$fork/util/pim-kernel/build.sh" "$fork/util/pim-kernel/config.fragment" \
        "$fork/util/pim-kernel/senvcfg.patch" "$STOCK_KERNEL" "$STOCK_IMAGE" > "$RUN_DIR/kernel-inputs.sha256"
    find "$fork/util/gem5_bridge" -type f \( -name '*.c' -o -name '*.h' -o -name Makefile \) \
        -print0 | sort -z | xargs -0 sha256sum >> "$RUN_DIR/kernel-inputs.sha256"
    riscv64-linux-gnu-gcc --version >> "$RUN_DIR/kernel-inputs.sha256"
    if [[ "${REBUILD_KERNEL:-0}" == 1 ]] || ! test -s "$WORK/vmlinux" || \
       ! test -s "$WORK/riscv-ubuntu-24.04-img-pim" || \
       ! cmp -s "$RUN_DIR/kernel-inputs.sha256" "$WORK/inputs.sha256"; then
        bash "$fork/util/pim-kernel/build.sh"
        cp "$RUN_DIR/kernel-inputs.sha256" "$WORK/inputs.sha256"
    else
        echo 'Reusing patched FS kernel and disk.'
    fi
    sha256sum "$WORK/vmlinux" "$WORK/riscv-ubuntu-24.04-img-pim" >> "$RUN_DIR/artifacts.sha256"
fi
if [[ "$STAGE" == PREPARE ]]; then
    echo 'NDP_PIM_PREPARE_PASS (build/resources only; no simulation)' > "$RUN_DIR/result.txt"
    exit 0
fi
if [[ "$STAGE" == STREAM ]]; then
    python3 "$scripts/se_stream.py" --directory "$RUN_DIR" \
        --worker "$out/riscv/pim-kernel-worker-fp16" --gem5 "$gem5" \
        --cpu "$cpu" --bind 0.0.0.0 --port 45240 --connect-timeout "${CONNECT_TIMEOUT:-0}" \
        2>&1 | tee "$RUN_DIR/stream.log"
    echo 'NDP_SE_TRANSPORT_CLOSED (verify external client verdict and PIM trace)' > "$RUN_DIR/result.txt"
    exit 0
fi
common=(--memory lpddr6 --channels 16 --policy RoRaBaChCo --num-fmt fp16 --acc-fmt fp32
        --layout "$out/lpddr6_fp16.h" --cpu "$cpu" --cores 1)
if [[ "$STAGE" == SE ]]; then
    args=(--mode se --binary "$binary")
    limit=${SE_TIMEOUT:-180}
else
    args=(--mode fs --boot no-systemd --boot-cpu atomic --switch-at work --way 1 --kernel "$WORK/vmlinux"
          --disk-image "$WORK/riscv-ubuntu-24.04-img-pim"
          --fs-program "$binary" --fs-env PIM_MODE=fs)
    limit=${BOOT_TIMEOUT:-3600}
fi
config="$fork/configs/example/gem5_library/riscv-pim.py"
if [[ "$STAGE" == FSSTREAM ]]; then
    case "$FS_CASE" in gemv|chain|partial|lost|decoder) ;; *) exit 2 ;; esac
    binary="$RUN_DIR/fs-worker"
    make -C "$scripts" riscv-fp16 WORKER_SOURCE=fs_stream.cpp RISCV_WORKER="$binary" \
        EXTRA_SOURCE="$fork/util/m5/src/abi/riscv/m5op.S"
    args=(--mode fs --boot no-systemd --boot-cpu atomic --switch-at marker --way 1 --kernel "$WORK/vmlinux"
          --disk-image "$WORK/riscv-ubuntu-24.04-img-pim" --fs-program "$binary" --fs-env PIM_MODE=fs)
    sha256sum "$gem5" "$binary" "$out/lpddr6_fp16.h" > "$RUN_DIR/artifacts.sha256"
    config="$scripts/fs_stream_config.py"
fi
printf '%q ' "$gem5" "${common[@]}" "${args[@]}" > "$RUN_DIR/options.txt"
timeout --kill-after=30s "$limit" "$gem5" --outdir="$RUN_DIR/sim" \
    --debug-flags=PIM --debug-file=pim.trace \
    "$config" "${common[@]}" "${args[@]}" \
    2>&1 | tee "$RUN_DIR/simulation.log"
if [[ "$STAGE" == FSSTREAM ]]; then
    grep -q "NDP_FS_SESSION_CLOSED case=$FS_CASE " "$RUN_DIR/simulation.log"
    grep -q 'NDP_FS_WORKER_CLOSED' "$RUN_DIR/sim/board.platform.terminal"
    grep -q 'PIM-FS exit 0' "$RUN_DIR/sim/board.platform.terminal"
    PYTHONPATH="$scripts" python3 "$scripts/fs_trace.py" "$RUN_DIR/sim" "$FS_CASE" > "$RUN_DIR/result.txt"
    exit 0
fi
python3 "$scripts/check_smoke.py" "$RUN_DIR/sim" --mode "$STAGE" --cpu "$cpu" --cases "$RUN_DIR/cases.json" > "$RUN_DIR/result.txt"
