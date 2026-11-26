#!/usr/bin/env bash
# Run the command-supply baseline with the same SE Timing configuration as Qwen.
set -Eeuo pipefail
cd /workspace
scripts=/workspace/tools/gem5/pim_kernel
fork=/workspace/gem5-ParallelPIM-main
out=/workspace/build/extended-gem5
gem5=/workspace/build/RISCV_RAMULATOR2/gem5.opt
binary="$RUN_DIR/supply-roi"
make -C "$scripts" profile
make -C "$scripts" riscv-fp16 WORKER_SOURCE=supply_roi.cpp RISCV_WORKER="$binary" \
    EXTRA_SOURCE="$fork/util/m5/src/abi/riscv/m5op.S"
scons -C "$fork" "$gem5" -j"${JOBS:-2}"
cp "$out/lpddr6_fp16.h" "$RUN_DIR/"
sha256sum "$binary" "$gem5" "$RUN_DIR/lpddr6_fp16.h" > "$RUN_DIR/artifacts.sha256"
debug=()
if [[ "$SUPPLY_CASE" == small ]]; then debug=(--debug-flags=PIM --debug-file=pim.trace); fi
timeout --kill-after=30s "$ROI_TIMEOUT" "$gem5" --outdir="$RUN_DIR/sim" "${debug[@]}" \
    "$fork/configs/example/gem5_library/riscv-pim.py" --mode se --cpu timing --cores 1 \
    --memory lpddr6 --channels 16 --policy RoRaBaChCo --num-fmt fp16 --acc-fmt fp32 \
    --layout "$RUN_DIR/lpddr6_fp16.h" --binary "$binary" --args "$SUPPLY_CASE" \
    2>&1 | tee "$RUN_DIR/simulation.log"
python3 "$scripts/supply_stats.py" "$RUN_DIR" "$SUPPLY_CASE" | tee "$RUN_DIR/result.txt"
