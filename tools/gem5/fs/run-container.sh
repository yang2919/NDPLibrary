#!/usr/bin/env bash
# Compile the selected guest probe and verify its output after full-system boot.
set -euo pipefail
scripts=/workspace/tools/gem5/fs
test "$(git -C /opt/gem5 rev-parse HEAD)" = b9da2bfe1e21f6efa2730edd4e80e58c5d965458
stage=${STAGE:-F0}
execution_sources=(/workspace/src/l1/gem5/execution.cpp /workspace/src/l1/gem5/metadata.cpp
    /workspace/src/l1/ndp/codec.cpp /workspace/src/l1/ndp/transport.cpp
    /workspace/src/l1/ndp/emulator.cpp /workspace/src/l1/ndp/pnm.cpp
    /workspace/src/l1/ndp/program.cpp /workspace/src/l1/ndp/launch.cpp
    /workspace/src/common/fp16.cpp)
receiver_sources=()
boot_args=()
case "$stage" in
  K4) sources=(/workspace/tools/gem5/pim_kernel/worker.cpp /workspace/src/common/fp16.cpp) ;;
  F0) sources=("$scripts/smoke.cpp") ;;
  F2) sources=(/workspace/tools/gem5/guest/metadata_probe.cpp
               /workspace/src/l1/gem5/metadata.cpp /workspace/src/l1/ndp/codec.cpp) ;;
  F3) sources=(/workspace/tools/gem5/guest/execution_probe.cpp "${execution_sources[@]}") ;;
  F4|F5) sources=(/workspace/tools/gem5/guest/session_probe.cpp
               /workspace/src/l1/gem5/backend.cpp /workspace/src/l1/ndp/backend.cpp
               /workspace/src/l1/ndp/command_builder.cpp /workspace/src/l2/session.cpp
               /workspace/src/common/tensor_metadata.cpp /workspace/src/common/compute.cpp
               "${execution_sources[@]}")
      if [[ "$stage" == F5 ]]; then
          sources[0]=/workspace/tools/gem5/guest/decoder_probe.cpp
          sources+=(/workspace/src/l3/llm.cpp /workspace/src/l3/weight.cpp)
      fi
      receiver_sources=(/workspace/tools/gem5/guest/execution_receiver.cpp "${execution_sources[@]}")
      boot_args=(--receiver-binary=/output/receiver) ;;
  *) echo "STAGE must be F0, F2, F3, F4, F5 or K4" >&2; exit 2 ;;
esac
if [[ "${K4_BUILD_ONLY:-0}" != 1 ]]; then
    python3 "$scripts/fetch.py" /output/resources
fi
# Jammy's cross libm linker script contains native absolute paths. Use the
# installed cross archives through a local script without changing the image.
mkdir -p /output/guest-link
sed 's|/usr/lib/x86_64-linux-gnu/|/usr/x86_64-linux-gnu/lib/|g' \
    /usr/x86_64-linux-gnu/lib/libm.a > /output/guest-link/libm.a
extra_flags=()
if [[ "$stage" == K4 ]]; then
    supplier=/workspace/pim-isa-library-2026-11-10
    ref="$supplier/util/pim/src"
    layout=/workspace/tools/gem5/pim_kernel/lpddr6_fp16_native.h
    extra_flags=(-I"$ref" -I"$supplier/include" "-DPIM_LAYOUT_HEADER=\"$layout\"" -DPIM_BACKEND_REF)
    objects=$(mktemp -d /output/k4-objects-XXXXXXXX)
    trap 'rm -rf "$objects"' EXIT
    for unit in layout init region alloc regs bf16 copy sync tensor ops ref/model ref/numfmt; do
        obj="$objects/${unit##*/}.o"
        x86_64-linux-gnu-gcc -std=gnu11 -O2 -ffp-contract=off "${extra_flags[@]}" \
            -c "$ref/$unit.c" -o "$obj"
        sources+=("$obj")
    done
fi
x86_64-linux-gnu-g++ -std=c++20 -O2 -fno-fast-math -ffp-contract=off -s -static -pthread -L/output/guest-link \
    -I/workspace/include -I/workspace/src "${extra_flags[@]}" "${sources[@]}" -lm -o /output/smoke
if [[ "$stage" == F4 || "$stage" == F5 ]]; then
    x86_64-linux-gnu-g++ -std=c++20 -O2 -fno-fast-math -ffp-contract=off -s -static -pthread \
        -L/output/guest-link -I/workspace/include -I/workspace/src \
        "${receiver_sources[@]}" -o /output/receiver
fi
if [[ "$stage" == K4 && "${K4_BUILD_ONLY:-0}" == 1 ]]; then
    x86_64-linux-gnu-readelf -h /output/smoke
    exit 0
fi
run_dir=$(mktemp -d "/output/run-$stage-XXXXXXXX")
printf '%s\n' "$run_dir" > /output/latest-run.txt
{
    git -C /opt/gem5 rev-parse HEAD
    git -C /opt/gem5 diff --exit-code
    /opt/gem5/build/X86/gem5.opt --outdir="$run_dir" --build-info
    x86_64-linux-gnu-g++ --version
    sha256sum /output/smoke /output/resources/vmlinux-5.4.49 /output/resources/x86-ubuntu.img
    if [[ "$stage" == F4 || "$stage" == F5 ]]; then sha256sum /output/receiver "${receiver_sources[@]}"; fi
    sha256sum "$scripts/boot.py" "$scripts/resources.json" "${sources[@]}"
} > "$run_dir/manifest.txt"
timeout --kill-after=30s "${BOOT_TIMEOUT:-3600}" /opt/gem5/build/X86/gem5.opt \
    --outdir="$run_dir" "$scripts/boot.py" \
    --binary=/output/smoke --resource-directory=/output/resources --stage="$stage" "${boot_args[@]}" \
    2>&1 | tee "$run_dir/console.log"
serial="$run_dir/board.pc.com_1.device"
if [[ "$stage" == F0 ]]; then
    grep -qx 'NDP_FS_SMOKE_PASS' <(tr -d '\r' < "$serial")
elif [[ "$stage" == F2 ]]; then
    grep -qx 'NDP_METADATA_ACCEPTED batches=3 exact_bytes=yes' <(tr -d '\r' < "$serial")
    grep -qx 'NDP_METADATA_REJECTIONS duplicate_order_epoch_opcode_range=5' <(tr -d '\r' < "$serial")
    grep -qx 'NDP_METADATA_ONLY execution=not_performed' <(tr -d '\r' < "$serial")
    grep -qx 'NDP_F2_PASS' <(tr -d '\r' < "$serial")
elif [[ "$stage" == F3 ]]; then
    grep -qx 'NDP_EXECUTION_COMPLETE pnm=pim=verified' <(tr -d '\r' < "$serial")
    grep -qx 'NDP_PERSISTENT_STATE mac_memory=verified' <(tr -d '\r' < "$serial")
    grep -qx 'NDP_REFERENCE_MATCH exact_fp16=yes' <(tr -d '\r' < "$serial")
    grep -qx 'NDP_EXECUTION_ERRORS preflight_download_recovery=verified' <(tr -d '\r' < "$serial")
    grep -qx 'NDP_F3_PASS' <(tr -d '\r' < "$serial")
elif [[ "$stage" == F4 ]]; then
    for marker in \
        'NDP_SESSION_DEPENDENCIES pnm_pim=verified' \
        'NDP_RESIDENT_BUFFERS copy_fill_snapshot=verified' \
        'NDP_SESSION_CLOSE drain_live_handles_reaped=verified' \
        'NDP_SESSION_FAULTS timeout_disconnect_protocol=verified' \
        'NDP_SESSION_LIFETIME idle_destructor=verified' 'NDP_F4_PASS'; do
        grep -qx "$marker" <(tr -d '\r' < "$serial")
    done
elif [[ "$stage" == K4 ]]; then
    grep -qx 'K4_GUEST_WORKER_CLOSED' <(tr -d '\r' < "$serial")
    grep -qx 'K4_HOST_SESSION_CLOSED' "$run_dir/console.log"
else
    for marker in \
        'NDP_DECODER_EXECUTION prefill_pnm=3 decode_pim=9' \
        'NDP_DECODER_REFERENCE outputs=12 exact_fp16=yes' \
        'NDP_DECODER_REQUESTS sequence=A_B_A kv_reset=verified' \
        'NDP_DECODER_RECLAMATION temporaries_model_cache_workspace=verified' 'NDP_F5_PASS'; do
        grep -qx "$marker" <(tr -d '\r' < "$serial")
    done
fi
grep -qx 'NDP_FS_SCRIPT_COMPLETE' <(tr -d '\r' < "$serial")
if [[ "$stage" == F0 ]]; then
    printf 'F0 PASS: Linux guest executed the C++ smoke program.\n'
elif [[ "$stage" == F2 ]]; then
    printf 'F2 PASS: Guest metadata IPC verified; no device execution.\n'
elif [[ "$stage" == F3 ]]; then
    printf 'F3 PASS: Persistent guest emulator execution verified.\n'
elif [[ "$stage" == F4 ]]; then
    printf 'F4 PASS: Session lifecycle and receiver fault cleanup verified.\n'
elif [[ "$stage" == K4 ]]; then
    printf 'K4 PASS: stock x86 guest reference worker session closed. Verify K4_CLIENT_PASS on host.\n'
else
    printf 'F5 PASS: Guest decoder, KV reset and memory reclamation verified.\n'
fi | tee "$run_dir/result.txt"
