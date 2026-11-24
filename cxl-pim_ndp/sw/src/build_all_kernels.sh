#!/usr/bin/env bash
# Build every board kernel. fp16 (_hp) REQUIRES GCC 16 (the Xilinx GCC 13.4
# lacks fp16 vector intrinsics); fp32 builds with either, so use GCC 16 for all.
#   riscv64-unknown-elf-
# Converted kernels (dtype.h + kernel.mk) emit <k>_sp.bin AND <k>_hp.bin;
# 06-08 kernels emit a single <k>.bin.
set -u
G16=${G16:-riscv64-unknown-elf-}
SW=$(cd "$(dirname "$0")" && pwd)
KERNELS="fmatmul_sp fmatmul_hp exp_sp fdotproduct_sp vfredsum_sp layernorm_sp softmax_sp gelu_sp attention_sp gqa_attention redmax_soak corner lmul_test matmul_iso roofline imatmul_i8 gather_bench spm_diag rag bf16_diag fp8_diag fp8_widen_diag lsu_bench"
fail=0
for k in $KERNELS; do
  [ -d "$SW/$k" ] || continue
  echo "=== $k ==="
  ( cd "$SW/$k" && make clean >/dev/null 2>&1
    make CROSS_COMPILE="$G16" 2>&1 | grep -E "====|error:|Error|undefined|redefinition|implicit" | grep -v "RWX" )
  # authoritative check: a .bin must have been produced (grep-on-output missed
  # failures because the "====" size line always matched)
  if ! ls "$SW/$k"/build/*.bin >/dev/null 2>&1; then
    echo "  BUILD FAILED: $k (no .bin produced)"; fail=$((fail+1))
  fi
done
echo ""
echo "=== built bins ==="
# imatmul_i8, bf16_diag, fp8_diag, fp8_widen_diag and lsu_bench are not on kernel.mk's
# {fp32, fp16} axis, so they have no _sp/_hp suffix to match -- list them by name.
ls -1 "$SW"/*/build/*_sp.bin "$SW"/*/build/*_hp.bin "$SW"/fmatmul_*/build/fmatmul_*.bin \
      "$SW"/imatmul_i8/build/imatmul_i8.bin \
      "$SW"/bf16_diag/build/bf16_diag.bin "$SW"/fp8_diag/build/fp8_diag.bin \
      "$SW"/fp8_widen_diag/build/fp8_widen_diag.bin "$SW"/lsu_bench/build/lsu_bench.bin 2>/dev/null | sort -u
echo "build failures: $fail"
