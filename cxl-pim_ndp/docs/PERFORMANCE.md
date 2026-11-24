# Roofline analysis result (2026-09-29)

**Conditions.**
VHK158 2-lane, VLEN 2048, 64-bit Ara AXI, soc_clk **120 MHz**, bitstream `pdi/ara_vhk158_2lane_120mhz_vlsu_spm_2026-09-29.pdi`.
`sw/src/roofline`, one binary per dtype, cycles from `rdcycle`.
The ideal is the arithmetic peak of 8 FLOP/cycle at fp32 and 16 at fp16 for matrix multiply and attention.
The ideal of the dot product is the memory limit of the 64-bit port, 2 FLOP/cycle at fp32 and 4 at fp16.
Code names in parentheses are the record tag in `sw/src/roofline/main.c` and the matching standalone kernel image in `sw/bin`.

Cycle counts are the mean of two runs, with a median run-to-run spread of 0.20 % above 3,000 cycles and 0.94 % below 3,000 cycles.

## Compute-bound kernels, fp32

| Kernel | Size | Cycles | GFLOP/s | % of ideal |
|---|---|---|---|---|
| matrix multiply (`mm`, `fmatmul_sp`) | 16x16 | 4,610 | 0.213 | 22.2 % |
| matrix multiply (`mm`, `fmatmul_sp`) | 32x32 | 13,722 | 0.573 | 59.7 % |
| matrix multiply (`mm`, `fmatmul_sp`) | 64x64 | 79,467 | 0.792 | 82.5 % |
| matrix multiply (`mm`, `fmatmul_sp`) | 128x128 | 537,338 | 0.937 | 97.6 % |
| matrix multiply (`mm`, `fmatmul_sp`) | 256x256 | 4,233,938 | 0.951 | 99.1 % |
| dot product (`dp`, `fdotproduct_sp`) | 256 | 770 | 0.080 | 33.3 % |
| dot product (`dp`, `fdotproduct_sp`) | 1024 | 1,664 | 0.148 | 61.5 % |
| dot product (`dp`, `fdotproduct_sp`) | 4096 | 4,832 | 0.203 | 84.8 % |
| dot product (`dp`, `fdotproduct_sp`) | 16384 | 17,641 | 0.223 | 92.9 % |
| attention (`attn`, `attention_sp`) | seq 64 | 1,018,262 | 0.124 | 12.9 % |
| attention (`attn`, `attention_sp`) | seq 128 | 3,985,934 | 0.126 | 13.2 % |

## Compute-bound kernels, fp16

| Kernel | Size | Cycles | GFLOP/s | % of ideal |
|---|---|---|---|---|
| matrix multiply (`mm`, `fmatmul_hp`) | 16x16 | 3,960 | 0.248 | 12.9 % |
| matrix multiply (`mm`, `fmatmul_hp`) | 32x32 | 8,849 | 0.889 | 46.3 % |
| matrix multiply (`mm`, `fmatmul_hp`) | 64x64 | 45,393 | 1.386 | 72.2 % |
| matrix multiply (`mm`, `fmatmul_hp`) | 128x128 | 293,232 | 1.716 | 89.4 % |
| matrix multiply (`mm`, `fmatmul_hp`) | 256x256 | 2,145,285 | 1.877 | 97.8 % |
| dot product (`dp`, `fdotproduct_hp`) | 256 | 458 | 0.134 | 27.9 % |
| dot product (`dp`, `fdotproduct_hp`) | 1024 | 1,012 | 0.243 | 50.6 % |
| dot product (`dp`, `fdotproduct_hp`) | 4096 | 2,618 | 0.376 | 78.2 % |
| dot product (`dp`, `fdotproduct_hp`) | 16384 | 8,958 | 0.439 | 91.4 % |
| attention (`attn`, `attention_hp`) | seq 64 | 745,613 | 0.169 | 8.8 % |
| attention (`attn`, `attention_hp`) | seq 128 | 2,924,506 | 0.172 | 9.0 % |

## Element-wise and transcendental, elem/cycle

### fp32

| Kernel | Size | Cycles | elem/cycle |
|---|---|---|---|
| sum reduction (`red`, `vfredsum_sp`) | 256 | 712 | 0.360 |
| sum reduction (`red`, `vfredsum_sp`) | 1024 | 1,080 | 0.949 |
| sum reduction (`red`, `vfredsum_sp`) | 4096 | 2,596 | 1.578 |
| sum reduction (`red`, `vfredsum_sp`) | 16384 | 9,060 | 1.808 |
| exponential (`exp`, `exp_sp`) | 256 | 2,196 | 0.117 |
| exponential (`exp`, `exp_sp`) | 1024 | 7,661 | 0.134 |
| exponential (`exp`, `exp_sp`) | 4096 | 30,154 | 0.136 |
| exponential (`exp`, `exp_sp`) | 16384 | 119,390 | 0.137 |
| GELU (`gelu`, `gelu_sp`) | 256 | 4,078 | 0.063 |
| GELU (`gelu`, `gelu_sp`) | 1024 | 15,421 | 0.066 |
| GELU (`gelu`, `gelu_sp`) | 4096 | 61,416 | 0.067 |
| GELU (`gelu`, `gelu_sp`) | 16384 | 246,021 | 0.067 |
| softmax (`sfx`, `softmax_sp`) | 256 | 4,250 | 0.060 |
| softmax (`sfx`, `softmax_sp`) | 1024 | 13,272 | 0.077 |
| softmax (`sfx`, `softmax_sp`) | 4096 | 50,630 | 0.081 |
| softmax (`sfx`, `softmax_sp`) | 16384 | 200,394 | 0.082 |
| layer normalization (`lyn`, `layernorm_sp`) | 256 | 2,960 | 0.086 |
| layer normalization (`lyn`, `layernorm_sp`) | 1024 | 8,238 | 0.124 |
| layer normalization (`lyn`, `layernorm_sp`) | 4096 | 29,320 | 0.140 |
| layer normalization (`lyn`, `layernorm_sp`) | 16384 | 114,096 | 0.144 |

### fp16

| Kernel | Size | Cycles | elem/cycle |
|---|---|---|---|
| sum reduction (`red`, `vfredsum_hp`) | 256 | 328 | 0.780 |
| sum reduction (`red`, `vfredsum_hp`) | 1024 | 606 | 1.688 |
| sum reduction (`red`, `vfredsum_hp`) | 4096 | 1,378 | 2.971 |
| sum reduction (`red`, `vfredsum_hp`) | 16384 | 4,616 | 3.549 |
| exponential (`exp`, `exp_hp`) | 256 | 1,328 | 0.193 |
| exponential (`exp`, `exp_hp`) | 1024 | 4,146 | 0.247 |
| exponential (`exp`, `exp_hp`) | 4096 | 15,526 | 0.264 |
| exponential (`exp`, `exp_hp`) | 16384 | 61,368 | 0.267 |
| GELU (`gelu`, `gelu_hp`) | 256 | 2,088 | 0.123 |
| GELU (`gelu`, `gelu_hp`) | 1024 | 7,510 | 0.136 |
| GELU (`gelu`, `gelu_hp`) | 4096 | 29,294 | 0.140 |
| GELU (`gelu`, `gelu_hp`) | 16384 | 116,450 | 0.141 |
| softmax (`sfx`, `softmax_hp`) | 256 | 2,366 | 0.108 |
| softmax (`sfx`, `softmax_hp`) | 1024 | 7,282 | 0.141 |
| softmax (`sfx`, `softmax_hp`) | 4096 | 26,100 | 0.157 |
| softmax (`sfx`, `softmax_hp`) | 16384 | 101,748 | 0.161 |
| layer normalization (`lyn`, `layernorm_hp`) | 256 | 1,528 | 0.167 |
| layer normalization (`lyn`, `layernorm_hp`) | 1024 | 4,419 | 0.232 |
| layer normalization (`lyn`, `layernorm_hp`) | 4096 | 14,991 | 0.273 |
| layer normalization (`lyn`, `layernorm_hp`) | 16384 | 57,212 | 0.286 |

## Operands in the scratchpad

`roofline_spm_sp` and `roofline_spm_hp` run every kernel with the operands in DRAM and in the 512 KiB scratchpad, as the median of 5 calls averaged over two runs, with bit-identical outputs for all 62 records.
Operands accessed by CVA6 scalar loads and stores stay in DRAM, such as A of matrix multiply (C = A × B) and the attention score and softmax matrices.

| Kernels | fp32 | fp16 |
|---|---|---|
| element-wise (dp, red, exp, gelu, sfx, lyn), median of 24 records | -14.6 % | -14.0 % |
| attention, seq 64 and 128 | -12.1 %, -12.3 % | -16.5 %, -16.9 % |
| matrix multiply, N 16 to 256 | +47.9 % to -0.2 % | +28.9 % to -1.2 % |

Since the scratchpad has the same 64-bit width as the DRAM port, the fp32 dot product takes only 0.9 % fewer cycles at 16384 elements, but 19.3 % fewer at 256 elements where the DRAM round trip takes a larger part of the call.
Kernels that store intermediate results and load them back take fewer cycles at every size, by 35 % for layer normalization, 17 % for GELU and 15 % for softmax at 16384 elements (fp32).

| Kernel | dtype | Size | DRAM (cycles) | scratchpad (cycles) | change | copy (cycles) |
|---|---|---|---|---|---|---|
| matrix multiply (`mm`) | fp32 | 256x256 | 4,233,512 | 4,225,994 | -0.2 % | 141,075 |
| dot product (`dp`) | fp32 | 16384 | 17,386 | 17,227 | -0.9 % | 35,964 |
| sum reduction (`red`) | fp32 | 16384 | 8,838 | 8,727 | -1.3 % | 17,934 |
| exponential (`exp`) | fp32 | 16384 | 119,712 | 116,137 | -3.0 % | 35,365 |
| GELU (`gelu`) | fp32 | 16384 | 251,386 | 209,107 | -16.8 % | 35,459 |
| softmax (`sfx`) | fp32 | 16384 | 200,724 | 170,819 | -14.9 % | 35,401 |
| layer normalization (`lyn`) | fp32 | 16384 | 113,827 | 73,824 | -35.1 % | 35,486 |
| attention (`attn`) | fp32 | seq 128 | 3,989,492 | 3,499,590 | -12.3 % | 35,720 |
| matrix multiply (`mm`) | fp16 | 256x256 | 2,144,858 | 2,120,144 | -1.2 % | 70,514 |
| dot product (`dp`) | fp16 | 16384 | 8,814 | 8,769 | -0.5 % | 17,868 |
| sum reduction (`red`) | fp16 | 16384 | 4,495 | 4,457 | -0.8 % | 8,942 |
| exponential (`exp`) | fp16 | 16384 | 62,146 | 60,310 | -3.0 % | 17,812 |
| GELU (`gelu`) | fp16 | 16384 | 120,091 | 98,917 | -17.6 % | 17,865 |
| softmax (`sfx`) | fp16 | 16384 | 102,526 | 87,881 | -14.3 % | 17,746 |
| layer normalization (`lyn`) | fp16 | 16384 | 57,236 | 37,286 | -34.9 % | 17,810 |
| attention (`attn`) | fp16 | seq 128 | 2,923,689 | 2,429,287 | -16.9 % | 17,924 |

Every record: [`../results/scratchpad_roofline_120mhz.md`](../results/scratchpad_roofline_120mhz.md).

Raw sweep of the 2026-09-29 bitstream: [`../results/roofline_sweep_120mhz.md`](../results/roofline_sweep_120mhz.md).
