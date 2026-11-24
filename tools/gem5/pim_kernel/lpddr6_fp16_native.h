// Native reference layout: FP16 has the same width as the baseline BF16 operand.
#pragma once
#include <gem5/pim/pim_layout_lpddr6.h>
#undef PIM_LAYOUT_NUM_FMT
#define PIM_LAYOUT_NUM_FMT 1
#undef GEM5_PIM_LAYOUT_ID
#define GEM5_PIM_LAYOUT_ID ndp_lpddr6_fp16_native_layout
