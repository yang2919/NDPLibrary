/*
 * Processing-in-Memory (PIM) layout selector.
 *
 * Includes the generated layout header of one PIM memory configuration
 * (util/pim/gen_pim_layout.py) and checks its constants at compile time.
 *
 * - PIM_LAYOUT_HEADER, a quoted file name, selects a generated header of
 *   a non-default configuration, for example
 *   -DPIM_LAYOUT_HEADER='"my_layout.h"'.
 * - Otherwise PIM_STD selects one of the committed default headers:
 *   ddr5, lpddr6 (the default), gddr7 or hbm4, for example -DPIM_STD=hbm4.
 *
 * Usable from C11 and C++17. Ends by including <gem5/pim/pim_addr.h>,
 * whose pim_encode() family then uses this layout's constants.
 */

#ifndef __GEM5_PIM_PIM_LAYOUT_H__
#define __GEM5_PIM_PIM_LAYOUT_H__

#if defined(PIM_LAYOUT_HEADER)
#define PIM__LAYOUT_FILE PIM_LAYOUT_HEADER
#else
#ifndef PIM_STD
#define PIM_STD lpddr6
#endif
#define PIM__STD_ddr5 1
#define PIM__STD_lpddr6 2
#define PIM__STD_gddr7 3
#define PIM__STD_hbm4 4
#define PIM__CAT(a, b) a##b
#define PIM__XCAT(a, b) PIM__CAT(a, b)
#if PIM__XCAT(PIM__STD_, PIM_STD) == 1
#define PIM__LAYOUT_FILE "pim_layout_ddr5.h"
#elif PIM__XCAT(PIM__STD_, PIM_STD) == 2
#define PIM__LAYOUT_FILE "pim_layout_lpddr6.h"
#elif PIM__XCAT(PIM__STD_, PIM_STD) == 3
#define PIM__LAYOUT_FILE "pim_layout_gddr7.h"
#elif PIM__XCAT(PIM__STD_, PIM_STD) == 4
#define PIM__LAYOUT_FILE "pim_layout_hbm4.h"
#else
#error "PIM_STD must be ddr5, lpddr6, gddr7 or hbm4"
#endif
#endif
#include PIM__LAYOUT_FILE

#if !defined(PIM_LAYOUT_POLICY) || !defined(GEM5_PIM_LAYOUT_ID)
#error "the selected PIM layout header is incomplete"
#endif

/* Policy numbers (PIM_POLICY_*) are in pim_addr.h. */

/* Number formats (PIM_LAYOUT_NUM_FMT, PIM_LAYOUT_ACC_FMT). */
#define PIM_FMT_BF16 0
#define PIM_FMT_FP16 1
#define PIM_FMT_FP32 2
#define PIM_FMT_INT8 3
#define PIM_FMT_INT16 4
#define PIM_FMT_INT32 5

/* Compile-time checks of the constants. */
#ifdef __cplusplus
#define PIM__STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define PIM__STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
#define PIM__POW2(x) ((x) > 0 && (((x) & ((x) - 1)) == 0))
#define PIM__AT_LEAST_1(x) ((x) > 0 ? (x) : 1)
#define PIM__ULL(x) ((unsigned long long)(x))

PIM__STATIC_ASSERT(PIM_LAYOUT_STD >= 1 && PIM_LAYOUT_STD <= 4,
                   "PIM_LAYOUT_STD out of range");
PIM__STATIC_ASSERT(PIM_LAYOUT_POLICY >= 0 && PIM_LAYOUT_POLICY <= 5,
                   "PIM_LAYOUT_POLICY out of range");
PIM__STATIC_ASSERT(
    PIM__POW2(PIM_LAYOUT_CHANNELS) && PIM__POW2(PIM_LAYOUT_INTLV_BYTES) &&
        PIM__POW2(PIM_LAYOUT_ROWS) && PIM__POW2(PIM_LAYOUT_COLS) &&
        PIM__POW2(PIM_LAYOUT_COL_BYTES) && PIM__POW2(PIM_LAYOUT_ROW_BYTES) &&
        PIM__POW2(PIM_LAYOUT_BANKS_PER_GROUP) && PIM__POW2(PIM_LAYOUT_UNITS) &&
        PIM__POW2(PIM_LAYOUT_BANKS_PER_UNIT) &&
        PIM__POW2(PIM_LAYOUT_CONTIG_BYTES) &&
        PIM__POW2(PIM_LAYOUT_WINDOW_SIZE),
    "PIM layout counts and sizes must be powers of two");
PIM__STATIC_ASSERT(PIM_LAYOUT_ROW_BYTES ==
                       PIM_LAYOUT_COLS * PIM_LAYOUT_COL_BYTES,
                   "a row is COLS columns of COL_BYTES");
PIM__STATIC_ASSERT(PIM_LAYOUT_BANKS_PER_UNIT ==
                       PIM__AT_LEAST_1(PIM_LAYOUT_SIDS) *
                           PIM__AT_LEAST_1(PIM_LAYOUT_BANK_GROUPS) *
                           PIM_LAYOUT_BANKS_PER_GROUP,
                   "BANKS_PER_UNIT does not match the bank levels");
PIM__STATIC_ASSERT(PIM_LAYOUT_UNITS ==
                       PIM__AT_LEAST_1(PIM_LAYOUT_RANKS) *
                           PIM__AT_LEAST_1(PIM_LAYOUT_PSEUDO_CHANNELS),
                   "UNITS does not match the rank or pseudo channel count");
PIM__STATIC_ASSERT(PIM__ULL(PIM_LAYOUT_WINDOW_SIZE) ==
                       PIM__ULL(PIM_LAYOUT_CHANNELS) * PIM_LAYOUT_UNITS *
                           PIM_LAYOUT_BANKS_PER_UNIT * PIM_LAYOUT_ROWS *
                           PIM_LAYOUT_ROW_BYTES,
                   "WINDOW_SIZE is not the capacity of every channel");
PIM__STATIC_ASSERT(PIM_LAYOUT_COL_BYTES <= PIM_LAYOUT_CONTIG_BYTES &&
                       PIM_LAYOUT_CONTIG_BYTES <= PIM_LAYOUT_ROW_BYTES,
                   "CONTIG_BYTES must lie between a column and a row");
PIM__STATIC_ASSERT(PIM_LAYOUT_INTLV_BYTES >= PIM_LAYOUT_COL_BYTES,
                   "the channel interleave is below one column");
PIM__STATIC_ASSERT((1ULL << PIM_LAYOUT_COL_SHIFT) >= PIM_LAYOUT_COL_BYTES,
                   "the column field lies above the column bytes");
PIM__STATIC_ASSERT((1ULL << PIM_LAYOUT_CH_SHIFT) ==
                       PIM__ULL(PIM_LAYOUT_INTLV_BYTES),
                   "the channel field starts at the interleave");
PIM__STATIC_ASSERT(
    PIM__ULL(PIM_LAYOUT_WINDOW_BASE_SE) % PIM_LAYOUT_WINDOW_SIZE == 0 &&
        PIM__ULL(PIM_LAYOUT_WINDOW_BASE_FS) % PIM_LAYOUT_WINDOW_SIZE == 0,
    "the PIM memory base must be a multiple of its size");
PIM__STATIC_ASSERT(PIM_LAYOUT_WINDOW_ROW0 == 0,
                   "the whole PIM memory is the window");
PIM__STATIC_ASSERT(PIM_LAYOUT_NUM_REGS >= 1 &&
                       PIM_LAYOUT_REG_ROW0 ==
                           PIM_LAYOUT_ROWS - PIM_LAYOUT_NUM_REGS - 1,
                   "the register rows are the top NUM_REGS + 1 rows");
PIM__STATIC_ASSERT((PIM_LAYOUT_ELEMS_PER_COL * PIM_LAYOUT_ELEM_BYTES) ==
                           PIM_LAYOUT_COL_BYTES &&
                       PIM_LAYOUT_ACC_BYTES >= PIM_LAYOUT_ELEM_BYTES,
                   "the number formats do not fit a column");
PIM__STATIC_ASSERT(PIM_LAYOUT_PUS_PER_BANK == 1, "one PU per bank");

#include "pim_addr.h"

#endif // __GEM5_PIM_PIM_LAYOUT_H__
