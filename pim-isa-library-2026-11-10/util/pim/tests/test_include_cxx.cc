/*
 * L0: the same as test_include.c, compiled as C++17.
 */

#include <cstdio>

#include <gem5/pim/pim.h>

static_assert(PIM_LAYOUT_ROW_BYTES == PIM_LAYOUT_COLS * PIM_LAYOUT_COL_BYTES,
              "row bytes");

int
main()
{
    std::printf("C++: std %d row %d bytes\n", PIM_LAYOUT_STD,
                PIM_LAYOUT_ROW_BYTES);
    return GEM5_PIM_LAYOUT_ID == PIM_LAYOUT_STD ? 0 : 1;
}
