/*
 * A C11 program that includes <gem5/pim/pim.h> and links the archive
 * of its layout. Prints the main constants and checks the layout symbol.
 */

#include <stdio.h>

#include <gem5/pim/pim.h>

int
main(void)
{
    printf("std %d policy %d channels %d row %d bytes cols %d "
           "row_shift %d contig %d\n",
           PIM_LAYOUT_STD, PIM_LAYOUT_POLICY, PIM_LAYOUT_CHANNELS,
           PIM_LAYOUT_ROW_BYTES, PIM_LAYOUT_COLS, PIM_LAYOUT_ROW_SHIFT,
           PIM_LAYOUT_CONTIG_BYTES);
    printf("window 0x%llx bytes, base SE 0x%llx, FS 0x%llx, reg row %d\n",
           (unsigned long long)PIM_LAYOUT_WINDOW_SIZE,
           (unsigned long long)PIM_LAYOUT_WINDOW_BASE_SE,
           (unsigned long long)PIM_LAYOUT_WINDOW_BASE_FS, PIM_LAYOUT_REG_ROW0);
    if (GEM5_PIM_LAYOUT_ID != PIM_LAYOUT_STD) {
        printf("FAIL: layout symbol %u\n", GEM5_PIM_LAYOUT_ID);
        return 1;
    }
    return 0;
}
