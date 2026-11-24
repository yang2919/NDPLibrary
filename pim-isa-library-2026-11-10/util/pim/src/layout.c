/*
 * The layout this archive was built for (DP-B (a), P-LIB section 14).
 *
 * pim.h declares GEM5_PIM_LAYOUT_ID, a name derived from every constant
 * of the layout header, and the library's entry points refer to it, so
 * that a program built for one layout does not link against the archive
 * of another one.
 */

#include <gem5/pim/pim.h>

const unsigned GEM5_PIM_LAYOUT_ID = PIM_LAYOUT_STD;
