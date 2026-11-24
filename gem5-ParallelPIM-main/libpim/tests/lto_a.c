/*
 * With lto_b.c: two translation units that include pim_isa.h, linked
 * with -flto, which assembles their top-level asm as one file; the
 * vector wrappers' macro must be defined once.
 */
#include <stddef.h>
#include <stdint.h>

#include <gem5/pim/pim_isa.h>

size_t lto_move(void *p);

size_t
lto_move(void *p)
{
    return pim_sb_mov_gbufbk(p, 32);
}
