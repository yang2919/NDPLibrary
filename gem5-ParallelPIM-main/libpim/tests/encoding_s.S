/*
 * The assembler macros of pim_isa.h with the registers of
 * test_encoding.sh's reference words: scalar rs2 = a2, imm = 0, rs1 = a1;
 * vector v8, rs1 = a1, e64.
 */

#include <gem5/pim/pim_isa.h>

        .text
        .option arch, +v

#define SCALAR(name)                                                        \
        .globl s_##name; s_##name: name a2, 0, a1; ret
#define VECTOR(name)                                                        \
        .globl s_##name; s_##name: name v8, a1, PIM_E64; ret

SCALAR(pim_sb_ewadd)
SCALAR(pim_sb_ewmul)
SCALAR(pim_sb_ewadd_gbuf)
SCALAR(pim_sb_ewmul_gbuf)
SCALAR(pim_sb_mac)
VECTOR(pim_sb_mov_bkgbuf)
VECTOR(pim_sb_mov_gbufbk)
VECTOR(pim_ab_ewadd)
VECTOR(pim_ab_ewmul)
VECTOR(pim_ab_ewadd_gbuf)
VECTOR(pim_ab_ewmul_gbuf)
VECTOR(pim_ab_mac)
VECTOR(pim_ab_mov_regbk)
VECTOR(pim_ab_mov_bkreg)
VECTOR(pim_rd_gb)
VECTOR(pim_wr_gb)
VECTOR(pim_rd_reg)
VECTOR(pim_wr_reg)

/* More encodings: imm = -32 and the e8/e16/e32 widths. */
        .globl s_extra
s_extra:
        pim_sb_mac a2, -32, a1
        pim_wr_gb v8, a1, PIM_E8
        pim_wr_gb v8, a1, PIM_E16
        pim_wr_gb v8, a1, PIM_E32
        ret
