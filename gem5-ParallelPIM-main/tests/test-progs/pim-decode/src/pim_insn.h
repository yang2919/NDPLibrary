/*
 * Encodings of the processing-in-memory (PIM) instructions, for the
 * decode test. A test aid, written independently of the PIM library's
 * include/gem5/pim/pim_isa.h, so that the two can be checked against
 * each other.
 *
 * Assembler: scalar forms are "pim_<name> rs2, off, rs1" for
 * "pim.<name> rs2, off(rs1)"; vector forms are "pim_<name> eew, vreg,
 * rs1" with the vector register as a number, since .insn names only
 * x registers.
 */

#ifndef __PIM_DECODE_PIM_INSN_H__
#define __PIM_DECODE_PIM_INSN_H__

/* Descriptor of the scalar forms: ncols in bits 15:0. */
#define PIM_DESC_NCOLS_SHIFT 0
#define PIM_DESC_NCOLS_MASK 0xffff
#define PIM_DESC(ncols) \
    ((((unsigned long)(ncols)) & PIM_DESC_NCOLS_MASK) << PIM_DESC_NCOLS_SHIFT)

#ifdef __ASSEMBLER__

.equ PIM_E8, 0
.equ PIM_E16, 5
.equ PIM_E32, 6
.equ PIM_E64, 7

/* custom-1 (0x2b), funct3 = operation */
.macro pim_sb_ewadd rs2, off, rs1
    .insn s 0x2b, 0, \rs2, \off(\rs1)
.endm
.macro pim_sb_ewmul rs2, off, rs1
    .insn s 0x2b, 1, \rs2, \off(\rs1)
.endm
.macro pim_sb_ewadd_gbuf rs2, off, rs1
    .insn s 0x2b, 2, \rs2, \off(\rs1)
.endm
.macro pim_sb_ewmul_gbuf rs2, off, rs1
    .insn s 0x2b, 3, \rs2, \off(\rs1)
.endm
.macro pim_sb_mac rs2, off, rs1
    .insn s 0x2b, 4, \rs2, \off(\rs1)
.endm

/* custom-2 (0x5b), sumop = operation; funct7 1 = vm 1, mop 0, mew 0, nf 0 */
.macro pim_vs op, eew, vs3, rs1
    .insn r 0x5b, \eew, 1, x\vs3, \rs1, x\op
.endm
.macro pim_sb_mov_bkgbuf eew, vs3, rs1
    pim_vs 0, \eew, \vs3, \rs1
.endm
.macro pim_sb_mov_gbufbk eew, vs3, rs1
    pim_vs 1, \eew, \vs3, \rs1
.endm
.macro pim_ab_ewadd eew, vs3, rs1
    pim_vs 2, \eew, \vs3, \rs1
.endm
.macro pim_ab_ewmul eew, vs3, rs1
    pim_vs 3, \eew, \vs3, \rs1
.endm
.macro pim_ab_ewadd_gbuf eew, vs3, rs1
    pim_vs 4, \eew, \vs3, \rs1
.endm
.macro pim_ab_ewmul_gbuf eew, vs3, rs1
    pim_vs 5, \eew, \vs3, \rs1
.endm
.macro pim_ab_mac eew, vs3, rs1
    pim_vs 6, \eew, \vs3, \rs1
.endm
.macro pim_ab_mov_regbk eew, vs3, rs1
    pim_vs 7, \eew, \vs3, \rs1
.endm
.macro pim_ab_mov_bkreg eew, vs3, rs1
    pim_vs 8, \eew, \vs3, \rs1
.endm
.macro pim_wr_gb eew, vs3, rs1
    pim_vs 9, \eew, \vs3, \rs1
.endm
.macro pim_wr_reg eew, vs3, rs1
    pim_vs 10, \eew, \vs3, \rs1
.endm

/* custom-0 (0x0b), lumop = operation */
.macro pim_rd_gb eew, vd, rs1
    .insn r 0x0b, \eew, 1, x\vd, \rs1, x0
.endm
.macro pim_rd_reg eew, vd, rs1
    .insn r 0x0b, \eew, 1, x\vd, \rs1, x1
.endm

#endif /* __ASSEMBLER__ */

#endif /* __PIM_DECODE_PIM_INSN_H__ */
