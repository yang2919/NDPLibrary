/*
 * Kernels of the processing-in-memory (PIM) decode test:
 * fixed sequences of PIM instructions, each group with its own vsetvli.
 * tests/gem5/pim/check_exec_trace.py mirrors these sequences; change both
 * together. a0 is the PIM address (rs1) unless noted.
 */

#include "pim_insn.h"

    .option arch, +v
    .text

/* void pim_scalar(void *buf, unsigned long desc): the 5 scalar forms with
 * rs2 = desc and imm = 0, then pim.sb.mac with imm = -32 and rs2 = x0. */
    .globl pim_scalar
pim_scalar:
    pim_sb_ewadd a1, 0, a0
    pim_sb_ewmul a1, 0, a0
    pim_sb_ewadd_gbuf a1, 0, a0
    pim_sb_ewmul_gbuf a1, 0, a0
    pim_sb_mac a1, 0, a0
    addi t0, a0, 32
    pim_sb_mac x0, -32, t0
    ret

/* void pim_vstore(void *buf, void *reg): the 11 vector store-like forms,
 * e64, m1, vl = VLMAX (one micro-op each); pim.wr.reg at reg (a1), the
 * others at buf, as a PIM memory serves register accesses only in its
 * register rows. */
    .globl pim_vstore
pim_vstore:
    vsetvli t0, zero, e64, m1, ta, ma
    pim_sb_mov_bkgbuf PIM_E64, 8, a0
    pim_sb_mov_gbufbk PIM_E64, 8, a0
    pim_ab_ewadd PIM_E64, 8, a0
    pim_ab_ewmul PIM_E64, 8, a0
    pim_ab_ewadd_gbuf PIM_E64, 8, a0
    pim_ab_ewmul_gbuf PIM_E64, 8, a0
    pim_ab_mac PIM_E64, 8, a0
    pim_ab_mov_regbk PIM_E64, 8, a0
    pim_ab_mov_bkreg PIM_E64, 8, a0
    pim_wr_gb PIM_E64, 8, a0
    pim_wr_reg PIM_E64, 8, a1
    ret

/* void pim_vmulti(void *buf): pim.ab.mac and pim.wr.gb at e64/m2 VLMAX
 * (two micro-ops), at e8, e16, e32 m1 VLMAX (one each), and at vl = 0
 * (no request). */
    .globl pim_vmulti
pim_vmulti:
    vsetvli t0, zero, e64, m2, ta, ma
    pim_ab_mac PIM_E64, 8, a0
    pim_wr_gb PIM_E64, 8, a0
    vsetvli t0, zero, e8, m1, ta, ma
    pim_ab_mac PIM_E8, 8, a0
    pim_wr_gb PIM_E8, 8, a0
    vsetvli t0, zero, e16, m1, ta, ma
    pim_ab_mac PIM_E16, 8, a0
    pim_wr_gb PIM_E16, 8, a0
    vsetvli t0, zero, e32, m1, ta, ma
    pim_ab_mac PIM_E32, 8, a0
    pim_wr_gb PIM_E32, 8, a0
    vsetivli t0, 0, e64, m1, ta, ma
    pim_ab_mac PIM_E64, 8, a0
    pim_wr_gb PIM_E64, 8, a0
    ret

/* void pim_vsplit(void *addr): pim.wr.gb of 32 bytes at addr (a line
 * base + 48, so the request is split at the line). */
    .globl pim_vsplit
pim_vsplit:
    vsetivli t0, 4, e64, m1, ta, ma
    pim_wr_gb PIM_E64, 8, a0
    ret

/* void pim_vload(void *buf, void *reg): pim.rd.gb at buf and pim.rd.reg
 * at reg (a1), each at e64/m1 and e64/m2, VLMAX. */
    .globl pim_vload
pim_vload:
    vsetvli t0, zero, e64, m1, ta, ma
    pim_rd_gb PIM_E64, 8, a0
    pim_rd_reg PIM_E64, 8, a1
    vsetvli t0, zero, e64, m2, ta, ma
    pim_rd_gb PIM_E64, 8, a0
    pim_rd_reg PIM_E64, 8, a1
    ret

/* void pim_bad(long n, void *buf): one reserved encoding;
 * each is expected to decode as Unknown. */
    .globl pim_bad
pim_bad:
    vsetvli t0, zero, e64, m1, ta, ma
    la t0, bad_table
    slli a0, a0, 3
    add t0, t0, a0
    ld t0, 0(t0)
    jr t0
bad_0:  .insn s 0x2b, 5, a2, 0(a1)          /* funct3 5 */
        ret
bad_1:  .insn s 0x2b, 6, a2, 0(a1)          /* funct3 6 */
        ret
bad_2:  .insn s 0x2b, 7, a2, 0(a1)          /* funct3 7 */
        ret
bad_3:  .insn r 0x5b, 7, 1, x8, a1, x11     /* sumop 0x0b */
        ret
bad_4:  .insn r 0x5b, 7, 1, x8, a1, x31     /* sumop 0x1f */
        ret
bad_5:  .insn r 0x0b, 7, 1, x8, a1, x2      /* lumop 0x02 */
        ret
bad_6:  .insn r 0x0b, 7, 1, x8, a1, x31     /* lumop 0x1f */
        ret
bad_7:  .insn r 0x5b, 7, 0, x8, a1, x6      /* vm 0 */
        ret
bad_8:  .insn r 0x5b, 7, 3, x8, a1, x6      /* mop 1 */
        ret
bad_9:  .insn r 0x5b, 7, 9, x8, a1, x6      /* mew 1 */
        ret
bad_10: .insn r 0x5b, 7, 33, x8, a1, x6     /* nf != 0 */
        ret
bad_11: .insn r 0x5b, 1, 1, x8, a1, x6      /* width 1 */
        ret
bad_12: .insn r 0x5b, 2, 1, x8, a1, x6      /* width 2 */
        ret
bad_13: .insn r 0x5b, 3, 1, x8, a1, x6      /* width 3 */
        ret
bad_14: .insn r 0x5b, 4, 1, x8, a1, x6      /* width 4 */
        ret
bad_15: .insn r 0x0b, 7, 0, x8, a1, x0      /* load-like vm 0 */
        ret

/* void pim_guard(long n, void *buf, unsigned long desc): the address
 * guards. 0: pim.sb.mac at buf + 4 (misaligned); 1:
 * pim.ab.mac of 32 bytes at buf + 4096 - 16 (crosses a page). */
    .globl pim_guard
pim_guard:
    bnez a0, 1f
    addi t0, a1, 4
    pim_sb_mac a2, 0, t0
    ret
1:  li t0, 4096 - 16
    add t0, a1, t0
    vsetivli t1, 4, e64, m1, ta, ma
    pim_ab_mac PIM_E64, 8, t0
    ret

    .section .rodata
    .balign 8
bad_table:
    .dword bad_0, bad_1, bad_2, bad_3, bad_4, bad_5, bad_6, bad_7
    .dword bad_8, bad_9, bad_10, bad_11, bad_12, bad_13, bad_14, bad_15

    .section .note.GNU-stack, "", @progbits
