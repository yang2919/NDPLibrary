/* M-mode start-up and trap entry of pim-fs-bm. */

    .section .text.start
    .globl _start
_start:
    la sp, stack_top
    la t0, mstack_top
    csrw mscratch, t0
    la t0, trap_entry
    csrw mtvec, t0
    /* PMP entry 7: NAPOT over all memory, RWX, for S-mode. */
    li t0, -1
    csrw pmpaddr7, t0
    li t0, 0x1f
    slli t0, t0, 56
    csrw pmpcfg0, t0
    call m_main
1:  j 1b

/* void enter_s(void (*entry)(void)): mret to entry in S-mode. */
    .globl enter_s
enter_s:
    csrw mepc, a0
    mret

/* Every trap: to m_trap(regs) on the M-mode stack. regs[i] is x(i). */
    .balign 4
trap_entry:
    csrrw sp, mscratch, sp
    addi sp, sp, -256
    sd x1, 8(sp)
    sd x5, 40(sp)
    sd x6, 48(sp)
    sd x7, 56(sp)
    sd x10, 80(sp)
    sd x11, 88(sp)
    sd x12, 96(sp)
    sd x13, 104(sp)
    sd x14, 112(sp)
    sd x15, 120(sp)
    sd x16, 128(sp)
    sd x17, 136(sp)
    sd x28, 224(sp)
    sd x29, 232(sp)
    sd x30, 240(sp)
    sd x31, 248(sp)
    mv a0, sp
    call m_trap
    ld x1, 8(sp)
    ld x5, 40(sp)
    ld x6, 48(sp)
    ld x7, 56(sp)
    ld x10, 80(sp)
    ld x11, 88(sp)
    ld x12, 96(sp)
    ld x13, 104(sp)
    ld x14, 112(sp)
    ld x15, 120(sp)
    ld x16, 128(sp)
    ld x17, 136(sp)
    ld x28, 224(sp)
    ld x29, 232(sp)
    ld x30, 240(sp)
    ld x31, 248(sp)
    addi sp, sp, 256
    csrrw sp, mscratch, sp
    mret

#ifndef PIM_FS_BM_NOPIM
/* void f1_kernel(void *win4k, void *win2m, unsigned long desc): case F1
 * between a statistics reset and a dump, with no normal data access:
 * 3 store-like PIM lookups (pim.sb.mac and pim.ab.mac on 4 KiB pages,
 * pim.ab.mac on the 2 MiB page) and 2 load-like ones (pim.rd.gb). */
    .globl f1_kernel
f1_kernel:
    .option push
    .option arch, +v
    mv t1, a0
    mv t2, a1
    mv t3, a2
    li a0, 0
    li a1, 0
    .long 0x0000007b | (0x40 << 25)     /* m5_reset_stats(0, 0) */
    vsetivli t0, 4, e64, m1, ta, ma
    .insn s 0x2b, 4, t3, 0(t1)          /* pim.sb.mac t3, 0(t1) */
    addi t4, t1, 0x40
    .insn r 0x5b, 7, 1, x8, t4, x6      /* pim.ab.mac v8, (t4) */
    addi t4, t1, 0x80
    .insn r 0x0b, 7, 1, x8, t4, x0      /* pim.rd.gb v8, (t4) */
    .insn r 0x5b, 7, 1, x8, t2, x6      /* pim.ab.mac v8, (t2) */
    li t4, 0x1000
    add t4, t2, t4
    .insn r 0x0b, 7, 1, x8, t4, x0      /* pim.rd.gb v8, (t4) */
    li a0, 0
    li a1, 0
    .long 0x0000007b | (0x41 << 25)     /* m5_dump_stats(0, 0) */
    .option pop
    ret

/* F13: cache-block operations in U-mode. Each
 * u_cbo_* runs one operation on (a0) and ends with ecall, which M-mode
 * turns into a return to run_user's caller in S-mode; it runs at its
 * alias + USER_ALIAS (pim_fs_bm.c), a U = 1 mapping of the code. */
    .globl u_cbo_inval, u_cbo_clean, u_cbo_flush, u_cbo_zero, run_user
u_cbo_inval:
    .insn i 0x0f, 2, x0, 0(a0)          /* cbo.inval (a0) */
    ecall
u_cbo_clean:
    .insn i 0x0f, 2, x0, 1(a0)          /* cbo.clean (a0) */
    ecall
u_cbo_flush:
    .insn i 0x0f, 2, x0, 2(a0)          /* cbo.flush (a0) */
    ecall
u_cbo_zero:
    .insn i 0x0f, 2, x0, 4(a0)          /* cbo.zero (a0) */
    ecall

/* void run_user(uint64_t entry, uint64_t arg): sret to entry in U-mode
 * with a0 = arg. */
run_user:
    csrw sepc, a0
    mv a0, a1
    li t0, 0x100                        /* sstatus.SPP = U */
    csrc sstatus, t0
    sret
#endif /* !PIM_FS_BM_NOPIM */

    .section .note.GNU-stack, "", @progbits
