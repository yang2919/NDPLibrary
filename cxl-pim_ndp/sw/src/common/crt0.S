// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0
//
// Minimal CRT0 for ara_vhk158 bare-metal SW.
//
// Entry sequence:
//   1. Set up global pointer (relaxation-friendly auipc/addi pair).
//   2. Set sp to top of stack region (defined in link.ld as _stack_top).
//   3. Zero .bss.
//   4. Call main(); on return, spin in _exit.
//
// No interrupt setup, no PMP, no FPU enable yet — main() does anything
// it needs. CVA6 boots from boot_addr (= DRAMBase 0x4000_0000); the
// linker script places .text.init here so _start is the very first
// instruction fetched.

    .section .text.init, "ax"
    .globl _start
    .globl _exit
    .extern main

_start:
    // gp (relaxation-friendly setup)
    .option push
    .option norelax
1:  auipc gp, %pcrel_hi(__global_pointer$)
    addi  gp, gp, %pcrel_lo(1b)
    .option pop

    // sp = _stack_top (defined in link.ld)
    la    sp, _stack_top

    // Enable scalar FPU (mstatus.FS = 01) AND RVV (mstatus.VS = 01).
    // Scalar FS is required for ANY fp load/store, fcvt, fmv.x.w etc.
    // — even just casting (float)i in C uses fp registers and traps
    // without FS. VS alone is not enough for fp main.c — only for
    // integer-only programs.
    // mstatus layout: FS = bits [14:13], VS = bits [10:9].
    // Initial state = 01 in each.
    li    t0, (1 << 9) | (1 << 13)
    csrs  mstatus, t0

    // Trap vector. Without this, mtvec keeps its reset value and ANY exception
    // jumps to address 0, which on this SoC re-enters the boot path — the
    // program silently restarts in a loop instead of stopping, and the symptom
    // reads as a hang at whatever heartbeat the loop happens to be sampled at
    // (2026-08-31: an spm_diag fault looked like a wedge at CAFE0020 for four
    // runs). The handler records the cause and PARKS, so the evidence survives.
    la    t0, _trap_entry
    csrw  mtvec, t0

    // zero .bss
    la    t0, __bss_start
    la    t1, __bss_end
2:  bgeu  t0, t1, 3f
    sd    zero, 0(t0)
    addi  t0, t0, 8
    j     2b
3:

    // jump into C
    call  main

_exit:
    // park here; HW reset / JTAG halt is the way out
    j     _exit

    // Trap record @ 0x4010_0300 (inside the reserved host gap, clear of
    // HB 0x..0000 / VERIFY 0x..0100 / RES 0x..0200):
    //   +0x00 magic 0x7A9B0001   +0x04 mcause_lo   +0x08 mcause_hi
    //   +0x0c mepc_lo            +0x10 mepc_hi     +0x14 mtval_lo
    //   +0x18 mtval_hi           +0x1c mstatus_lo  +0x20 vtype  +0x24 vl
    //   +0x28 vstart            +0x2c vcsr
    // Read it with sw/scripts/trap_read.tcl. mcause 2 = illegal instruction,
    // 5 = load access fault, 7 = store access fault, 1 = instruction access.
    .globl _trap_entry
    .align 4
_trap_entry:
    li    t0, 0x40100300
    li    t1, 0x7A9B0001
    sw    t1, 0(t0)
    csrr  t1, mcause
    sw    t1, 4(t0)
    srli  t1, t1, 32
    sw    t1, 8(t0)
    csrr  t1, mepc
    sw    t1, 12(t0)
    srli  t1, t1, 32
    sw    t1, 16(t0)
    csrr  t1, mtval
    sw    t1, 20(t0)
    srli  t1, t1, 32
    sw    t1, 24(t0)
    csrr  t1, mstatus
    sw    t1, 28(t0)
    csrr  t1, vtype
    sw    t1, 32(t0)
    csrr  t1, vl
    sw    t1, 36(t0)
    csrr  t1, vstart
    sw    t1, 40(t0)
    csrr  t1, vcsr
    sw    t1, 44(t0)
1:  j     1b
