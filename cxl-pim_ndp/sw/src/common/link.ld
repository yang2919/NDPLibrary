/* Copyright 2026 Jae
 * SPDX-License-Identifier: Apache-2.0
 *
 * Linker script for ara_vhk158 bare-metal SW.
 *
 * Memory:
 *   DRAM @ 0x4000_0000, 1 GiB (matches DRAMLength in ara_soc_vhk158.sv).
 *   CVA6 boot_addr = DRAMBase (see rtl/ara_soc_vhk158.sv:509), so
 *   placing .text.init at the very start of DRAM means _start is the
 *   first instruction the CPU fetches after reset.
 *
 * Sections layout:
 *   [.text.init][.text][.rodata][.data][.bss][heap][stack ↓]
 *   The stack top follows .bss instead of sitting at a fixed address, so
 *   .bss size is bounded only by DRAM.
 */

OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY {
  DRAM (rwx) : ORIGIN = 0x40000000, LENGTH = 0x40000000  /* 1 GiB */
}

SECTIONS {
  .text.init : {
    *(.text.init)
  } > DRAM

  .text : ALIGN(8) {
    *(.text .text.*)
  } > DRAM

  /* Kernels whose .rodata is multi-MiB (sw/rag: a 4 MiB FAISS database) would
     otherwise grow across the host scratchpad gap at 0x40100000, so the HB[0]
     write lands inside their own read-only data. Such a kernel links with
     -Wl,--section-start=.rodata=0x40200000 to start .rodata above the gap;
     with no flag the layout is unchanged (.rodata immediately after .text) for
     every other kernel, byte-for-byte.
     NOT --defsym + DEFINED(): --defsym symbols are not visible to DEFINED()
     while the script is evaluated, so that form silently took the else branch
     and left .rodata across the gap (measured 2026-08-30). */
  .rodata : ALIGN(8) {
    *(.rodata .rodata.* .gnu.linkonce.r.*)
    *(.srodata .srodata.*)
  } > DRAM

  .data : ALIGN(8) {
    __global_pointer$ = . + 0x800;
    *(.data .data.*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
  } > DRAM

  /* Reserve the host scratchpad gap. HB=0x40100000 / VERIFY=0x40100100 are plain
     DRAM addresses the host reads over JTAG; they sit between .data (code+data is
     well under 1 MiB, ends far below 0x40100000) and .bss. Force .bss to DRAMBase +
     2 MiB so no kernel array can grow across 0x40100000 — fp32 attention's X[128*768]
     did exactly that (HB[0] write landed in X[45888] -> Q=1.3M garbage). Safe with
     -mno-relax (medany PC-relative globals; no gp-relative range limit on .bss). */
  /* Pinned at DRAMBase + 2 MiB, or just above .rodata/.data when a big-.rodata
     kernel already runs past that. MAX() inside the section-address expression,
     NOT a `. = ...` assignment before the section: a section with `> DRAM` takes
     its address from the region allocator and silently ignores the assignment
     (measured 2026-08-30 — it unpinned .bss to 0x40001660 and put gather_bench's
     32 MiB DATA[] straight across the 0x40100000 gap). */
  .bss MAX(., 0x40200000) (NOLOAD) : ALIGN(8) {   /* see scratchpad note above */
    __bss_start = .;
    *(.bss .bss.* .sbss .sbss.*)
    *(COMMON)
    __bss_end = ALIGN(8);
  } > DRAM

  . = ALIGN(16);
  _heap_start = .;

  /* Stack top sits ABOVE .bss and grows down toward _heap_start. The previous
     fixed top (DRAMBase + 16 MiB) silently capped .bss at 14 MiB; gqa_attention
     needs 28 MiB of weights/activations, which would have had the stack writing
     into the middle of them. 4 MiB of stack is far beyond what these leaf-call
     kernels use (no recursion, no alloca). */
  . = . + 0x00400000;
  . = ALIGN(16);
  _stack_top = .;

  ASSERT(_stack_top <= ORIGIN(DRAM) + LENGTH(DRAM),
         "link.ld: .bss + stack overflows the 1 GiB DRAM region")

  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
