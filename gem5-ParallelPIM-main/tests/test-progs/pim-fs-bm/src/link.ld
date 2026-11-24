/* Bare-metal layout of pim-fs-bm: code and data at 1 MiB in the
 * bare-metal system's memory, which starts at 0. */
OUTPUT_ARCH(riscv)
ENTRY(_start)
SECTIONS
{
    . = 0x100000;
    .text : { *(.text.start) *(.text*) }
    .rodata : { *(.rodata*) *(.srodata*) }
    .data : { *(.data*) *(.sdata*) }
    .bss : { *(.bss*) *(.sbss*) *(COMMON) }
    . = ALIGN(16);
    . += 0x10000;
    stack_top = .;
    . += 0x4000;
    mstack_top = .;
    /DISCARD/ : { *(.note*) *(.comment*) *(.eh_frame*) }
}
