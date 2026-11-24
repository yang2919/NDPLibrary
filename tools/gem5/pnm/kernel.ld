/* Native N0 fixture program window. The target team supplies its actual map. */
ENTRY(pnm_vector_add)
PHDRS { code PT_LOAD FLAGS(5); }
SECTIONS {
    . = 0x10000000;
    .text : { *(.text.entry) *(.text*) } :code
    /DISCARD/ : { *(.comment) *(.note*) }
}
