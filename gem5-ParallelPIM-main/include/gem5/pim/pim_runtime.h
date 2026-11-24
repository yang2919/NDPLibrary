/*
 * Processing-in-Memory (PIM) library runtime.
 *
 * Initialization, the Way 1 region (the whole PIM memory mapped linearly
 * through /dev/mem, or the emulated driver in Syscall Emulation (SE)
 * mode), Way 2 (Full System (FS) only: the PIM memory a NUMA node, row
 * slices from 2 MiB huge pages bound to it, the register rows through
 * /dev/mem), the row-slice allocator, register addresses and status
 * codes.
 * Implemented in libpim_<std>.a; the layout is the one the program and
 * the archive were built for (<gem5/pim/pim_layout.h>).
 *
 * A buffer is a set of row slices (one row index in every bank of every
 * channel). Inside a buffer, a coordinate's row is the slice index;
 * pim_buf_addr() gives the address of any (channel, unit, stack ID, bank
 * group, bank, slice, column), so the caller chooses the placement.
 */

#ifndef __GEM5_PIM_PIM_RUNTIME_H__
#define __GEM5_PIM_PIM_RUNTIME_H__

#include <stddef.h>
#include <stdint.h>

#include <gem5/pim/pim_isa.h>
#include <gem5/pim/pim_layout.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes. */
typedef enum pim_status
{
    PIM_OK = 0,
    PIM_E_ARG,      /* bad argument or shape */
    PIM_E_NOMEM,    /* no free row slices */
    PIM_E_DEVICE,   /* /dev/mem or the driver cannot be opened or mapped */
    PIM_E_MODE,     /* the run-time mode contradicts the system */
    PIM_E_NODE,     /* Way 2: no PIM NUMA node */
    PIM_E_HUGEPAGE, /* Way 2: no free huge page */
    PIM_E_STATE,    /* called before pim_init() or after pim_finalize() */
} pim_status_t;

/* SE or Full System (FS) mode: selects the PIM memory's base. */
typedef enum pim_mode
{
    PIM_MODE_DEFAULT = 0, /* PIM_MODE in the environment, else the build */
    PIM_MODE_SE = 1,
    PIM_MODE_FS = 2,
} pim_mode_t;

typedef struct pim_config
{
    pim_mode_t mode;
    int way;            /* 0: PIM_WAY in the environment (1 or 2), else 1;
                           1: the linear region; 2: huge pages on the PIM
                           memory's NUMA node (FS; PIM_NODE, else the
                           device tree, names the node) */
    unsigned flags;     /* none yet */
    const char *device; /* NULL: /dev/mem */
} pim_config_t;

#define PIM_CONFIG_INIT {PIM_MODE_DEFAULT, 0, 0, NULL}

/*
 * Defined by the archive built for the same layout (libpim_<std>.a), and
 * referenced by pim_init(), so that a program built for another layout
 * fails to link.
 */
extern const unsigned GEM5_PIM_LAYOUT_ID;

/* A set of row slices owned by the program. */
typedef struct pim_buf pim_buf_t;

/* Library entry points; the program calls the inline wrappers below. */
pim_status_t pim__init(const pim_config_t *cfg, const unsigned *layout_id);

/*
 * Map the PIM memory and start the library. NULL takes the defaults:
 * the mode from PIM_MODE (se or fs), else the default libpim was built
 * with; the way from PIM_WAY, else 1. Refuses (PIM_E_MODE) a mode that
 * uname() contradicts (SE reports the node name sim.gem5.org), and in FS
 * a range that /proc/iomem shows as System RAM (Way 2: the register
 * rows). Way 2 maps huge pages as buffers need them; PIM_VERBOSE prints
 * the node and every page's frame.
 */
static inline pim_status_t
pim_init(const pim_config_t *cfg)
{
    /* Referring to the layout's symbol makes a program built for another
     * layout fail to link against this archive. */
    return pim__init(cfg, &GEM5_PIM_LAYOUT_ID);
}

void pim_finalize(void);
const char *pim_strerror(pim_status_t s);
/* More detail on the last error, or "". */
const char *pim_error_detail(void);

/* The mode pim_init() chose, and the region (Way 2: none, NULL and 0). */
pim_mode_t pim_mode(void);
void *pim_region_base(void);
uint64_t pim_region_size(void);
/* The PIM memory's physical base (WINDOW_BASE_SE or WINDOW_BASE_FS). */
uint64_t pim_region_phys(void);

/* Buffers of whole row slices. */
pim_status_t pim_buf_alloc(pim_buf_t **b, uint32_t n_slices);
void pim_buf_free(pim_buf_t *b);
uint32_t pim_buf_slices(const pim_buf_t *b);
/* The absolute row index of a slice of the buffer. */
uint32_t pim_buf_row(const pim_buf_t *b, uint32_t slice);
/* The address of coordinates in the buffer; c->row is the slice index.
 * NULL when the slice is outside the buffer. */
void *pim_buf_addr(const pim_buf_t *b, const pim_coord_t *c);
/* Free row slices left (Way 2: in the huge pages mapped so far). */
uint32_t pim_free_slices(void);

/*
 * The register window: row REG_ROW0 + j of every bank
 * is register j of its PU, and row REG_ROW0 + NUM_REGS the accumulator
 * row. pim_reg_addr() takes the bank and column of c (c->row ignored);
 * j = NUM_REGS gives the accumulator row; NULL for j > NUM_REGS.
 */
void *pim_reg_addr(const pim_coord_t *c, uint32_t j);
void *pim_acc_addr(const pim_coord_t *c);
/* An address of the global buffer (GB) of a unit, at a column (the GB
 * commands decode only the channel and unit, and the column). */
void *pim_gb_addr(uint32_t ch, uint32_t unit, uint32_t col);

/*
 * Number formats. Host data are fp32; PIM operands are NUM_FMT,
 * accumulators ACC_FMT. Narrowing rounds to nearest even; integer
 * formats round to the nearest integer and wrap.
 */
uint16_t pim_f32_to_bf16(float f);
float pim_bf16_to_f32(uint16_t b);
/* One element of PIM_LAYOUT_NUM_FMT (PIM_LAYOUT_ELEM_BYTES bytes). */
void pim_f32_to_elem(void *dst, float f);
float pim_elem_to_f32(const void *src);
/* One accumulator of PIM_LAYOUT_ACC_FMT (PIM_LAYOUT_ACC_BYTES bytes). */
float pim_acc_to_f32(const void *src);

/*
 * Copying: normal loads and stores at the encoder's
 * addresses, one address-contiguous piece (at most
 * PIM_LAYOUT_CONTIG_BYTES, inside one bank row) at a time.
 */
void pim_copy_piece(void *pim, const void *host, size_t n);
void pim_copy_piece_out(void *host, const void *pim, size_t n);
/* Bytes [off, off + n) of the bank row of c (c->row is the slice index,
 * c->col ignored), to or from host memory. */
pim_status_t pim_write_bank_row(const pim_buf_t *b, const pim_coord_t *c,
                                size_t off, const void *src, size_t n);
pim_status_t pim_read_bank_row(const pim_buf_t *b, const pim_coord_t *c,
                               size_t off, void *dst, size_t n);
/* The same for n elements from element index first, converting fp32 to
 * and from NUM_FMT. */
pim_status_t pim_write_bank_row_f32(const pim_buf_t *b, const pim_coord_t *c,
                                    size_t first, const float *src, size_t n);
pim_status_t pim_read_bank_row_f32(const pim_buf_t *b, const pim_coord_t *c,
                                   size_t first, float *dst, size_t n);

/*
 * Coherence: normal accesses to the PIM memory are cacheable, PIM
 * requests are not, so software flushes the lines a PIM command touches
 * (Zicbom). The flushes cover every cache line of the rows a command
 * touches, once, and add no fence:
 * - pim_flush_rows: slices [first, first + n) of the buffer, the rows of
 *   all-bank commands in every unit;
 * - pim_flush_unit_row: one slice in one unit (an all-bank command);
 * - pim_flush_bank_row: one bank row (a single-bank command; flush the
 *   pair bank 2k + 1 too for element-wise pairs).
 */
pim_status_t pim_flush_rows(const pim_buf_t *b, uint32_t first, uint32_t n);
pim_status_t pim_flush_unit_row(const pim_buf_t *b, uint32_t ch, uint32_t unit,
                                uint32_t slice);
pim_status_t pim_flush_bank_row(const pim_buf_t *b, const pim_coord_t *c);
/* Before PIM commands: flush every row of the buffer, then fence. */
pim_status_t pim_sync_to_device(const pim_buf_t *b);
/* After PIM commands, before normal loads: fence, flush, fence. */
pim_status_t pim_sync_from_device(const pim_buf_t *b);

#ifdef __cplusplus
}
#endif

#endif // __GEM5_PIM_PIM_RUNTIME_H__
