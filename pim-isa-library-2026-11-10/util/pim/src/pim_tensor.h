/*
 * Internal tensor state; private to util/pim.
 */

#ifndef __PIM_TENSOR_H__
#define __PIM_TENSOR_H__

#include <gem5/pim/pim.h>

#include "pim_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Row slices shared by the two tensors of a pair. */
typedef struct pim_slices
{
    pim_buf_t *buf;
    int refs;
} pim_slices_t;

struct pim_tensor
{
    pim_kind_t kind;
    pim_dtype_t dtype;
    size_t rows, cols;
    size_t esize;         /* bytes per element */
    size_t per_row;       /* elements per bank row */
    size_t tiles, groups; /* PIM_MATRIX_AB: column tiles, row groups */
    pim_slices_t *slices; /* NULL for PIM_VECTOR_GB */
    uint8_t *host;        /* PIM_VECTOR_GB: the elements */
    int side;             /* the bank of each pair: 0 even, 1 odd */
    pim_tensor_t *partner;
    int host_dirty;  /* the host wrote rows without flushing them */
    int pim_written; /* PIM commands wrote rows the host may cache */
};

/* Units, banks per unit, and pairs per row slice. */
#define PIM_T_UNITS (PIM_LAYOUT_CHANNELS * PIM_LAYOUT_UNITS)
#define PIM_T_BANKS PIM_LAYOUT_BANKS_PER_UNIT

/* (ch, unit, sid, bg, bank) of bank k of unit u; row and column 0. */
static inline void
pim__unit_bank(uint32_t u, uint32_t k, pim_coord_t *c)
{
    c->row = 0;
    c->col = 0;
    c->ch = u / PIM_LAYOUT_UNITS;
    c->unit = u % PIM_LAYOUT_UNITS;
    c->bank = k % PIM_LAYOUT_BANKS_PER_GROUP;
    k /= PIM_LAYOUT_BANKS_PER_GROUP;
    const uint32_t bgs = PIM_LAYOUT_BANK_GROUPS ? PIM_LAYOUT_BANK_GROUPS : 1;
    c->bg = k % bgs;
    c->sid = k / bgs;
}
/* Element i (row-major) of t: its bank coordinates (c->row is the slice
 * index in t's buffer) and its byte offset in the bank row. */
void pim__elem_place(const pim_tensor_t *t, size_t i, pim_coord_t *c,
                     size_t *off);
/* Flush a tensor's rows before PIM commands read or write them. */
pim_status_t pim__before_pim(pim_tensor_t *t);

#ifdef __cplusplus
}
#endif

#endif // __PIM_TENSOR_H__
