/*
 * Buffers, the addresses of buffer coordinates, of the register window
 * and of the global buffers, and the PIM_DEBUG range check of the
 * wrappers.
 */

#include <stdio.h>
#include <stdlib.h>

#include "pim_internal.h"

pim_status_t
pim_buf_alloc(pim_buf_t **b, uint32_t n_slices)
{
    if (!pim__g.initialized) {
        return pim__fail(PIM_E_STATE, "pim_buf_alloc() before pim_init()");
    }
    if (!b || n_slices == 0) {
        return pim__fail(PIM_E_ARG, "pim_buf_alloc() of %u slices", n_slices);
    }
    pim_buf_t *nb =
        (pim_buf_t *)malloc(sizeof(pim_buf_t) + n_slices * sizeof(uint32_t));
    if (!nb) {
        return pim__fail(PIM_E_NOMEM, "no host memory for a buffer");
    }
    /* Way 2: map huge pages on the PIM node until enough slices are
     * free. */
    if (pim__g.way == 2 && pim__rows_free_count() < n_slices) {
        const pim_status_t g =
            pim__way2_grow(n_slices - pim__rows_free_count());
        if (g != PIM_OK) {
            free(nb);
            return g;
        }
    }
    const pim_status_t s = pim__rows_alloc(n_slices, nb->rows);
    if (s != PIM_OK) {
        free(nb);
        return s;
    }
    nb->n = n_slices;
    nb->session = pim__g.session;
    *b = nb;
    return PIM_OK;
}

void
pim_buf_free(pim_buf_t *b)
{
    if (!b) {
        return;
    }
    /* A buffer of an earlier pim_init() session owns no rows of this
     * one: only its host memory is freed. */
    if (pim__g.initialized && b->session == pim__g.session) {
        pim__rows_free(b->rows, b->n);
    }
    free(b);
}

uint32_t
pim_buf_slices(const pim_buf_t *b)
{ return b ? b->n : 0; }

uint32_t
pim_buf_row(const pim_buf_t *b, uint32_t slice)
{ return b && slice < b->n ? b->rows[slice] : UINT32_MAX; }

uint32_t
pim_free_slices(void)
{ return pim__g.initialized ? pim__rows_free_count() : 0; }

void *
pim_buf_addr(const pim_buf_t *b, const pim_coord_t *c)
{
    if (!pim__g.initialized || !b || c->row >= b->n) {
#ifdef PIM_DEBUG
        fprintf(stderr, "pim_buf_addr: slice %u outside the buffer\n", c->row);
        abort();
#endif
        return NULL;
    }
    pim_coord_t p = *c;
    p.row = b->rows[c->row];
    return pim__addr(pim_encode(&p));
}

void *
pim_reg_addr(const pim_coord_t *c, uint32_t j)
{
    if (!pim__g.initialized || j >= PIM_LAYOUT_NUM_REGS) {
#ifdef PIM_DEBUG
        fprintf(stderr, "pim_reg_addr: register %u of %u\n", j,
                PIM_LAYOUT_NUM_REGS);
        abort();
#endif
        return NULL;
    }
    pim_coord_t p = *c;
    p.row = PIM_LAYOUT_REG_ROW0 + j;
    return pim__addr(pim_encode(&p));
}

void *
pim_acc_addr(const pim_coord_t *c)
{
    if (!pim__g.initialized) {
        return NULL;
    }
    pim_coord_t p = *c;
    p.row = PIM_LAYOUT_REG_ROW0 + PIM_LAYOUT_NUM_REGS;
    return pim__addr(pim_encode(&p));
}

void *
pim_gb_addr(uint32_t ch, uint32_t unit, uint32_t col)
{
    if (!pim__g.initialized) {
        return NULL;
    }
    const pim_coord_t p = {ch, unit, 0, 0, 0, PIM_LAYOUT_REG_ROW0, col};
    return pim__addr(pim_encode(&p));
}

/*
 * The PIM_DEBUG check of a wrapper (pim_isa.h): the address lies in the
 * PIM memory (one mapping of it, under Way 2); a bank operation
 * addresses a row of some buffer; a
 * register operation the register window; a vector range stays inside
 * one PIM_LAYOUT_CONTIG_BYTES piece. Aborts with a message.
 */
void
pim__debug_range(const char *name, const void *addr, size_t n, int kind)
{
    const uint8_t *a = (const uint8_t *)addr;
    uint64_t off, last;
    if (!pim__g.initialized || pim__offset(a, &off) ||
        (n && (pim__offset(a + n - 1, &last) || last != off + n - 1))) {
        fprintf(stderr, "%s: %p is outside the PIM memory\n", name, addr);
        abort();
    }
    pim_coord_t c;
    pim_decode(off, &c);
    if (kind == PIM_DEBUG_KIND_BANK && !pim__row_used(c.row)) {
        fprintf(stderr, "%s: %p (row %u) is outside every buffer\n", name,
                addr, c.row);
        abort();
    }
    if (kind == PIM_DEBUG_KIND_REG && c.row < PIM_LAYOUT_REG_ROW0) {
        fprintf(stderr, "%s: %p (row %u) is outside the register window\n",
                name, addr, c.row);
        abort();
    }
    if (n && off / PIM_LAYOUT_CONTIG_BYTES !=
                 (off + n - 1) / PIM_LAYOUT_CONTIG_BYTES) {
        fprintf(stderr,
                "%s: %p + %zu crosses a %d-byte contiguous piece "
                "(PIM_LAYOUT_CONTIG_BYTES)\n",
                name, addr, n, PIM_LAYOUT_CONTIG_BYTES);
        abort();
    }
}
