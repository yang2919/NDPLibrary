/*
 * Coherence: normal accesses to the PIM memory are cacheable and PIM
 * requests are not, so software flushes (cbo.flush) every cache line of
 * every row a command touches, and fences around PIM commands.
 *
 * A row slice is one range when the policy keeps it contiguous
 * (pim_geom_slice_contiguous()); otherwise it is flushed piece by piece.
 * A piece smaller than a cache line shares its line with other columns,
 * possibly of other banks or units (MOP4CLXOR XORs column bits into
 * them): the line is flushed by the lowest of its columns that belongs to
 * the rows being flushed, so every line is flushed once.
 */

#include "pim_internal.h"

/* The rows being flushed: row r of every bank, of one unit, or of one
 * bank. */
typedef struct flush_set
{
    uint32_t row;
    int ch, unit;            /* -1: every channel and unit */
    const pim_coord_t *bank; /* NULL: every bank of those units */
} flush_set_t;

static int
in_set(const flush_set_t *f, const pim_coord_t *d)
{
    if (d->row != f->row) {
        return 0;
    }
    if (f->bank) {
        return d->ch == f->bank->ch && d->unit == f->bank->unit &&
               d->sid == f->bank->sid && d->bg == f->bank->bg &&
               d->bank == f->bank->bank;
    }
    return f->ch < 0 ||
           (d->ch == (uint32_t)f->ch && d->unit == (uint32_t)f->unit);
}

/* Flush the lines of one contiguous piece of the set, at offset off. */
static void
flush_piece(const flush_set_t *f, uint64_t off, size_t n)
{
    const uint8_t *p = pim__addr(off);
    if (n >= PIM_CACHE_LINE) {
        pim_flush_range(p, n);
        return;
    }
    const uint64_t line = off & ~(uint64_t)(PIM_CACHE_LINE - 1);
    for (uint64_t o = line; o < off; o += PIM_LAYOUT_COL_BYTES) {
        pim_coord_t d;
        pim_decode(o, &d);
        if (in_set(f, &d)) {
            return; /* a lower column of the set flushes this line */
        }
    }
    pim_cbo_flush(p);
}

/* Flush the bank row of c (absolute row in c->row). */
static void
flush_abs_bank_row(const flush_set_t *f, const pim_coord_t *c)
{
    const pim_geom_t *g = pim_geom();
    for (uint32_t i = 0; i < pim_geom_pieces_per_row(g); i++) {
        flush_piece(f, pim_geom_piece_offset(g, c, i),
                    PIM_LAYOUT_CONTIG_BYTES);
    }
}

/* Flush the rows of a set. */
static void
flush_set(const flush_set_t *f)
{
    const pim_geom_t *g = pim_geom();
    if (f->bank) {
        flush_abs_bank_row(f, f->bank);
        return;
    }
    if (f->ch < 0 && pim_geom_slice_contiguous(g)) {
        /* One slice is one range of one mapping (Way 2: inside a huge
         * page). */
        pim_flush_range(pim__addr(pim_geom_slice_offset(g, f->row)),
                        pim_geom_slice_bytes(g));
        return;
    }
    pim_coord_t c = {0, 0, 0, 0, 0, f->row, 0};
    const uint32_t ch0 = f->ch < 0 ? 0 : (uint32_t)f->ch;
    const uint32_t ch1 = f->ch < 0 ? g->channels : ch0 + 1;
    const uint32_t u0 = f->ch < 0 ? 0 : (uint32_t)f->unit;
    const uint32_t u1 = f->ch < 0 ? g->units : u0 + 1;
    for (c.ch = ch0; c.ch < ch1; c.ch++) {
        for (c.unit = u0; c.unit < u1; c.unit++) {
            for (c.sid = 0; c.sid < g->sids; c.sid++) {
                for (c.bg = 0; c.bg < g->bank_groups; c.bg++) {
                    for (c.bank = 0; c.bank < g->banks_per_group; c.bank++) {
                        flush_abs_bank_row(f, &c);
                    }
                }
            }
        }
    }
}

static pim_status_t
check_buf(const pim_buf_t *b, uint32_t first, uint32_t n)
{
    if (!pim__g.initialized) {
        return pim__fail(PIM_E_STATE, "flush before pim_init()");
    }
    if (!b || first + n > pim_buf_slices(b) || first + n < first) {
        return pim__fail(PIM_E_ARG, "slices %u + %u outside the buffer", first,
                         n);
    }
    return PIM_OK;
}

pim_status_t
pim_flush_rows(const pim_buf_t *b, uint32_t first, uint32_t n)
{
    const pim_status_t s = check_buf(b, first, n);
    if (s != PIM_OK) {
        return s;
    }
    for (uint32_t i = first; i < first + n; i++) {
        const flush_set_t f = {pim_buf_row(b, i), -1, -1, NULL};
        flush_set(&f);
    }
    return PIM_OK;
}

pim_status_t
pim_flush_unit_row(const pim_buf_t *b, uint32_t ch, uint32_t unit,
                   uint32_t slice)
{
    const pim_status_t s = check_buf(b, slice, 1);
    if (s != PIM_OK) {
        return s;
    }
    if (ch >= PIM_LAYOUT_CHANNELS || unit >= PIM_LAYOUT_UNITS) {
        return pim__fail(PIM_E_ARG, "unit %u of channel %u", unit, ch);
    }
    const flush_set_t f = {pim_buf_row(b, slice), (int)ch, (int)unit, NULL};
    flush_set(&f);
    return PIM_OK;
}

pim_status_t
pim_flush_bank_row(const pim_buf_t *b, const pim_coord_t *c)
{
    const pim_status_t s = check_buf(b, c ? c->row : 0, 1);
    if (s != PIM_OK) {
        return s;
    }
    pim_coord_t a = *c;
    a.row = pim_buf_row(b, c->row);
    const flush_set_t f = {a.row, -1, -1, &a};
    flush_set(&f);
    return PIM_OK;
}

pim_status_t
pim_sync_to_device(const pim_buf_t *b)
{
    const pim_status_t s = pim_flush_rows(b, 0, pim_buf_slices(b));
    pim_fence();
    return s;
}

pim_status_t
pim_sync_from_device(const pim_buf_t *b)
{
    pim_fence();
    const pim_status_t s = pim_flush_rows(b, 0, pim_buf_slices(b));
    pim_fence();
    return s;
}
