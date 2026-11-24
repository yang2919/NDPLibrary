/*
 * Copy in and out: normal stores and
 * loads at the encoder's addresses, one address-contiguous piece at a
 * time, so the DRAM content is the same under every policy.
 */

#include <string.h>

#include "pim_internal.h"

void
pim_copy_piece(void *pim, const void *host, size_t n)
{ memcpy(pim, host, n); }

void
pim_copy_piece_out(void *host, const void *pim, size_t n)
{ memcpy(host, pim, n); }

/* The address of byte off of the bank row of c, and how many bytes
 * from there stay inside its contiguous piece. */
static uint8_t *
piece_at(const pim_buf_t *b, const pim_coord_t *c, size_t off, size_t *left)
{
    pim_coord_t p = *c;
    p.col = (uint32_t)(off / PIM_LAYOUT_COL_BYTES);
    uint8_t *a = (uint8_t *)pim_buf_addr(b, &p);
    *left = PIM_LAYOUT_CONTIG_BYTES - off % PIM_LAYOUT_CONTIG_BYTES;
    return a ? a + off % PIM_LAYOUT_COL_BYTES : NULL;
}

static pim_status_t
check_range(const pim_buf_t *b, const pim_coord_t *c, size_t off, size_t n)
{
    if (!pim__g.initialized) {
        return pim__fail(PIM_E_STATE, "copy before pim_init()");
    }
    if (!b || !c || c->row >= pim_buf_slices(b) ||
        off + n > PIM_LAYOUT_ROW_BYTES || c->ch >= PIM_LAYOUT_CHANNELS ||
        c->unit >= PIM_LAYOUT_UNITS || c->bank >= PIM_LAYOUT_BANKS_PER_GROUP ||
        c->bg >= (PIM_LAYOUT_BANK_GROUPS ? PIM_LAYOUT_BANK_GROUPS : 1) ||
        c->sid >= (PIM_LAYOUT_SIDS ? PIM_LAYOUT_SIDS : 1)) {
        return pim__fail(PIM_E_ARG,
                         "bytes %zu + %zu of a bank row outside "
                         "the buffer or the row",
                         off, n);
    }
    return PIM_OK;
}

pim_status_t
pim_write_bank_row(const pim_buf_t *b, const pim_coord_t *c, size_t off,
                   const void *src, size_t n)
{
    const pim_status_t s = check_range(b, c, off, n);
    if (s != PIM_OK) {
        return s;
    }
    const uint8_t *from = (const uint8_t *)src;
    while (n) {
        size_t left;
        uint8_t *to = piece_at(b, c, off, &left);
        const size_t k = n < left ? n : left;
        pim_copy_piece(to, from, k);
        from += k;
        off += k;
        n -= k;
    }
    return PIM_OK;
}

pim_status_t
pim_read_bank_row(const pim_buf_t *b, const pim_coord_t *c, size_t off,
                  void *dst, size_t n)
{
    const pim_status_t s = check_range(b, c, off, n);
    if (s != PIM_OK) {
        return s;
    }
    uint8_t *to = (uint8_t *)dst;
    while (n) {
        size_t left;
        const uint8_t *from = piece_at(b, c, off, &left);
        const size_t k = n < left ? n : left;
        pim_copy_piece_out(to, from, k);
        to += k;
        off += k;
        n -= k;
    }
    return PIM_OK;
}

pim_status_t
pim_write_bank_row_f32(const pim_buf_t *b, const pim_coord_t *c, size_t first,
                       const float *src, size_t n)
{
    const size_t e = PIM_LAYOUT_ELEM_BYTES;
    pim_status_t s = check_range(b, c, first * e, n * e);
    if (s != PIM_OK) {
        return s;
    }
    uint8_t buf[256];
    const size_t per = sizeof(buf) / e;
    while (n) {
        const size_t k = n < per ? n : per;
        for (size_t i = 0; i < k; i++) {
            pim_f32_to_elem(buf + i * e, src[i]);
        }
        s = pim_write_bank_row(b, c, first * e, buf, k * e);
        if (s != PIM_OK) {
            return s;
        }
        src += k;
        first += k;
        n -= k;
    }
    return PIM_OK;
}

pim_status_t
pim_read_bank_row_f32(const pim_buf_t *b, const pim_coord_t *c, size_t first,
                      float *dst, size_t n)
{
    const size_t e = PIM_LAYOUT_ELEM_BYTES;
    pim_status_t s = check_range(b, c, first * e, n * e);
    if (s != PIM_OK) {
        return s;
    }
    uint8_t buf[256];
    const size_t per = sizeof(buf) / e;
    while (n) {
        const size_t k = n < per ? n : per;
        s = pim_read_bank_row(b, c, first * e, buf, k * e);
        if (s != PIM_OK) {
            return s;
        }
        for (size_t i = 0; i < k; i++) {
            dst[i] = pim_elem_to_f32(buf + i * e);
        }
        dst += k;
        first += k;
        n -= k;
    }
    return PIM_OK;
}
