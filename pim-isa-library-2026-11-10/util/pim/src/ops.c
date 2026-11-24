/*
 * The operations of P-LIB section 7 (LQ7 (b)): GEMV with the all-bank
 * MAC, the GB and the registers; element-wise add and multiply with the
 * EW commands on paired banks and the register-to-row copy.
 *
 * Every vector command covers one address-contiguous piece at most
 * (PIM_LAYOUT_CONTIG_BYTES, inside one bank row); the wrappers cap each
 * piece at VLMAX and return the bytes covered.
 */

#include <string.h>

#include "pim_tensor.h"

/* The address of byte off of the bank row of c (c->row a slice index of
 * b), inline: the folded encoder plus the region base. */
static inline uint8_t *
row_addr(const pim_buf_t *b, const pim_coord_t *c, size_t off)
{
    pim_coord_t p = *c;
    p.row = b->rows[c->row];
    p.col = (uint32_t)(off / PIM_LAYOUT_COL_BYTES);
    return pim__g.base + pim_encode(&p) + off % PIM_LAYOUT_COL_BYTES;
}

/* The bytes from off that stay inside one contiguous piece. */
static inline size_t
piece_left(size_t off, size_t n)
{
    const size_t left =
        PIM_LAYOUT_CONTIG_BYTES - off % PIM_LAYOUT_CONTIG_BYTES;
    return left < n - off ? left : n - off;
}

/*
 * A vector command on bytes [0, n) of the rows at c (one bank, or the
 * unit for all-bank commands), piece by piece; a macro, so that each
 * wrapper is inlined (LQ3 (a)).
 */
#define EACH_PIECE(cmd, b, c, n)                                              \
    do {                                                                      \
        size_t off_ = 0;                                                      \
        while (off_ < (n)) {                                                  \
            off_ += cmd(row_addr((b), (c), off_), piece_left(off_, (n)));     \
        }                                                                     \
    } while (0)

/* The accumulator row of a bank (c->row, c->col ignored). */
static inline uint8_t *
acc_addr(const pim_coord_t *c)
{
    pim_coord_t p = *c;
    p.row = PIM_LAYOUT_REG_ROW0 + PIM_LAYOUT_NUM_REGS;
    p.col = 0;
    return pim__g.base + pim_encode(&p);
}

/* Host bytes [0, n) into the GB of one unit, piece by piece. */
static inline void
write_gb(uint32_t ch, uint32_t unit, const uint8_t *src, size_t n)
{
    size_t off = 0;
    while (off < n) {
        const pim_coord_t p = {ch,
                               unit,
                               0,
                               0,
                               0,
                               PIM_LAYOUT_REG_ROW0,
                               (uint32_t)(off / PIM_LAYOUT_COL_BYTES)};
        uint8_t *a = pim__g.base + pim_encode(&p) + off % PIM_LAYOUT_COL_BYTES;
        off += pim_wr_gb_mem(a, src + off, piece_left(off, n));
    }
}

pim_status_t
pim_gemv(pim_tensor_t *y, const pim_tensor_t *A, const pim_tensor_t *x)
{
    if (!pim__g.initialized) {
        return pim__fail(PIM_E_STATE, "pim_gemv() before pim_init()");
    }
    if (!y || !A || !x || A->kind != PIM_MATRIX_AB ||
        x->kind != PIM_VECTOR_GB || y->kind != PIM_VECTOR_SB ||
        x->cols != A->cols || y->cols != A->rows ||
        A->dtype != (pim_dtype_t)PIM_LAYOUT_NUM_FMT ||
        x->dtype != (pim_dtype_t)PIM_LAYOUT_NUM_FMT ||
        y->dtype != (pim_dtype_t)PIM_LAYOUT_ACC_FMT ||
        (PIM_LAYOUT_ACC_FMT != PIM_FMT_FP32 &&
         PIM_LAYOUT_ACC_FMT != PIM_LAYOUT_NUM_FMT)) {
        return pim__fail(PIM_E_ARG,
                         "pim_gemv(): y must be a 1 x %zu "
                         "PIM_VECTOR_SB of the accumulator type, A a "
                         "PIM_MATRIX_AB and x a 1 x %zu PIM_VECTOR_GB of "
                         "the operand type",
                         A ? A->rows : 0, A ? A->cols : 0);
    }
    pim_status_t s = pim__before_pim((pim_tensor_t *)A);
    if (s != PIM_OK) {
        return s;
    }
    const pim_buf_t *ab = A->slices->buf;
    const size_t ub = (size_t)PIM_T_UNITS * PIM_T_BANKS;
    const uint8_t zero[8] = {0};
    for (size_t g = 0; g < A->groups; g++) {
        /* Reset every accumulator of every unit (D9.5). */
        for (uint32_t u = 0; u < PIM_T_UNITS; u++) {
            for (uint32_t k = 0; k < PIM_T_BANKS; k++) {
                pim_coord_t c;
                pim__unit_bank(u, k, &c);
                pim_wr_reg_mem(acc_addr(&c), zero, PIM_LAYOUT_ACC_BYTES);
            }
        }
        for (size_t t = 0; t < A->tiles; t++) {
            const size_t first = t * A->per_row;
            const size_t ncols =
                A->cols - first < A->per_row ? A->cols - first : A->per_row;
            const size_t bytes = ncols * A->esize;
            for (uint32_t u = 0; u < PIM_T_UNITS; u++) {
                pim_coord_t c;
                pim__unit_bank(u, 0, &c);
                write_gb(c.ch, c.unit, x->host + first * x->esize, bytes);
                c.row = (uint32_t)(g * A->tiles + t);
                EACH_PIECE(pim_ab_mac, ab, &c, bytes);
            }
        }
        /* Gather the accumulators: matrix row m = g U B + u B + k. */
        for (uint32_t u = 0; u < PIM_T_UNITS; u++) {
            for (uint32_t k = 0; k < PIM_T_BANKS; k++) {
                const size_t m = g * ub + (size_t)u * PIM_T_BANKS + k;
                if (m >= A->rows) {
                    continue;
                }
                pim_coord_t c;
                pim__unit_bank(u, k, &c);
                uint8_t acc[8];
                pim_rd_reg_mem(acc_addr(&c), acc, PIM_LAYOUT_ACC_BYTES);
                pim_coord_t yc;
                size_t off;
                pim__elem_place(y, m, &yc, &off);
                s = pim_write_bank_row(y->slices->buf, &yc, off, acc,
                                       PIM_LAYOUT_ACC_BYTES);
                if (s != PIM_OK) {
                    return s;
                }
            }
        }
    }
    y->host_dirty = 1; /* normal stores, cached */
    y->pim_written = 0;
    return PIM_OK;
}

/* c = a (+ or *) b; see pim.h. */
static pim_status_t
eltwise(int mul, pim_tensor_t *c, const pim_tensor_t *a, const pim_tensor_t *b)
{
    if (!pim__g.initialized) {
        return pim__fail(PIM_E_STATE, "pim_add() before pim_init()");
    }
    if (!c || !a || !b || a->partner != b || a->side != 0 ||
        (a->kind != PIM_VECTOR_PAIRED && a->kind != PIM_VECTOR_SB) ||
        c->kind != a->kind || c->partner || c->cols != a->cols ||
        c->dtype != a->dtype || a->dtype != (pim_dtype_t)PIM_LAYOUT_NUM_FMT) {
        return pim__fail(PIM_E_ARG,
                         "pim_add()/pim_mul(): a and b must be the first and "
                         "second tensor of one pim_tensor_create_pair() of "
                         "the operand type, c a tensor of the same kind, "
                         "shape and type that is not part of a pair");
    }
    if (PIM_LAYOUT_NUM_REGS != 1) {
        return pim__fail(PIM_E_ARG, "element-wise commands need one register "
                                    "per PU");
    }
    pim_status_t s = pim__before_pim((pim_tensor_t *)a);
    if (s == PIM_OK) {
        s = pim__before_pim(c);
    }
    if (s != PIM_OK) {
        return s;
    }
    const pim_buf_t *src = a->slices->buf;
    const pim_buf_t *dst = c->slices->buf;
    const size_t n = a->cols;
    const size_t row = a->per_row * a->esize;
    const uint32_t nslices = pim_buf_slices(src);
    if (a->kind == PIM_VECTOR_PAIRED) {
        /* Every unit's pairs at once: pim.ab.ew*, then the registers into
         * c's rows (the even banks hold the results). */
        const size_t pairs = (size_t)PIM_T_UNITS * PIM_T_BANKS / 2;
        const size_t per_slice = pairs * a->per_row;
        for (uint32_t slice = 0; slice < nslices; slice++) {
            const size_t left = n - slice * per_slice;
            for (uint32_t u = 0; u < PIM_T_UNITS; u++) {
                const size_t first =
                    (size_t)u * (PIM_T_BANKS / 2) * a->per_row;
                if (first >= left) {
                    break;
                }
                pim_coord_t c0;
                pim__unit_bank(u, 0, &c0);
                c0.row = slice;
                if (mul) {
                    EACH_PIECE(pim_ab_ewmul, src, &c0, row);
                } else {
                    EACH_PIECE(pim_ab_ewadd, src, &c0, row);
                }
                EACH_PIECE(pim_ab_mov_regbk, dst, &c0, row);
            }
        }
    } else {
        /* One bank pair: pim.sb.ew* over the used columns of each row,
         * then the register of bank 0's PU into c's row. */
        for (uint32_t slice = 0; slice < nslices; slice++) {
            const size_t elems = n - slice * a->per_row < a->per_row
                                     ? n - slice * a->per_row
                                     : a->per_row;
            const size_t bytes = elems * a->esize;
            const uint32_t ncols =
                (uint32_t)((bytes + PIM_LAYOUT_COL_BYTES - 1) /
                           PIM_LAYOUT_COL_BYTES);
            pim_coord_t c0;
            pim__unit_bank(0, 0, &c0);
            c0.row = slice;
            c0.col = 0;
            void *addr = row_addr(src, &c0, 0);
            if (mul) {
                pim_sb_ewmul(addr, PIM_DESC(ncols));
            } else {
                pim_sb_ewadd(addr, PIM_DESC(ncols));
            }
            const size_t cbytes = (size_t)ncols * PIM_LAYOUT_COL_BYTES;
            EACH_PIECE(pim_ab_mov_regbk, dst, &c0, cbytes);
        }
    }
    c->pim_written = 1;
    c->host_dirty = 0;
    return PIM_OK;
}

pim_status_t
pim_add(pim_tensor_t *c, const pim_tensor_t *a, const pim_tensor_t *b)
{ return eltwise(0, c, a, b); }

pim_status_t
pim_mul(pim_tensor_t *c, const pim_tensor_t *a, const pim_tensor_t *b)
{ return eltwise(1, c, a, b); }
