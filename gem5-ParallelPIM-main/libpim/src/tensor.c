/*
 * Tensors: shapes, layout kinds, pairs (element i of both tensors in the
 * same row and column of the two banks of a bank pair), and copying to
 * and from the host.
 */

#include <stdlib.h>
#include <string.h>

#include "pim_tensor.h"

static size_t
elem_bytes(pim_dtype_t d)
{
    switch (d) {
        case PIM_INT8:
            return 1;
        case PIM_BF16:
        case PIM_FP16:
        case PIM_INT16:
            return 2;
        default:
            return 4;
    }
}

void
pim__elem_place(const pim_tensor_t *t, size_t i, pim_coord_t *c, size_t *off)
{
    const size_t e = t->per_row;
    size_t slice;
    uint32_t u, k;
    if (t->kind == PIM_MATRIX_AB) {
        const size_t m = i / t->cols, j = i % t->cols;
        const size_t ub = (size_t)PIM_T_UNITS * PIM_T_BANKS;
        u = (uint32_t)(m % ub / PIM_T_BANKS);
        k = (uint32_t)(m % ub % PIM_T_BANKS);
        slice = m / ub * t->tiles + j / e;
        *off = (j % e) * t->esize;
    } else if (t->kind == PIM_VECTOR_PAIRED) {
        const size_t q = i / e;
        const size_t pairs = (size_t)PIM_T_UNITS * PIM_T_BANKS / 2;
        const size_t w = q % pairs;
        slice = q / pairs;
        u = (uint32_t)(w / (PIM_T_BANKS / 2));
        k = (uint32_t)(2 * (w % (PIM_T_BANKS / 2)) + t->side);
        *off = (i % e) * t->esize;
    } else { /* PIM_VECTOR_SB */
        slice = i / e;
        u = 0;
        k = (uint32_t)t->side;
        *off = (i % e) * t->esize;
    }
    pim__unit_bank(u, k, c);
    c->row = (uint32_t)slice;
    c->col = (uint32_t)(*off / PIM_LAYOUT_COL_BYTES);
}

static pim_status_t
make(pim_tensor_t **out, pim_kind_t kind, size_t rows, size_t cols,
     pim_dtype_t dtype, int side, pim_slices_t *shared)
{
    if (!pim__g.initialized) {
        return pim__fail(PIM_E_STATE, "tensor before pim_init()");
    }
    if (!out || rows == 0 || cols == 0 || kind > PIM_VECTOR_SB ||
        dtype > PIM_INT32 || (kind != PIM_MATRIX_AB && rows != 1)) {
        return pim__fail(PIM_E_ARG, "tensor kind %d, %zu x %zu", (int)kind,
                         rows, cols);
    }
    if (kind == PIM_VECTOR_PAIRED && PIM_T_BANKS < 2) {
        return pim__fail(PIM_E_ARG, "no bank pairs in this layout");
    }
    pim_tensor_t *t = (pim_tensor_t *)calloc(1, sizeof(*t));
    if (!t) {
        return pim__fail(PIM_E_NOMEM, "no host memory for a tensor");
    }
    t->kind = kind;
    t->dtype = dtype;
    t->rows = rows;
    t->cols = cols;
    t->esize = elem_bytes(dtype);
    t->per_row = PIM_LAYOUT_ROW_BYTES / t->esize;
    t->side = side;
    const size_t n = rows * cols;
    size_t slices = 0;
    if (kind == PIM_MATRIX_AB) {
        const size_t ub = (size_t)PIM_T_UNITS * PIM_T_BANKS;
        t->tiles = (cols + t->per_row - 1) / t->per_row;
        t->groups = (rows + ub - 1) / ub;
        slices = t->tiles * t->groups;
    } else if (kind == PIM_VECTOR_PAIRED) {
        const size_t per_slice =
            (size_t)PIM_T_UNITS * PIM_T_BANKS / 2 * t->per_row;
        slices = (n + per_slice - 1) / per_slice;
    } else if (kind == PIM_VECTOR_SB) {
        slices = (n + t->per_row - 1) / t->per_row;
    }
    if (kind == PIM_VECTOR_GB) {
        t->host = (uint8_t *)calloc(n, t->esize);
        if (!t->host) {
            free(t);
            return pim__fail(PIM_E_NOMEM, "no host memory for a vector");
        }
    } else if (shared) {
        t->slices = shared;
        shared->refs++;
    } else {
        t->slices = (pim_slices_t *)calloc(1, sizeof(pim_slices_t));
        const pim_status_t s =
            t->slices ? pim_buf_alloc(&t->slices->buf, (uint32_t)slices)
                      : PIM_E_NOMEM;
        if (s != PIM_OK) {
            free(t->slices);
            free(t);
            return s;
        }
        t->slices->refs = 1;
    }
    *out = t;
    return PIM_OK;
}

pim_status_t
pim_tensor_create(pim_tensor_t **t, pim_kind_t kind, size_t rows, size_t cols,
                  pim_dtype_t dtype)
{ return make(t, kind, rows, cols, dtype, 0, NULL); }

pim_status_t
pim_tensor_create_pair(pim_tensor_t **a, pim_tensor_t **b, pim_kind_t kind,
                       size_t rows, size_t cols, pim_dtype_t dtype)
{
    if (kind != PIM_VECTOR_PAIRED && kind != PIM_VECTOR_SB) {
        return pim__fail(PIM_E_ARG, "pairs are PIM_VECTOR_PAIRED or "
                                    "PIM_VECTOR_SB tensors");
    }
    if (!a || !b) {
        return pim__fail(PIM_E_ARG, "NULL tensor pointer");
    }
    pim_status_t s = make(a, kind, rows, cols, dtype, 0, NULL);
    if (s != PIM_OK) {
        return s;
    }
    s = make(b, kind, rows, cols, dtype, 1, (*a)->slices);
    if (s != PIM_OK) {
        pim_tensor_destroy(*a);
        *a = NULL;
        return s;
    }
    (*a)->partner = *b;
    (*b)->partner = *a;
    return PIM_OK;
}

void
pim_tensor_destroy(pim_tensor_t *t)
{
    if (!t) {
        return;
    }
    if (t->partner) {
        t->partner->partner = NULL;
    }
    if (t->slices && --t->slices->refs == 0) {
        pim_buf_free(t->slices->buf);
        free(t->slices);
    }
    free(t->host);
    free(t);
}

const pim_buf_t *
pim_tensor_buf(const pim_tensor_t *t)
{ return t && t->slices ? t->slices->buf : NULL; }

pim_status_t
pim_tensor_locate(const pim_tensor_t *t, size_t row, size_t col,
                  pim_coord_t *c, uint32_t *byte)
{
    if (!t || !t->slices || row >= t->rows || col >= t->cols || !c) {
        return pim__fail(PIM_E_ARG, "no element (%zu, %zu) in PIM memory", row,
                         col);
    }
    size_t off;
    pim__elem_place(t, row * t->cols + col, c, &off);
    c->row = pim_buf_row(t->slices->buf, c->row);
    if (byte) {
        *byte = (uint32_t)(off % PIM_LAYOUT_COL_BYTES);
    }
    return PIM_OK;
}

pim_status_t
pim__before_pim(pim_tensor_t *t, int pim_writes)
{
    if (t->slices && (t->host_dirty || (pim_writes && t->host_cached))) {
        const pim_status_t s = pim_sync_to_device(t->slices->buf);
        if (s != PIM_OK) {
            return s;
        }
        /* The flush covers the whole buffer, so both tensors of a pair. */
        t->host_dirty = t->host_cached = 0;
        if (t->partner) {
            t->partner->host_dirty = t->partner->host_cached = 0;
        }
    }
    return PIM_OK;
}

/* One element of the tensor's type, NUM_FMT or fp32, from and to fp32. */
static void
store_elem(const pim_tensor_t *t, uint8_t *dst, float f)
{
    if (t->dtype == (pim_dtype_t)PIM_LAYOUT_NUM_FMT) {
        pim_f32_to_elem(dst, f);
    } else {
        memcpy(dst, &f, 4);
    }
}

static float
load_elem(const pim_tensor_t *t, const uint8_t *src)
{
    if (t->dtype == (pim_dtype_t)PIM_LAYOUT_NUM_FMT) {
        return pim_elem_to_f32(src);
    }
    float f;
    memcpy(&f, src, 4);
    return f;
}

static int
dtype_ok(const pim_tensor_t *t)
{ return t->dtype == (pim_dtype_t)PIM_LAYOUT_NUM_FMT || t->dtype == PIM_FP32; }

pim_status_t
pim_copy_to(pim_tensor_t *t, const float *host)
{
    if (!pim__g.initialized) {
        return pim__fail(PIM_E_STATE, "copy before pim_init()");
    }
    if (!t || !host || !dtype_ok(t)) {
        return pim__fail(PIM_E_ARG, "copy of a tensor of type %d",
                         t ? (int)t->dtype : -1);
    }
    const size_t n = t->rows * t->cols;
    if (t->kind == PIM_VECTOR_GB) {
        for (size_t i = 0; i < n; i++) {
            store_elem(t, t->host + i * t->esize, host[i]);
        }
        return PIM_OK;
    }
    /* Runs of consecutive elements that share a bank row. */
    uint8_t buf[512];
    size_t i = 0;
    while (i < n) {
        pim_coord_t c;
        size_t off;
        pim__elem_place(t, i, &c, &off);
        size_t run = t->per_row - off / t->esize;
        if (t->kind == PIM_MATRIX_AB) {
            const size_t left_in_row = t->cols - i % t->cols;
            run = run < left_in_row ? run : left_in_row;
        }
        run = run < n - i ? run : n - i;
        run = run < sizeof(buf) / t->esize ? run : sizeof(buf) / t->esize;
        for (size_t k = 0; k < run; k++) {
            store_elem(t, buf + k * t->esize, host[i + k]);
        }
        const pim_status_t s =
            pim_write_bank_row(t->slices->buf, &c, off, buf, run * t->esize);
        if (s != PIM_OK) {
            return s;
        }
        i += run;
    }
    /* Flush now, so that PIM commands see the data. */
    t->host_dirty = 1;
    t->pim_written = 0;
    return pim__before_pim(t, 0);
}

pim_status_t
pim_copy_from(const pim_tensor_t *ct, float *host)
{
    pim_tensor_t *t = (pim_tensor_t *)ct;
    if (!pim__g.initialized) {
        return pim__fail(PIM_E_STATE, "copy before pim_init()");
    }
    if (!t || !host || !dtype_ok(t)) {
        return pim__fail(PIM_E_ARG, "copy of a tensor of type %d",
                         t ? (int)t->dtype : -1);
    }
    const size_t n = t->rows * t->cols;
    if (t->kind == PIM_VECTOR_GB) {
        for (size_t i = 0; i < n; i++) {
            host[i] = load_elem(t, t->host + i * t->esize);
        }
        return PIM_OK;
    }
    if (t->pim_written) {
        /* PIM wrote the rows: drop stale cached copies. */
        const pim_status_t s = pim_sync_from_device(t->slices->buf);
        if (s != PIM_OK) {
            return s;
        }
        t->pim_written = 0;
    }
    uint8_t buf[512];
    size_t i = 0;
    while (i < n) {
        pim_coord_t c;
        size_t off;
        pim__elem_place(t, i, &c, &off);
        size_t run = t->per_row - off / t->esize;
        if (t->kind == PIM_MATRIX_AB) {
            const size_t left_in_row = t->cols - i % t->cols;
            run = run < left_in_row ? run : left_in_row;
        }
        run = run < n - i ? run : n - i;
        run = run < sizeof(buf) / t->esize ? run : sizeof(buf) / t->esize;
        const pim_status_t s =
            pim_read_bank_row(t->slices->buf, &c, off, buf, run * t->esize);
        if (s != PIM_OK) {
            return s;
        }
        for (size_t k = 0; k < run; k++) {
            host[i + k] = load_elem(t, buf + k * t->esize);
        }
        i += run;
    }
    /* Normal loads leave clean copies, flushed before PIM writes the rows
     * again. */
    t->host_cached = 1;
    return PIM_OK;
}
