/*
 * L6: the user API (pim.h) on the host reference backend, for the layout
 * this program is built for (run for every policy).
 *
 * pim_gemv, pim_add and pim_mul against plain C loops: exactly against
 * the same rounding (operands rounded to bf16; GEMV sums each column in
 * fp32 as the PU does), and within bf16 tolerance against the
 * unrounded inputs. Also the layout of every kind (pim_tensor_locate),
 * the pair's bank designation (DP-F (b)), the number of PIM requests and
 * the errors.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <gem5/pim/pim.h>

#include "pim_test.h"
#include "ref/pim_ref.h"

static uint64_t rng = 0x2545f4914f6cdd1dULL;

static float
frand(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (float)((rng >> 11) % 2001) / 1000.0f - 1.0f; /* [-1, 1] */
}

static float
q(float f) /* the operand format's value of f */
{
    uint8_t b[4];
    pim_f32_to_elem(b, f);
    return pim_elem_to_f32(b);
}

/* The operand type of the layout. */
#define OPDT ((pim_dtype_t)PIM_LAYOUT_NUM_FMT)

#define UB                                                                    \
    ((size_t)PIM_LAYOUT_CHANNELS * PIM_LAYOUT_UNITS *                         \
     PIM_LAYOUT_BANKS_PER_UNIT)
#define PER_ROW ((size_t)PIM_LAYOUT_ROW_BYTES / PIM_LAYOUT_ELEM_BYTES)

static void
check_gemv(size_t rows, size_t cols)
{
    pim_tensor_t *A, *x, *y;
    CHECK(pim_tensor_create(&A, PIM_MATRIX_AB, rows, cols, OPDT) == PIM_OK,
          "A %zu x %zu: %s", rows, cols, pim_error_detail());
    CHECK(pim_tensor_create(&x, PIM_VECTOR_GB, 1, cols, OPDT) == PIM_OK, "x");
    CHECK(pim_tensor_create(&y, PIM_VECTOR_SB, 1, rows, PIM_FP32) == PIM_OK,
          "y");
    float *a = (float *)malloc(rows * cols * sizeof(float));
    float *xv = (float *)malloc(cols * sizeof(float));
    float *yv = (float *)malloc(rows * sizeof(float));
    for (size_t i = 0; i < rows * cols; i++) {
        a[i] = frand();
    }
    for (size_t j = 0; j < cols; j++) {
        xv[j] = frand();
    }
    CHECK(pim_copy_to(A, a) == PIM_OK, "copy A");
    CHECK(pim_copy_to(x, xv) == PIM_OK, "copy x");
    const uint64_t macs = pim_ref_count(PIM_OP_AB_MAC);
    CHECK(pim_gemv(y, A, x) == PIM_OK, "gemv: %s", pim_error_detail());
    CHECK(pim_copy_from(y, yv) == PIM_OK, "copy y");
    /* The PU's arithmetic: per column, products of bf16 operands summed
     * in fp32 in element order; then accumulated. */
    const size_t per_col = PIM_LAYOUT_COL_BYTES / PIM_LAYOUT_ELEM_BYTES;
    long exact = 0, close = 0;
    for (size_t m = 0; m < rows; m++) {
        float acc = 0.0f;
        double ref = 0, mag = 0;
        for (size_t j0 = 0; j0 < cols; j0 += per_col) {
            float part = 0.0f;
            for (size_t j = j0; j < j0 + per_col && j < cols; j++) {
                const float p = q(a[m * cols + j]) * q(xv[j]);
                part = j == j0 ? p : part + p;
                ref += (double)a[m * cols + j] * xv[j];
                mag += fabs((double)a[m * cols + j] * xv[j]);
            }
            acc += part;
        }
        exact += memcmp(&acc, &yv[m], 4) != 0;
        close += fabs(yv[m] - ref) > 1.0 / 128 * mag + 1e-6;
    }
    CHECK(exact == 0,
          "gemv %zu x %zu: %ld rows differ from the PU's "
          "arithmetic",
          rows, cols, exact);
    CHECK(close == 0, "gemv %zu x %zu: %ld rows outside bf16 tolerance", rows,
          cols, close);
    /* One pim.ab.mac per unit, tile and piece of the tile. */
    const size_t groups = (rows + UB - 1) / UB;
    uint64_t want = 0;
    for (size_t t = 0; t * PER_ROW < cols; t++) {
        const size_t bytes =
            (cols - t * PER_ROW < PER_ROW ? cols - t * PER_ROW : PER_ROW) *
            PIM_LAYOUT_ELEM_BYTES;
        const size_t vlmax = pim_ref_vlmax((size_t)-1);
        const size_t step =
            PIM_LAYOUT_CONTIG_BYTES < vlmax ? PIM_LAYOUT_CONTIG_BYTES : vlmax;
        want += (bytes + step - 1) / step;
    }
    want *= groups * PIM_LAYOUT_CHANNELS * PIM_LAYOUT_UNITS;
    CHECK(pim_ref_count(PIM_OP_AB_MAC) - macs == want,
          "gemv %zu x %zu: %llu pim.ab.mac, want %llu", rows, cols,
          (unsigned long long)(pim_ref_count(PIM_OP_AB_MAC) - macs),
          (unsigned long long)want);
    /* Matrix rows lie one per bank row, matrix row m in bank m % B. */
    pim_coord_t c;
    uint32_t byte;
    const size_t m = rows - 1, j = cols - 1;
    CHECK(pim_tensor_locate(A, m, j, &c, &byte) == PIM_OK, "locate A");
    const pim_coord_t *loc = &c;
    uint8_t want_bytes[4];
    pim_f32_to_elem(want_bytes, a[m * cols + j]);
    CHECK(memcmp((uint8_t *)pim_region_base() + pim_encode(loc) + byte,
                 want_bytes, PIM_LAYOUT_ELEM_BYTES) == 0,
          "A(%zu, %zu) not at its location", m, j);
    CHECK(pim_tensor_locate(x, 0, 0, &c, &byte) == PIM_E_ARG,
          "x is on the host");
    free(a);
    free(xv);
    free(yv);
    pim_tensor_destroy(A);
    pim_tensor_destroy(x);
    pim_tensor_destroy(y);
}

static void
check_eltwise(pim_kind_t kind, size_t n)
{
    pim_tensor_t *a, *b, *c, *d;
    CHECK(pim_tensor_create_pair(&a, &b, kind, 1, n, OPDT) == PIM_OK,
          "pair %d x %zu: %s", (int)kind, n, pim_error_detail());
    CHECK(pim_tensor_create(&c, kind, 1, n, OPDT) == PIM_OK, "c");
    CHECK(pim_tensor_create(&d, kind, 1, n, OPDT) == PIM_OK, "d");
    float *av = (float *)malloc(n * sizeof(float));
    float *bv = (float *)malloc(n * sizeof(float));
    float *cv = (float *)malloc(n * sizeof(float));
    float *dv = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        av[i] = frand();
        bv[i] = frand();
    }
    CHECK(pim_copy_to(a, av) == PIM_OK && pim_copy_to(b, bv) == PIM_OK,
          "copy a, b");
    /* DP-F (b): element i of a and b in the same row and column of the
     * two banks of one pair, a in the even bank, b in the odd one. */
    long misplaced = 0;
    for (size_t i = 0; i < n; i += 1 + n / 50) {
        pim_coord_t ca, cb;
        uint32_t ba, bb;
        pim_tensor_locate(a, 0, i, &ca, &ba);
        pim_tensor_locate(b, 0, i, &cb, &bb);
        misplaced += ca.bank % 2 != 0 || cb.bank != ca.bank + 1 ||
                     ca.ch != cb.ch || ca.unit != cb.unit ||
                     ca.sid != cb.sid || ca.bg != cb.bg || ca.row != cb.row ||
                     ca.col != cb.col || ba != bb;
    }
    CHECK(misplaced == 0, "%ld elements of the pair are not in paired banks",
          misplaced);
    const uint64_t ew = pim_ref_count(kind == PIM_VECTOR_SB ? PIM_OP_SB_EWADD
                                                            : PIM_OP_AB_EWADD);
    CHECK(pim_add(c, a, b) == PIM_OK, "add: %s", pim_error_detail());
    CHECK(pim_mul(d, a, b) == PIM_OK, "mul: %s", pim_error_detail());
    CHECK(pim_ref_count(kind == PIM_VECTOR_SB ? PIM_OP_SB_EWADD
                                              : PIM_OP_AB_EWADD) > ew,
          "element-wise commands issued");
    CHECK(pim_copy_from(c, cv) == PIM_OK && pim_copy_from(d, dv) == PIM_OK,
          "copy c, d");
    long exact = 0, close = 0;
    for (size_t i = 0; i < n; i++) {
        const float s = q(q(av[i]) + q(bv[i]));
        const float p = q(q(av[i]) * q(bv[i]));
        exact += memcmp(&s, &cv[i], 4) != 0 || memcmp(&p, &dv[i], 4) != 0;
        const double rs = (double)av[i] + bv[i], rp = (double)av[i] * bv[i];
        close += fabs(cv[i] - rs) > 3.0 / 256 * (fabs(av[i]) + fabs(bv[i])) ||
                 fabs(dv[i] - rp) > 3.0 / 256 * fabs(rp) + 1e-6;
    }
    CHECK(exact == 0, "kind %d, %zu elements: %ld results differ", (int)kind,
          n, exact);
    CHECK(close == 0, "kind %d, %zu elements: %ld outside bf16 tolerance",
          (int)kind, n, close);
    /* Misuse: operands not one pair, in the wrong order, c in a pair. */
    CHECK(pim_add(c, b, a) == PIM_E_ARG, "operands swapped");
    CHECK(pim_add(c, a, d) == PIM_E_ARG, "not a pair");
    CHECK(pim_add(b, a, b) == PIM_E_ARG, "c in the pair");
    /* A result used again: c is written by PIM, then read back. */
    CHECK(pim_add(c, a, b) == PIM_OK, "add twice");
    CHECK(pim_copy_from(c, cv) == PIM_OK, "copy c again");
    exact = 0;
    for (size_t i = 0; i < n; i++) {
        const float s = q(q(av[i]) + q(bv[i]));
        exact += memcmp(&s, &cv[i], 4) != 0;
    }
    CHECK(exact == 0, "second add differs");
    free(av);
    free(bv);
    free(cv);
    free(dv);
    pim_tensor_destroy(b); /* the shared slices survive a */
    pim_tensor_destroy(a);
    pim_tensor_destroy(c);
    pim_tensor_destroy(d);
}

static void
check_errors(void)
{
    pim_tensor_t *t, *u;
    CHECK(pim_tensor_create(&t, PIM_VECTOR_SB, 2, 8, OPDT) == PIM_E_ARG,
          "a vector of two rows");
    CHECK(pim_tensor_create(&t, PIM_MATRIX_AB, 0, 8, OPDT) == PIM_E_ARG,
          "an empty matrix");
    CHECK(pim_tensor_create_pair(&t, &u, PIM_MATRIX_AB, 4, 4, OPDT) ==
              PIM_E_ARG,
          "a matrix pair");
    pim_tensor_t *A, *x, *y;
    pim_tensor_create(&A, PIM_MATRIX_AB, 4, 8, OPDT);
    pim_tensor_create(&x, PIM_VECTOR_GB, 1, 7, OPDT);
    pim_tensor_create(&y, PIM_VECTOR_SB, 1, 4, PIM_FP32);
    CHECK(pim_gemv(y, A, x) == PIM_E_ARG, "gemv shape");
    pim_tensor_destroy(x);
    pim_tensor_create(&x, PIM_VECTOR_GB, 1, 8, OPDT);
    CHECK(pim_gemv(y, A, A) == PIM_E_ARG, "gemv kinds");
    pim_tensor_destroy(A);
    pim_tensor_destroy(x);
    pim_tensor_destroy(y);
}

int
main(void)
{
    pim_tensor_t *t;
    CHECK(pim_tensor_create(&t, PIM_VECTOR_SB, 1, 4, OPDT) == PIM_E_STATE,
          "before init");
    if (pim_init(NULL) != PIM_OK) {
        printf("init: %s\n", pim_error_detail());
        return 1;
    }
    if (PIM_LAYOUT_NUM_FMT != PIM_FMT_BF16 &&
        PIM_LAYOUT_NUM_FMT != PIM_FMT_FP16) {
        printf("test_api: operands are not bf16 or fp16; skipped\n");
        pim_finalize();
        return 0;
    }
    const uint32_t free0 = pim_free_slices();
    check_gemv(1, 1);
    check_gemv(37, 5);
    check_gemv(UB + 3, PER_ROW + 7);
    check_gemv(2 * UB, 2 * PER_ROW);
    check_eltwise(PIM_VECTOR_SB, 1);
    check_eltwise(PIM_VECTOR_SB, 2 * PER_ROW + 3);
    check_eltwise(PIM_VECTOR_PAIRED, 5);
    check_eltwise(PIM_VECTOR_PAIRED, UB / 2 * PER_ROW + 9);
    check_errors();
    CHECK(pim_free_slices() == free0, "slices leaked: %u of %u",
          pim_free_slices(), free0);
    CHECK(pim_ref_errors() == 0, "model errors %llu",
          (unsigned long long)pim_ref_errors());
    pim_finalize();
    return pim_test_summary("test_api");
}
