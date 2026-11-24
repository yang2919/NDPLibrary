/*
 * Processing-in-Memory (PIM) library: the user application programming
 * interface (API).
 *
 * A program includes only this header and links the archive built for
 * its layout, libpim_<std>.a (libpim/README.md). Usable from C11 and
 * C++17. The lower layers (pim_runtime.h, pim_isa.h, pim_addr.h) stay
 * available for code that places data itself.
 *
 *   pim_init(NULL);
 *   pim_tensor_t *A, *x, *y;
 *   pim_tensor_create(&A, PIM_MATRIX_AB, 1024, 1024, PIM_BF16);
 *   pim_tensor_create(&x, PIM_VECTOR_GB, 1, 1024, PIM_BF16);
 *   pim_tensor_create(&y, PIM_VECTOR_SB, 1, 1024, PIM_FP32);
 *   pim_copy_to(A, a_host);           // converts, places, flushes
 *   pim_copy_to(x, x_host);
 *   pim_gemv(y, A, x);                // WR_GB, AB_MAC per row slice, RD_REG
 *   pim_copy_from(y, y_host);
 *   pim_finalize();
 */

#ifndef __GEM5_PIM_PIM_H__
#define __GEM5_PIM_PIM_H__

#include <stddef.h>

#include <gem5/pim/pim_layout.h>
#include <gem5/pim/pim_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * How a tensor is laid out; the library chooses its banks, rows and
 * columns (pim_tensor_locate() says where each element is).
 *
 * With U units (channels x ranks or pseudo channels), B banks per unit
 * and E elements per bank row:
 * - PIM_MATRIX_AB: matrix row m in bank (m % (U B)) % B of unit
 *   (m % (U B)) / B; columns in tiles of E, tile t of row group
 *   m / (U B) in its own row slice. For pim.ab.mac (GEMV).
 * - PIM_VECTOR_GB: a vector held by the library on the host and written
 *   into every unit's global buffer (GB) when an operation needs it.
 * - PIM_VECTOR_PAIRED: bank rows of E elements in the paired banks of
 *   every unit, for the all-bank element-wise (EW) commands. A tensor
 *   from pim_tensor_create() uses the even bank 2k of each pair; the two
 *   tensors of pim_tensor_create_pair() share row slices, the first in
 *   bank 2k and the second in bank 2k + 1 of the same pair and row.
 * - PIM_VECTOR_SB: one bank, bank 0 of unit 0, for single-bank (SB)
 *   commands; the two tensors of pim_tensor_create_pair() are in banks
 *   0 and 1 (one pair) of the same rows.
 */
typedef enum pim_kind
{
    PIM_MATRIX_AB = 0,
    PIM_VECTOR_GB = 1,
    PIM_VECTOR_PAIRED = 2,
    PIM_VECTOR_SB = 3,
} pim_kind_t;

/* Element types, the number formats of PIM_LAYOUT_NUM_FMT. */
typedef enum pim_dtype
{
    PIM_BF16 = PIM_FMT_BF16,
    PIM_FP16 = PIM_FMT_FP16,
    PIM_FP32 = PIM_FMT_FP32,
    PIM_INT8 = PIM_FMT_INT8,
    PIM_INT16 = PIM_FMT_INT16,
    PIM_INT32 = PIM_FMT_INT32,
} pim_dtype_t;

typedef struct pim_tensor pim_tensor_t;

/*
 * A rows x cols tensor of a kind (vectors: rows = 1). Operands of PIM
 * commands use PIM_LAYOUT_NUM_FMT; a GEMV result uses
 * PIM_LAYOUT_ACC_FMT.
 */
pim_status_t pim_tensor_create(pim_tensor_t **t, pim_kind_t kind, size_t rows,
                               size_t cols, pim_dtype_t dtype);
/*
 * Two tensors of the same shape whose elements share row slices, for
 * pim_add() and pim_mul(): element i of *a and element i of *b lie in
 * the same row and column of the two banks of one pair, *a in the even
 * bank 2k and *b in the odd bank 2k + 1 (PIM_VECTOR_PAIRED or
 * PIM_VECTOR_SB).
 */
pim_status_t pim_tensor_create_pair(pim_tensor_t **a, pim_tensor_t **b,
                                    pim_kind_t kind, size_t rows, size_t cols,
                                    pim_dtype_t dtype);
void pim_tensor_destroy(pim_tensor_t *t);

/*
 * Where element (row, col) is stored: its DRAM coordinates (c->row the
 * absolute DRAM row, c->col its column) and the byte within the column.
 * PIM_E_ARG for a PIM_VECTOR_GB tensor (held on the host).
 */
pim_status_t pim_tensor_locate(const pim_tensor_t *t, size_t row, size_t col,
                               pim_coord_t *c, uint32_t *byte);
/* The row slices holding the tensor (NULL for PIM_VECTOR_GB). */
const pim_buf_t *pim_tensor_buf(const pim_tensor_t *t);

/* Host fp32 data to the tensor (converted, placed, flushed), and back. */
pim_status_t pim_copy_to(pim_tensor_t *t, const float *host);
pim_status_t pim_copy_from(const pim_tensor_t *t, float *host);

/*
 * y = A x: A a PIM_MATRIX_AB tensor (rows x cols), x a PIM_VECTOR_GB
 * tensor (1 x cols), both of PIM_LAYOUT_NUM_FMT; y a PIM_VECTOR_SB
 * tensor (1 x rows) of PIM_LAYOUT_ACC_FMT. Per row group: the
 * accumulators reset with pim.wr.reg, pim.wr.gb of each column tile into
 * every unit's GB, pim.ab.mac over the tile, pim.rd.reg of the
 * accumulators.
 */
pim_status_t pim_gemv(pim_tensor_t *y, const pim_tensor_t *A,
                      const pim_tensor_t *x);
/*
 * c = a + b and c = a * b, element-wise: a and b from one
 * pim_tensor_create_pair(), c from pim_tensor_create() with the same
 * kind, shape and type. PIM_VECTOR_PAIRED uses pim.ab.ewadd /
 * pim.ab.ewmul, PIM_VECTOR_SB pim.sb.ewadd / pim.sb.ewmul; both then
 * copy the registers into c's rows with pim.ab.mov.regbk.
 */
pim_status_t pim_add(pim_tensor_t *c, const pim_tensor_t *a,
                     const pim_tensor_t *b);
pim_status_t pim_mul(pim_tensor_t *c, const pim_tensor_t *a,
                     const pim_tensor_t *b);

#ifdef __cplusplus
}
#endif

#endif // __GEM5_PIM_PIM_H__
