// Freestanding N0 GEMV issue loop shared by the PNM ELF and supplier reference model.
#include "gemv.h"
#include <gem5/pim/pim_layout.h>
#include <gem5/pim/pim_isa.h>

_Static_assert(PIM_LAYOUT_NUM_FMT == PIM_FMT_FP16 && PIM_LAYOUT_ACC_FMT == PIM_FMT_FP32,
               "expected FP16 inputs and FP32 accumulators");

static void *address(uintptr_t base, unsigned unit, unsigned lane, unsigned row, size_t offset)
{
    pim_coord_t coord = {
        .ch = unit / PIM_LAYOUT_UNITS, .unit = unit % PIM_LAYOUT_UNITS,
        .bank = lane % PIM_LAYOUT_BANKS_PER_GROUP,
        .bg = lane / PIM_LAYOUT_BANKS_PER_GROUP % PIM_LAYOUT_BANK_GROUPS,
        .sid = lane / (PIM_LAYOUT_BANKS_PER_GROUP * PIM_LAYOUT_BANK_GROUPS),
        .row = row, .col = offset / PIM_LAYOUT_COL_BYTES
    };
    return (void *)(base + pim_encode(&coord) + offset % PIM_LAYOUT_COL_BYTES);
}

static size_t piece(size_t offset, size_t end)
{
    size_t remaining = PIM_LAYOUT_CONTIG_BYTES - offset % PIM_LAYOUT_CONTIG_BYTES;
    return remaining < end - offset ? remaining : end - offset;
}

int gemv_run(const struct gemv_args *a, size_t size, uintptr_t base)
{
    if (size != sizeof(*a) || a->version != 1 || a->bytes != size ||
        !a->rows || !a->cols || !a->matrix || !a->x || !a->y)
        return 1;
    const unsigned units = PIM_LAYOUT_CHANNELS * PIM_LAYOUT_UNITS;
    const unsigned lanes = units * PIM_LAYOUT_BANKS_PER_UNIT;
    const size_t tile_cols = PIM_LAYOUT_ROW_BYTES / sizeof(*a->x);
    size_t tiles = (a->cols - 1) / tile_cols + 1;
    size_t groups = (a->rows - 1) / lanes + 1;
    size_t required = tiles * groups;
    const struct matrix_rows *matrix = a->matrix;
    if (matrix->version != 1 || matrix->count > required || required > PIM_LAYOUT_REG_ROW0)
        return 1;
    size_t total = 0;
    for (unsigned i = 0; i < matrix->count; ++i) {
        struct row_slice s = matrix->slices[i];
        if (!s.count || s.first >= PIM_LAYOUT_REG_ROW0 || s.count > PIM_LAYOUT_REG_ROW0 - s.first)
            return 1;
        total += s.count;
    }
    if (total != required) return 1;

    const unsigned acc_row = PIM_LAYOUT_REG_ROW0 + PIM_LAYOUT_NUM_REGS;
    unsigned slice = 0, within = 0;
    const float zero = 0;
    pim_fence();
    for (size_t g = 0; g < groups; ++g) {
        for (unsigned u = 0; u < units; ++u)
            for (unsigned k = 0; k < PIM_LAYOUT_BANKS_PER_UNIT; ++k)
                pim_wr_reg_mem(address(base, u, k, acc_row, 0), &zero, sizeof(zero));

        for (size_t t = 0; t < tiles; ++t) {
            unsigned row = matrix->slices[slice].first + within;
            if (++within == matrix->slices[slice].count) { ++slice; within = 0; }
            size_t first = t * tile_cols;
            size_t n = a->cols - first;
            size_t bytes = (n < tile_cols ? n : tile_cols) * sizeof(*a->x);
            const uint8_t *x = (const uint8_t *)(a->x + first);
            for (unsigned u = 0; u < units; ++u) {
                for (size_t off = 0; off < bytes;)
                    off += pim_wr_gb_mem(address(base, u, 0, PIM_LAYOUT_REG_ROW0, off),
                                         x + off, piece(off, bytes));
                for (size_t off = 0; off < bytes;)
                    off += pim_ab_mac(address(base, u, 0, row, off), piece(off, bytes));
            }
        }
        for (unsigned lane = 0; lane < lanes && g * lanes + lane < a->rows; ++lane) {
            float acc;
            pim_rd_reg_mem(address(base, lane / PIM_LAYOUT_BANKS_PER_UNIT,
                                   lane % PIM_LAYOUT_BANKS_PER_UNIT, acc_row, 0), &acc, sizeof(acc));
            a->y[g * lanes + lane] = (_Float16)acc;
        }
    }
    pim_fence();
    return 0;
}

#ifndef PIM_BACKEND_REF
__attribute__((section(".text.entry")))
int pnm_pim_gemv(const struct gemv_args *args, size_t size)
{
    return gemv_run(args, size, PIM_LAYOUT_WINDOW_BASE_FS);
}
#endif
