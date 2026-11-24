// Pack rows independently using the LPDDR6 fixture bit map and check the real issue loop.
#include "gemv.h"
#include <gem5/pim/pim_layout.h>
#include "ref/pim_ref.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

static int weight(unsigned r, unsigned k) {
    return (int)((r * 13 + k * 7 + r * k) % 17) - 8;
}

static void check_case(uint8_t *window, unsigned rows, unsigned cols)
{
    size_t tiles = (cols + 1023) / 1024, count = ((rows + 127) / 128) * tiles;
    struct matrix_rows *matrix = malloc(sizeof(*matrix) + count * sizeof(struct row_slice));
    _Float16 *x = calloc(cols, sizeof(*x)), *guarded = calloc(rows + 2, sizeof(*guarded));
    assert(matrix && x && guarded);
    matrix->version = 1;
    // For four logical rows use [16,17], [10], [7] to exercise both
    // advancing inside a slice and following noncontiguous slice entries.
    uint32_t physical[4] = {16, 17, 10, 7};
    assert(count <= 4);
    matrix->count = count == 4 ? 3 : count;
    for (size_t i = 0; i < matrix->count; ++i) {
        size_t first = count == 4 && i > 0 ? i + 1 : i;
        matrix->slices[i] = (struct row_slice){physical[first], count == 4 && i == 0 ? 2u : 1u};
    }
    for (unsigned r = 0; r < rows; ++r) {
        unsigned lane = r % 128, bank = lane % 16;
        for (unsigned k = 0; k < cols; ++k) {
            size_t logical = r / 128 * tiles + k / 1024;
            uint64_t offset = ((uint64_t)physical[logical] << 18)
                | ((bank % 4) << 16) | ((bank / 4) << 14) | ((lane / 16) << 11)
                | ((k % 1024) * 2);
            *(_Float16 *)(window + offset) = (_Float16)(weight(r,k) / 8.0f);
        }
    }
    guarded[0] = guarded[rows+1] = -123;
    struct gemv_args args = {1, 40, rows, cols, matrix, x, guarded + 1};
    uint64_t before = pim_ref_count(PIM_OP_AB_MAC);
    unsigned saved = matrix->slices[0].first;
    matrix->slices[0].first = PIM_LAYOUT_REG_ROW0;
    assert(gemv_run(&args, sizeof(args), (uintptr_t)window) != 0);
    assert(pim_ref_count(PIM_OP_AB_MAC) == before);
    matrix->slices[0].first = saved;
    for (unsigned pass = 0; pass < 2; ++pass) {
        for (unsigned k = 0; k < cols; ++k)
            x[k] = (_Float16)(((int)((k + pass * 3) % 11) - 5) / 8.0f);
        assert(gemv_run(&args, sizeof(args), (uintptr_t)window) == 0);
        for (unsigned r = 0; r < rows; ++r) {
            double expected = 0;
            for (unsigned k = 0; k < cols; ++k)
                expected += weight(r,k) / 8.0 * (double)x[k];
            assert(guarded[r+1] == (_Float16)expected);
        }
        assert(guarded[0] == -123 && guarded[rows+1] == -123);
    }
    // e8,m8 at VLEN 128 covers 128 bytes per command.
    size_t commands_per_group = (cols * 2 + 127) / 128;
    size_t expected_mac = 2 * ((rows + 127) / 128) * 8 * commands_per_group;
    assert(!pim_ref_errors() && pim_ref_count(PIM_OP_AB_MAC) - before == expected_mac);
    printf("%ux%u: two GEMVs matched, %llu all-bank MAC requests\n", rows, cols,
           (unsigned long long)(pim_ref_count(PIM_OP_AB_MAC) - before));
    free(matrix);
    free(x);
    free(guarded);
}

int main(void)
{
    size_t bytes = PIM_LAYOUT_WINDOW_SIZE;
    uint8_t *window = mmap(NULL, bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    assert(window != MAP_FAILED);
    pim_ref_config_t config = {PIM_GEOM_INIT, PIM_LAYOUT_NUM_REGS, PIM_LAYOUT_NUM_FMT,
                               PIM_LAYOUT_ACC_FMT, 16, 1};
    assert(pim_ref_attach(window, &config) == 0);
    check_case(window, 2, 3);
    check_case(window, 128, 1024);
    check_case(window, 129, 1025);
    pim_ref_detach();
    munmap(window, bytes);
    return 0;
}
