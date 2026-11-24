// N0 GEMV ABI. All pointers in the relocated argument block are PNM-visible.
#pragma once
#include <stddef.h>
#include <stdint.h>

struct row_slice { uint32_t first, count; };
struct matrix_rows {
    uint32_t version, count;
    struct row_slice slices[];
};
struct gemv_args {
    uint32_t version, bytes, rows, cols;
    const struct matrix_rows *matrix;
    const _Float16 *x;
    _Float16 *y;
};
_Static_assert(sizeof(struct gemv_args) == 40 && offsetof(struct gemv_args, matrix) == 16,
               "requires the RV64 launch ABI");
int gemv_run(const struct gemv_args *args, size_t size, uintptr_t window);
