/*
 * The row allocator (LQ5 (a)): whole row slices, one free list of the
 * usable rows 0 .. REG_ROW0 - 1; the register rows are never handed out.
 * First fit of a contiguous run, else the lowest free rows.
 */

#include <stdlib.h>
#include <string.h>

#include "pim_internal.h"

static uint8_t *used;
static uint32_t nrows;
static uint32_t nfree;

int
pim__rows_init(uint32_t rows)
{
    free(used);
    used = (uint8_t *)calloc(rows ? rows : 1, 1);
    nrows = rows;
    nfree = rows;
    return used ? 0 : -1;
}

void
pim__rows_fini(void)
{
    free(used);
    used = NULL;
    nrows = nfree = 0;
}

pim_status_t
pim__rows_alloc(uint32_t n, uint32_t *out)
{
    if (n == 0) {
        return PIM_E_ARG;
    }
    if (n > nfree) {
        return pim__fail(PIM_E_NOMEM, "%u row slices asked, %u free", n,
                         nfree);
    }
    /* First fit of n contiguous rows. */
    uint32_t run = 0;
    for (uint32_t r = 0; r < nrows; r++) {
        run = used[r] ? 0 : run + 1;
        if (run == n) {
            for (uint32_t i = 0; i < n; i++) {
                out[i] = r + 1 - n + i;
                used[out[i]] = 1;
            }
            nfree -= n;
            return PIM_OK;
        }
    }
    /* Else the lowest free rows. */
    uint32_t k = 0;
    for (uint32_t r = 0; r < nrows && k < n; r++) {
        if (!used[r]) {
            out[k++] = r;
            used[r] = 1;
        }
    }
    nfree -= n;
    return PIM_OK;
}

void
pim__rows_free(const uint32_t *rows, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++) {
        if (rows[i] < nrows && used[rows[i]]) {
            used[rows[i]] = 0;
            nfree++;
        }
    }
}

int
pim__row_used(uint32_t row)
{ return row < nrows && used[row]; }

uint32_t
pim__rows_free_count(void)
{ return nfree; }
