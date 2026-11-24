/*
 * Internal state and functions of the PIM runtime; private to util/pim.
 */

#ifndef __PIM_INTERNAL_H__
#define __PIM_INTERNAL_H__

#include <stddef.h>
#include <stdint.h>

#include <gem5/pim/pim_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The mode libpim falls back to (SE, the development path). */
#ifndef PIM_DEFAULT_MODE
#define PIM_DEFAULT_MODE PIM_MODE_SE
#endif

struct pim_buf
{
    uint32_t n;
    uint32_t rows[]; /* absolute row indices, one per slice */
};

typedef struct pim__state
{
    int initialized;
    pim_mode_t mode;
    uint8_t *base;
    uint64_t size;
    uint64_t phys;
    int ref_file; /* reference backend: region is a file mapping */
    char detail[512];
} pim__state_t;

extern pim__state_t pim__g;

/* Record the detail of an error and return the status. */
pim_status_t pim__fail(pim_status_t s, const char *fmt, ...);

/* The mode from the configuration, PIM_MODE (env), or the default. */
pim_status_t pim__resolve_mode(const pim_config_t *cfg, const char *env,
                               pim_mode_t *mode);
/* LQ4 (b)'s guards on injected inputs: the node name, and in FS the text
 * of /proc/iomem; base and size are the PIM memory's physical range. */
pim_status_t pim__mode_check(pim_mode_t mode, const char *nodename,
                             const char *iomem, uint64_t base, uint64_t size);

/* The Way 1 region. */
pim_status_t pim__region_open(pim_mode_t mode, const char *device);
void pim__region_close(void);

/* The row allocator over rows 0 .. REG_ROW0 - 1. */
int pim__rows_init(uint32_t rows);
void pim__rows_fini(void);
pim_status_t pim__rows_alloc(uint32_t n, uint32_t *out);
void pim__rows_free(const uint32_t *rows, uint32_t n);
int pim__row_used(uint32_t row);
uint32_t pim__rows_free_count(void);

#ifdef __cplusplus
}
#endif

#endif // __PIM_INTERNAL_H__
