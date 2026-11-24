/*
 * Internal state and functions of the PIM runtime; private to libpim.
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
    unsigned session; /* the pim_init() session that allocated it */
    uint32_t n;
    uint32_t rows[]; /* absolute row indices, one per slice */
};

/* Way 2: a mapping of part of the PIM memory, at offset off. */
typedef struct pim__map
{
    uint8_t *va;
    uint64_t off;
    uint64_t len;
} pim__map_t;

typedef struct pim__state
{
    int initialized;
    unsigned session; /* counts pim_init() calls */
    pim_mode_t mode;
    int way; /* 1: base + offset; 2: slice_va */
    uint8_t *base;
    uint64_t size;
    uint64_t phys;
    int ref_file; /* reference backend: region is a file mapping */
    /* Way 2: the virtual address of every row slice
     * (offset >> ROW_SHIFT) that a mapping holds, else NULL; the mappings
     * (the register rows, then the huge pages); the node. */
    uint8_t **slice_va;
    pim__map_t *maps;
    uint32_t nmaps, cap_maps;
    int node;
    char detail[512];
} pim__state_t;

extern pim__state_t pim__g;

/* Record the detail of an error and return the status. */
pim_status_t pim__fail(pim_status_t s, const char *fmt, ...);

/* The mode from the configuration, PIM_MODE (env), or the default. */
pim_status_t pim__resolve_mode(const pim_config_t *cfg, const char *env,
                               pim_mode_t *mode);
/* The mode guards on injected inputs: the node name, and in FS the text
 * of /proc/iomem; base and size are the PIM memory's physical range. */
pim_status_t pim__mode_check(pim_mode_t mode, const char *nodename,
                             const char *iomem, uint64_t base, uint64_t size);

/* The Way 1 region. */
pim_status_t pim__region_open(pim_mode_t mode, const char *device);
void pim__region_close(void);

/*
 * The address of byte off of the PIM memory: Way 1, the region's base
 * plus off; Way 2, the mapping that holds its row slice (NULL when none
 * does).
 */
static inline uint8_t *
pim__addr(uint64_t off)
{
    if (pim__g.way != 2) {
        return pim__g.base + off;
    }
    uint8_t *s = pim__g.slice_va[off >> PIM_LAYOUT_ROW_SHIFT];
    return s ? s + (off & (((uint64_t)1 << PIM_LAYOUT_ROW_SHIFT) - 1))
             : NULL;
}
/* The inverse: 0 and *off for an address in the PIM memory, else -1. */
int pim__offset(const void *a, uint64_t *off);

/* The way from the configuration, PIM_WAY (env), or 1. */
pim_status_t pim__resolve_way(const pim_config_t *cfg, const char *env,
                              int *way);

/*
 * Way 2 (way2.c). The pure parts, testable on the host:
 * - the node: PIM_NODE (env) when set, else the device tree's
 *   numa-node-id property (4 bytes, big-endian; dt NULL: none);
 * - a huge page's frame must lie in the PIM memory below the register
 *   rows, aligned to the page;
 * - the slice table: add a mapping of [off, off + len) at va.
 */
pim_status_t pim__way2_node(const char *env, const uint8_t *dt, size_t n,
                            int *node);
int pim__way2_frame_ok(uint64_t pa, uint64_t page);
pim_status_t pim__way2_tables_init(void);
pim_status_t pim__way2_add_map(uint8_t *va, uint64_t off, uint64_t len);
void pim__way2_tables_fini(void);
/* The guest parts: open (node, register rows) and close; grow the row
 * allocator by huge pages holding at least n more slices. */
pim_status_t pim__way2_open(const char *device);
void pim__way2_close(void);
pim_status_t pim__way2_grow(uint32_t n);

/* The row allocator over rows 0 .. REG_ROW0 - 1 (Way 2: empty until
 * pim__rows_add() makes a row of a mapped huge page available). */
int pim__rows_init(uint32_t rows);
int pim__rows_init_empty(uint32_t rows);
void pim__rows_add(uint32_t row);
void pim__rows_fini(void);
pim_status_t pim__rows_alloc(uint32_t n, uint32_t *out);
void pim__rows_free(const uint32_t *rows, uint32_t n);
int pim__row_used(uint32_t row);
uint32_t pim__rows_free_count(void);

#ifdef __cplusplus
}
#endif

#endif // __PIM_INTERNAL_H__
