/*
 * Way 2: the PIM memory is a Linux NUMA node, FS only, under a policy
 * with the row on top. Its row slices come from 2 MiB huge pages bound to
 * that node; a small /dev/mem mapping covers the register rows, which the
 * board keeps out of the node.
 *
 * Addresses: the library still computes a physical offset in the PIM
 * memory (pim_encode()), and pim__addr() finds the mapping that holds its
 * row slice in a table indexed by offset >> ROW_SHIFT. Under a policy
 * with the row on top a slice is one range (pim_geom_slice_contiguous()),
 * and a huge page holds whole slices, so a table entry is the page's
 * address plus the slice's place in it.
 *
 * Sequence per group of pages: mmap(MAP_SHARED | MAP_ANONYMOUS |
 * MAP_HUGETLB), mbind() to the node before the first touch, touch,
 * mlock(), then each page's frame from /proc/self/pagemap (root), which
 * must lie in the PIM memory below the register rows: a page elsewhere,
 * for example on the OS node, is refused with PIM_E_NODE.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "pim_internal.h"

#define HUGE_PAGE ((uint64_t)2 << 20)
#define SLICE_BYTES ((uint64_t)1 << PIM_LAYOUT_ROW_SHIFT)
#define SLICES (PIM_LAYOUT_WINDOW_SIZE >> PIM_LAYOUT_ROW_SHIFT)
/* The register rows: REG_ROW0 up, at the top of the PIM memory. */
#define REG_OFF ((uint64_t)PIM_LAYOUT_REG_ROW0 << PIM_LAYOUT_ROW_SHIFT)

pim_status_t
pim__resolve_way(const pim_config_t *cfg, const char *env, int *way)
{
    if (cfg && cfg->way != 0) {
        if (cfg->way != 1 && cfg->way != 2) {
            return pim__fail(PIM_E_ARG, "unknown way %d", cfg->way);
        }
        *way = cfg->way;
        return PIM_OK;
    }
    if (env && *env) {
        if (strcmp(env, "1") != 0 && strcmp(env, "2") != 0) {
            return pim__fail(PIM_E_ARG, "PIM_WAY=%s: use 1 or 2", env);
        }
        *way = env[0] - '0';
        return PIM_OK;
    }
    *way = 1;
    return PIM_OK;
}

pim_status_t
pim__way2_node(const char *env, const uint8_t *dt, size_t n, int *node)
{
    if (env && *env) {
        char *end;
        const long v = strtol(env, &end, 10);
        if (*end || v < 0 || v > 63) {
            return pim__fail(PIM_E_NODE, "PIM_NODE=%s: not a node number",
                             env);
        }
        *node = (int)v;
        return PIM_OK;
    }
    if (!dt || n != 4) {
        return pim__fail(PIM_E_NODE,
                         "no numa-node-id for the PIM memory in the device "
                         "tree (memory@%llx): is it a Way 2 board?",
                         (unsigned long long)PIM_LAYOUT_WINDOW_BASE_FS);
    }
    const uint32_t v = (uint32_t)dt[0] << 24 | (uint32_t)dt[1] << 16 |
                       (uint32_t)dt[2] << 8 | dt[3];
    if (v > 63) {
        return pim__fail(PIM_E_NODE, "numa-node-id %u", v);
    }
    *node = (int)v;
    return PIM_OK;
}

int
pim__way2_frame_ok(uint64_t pa, uint64_t page)
{
    const uint64_t lo = PIM_LAYOUT_WINDOW_BASE_FS;
    return pa % page == 0 && pa >= lo && pa - lo < REG_OFF &&
           pa - lo + page <= REG_OFF;
}

pim_status_t
pim__way2_tables_init(void)
{
    pim__way2_tables_fini();
    pim__g.slice_va = (uint8_t **)calloc(SLICES, sizeof(uint8_t *));
    if (!pim__g.slice_va) {
        return pim__fail(PIM_E_NOMEM, "no host memory for the slice table");
    }
    return PIM_OK;
}

pim_status_t
pim__way2_add_map(uint8_t *va, uint64_t off, uint64_t len)
{
    if (off % SLICE_BYTES || len % SLICE_BYTES ||
        off + len > PIM_LAYOUT_WINDOW_SIZE || off + len < off) {
        return pim__fail(PIM_E_ARG, "mapping %#llx + %#llx is not whole row "
                                    "slices of the PIM memory",
                         (unsigned long long)off, (unsigned long long)len);
    }
    if (pim__g.nmaps == pim__g.cap_maps) {
        const uint32_t cap = pim__g.cap_maps ? 2 * pim__g.cap_maps : 16;
        pim__map_t *m =
            (pim__map_t *)realloc(pim__g.maps, cap * sizeof(pim__map_t));
        if (!m) {
            return pim__fail(PIM_E_NOMEM, "no host memory for a mapping");
        }
        pim__g.maps = m;
        pim__g.cap_maps = cap;
    }
    pim__g.maps[pim__g.nmaps++] = (pim__map_t){va, off, len};
    for (uint64_t o = 0; o < len; o += SLICE_BYTES) {
        pim__g.slice_va[(off + o) >> PIM_LAYOUT_ROW_SHIFT] = va + o;
    }
    return PIM_OK;
}

void
pim__way2_tables_fini(void)
{
    free(pim__g.slice_va);
    free(pim__g.maps);
    pim__g.slice_va = NULL;
    pim__g.maps = NULL;
    pim__g.nmaps = pim__g.cap_maps = 0;
}

int
pim__offset(const void *a, uint64_t *off)
{
    const uint8_t *p = (const uint8_t *)a;
    if (pim__g.way != 2) {
        if (!pim__g.base || p < pim__g.base ||
            p >= pim__g.base + pim__g.size) {
            return -1;
        }
        *off = (uint64_t)(p - pim__g.base);
        return 0;
    }
    for (uint32_t i = 0; i < pim__g.nmaps; i++) {
        const pim__map_t *m = &pim__g.maps[i];
        if (p >= m->va && p < m->va + m->len) {
            *off = m->off + (uint64_t)(p - m->va);
            return 0;
        }
    }
    return -1;
}

#ifdef PIM_BACKEND_REF

/* The host reference backend has no NUMA node: Way 2 is FS only. */
pim_status_t
pim__way2_open(const char *device)
{
    (void)device;
    return pim__fail(PIM_E_ARG, "Way 2 runs only in gem5 full-system mode");
}

void
pim__way2_close(void)
{ pim__way2_tables_fini(); }

pim_status_t
pim__way2_grow(uint32_t n)
{
    (void)n;
    return pim__fail(PIM_E_STATE, "Way 2 is not open");
}

#else

/* The device tree's numa-node-id of memory@<WINDOW_BASE_FS>. */
static int
read_dt_node(uint8_t *out, size_t *n)
{
    char path[128];
    snprintf(path, sizeof(path),
             "/sys/firmware/devicetree/base/memory@%llx/numa-node-id",
             (unsigned long long)PIM_LAYOUT_WINDOW_BASE_FS);
    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return -1;
    }
    const ssize_t r = read(fd, out, 8);
    close(fd);
    *n = r > 0 ? (size_t)r : 0;
    return r > 0 ? 0 : -1;
}

pim_status_t
pim__way2_open(const char *device)
{
    if (!pim_geom_slice_contiguous(pim_geom()) || SLICE_BYTES > HUGE_PAGE) {
        return pim__fail(PIM_E_ARG,
                         "Way 2 needs a policy with the row on top and row "
                         "slices of at most 2 MiB (slices of %llu bytes)",
                         (unsigned long long)SLICE_BYTES);
    }
    uint8_t dt[8];
    size_t n = 0;
    const int have_dt = read_dt_node(dt, &n) == 0;
    pim_status_t s =
        pim__way2_node(getenv("PIM_NODE"), have_dt ? dt : NULL, n,
                       &pim__g.node);
    if (s != PIM_OK) {
        return s;
    }
    s = pim__way2_tables_init();
    if (s != PIM_OK) {
        return s;
    }
    /* The register rows, through /dev/mem (root). */
    const char *path = device ? device : "/dev/mem";
    const int fd = open(path, O_RDWR);
    if (fd < 0) {
        pim__way2_tables_fini();
        return pim__fail(PIM_E_DEVICE, "%s: %s", path, strerror(errno));
    }
    const uint64_t len = PIM_LAYOUT_WINDOW_SIZE - REG_OFF;
    void *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                   (off_t)(PIM_LAYOUT_WINDOW_BASE_FS + REG_OFF));
    const int err = errno;
    close(fd);
    if (p == MAP_FAILED) {
        pim__way2_tables_fini();
        return pim__fail(PIM_E_DEVICE, "mmap of the register rows: %s",
                         strerror(err));
    }
    s = pim__way2_add_map((uint8_t *)p, REG_OFF, len);
    if (s != PIM_OK) {
        munmap(p, len);
        pim__way2_tables_fini();
        return s;
    }
    if (getenv("PIM_VERBOSE")) {
        fprintf(stderr, "pim: way 2, node %d (%s), register rows at %p\n",
                pim__g.node, getenv("PIM_NODE") ? "PIM_NODE" : "device tree",
                p);
    }
    return PIM_OK;
}

void
pim__way2_close(void)
{
    for (uint32_t i = 0; i < pim__g.nmaps; i++) {
        munmap(pim__g.maps[i].va, pim__g.maps[i].len);
    }
    pim__way2_tables_fini();
}

pim_status_t
pim__way2_grow(uint32_t n)
{
    const uint32_t per_page = (uint32_t)(HUGE_PAGE / SLICE_BYTES);
    const uint64_t pages = (n + per_page - 1) / per_page;
    const uint64_t len = pages * HUGE_PAGE;
    uint8_t *p = (uint8_t *)mmap(NULL, len, PROT_READ | PROT_WRITE,
                                 MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB,
                                 -1, 0);
    if (p == MAP_FAILED) {
        return pim__fail(PIM_E_HUGEPAGE,
                         "%llu huge pages of 2 MiB: %s (reserve them on node "
                         "%d: nr_hugepages)",
                         (unsigned long long)pages, strerror(errno),
                         pim__g.node);
    }
    unsigned long mask = 1UL << pim__g.node;
    if (syscall(SYS_mbind, p, len, 2 /* MPOL_BIND */, &mask,
                8 * sizeof(mask), 0) != 0) {
        const int err = errno;
        munmap(p, len);
        return pim__fail(PIM_E_NODE, "mbind to node %d: %s", pim__g.node,
                         strerror(err));
    }
    /* Touch every page (the kernel takes a huge page of the node or
     * fails the access), then pin them. */
    for (uint64_t i = 0; i < pages; i++) {
        ((volatile uint8_t *)p)[i * HUGE_PAGE] = 0;
    }
    if (mlock(p, len) != 0) {
        const int err = errno;
        munmap(p, len);
        return pim__fail(PIM_E_HUGEPAGE, "mlock: %s", strerror(err));
    }
    const int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) {
        munmap(p, len);
        return pim__fail(PIM_E_NODE, "/proc/self/pagemap: %s",
                         strerror(errno));
    }
    uint64_t *pa = (uint64_t *)malloc(pages * sizeof(uint64_t));
    pim_status_t s = pa ? PIM_OK
                        : pim__fail(PIM_E_NOMEM, "no host memory for frames");
    for (uint64_t i = 0; s == PIM_OK && i < pages; i++) {
        const uintptr_t va = (uintptr_t)p + i * HUGE_PAGE;
        uint64_t e = 0;
        if (pread(fd, &e, sizeof(e), (off_t)(va / 4096 * sizeof(e))) !=
                (ssize_t)sizeof(e) ||
            !(e >> 63)) {
            s = pim__fail(PIM_E_NODE, "no frame for %p in pagemap (root?)",
                          (void *)va);
            break;
        }
        pa[i] = (e & (((uint64_t)1 << 55) - 1)) << 12;
        if (!pim__way2_frame_ok(pa[i], HUGE_PAGE)) {
            s = pim__fail(PIM_E_NODE,
                          "the huge page at %p is at %#llx, not in the PIM "
                          "memory %#llx-%#llx (node %d is not the PIM "
                          "memory's)",
                          (void *)va, (unsigned long long)pa[i],
                          (unsigned long long)PIM_LAYOUT_WINDOW_BASE_FS,
                          (unsigned long long)(PIM_LAYOUT_WINDOW_BASE_FS +
                                               REG_OFF - 1),
                          pim__g.node);
        }
    }
    close(fd);
    for (uint64_t i = 0; s == PIM_OK && i < pages; i++) {
        const uint64_t off = pa[i] - PIM_LAYOUT_WINDOW_BASE_FS;
        s = pim__way2_add_map(p + i * HUGE_PAGE, off, HUGE_PAGE);
        if (s != PIM_OK) {
            break;
        }
        for (uint64_t o = 0; o < HUGE_PAGE; o += SLICE_BYTES) {
            pim__rows_add((uint32_t)((off + o) >> PIM_LAYOUT_ROW_SHIFT));
        }
        if (getenv("PIM_VERBOSE")) {
            fprintf(stderr, "pim: way 2 page %p frame %#llx\n",
                    (void *)(p + i * HUGE_PAGE), (unsigned long long)pa[i]);
        }
    }
    free(pa);
    if (s != PIM_OK) {
        /* Pages already in the table stay mapped until pim_finalize(). */
        int listed = 0;
        for (uint32_t i = 0; i < pim__g.nmaps; i++) {
            listed |= pim__g.maps[i].va == p;
        }
        if (!listed) {
            munmap(p, len);
        }
    }
    return s;
}

#endif
