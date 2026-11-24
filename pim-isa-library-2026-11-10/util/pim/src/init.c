/*
 * pim_init() and pim_finalize(), the mode and its guards (LQ4 (b)), and
 * the status codes (LQ9 (c)).
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>

#include "pim_internal.h"

pim__state_t pim__g;

/* The node name gem5's SE mode reports
 * (src/arch/riscv/linux/se_workload.cc:116, :131). */
static const char se_nodename[] = "sim.gem5.org";

pim_status_t
pim__fail(pim_status_t s, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(pim__g.detail, sizeof(pim__g.detail), fmt, ap);
    va_end(ap);
    return s;
}

const char *
pim_strerror(pim_status_t s)
{
    switch (s) {
        case PIM_OK:
            return "success";
        case PIM_E_ARG:
            return "bad argument or shape";
        case PIM_E_NOMEM:
            return "no free PIM row slices";
        case PIM_E_DEVICE:
            return "cannot open or map the PIM memory (/dev/mem or the "
                   "driver)";
        case PIM_E_MODE:
            return "the PIM mode contradicts the system (set PIM_MODE=se or "
                   "fs)";
        case PIM_E_NODE:
            return "no PIM NUMA node";
        case PIM_E_HUGEPAGE:
            return "no free huge page";
        case PIM_E_STATE:
            return "called before pim_init() or after pim_finalize()";
    }
    return "unknown status";
}

const char *
pim_error_detail(void)
{ return pim__g.detail; }

pim_status_t
pim__resolve_mode(const pim_config_t *cfg, const char *env, pim_mode_t *mode)
{
    if (cfg && cfg->mode != PIM_MODE_DEFAULT) {
        if (cfg->mode != PIM_MODE_SE && cfg->mode != PIM_MODE_FS) {
            return pim__fail(PIM_E_ARG, "unknown mode %d", (int)cfg->mode);
        }
        *mode = cfg->mode;
        return PIM_OK;
    }
    if (env && *env) {
        if (strcmp(env, "se") == 0 || strcmp(env, "SE") == 0) {
            *mode = PIM_MODE_SE;
        } else if (strcmp(env, "fs") == 0 || strcmp(env, "FS") == 0) {
            *mode = PIM_MODE_FS;
        } else {
            return pim__fail(PIM_E_ARG, "PIM_MODE=%s: use se or fs", env);
        }
        return PIM_OK;
    }
    *mode = PIM_DEFAULT_MODE;
    return PIM_OK;
}

pim_status_t
pim__mode_check(pim_mode_t mode, const char *nodename, const char *iomem,
                uint64_t base, uint64_t size)
{
    const int se_node = nodename && strcmp(nodename, se_nodename) == 0;
    if (mode == PIM_MODE_SE && !se_node) {
        return pim__fail(PIM_E_MODE,
                         "PIM mode SE, but the node name is '%s', not %s: "
                         "set PIM_MODE=fs in full-system runs",
                         nodename ? nodename : "", se_nodename);
    }
    if (mode == PIM_MODE_FS && se_node) {
        return pim__fail(PIM_E_MODE,
                         "PIM mode FS, but the node name %s is gem5's SE "
                         "mode: set PIM_MODE=se",
                         se_nodename);
    }
    if (mode != PIM_MODE_FS) {
        return PIM_OK;
    }
    /* FS: no System RAM may overlap [base, base + size). */
    int ram_lines = 0, ram_known = 0;
    const char *p = iomem ? iomem : "";
    while (*p) {
        const char *eol = strchr(p, '\n');
        const size_t len = eol ? (size_t)(eol - p) : strlen(p);
        char line[256];
        const size_t n = len < sizeof(line) - 1 ? len : sizeof(line) - 1;
        memcpy(line, p, n);
        line[n] = 0;
        p += len + (eol ? 1 : 0);
        unsigned long long lo, hi;
        int name_at = 0;
        if (sscanf(line, " %llx-%llx : %n", &lo, &hi, &name_at) != 2 ||
            !name_at) {
            continue;
        }
        const char *name = line + name_at;
        if (strncmp(name, "System RAM", 10) != 0) {
            continue;
        }
        ram_lines++;
        if (lo == 0 && hi == 0) {
            continue; /* hidden: not root */
        }
        ram_known++;
        if (lo < base + size && hi >= base) {
            return pim__fail(PIM_E_MODE,
                             "PIM mode FS, but System RAM %llx-%llx overlaps "
                             "the PIM memory %llx-%llx: is PIM_MODE right?",
                             lo, hi, (unsigned long long)base,
                             (unsigned long long)(base + size - 1));
        }
    }
    if (!ram_lines || ram_known < ram_lines) {
        return pim__fail(PIM_E_MODE,
                         "PIM mode FS, but /proc/iomem shows no System RAM "
                         "addresses (run as root)");
    }
    return PIM_OK;
}

#ifndef PIM_BACKEND_REF
static char *
read_file(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        return NULL;
    }
    size_t cap = 4096, n = 0;
    char *buf = (char *)malloc(cap);
    while (buf) {
        n += fread(buf + n, 1, cap - n - 1, f);
        if (n < cap - 1) {
            break;
        }
        cap *= 2;
        char *grown = (char *)realloc(buf, cap);
        if (!grown) {
            free(buf);
            buf = NULL;
        }
        buf = grown;
    }
    if (buf) {
        buf[n] = 0;
    }
    fclose(f);
    return buf;
}
#endif

pim_status_t
pim__init(const pim_config_t *cfg, const unsigned *layout_id)
{
    (void)layout_id;
    pim__g.detail[0] = 0;
    if (pim__g.initialized) {
        return pim__fail(PIM_E_STATE, "pim_init() called twice");
    }
    const pim_config_t defaults = PIM_CONFIG_INIT;
    if (!cfg) {
        cfg = &defaults;
    }
    if (cfg->way != 1) {
        return pim__fail(PIM_E_ARG, "only Way 1 (the linear region) is "
                                    "implemented");
    }
    pim_mode_t mode;
    pim_status_t s = pim__resolve_mode(cfg, getenv("PIM_MODE"), &mode);
    if (s != PIM_OK) {
        return s;
    }
    const uint64_t base = mode == PIM_MODE_SE ? PIM_LAYOUT_WINDOW_BASE_SE
                                              : PIM_LAYOUT_WINDOW_BASE_FS;
#ifndef PIM_BACKEND_REF
    /* The guards read no PIM memory (LQ4 (b)); the host reference
     * backend is not gem5 and skips them. */
    struct utsname u;
    if (uname(&u) != 0) {
        return pim__fail(PIM_E_MODE, "uname() failed");
    }
    char *iomem = mode == PIM_MODE_FS ? read_file("/proc/iomem") : NULL;
    s = pim__mode_check(mode, u.nodename, iomem, base, PIM_LAYOUT_WINDOW_SIZE);
    free(iomem);
    if (s != PIM_OK) {
        return s;
    }
#endif
    s = pim__region_open(mode, cfg->device);
    if (s != PIM_OK) {
        return s;
    }
    if (pim__rows_init(PIM_LAYOUT_REG_ROW0)) {
        pim__region_close();
        return pim__fail(PIM_E_NOMEM, "no memory for the row allocator");
    }
    pim__g.mode = mode;
    pim__g.phys = base;
    pim__g.initialized = 1;
    return PIM_OK;
}

void
pim_finalize(void)
{
    if (!pim__g.initialized) {
        return;
    }
    pim__rows_fini();
    pim__region_close();
    pim__g.initialized = 0;
}

pim_mode_t
pim_mode(void)
{ return pim__g.initialized ? pim__g.mode : PIM_MODE_DEFAULT; }

uint64_t
pim_region_phys(void)
{ return pim__g.initialized ? pim__g.phys : 0; }
