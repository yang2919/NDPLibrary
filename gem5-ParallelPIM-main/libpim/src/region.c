/*
 * The Way 1 region: the whole PIM memory mapped
 * linearly, so the virtual offset from the base equals the physical
 * offset in the PIM memory.
 *
 * - RISC-V: open("/dev/mem") and mmap(..., MAP_SHARED, fd, base), base
 *   WINDOW_BASE_SE or WINDOW_BASE_FS; in SE the emulated driver answers.
 *   No O_SYNC, which can make the mapping uncached.
 * - Host reference backend: an anonymous mapping, or the file named by
 *   PIM_REF_FILE; the model attaches to it.
 */

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "pim_internal.h"
#include "ref/pim_ref.h"

pim_status_t
pim__region_open(pim_mode_t mode, const char *device)
{
    const uint64_t size = PIM_LAYOUT_WINDOW_SIZE;
    void *p;
#ifdef PIM_BACKEND_REF
    (void)mode;
    (void)device;
    const char *file = getenv("PIM_REF_FILE");
    if (file && *file) {
        const int fd = open(file, O_RDWR | O_CREAT, 0600);
        if (fd < 0 || ftruncate(fd, (off_t)size) != 0) {
            if (fd >= 0) {
                close(fd);
            }
            return pim__fail(PIM_E_DEVICE, "%s: %s", file, strerror(errno));
        }
        p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        pim__g.ref_file = 1;
    } else {
        p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        pim__g.ref_file = 0;
    }
    if (p == MAP_FAILED) {
        return pim__fail(PIM_E_DEVICE, "cannot map %llu bytes: %s",
                         (unsigned long long)size, strerror(errno));
    }
    pim_ref_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.geom = *pim_geom();
    cfg.num_regs = PIM_LAYOUT_NUM_REGS;
    cfg.num_fmt = PIM_LAYOUT_NUM_FMT;
    cfg.acc_fmt = PIM_LAYOUT_ACC_FMT;
    /* VLEN = 8 x the column bytes by default. */
    const char *vlenb = getenv("PIM_REF_VLENB");
    cfg.vlenb =
        vlenb && *vlenb ? strtoull(vlenb, NULL, 0) : PIM_LAYOUT_COL_BYTES;
    const char *errors = getenv("PIM_REF_ERRORS");
    cfg.errors_fatal = !(errors && strcmp(errors, "count") == 0);
    if (pim_ref_attach(p, &cfg) != 0) {
        munmap(p, size);
        return pim__fail(PIM_E_DEVICE, "the reference model refused the "
                                       "layout");
    }
#else
    const char *path = device ? device : "/dev/mem";
    const uint64_t base = mode == PIM_MODE_SE ? PIM_LAYOUT_WINDOW_BASE_SE
                                              : PIM_LAYOUT_WINDOW_BASE_FS;
    const int fd = open(path, O_RDWR);
    if (fd < 0) {
        return pim__fail(PIM_E_DEVICE, "%s: %s", path, strerror(errno));
    }
    p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)base);
    const int err = errno;
    close(fd);
    if (p == MAP_FAILED) {
        return pim__fail(PIM_E_DEVICE, "mmap of %s at %#llx, %llu bytes: %s",
                         path, (unsigned long long)base,
                         (unsigned long long)size, strerror(err));
    }
#endif
    pim__g.base = (uint8_t *)p;
    pim__g.size = size;
    return PIM_OK;
}

void
pim__region_close(void)
{
#ifdef PIM_BACKEND_REF
    pim_ref_detach();
#endif
    if (pim__g.base) {
        munmap(pim__g.base, pim__g.size);
    }
    pim__g.base = NULL;
    pim__g.size = 0;
}

void *
pim_region_base(void)
{ return pim__g.initialized ? pim__g.base : NULL; }

uint64_t
pim_region_size(void)
{ return pim__g.initialized ? pim__g.size : 0; }
