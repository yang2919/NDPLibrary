/*
 * L4: the runtime (pim_runtime.h) on the host reference backend: the
 * mode and its guards with injected uname() and /proc/iomem contents,
 * initialization, the row allocator and the addresses.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "pim_internal.h"
#include "pim_test.h"

static const char iomem_fs[] = "00000000-00000fff : Reserved\n"
                               "02000000-0200ffff : riscv,clint0\n"
                               "10000000-10000007 : serial\n"
                               "80000000-47fffffff : System RAM\n"
                               "  80200000-8130ffff : Kernel code\n"
                               "  81310000-81bfffff : Kernel data\n";

static void
check_guards(void)
{
    const uint64_t base_fs = PIM_LAYOUT_WINDOW_BASE_FS;
    const uint64_t size = PIM_LAYOUT_WINDOW_SIZE;
    /* SE: the node name must be gem5's. */
    CHECK(pim__mode_check(PIM_MODE_SE, "sim.gem5.org", NULL, 0, size) ==
              PIM_OK,
          "SE accepted");
    CHECK(pim__mode_check(PIM_MODE_SE, "ubuntu", NULL, 0, size) == PIM_E_MODE,
          "SE refused on another node");
    CHECK(strstr(pim_error_detail(), "PIM_MODE=fs") != NULL,
          "the message names PIM_MODE: %s", pim_error_detail());
    /* FS: never on gem5's SE node name. */
    CHECK(pim__mode_check(PIM_MODE_FS, "sim.gem5.org", iomem_fs, base_fs,
                          size) == PIM_E_MODE,
          "FS refused on the SE node name");
    CHECK(strstr(pim_error_detail(), "PIM_MODE=se") != NULL, "%s",
          pim_error_detail());
    /* FS: OS RAM below the PIM memory is fine ... */
    char iomem[1024];
    snprintf(iomem, sizeof(iomem),
             "80000000-%llx : System RAM\n  80200000-8130ffff : Kernel "
             "code\n",
             (unsigned long long)(base_fs - 1));
    CHECK(pim__mode_check(PIM_MODE_FS, "ubuntu", iomem, base_fs, size) ==
              PIM_OK,
          "FS accepted: %s", pim_error_detail());
    /* ... RAM overlapping it is refused (the SE base in an FS run) ... */
    snprintf(iomem, sizeof(iomem), "80000000-%llx : System RAM\n",
             (unsigned long long)(base_fs + 0x1000));
    CHECK(pim__mode_check(PIM_MODE_FS, "ubuntu", iomem, base_fs, size) ==
              PIM_E_MODE,
          "FS refused: RAM overlaps");
    snprintf(iomem, sizeof(iomem),
             "80000000-%llx : System RAM\n%llx-%llx : System RAM\n",
             (unsigned long long)(base_fs - 1),
             (unsigned long long)(base_fs + size - 1),
             (unsigned long long)(base_fs + size + 0xfff));
    CHECK(pim__mode_check(PIM_MODE_FS, "ubuntu", iomem, base_fs, size) ==
              PIM_E_MODE,
          "FS refused: RAM at the last byte");
    snprintf(iomem, sizeof(iomem),
             "80000000-%llx : System RAM\n%llx-%llx : Reserved\n",
             (unsigned long long)(base_fs - 1), (unsigned long long)base_fs,
             (unsigned long long)(base_fs + size - 1));
    CHECK(pim__mode_check(PIM_MODE_FS, "ubuntu", iomem, base_fs, size) ==
              PIM_OK,
          "FS accepted: a reserved range is not RAM");
    /* ... and hidden addresses (not root) or no RAM line are refused. */
    CHECK(pim__mode_check(PIM_MODE_FS, "ubuntu",
                          "00000000-00000000 : System RAM\n", base_fs,
                          size) == PIM_E_MODE,
          "FS refused: addresses hidden");
    CHECK(pim__mode_check(PIM_MODE_FS, "ubuntu", "", base_fs, size) ==
              PIM_E_MODE,
          "FS refused: no RAM line");
    CHECK(pim__mode_check(PIM_MODE_FS, "ubuntu", NULL, base_fs, size) ==
              PIM_E_MODE,
          "FS refused: no /proc/iomem");
}

static void
check_mode_resolution(void)
{
    pim_mode_t m;
    pim_config_t cfg = PIM_CONFIG_INIT;
    CHECK(pim__resolve_mode(NULL, NULL, &m) == PIM_OK && m == PIM_DEFAULT_MODE,
          "default");
    CHECK(pim__resolve_mode(&cfg, "fs", &m) == PIM_OK && m == PIM_MODE_FS,
          "PIM_MODE=fs");
    CHECK(pim__resolve_mode(&cfg, "se", &m) == PIM_OK && m == PIM_MODE_SE,
          "PIM_MODE=se");
    cfg.mode = PIM_MODE_SE;
    CHECK(pim__resolve_mode(&cfg, "fs", &m) == PIM_OK && m == PIM_MODE_SE,
          "the configuration overrides PIM_MODE");
    cfg.mode = PIM_MODE_DEFAULT;
    CHECK(pim__resolve_mode(&cfg, "both", &m) == PIM_E_ARG, "bad PIM_MODE");
}

static void
check_init_and_state(void)
{
    pim_buf_t *b = NULL;
    CHECK(pim_buf_alloc(&b, 1) == PIM_E_STATE, "alloc before init");
    CHECK(pim_region_base() == NULL, "no region before init");
    pim_config_t cfg = PIM_CONFIG_INIT;
    cfg.way = 2;
    CHECK(pim_init(&cfg) == PIM_E_ARG, "Way 2 not implemented");
    unsetenv("PIM_MODE");
    CHECK(pim_init(NULL) == PIM_OK, "init: %s", pim_error_detail());
    CHECK(pim_init(NULL) == PIM_E_STATE, "init twice");
    CHECK(pim_mode() == PIM_DEFAULT_MODE, "default mode");
    CHECK(pim_region_size() == PIM_LAYOUT_WINDOW_SIZE, "region size");
    CHECK(pim_region_phys() == PIM_LAYOUT_WINDOW_BASE_SE, "SE base");
    pim_finalize();
    CHECK(pim_buf_alloc(&b, 1) == PIM_E_STATE, "alloc after finalize");
    /* PIM_MODE=fs selects the FS base. */
    setenv("PIM_MODE", "fs", 1);
    CHECK(pim_init(NULL) == PIM_OK, "init FS: %s", pim_error_detail());
    CHECK(pim_mode() == PIM_MODE_FS &&
              pim_region_phys() == PIM_LAYOUT_WINDOW_BASE_FS,
          "FS base");
    pim_finalize();
    setenv("PIM_MODE", "x", 1);
    CHECK(pim_init(NULL) == PIM_E_ARG, "bad PIM_MODE refused");
    unsetenv("PIM_MODE");
    /* A file-backed region (PIM_REF_FILE). */
    char path[128];
    snprintf(path, sizeof(path), "build/native/test/ref_region_%d.bin",
             (int)getpid());
    setenv("PIM_REF_FILE", path, 1);
    CHECK(pim_init(NULL) == PIM_OK, "file region: %s", pim_error_detail());
    ((volatile uint8_t *)pim_region_base())[12345] = 7;
    pim_finalize();
    unsetenv("PIM_REF_FILE");
    FILE *f = fopen(path, "rb");
    int byte = -1;
    if (f) {
        fseek(f, 12345, SEEK_SET);
        byte = fgetc(f);
        fclose(f);
    }
    CHECK(byte == 7, "the file holds the region's bytes");
    unlink(path);
    /* Every status has its own text. */
    for (int i = PIM_OK; i <= PIM_E_STATE; i++) {
        for (int k = i + 1; k <= PIM_E_STATE; k++) {
            CHECK(strcmp(pim_strerror((pim_status_t)i),
                         pim_strerror((pim_status_t)k)) != 0,
                  "texts %d %d", i, k);
        }
    }
}

static void
check_allocator(void)
{
    CHECK(pim_init(NULL) == PIM_OK, "init: %s", pim_error_detail());
    const uint32_t usable = PIM_LAYOUT_REG_ROW0;
    CHECK(pim_free_slices() == usable, "free %u", pim_free_slices());
    pim_buf_t *a, *b, *c, *d;
    CHECK(pim_buf_alloc(&a, 3) == PIM_OK, "a");
    CHECK(pim_buf_alloc(&b, 5) == PIM_OK, "b");
    CHECK(pim_buf_alloc(&c, 2) == PIM_OK, "c");
    CHECK(pim_buf_alloc(&d, 0) == PIM_E_ARG, "zero slices");
    /* No overlap, contiguous first fit. */
    unsigned char *seen = (unsigned char *)calloc(PIM_LAYOUT_ROWS, 1);
    const pim_buf_t *bufs[3] = {a, b, c};
    int overlap = 0;
    for (int i = 0; i < 3; i++) {
        for (uint32_t s = 0; s < pim_buf_slices(bufs[i]); s++) {
            overlap += seen[pim_buf_row(bufs[i], s)]++ != 0;
        }
    }
    CHECK(!overlap, "buffers overlap");
    CHECK(pim_buf_row(b, 0) == 3 && pim_buf_row(b, 4) == 7, "first fit");
    /* A freed hole is reused; a larger request goes past it. */
    pim_buf_free(b);
    CHECK(pim_buf_alloc(&b, 6) == PIM_OK && pim_buf_row(b, 0) == 10,
          "6 slices after the hole");
    CHECK(pim_buf_alloc(&d, 5) == PIM_OK && pim_buf_row(d, 0) == 3,
          "the hole reused");
    /* Exhaustion: every usable row, never a register row. */
    const uint32_t left = pim_free_slices();
    pim_buf_t *rest;
    CHECK(pim_buf_alloc(&rest, left + 1) == PIM_E_NOMEM, "exhaustion");
    CHECK(pim_buf_alloc(&rest, left) == PIM_OK, "the rest");
    CHECK(pim_free_slices() == 0, "none left");
    pim_buf_t *more;
    CHECK(pim_buf_alloc(&more, 1) == PIM_E_NOMEM, "nothing left");
    int reg_rows = 0;
    for (uint32_t s = 0; s < pim_buf_slices(rest); s++) {
        reg_rows += pim_buf_row(rest, s) >= PIM_LAYOUT_REG_ROW0;
    }
    CHECK(reg_rows == 0, "register rows handed out");
    /* Scattered allocation when no run fits. */
    pim_buf_free(a);
    pim_buf_free(c);
    CHECK(pim_buf_alloc(&a, 5) == PIM_OK, "scattered 3 + 2 slices");
    CHECK(pim_buf_row(a, 0) == 0 && pim_buf_row(a, 2) == 2 &&
              pim_buf_row(a, 3) == 8 && pim_buf_row(a, 4) == 9,
          "scattered rows %u %u %u %u %u", pim_buf_row(a, 0),
          pim_buf_row(a, 1), pim_buf_row(a, 2), pim_buf_row(a, 3),
          pim_buf_row(a, 4));
    /* Addresses: region base + encode(absolute row), the window rows. */
    const pim_coord_t k = {
        PIM_LAYOUT_CHANNELS - 1, PIM_LAYOUT_UNITS - 1, 0, 0, 1, 4,
        PIM_LAYOUT_COLS - 1};
    pim_coord_t abs = k;
    abs.row = pim_buf_row(a, 4);
    CHECK((uint8_t *)pim_buf_addr(a, &k) ==
              (uint8_t *)pim_region_base() + pim_encode(&abs),
          "buffer address");
    pim_coord_t out = k;
    out.row = 5;
    CHECK(pim_buf_addr(a, &out) == NULL, "slice outside the buffer");
    abs.row = PIM_LAYOUT_REG_ROW0;
    CHECK((uint8_t *)pim_reg_addr(&k, 0) ==
              (uint8_t *)pim_region_base() + pim_encode(&abs),
          "register 0 address");
    CHECK(pim_reg_addr(&k, PIM_LAYOUT_NUM_REGS) == NULL,
          "register NUM_REGS refused");
    abs.row = PIM_LAYOUT_REG_ROW0 + PIM_LAYOUT_NUM_REGS;
    CHECK((uint8_t *)pim_acc_addr(&k) ==
              (uint8_t *)pim_region_base() + pim_encode(&abs),
          "accumulator row address");
    const pim_coord_t gb = {1 % PIM_LAYOUT_CHANNELS, 0, 0, 0, 0,
                            PIM_LAYOUT_REG_ROW0,     3};
    CHECK((uint8_t *)pim_gb_addr(gb.ch, 0, 3) ==
              (uint8_t *)pim_region_base() + pim_encode(&gb),
          "GB address");
    pim_buf_free(a);
    pim_buf_free(b);
    pim_buf_free(d);
    pim_buf_free(rest);
    CHECK(pim_free_slices() == usable, "all freed: %u", pim_free_slices());
    free(seen);
    pim_finalize();
}

int
main(void)
{
    check_guards();
    check_mode_resolution();
    check_init_and_state();
    check_allocator();
    return pim_test_summary("test_runtime");
}
