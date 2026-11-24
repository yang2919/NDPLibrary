/*
 * Copy, sync and number formats on the host reference backend, for
 * the layout this program is built for (run for every policy).
 *
 * 1. Random data round-trip through pim_write_bank_row() and
 *    pim_read_bank_row(), and every byte lies at its encoder address.
 * 2. The PIM view: a bank row copied to the GB by PIM commands, piece by
 *    piece, reads back in column order (placement does not depend on the
 *    policy).
 * 3. The library's conversions equal the reference model's.
 * 4. The flush lists cover exactly the lines of the touched rows, each
 *    once; the syncs add the fences between PIM and normal accesses.
 */

#include <stdlib.h>
#include <string.h>

#include "pim_internal.h"
#include "pim_test.h"
#include "ref/pim_ref.h"

static uint64_t rng = 0x9e3779b97f4a7c15ULL;

static uint64_t
rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return rng;
}

static pim_coord_t
random_bank(uint32_t slices)
{
    const pim_geom_t *g = pim_geom();
    pim_coord_t c = {(uint32_t)(rnd() % g->channels),
                     (uint32_t)(rnd() % g->units),
                     (uint32_t)(rnd() % g->sids),
                     (uint32_t)(rnd() % g->bank_groups),
                     (uint32_t)(rnd() % g->banks_per_group),
                     (uint32_t)(rnd() % slices),
                     0};
    return c;
}

static void
check_round_trip(pim_buf_t *b)
{
    const size_t row = PIM_LAYOUT_ROW_BYTES;
    uint8_t *data = (uint8_t *)malloc(row);
    uint8_t *back = (uint8_t *)malloc(row);
    long bad_rt = 0, bad_place = 0;
    for (int t = 0; t < 300; t++) {
        const pim_coord_t c = random_bank(pim_buf_slices(b));
        const size_t off = rnd() % row;
        const size_t n = 1 + rnd() % (row - off);
        for (size_t i = 0; i < n; i++) {
            data[i] = (uint8_t)rnd();
        }
        CHECK(pim_write_bank_row(b, &c, off, data, n) == PIM_OK, "write");
        memset(back, 0, n);
        CHECK(pim_read_bank_row(b, &c, off, back, n) == PIM_OK, "read");
        bad_rt += memcmp(data, back, n) != 0;
        pim_coord_t a = c;
        a.row = pim_buf_row(b, c.row);
        for (size_t i = 0; i < n; i++) {
            a.col = (uint32_t)((off + i) / PIM_LAYOUT_COL_BYTES);
            const uint8_t *p = (const uint8_t *)pim_region_base() +
                               pim_encode(&a) +
                               (off + i) % PIM_LAYOUT_COL_BYTES;
            bad_place += *p != data[i];
        }
    }
    CHECK(bad_rt == 0, "%ld round trips differ", bad_rt);
    CHECK(bad_place == 0, "%ld bytes not at their encoder address", bad_place);
    /* Errors. */
    const pim_coord_t c = {0, 0, 0, 0, 0, 0, 0};
    pim_coord_t far = c;
    far.row = pim_buf_slices(b);
    CHECK(pim_write_bank_row(b, &c, row - 1, data, 2) == PIM_E_ARG,
          "past the row");
    CHECK(pim_write_bank_row(b, &far, 0, data, 1) == PIM_E_ARG,
          "slice outside the buffer");
    free(data);
    free(back);
}

static void
check_pim_view(pim_buf_t *b)
{
    const size_t row = PIM_LAYOUT_ROW_BYTES;
    const size_t vlmax = pim_ref_vlmax((size_t)-1);
    uint8_t *data = (uint8_t *)malloc(row);
    uint8_t *gb = (uint8_t *)malloc(row);
    long bad = 0;
    for (int t = 0; t < 8; t++) {
        const pim_coord_t c = random_bank(pim_buf_slices(b));
        for (size_t i = 0; i < row; i++) {
            data[i] = (uint8_t)rnd();
        }
        pim_write_bank_row(b, &c, 0, data, row);
        pim_sync_to_device(b);
        /* pim.sb.mov.bkgbuf per piece, at most VLMAX bytes each. */
        for (size_t off = 0; off < row;) {
            pim_coord_t p = c;
            p.col = (uint32_t)(off / PIM_LAYOUT_COL_BYTES);
            size_t n = PIM_LAYOUT_CONTIG_BYTES - off % PIM_LAYOUT_CONTIG_BYTES;
            n = n < vlmax ? n : vlmax;
            uint8_t *a =
                (uint8_t *)pim_buf_addr(b, &p) + off % PIM_LAYOUT_COL_BYTES;
            off += pim_sb_mov_bkgbuf(a, n);
        }
        /* pim.rd.gb back, in column order, piece by piece: under a fine
         * interleave the next piece of a GB is at another address. */
        for (size_t off = 0; off < row;) {
            size_t n = PIM_LAYOUT_CONTIG_BYTES - off % PIM_LAYOUT_CONTIG_BYTES;
            n = n < vlmax ? n : vlmax;
            uint8_t *a =
                (uint8_t *)pim_gb_addr(
                    c.ch, c.unit, (uint32_t)(off / PIM_LAYOUT_COL_BYTES)) +
                off % PIM_LAYOUT_COL_BYTES;
            off += pim_rd_gb_mem(a, gb + off, n);
        }
        bad += memcmp(gb, data, row) != 0;
    }
    CHECK(bad == 0, "%ld bank rows differ in the PIM's view", bad);
    CHECK(pim_ref_errors() == 0, "model errors %llu",
          (unsigned long long)pim_ref_errors());
    free(data);
    free(gb);
}

static uint32_t
f2u(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

static void
check_formats(void)
{
    long bad = 0;
    for (int t = 0; t < 200000; t++) {
        uint32_t u = (uint32_t)rnd();
        if (t % 4 == 0) {
            u = (u & 0xffff0000u) | (t & 0x8000 ? 0x8000u : 0x7fffu);
        }
        float f;
        memcpy(&f, &u, sizeof(f));
        uint8_t mine[4] = {0}, ref[4] = {0};
        pim_f32_to_elem(mine, f);
        pim_ref_store_f(PIM_LAYOUT_NUM_FMT, ref, f);
        if (PIM_LAYOUT_NUM_FMT <= PIM_FMT_FP32) {
            bad += memcmp(mine, ref, PIM_LAYOUT_ELEM_BYTES) != 0;
            const float a = pim_elem_to_f32(mine);
            const float r = pim_ref_load_f(PIM_LAYOUT_NUM_FMT, ref);
            bad += f2u(a) != f2u(r) && !(a != a && r != r);
        }
    }
    CHECK(bad == 0, "%ld conversions differ from the model", bad);
    CHECK(pim_f32_to_bf16(1.5f) == 0x3fc0 && pim_bf16_to_f32(0x3fc0) == 1.5f,
          "bf16 1.5");
}

/* Lines a set of coordinates covers, sorted and unique. */
static int
cmp_u(const void *a, const void *b)
{
    const uintptr_t x = *(const uintptr_t *)a, y = *(const uintptr_t *)b;
    return x < y ? -1 : x > y;
}

static size_t
expected_lines(const pim_buf_t *b, uint32_t slice, int ch, int unit,
               const pim_coord_t *bank, uintptr_t *out)
{
    const pim_geom_t *g = pim_geom();
    size_t n = 0;
    pim_coord_t c;
    for (c.ch = 0; c.ch < g->channels; c.ch++) {
        for (c.unit = 0; c.unit < g->units; c.unit++) {
            if (ch >= 0 &&
                (c.ch != (uint32_t)ch || c.unit != (uint32_t)unit)) {
                continue;
            }
            for (c.sid = 0; c.sid < g->sids; c.sid++) {
                for (c.bg = 0; c.bg < g->bank_groups; c.bg++) {
                    for (c.bank = 0; c.bank < g->banks_per_group; c.bank++) {
                        if (bank &&
                            (c.ch != bank->ch || c.unit != bank->unit ||
                             c.sid != bank->sid || c.bg != bank->bg ||
                             c.bank != bank->bank)) {
                            continue;
                        }
                        c.row = pim_buf_row(b, slice);
                        for (c.col = 0; c.col < g->cols; c.col++) {
                            out[n++] = ((uintptr_t)pim_region_base() +
                                        pim_encode(&c)) &
                                       ~(uintptr_t)(PIM_CACHE_LINE - 1);
                        }
                    }
                }
            }
        }
    }
    qsort(out, n, sizeof(*out), cmp_u);
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        if (k == 0 || out[k - 1] != out[i]) {
            out[k++] = out[i];
        }
    }
    return k;
}

/* The flush log equals the expected lines, each once. */
static void
check_log(const char *what, const uintptr_t *want, size_t nwant,
          uintptr_t *scratch)
{
    size_t n;
    const uintptr_t *log = pim_ref_flush_log(&n);
    memcpy(scratch, log, n * sizeof(*log));
    qsort(scratch, n, sizeof(*scratch), cmp_u);
    size_t dup = 0;
    for (size_t i = 1; i < n; i++) {
        dup += scratch[i] == scratch[i - 1];
    }
    CHECK(dup == 0, "%s: %zu lines flushed twice", what, dup);
    CHECK(n == nwant && memcmp(scratch, want, n * sizeof(*want)) == 0,
          "%s: %zu lines flushed, %zu expected", what, n, nwant);
    pim_ref_flush_log_clear();
}

static void
check_flushes(pim_buf_t *b)
{
    const pim_geom_t *g = pim_geom();
    const size_t max = (size_t)pim_geom_bank_rows(g) * g->cols * 3;
    uintptr_t *want = (uintptr_t *)malloc(max * sizeof(uintptr_t));
    uintptr_t *tmp = (uintptr_t *)malloc(max * sizeof(uintptr_t));
    uintptr_t *all = (uintptr_t *)malloc(max * sizeof(uintptr_t));
    size_t n;

    pim_ref_flush_log_clear();
    CHECK(pim_flush_rows(b, 1, 1) == PIM_OK, "flush rows");
    n = expected_lines(b, 1, -1, -1, NULL, want);
    check_log("one slice", want, n, tmp);

    const uint32_t ch = g->channels - 1, unit = g->units - 1;
    CHECK(pim_flush_unit_row(b, ch, unit, 2) == PIM_OK, "unit row");
    n = expected_lines(b, 2, (int)ch, (int)unit, NULL, want);
    check_log("one unit's row", want, n, tmp);

    pim_coord_t k = {0, 0, 0, 0, 0, 0, 0};
    for (int t = 0; t < 4; t++) {
        k = random_bank(pim_buf_slices(b));
        CHECK(pim_flush_bank_row(b, &k) == PIM_OK, "bank row");
        n = expected_lines(b, k.row, -1, -1, &k, want);
        check_log("one bank row", want, n, tmp);
    }

    /* The syncs: every slice, and the fences around them. */
    size_t nall = 0;
    for (uint32_t s = 0; s < pim_buf_slices(b); s++) {
        nall += expected_lines(b, s, -1, -1, NULL, all + nall);
    }
    qsort(all, nall, sizeof(*all), cmp_u);
    const uint64_t f0 = pim_ref_fences();
    CHECK(pim_sync_to_device(b) == PIM_OK, "sync to device");
    check_log("sync to device", all, nall, tmp);
    CHECK(pim_ref_fences() == f0 + 1, "one fence after the flushes");
    CHECK(pim_sync_from_device(b) == PIM_OK, "sync from device");
    check_log("sync from device", all, nall, tmp);
    CHECK(pim_ref_fences() == f0 + 3, "fences around the flushes");
    CHECK(pim_flush_rows(b, 2, 2) == PIM_E_ARG, "slices outside");
    free(want);
    free(tmp);
    free(all);
}

int
main(void)
{
    pim_buf_t *b;
    CHECK(pim_flush_rows(NULL, 0, 1) == PIM_E_STATE, "flush before init");
    if (pim_init(NULL) != PIM_OK || pim_buf_alloc(&b, 3) != PIM_OK) {
        printf("init: %s\n", pim_error_detail());
        return 1;
    }
    check_round_trip(b);
    check_pim_view(b);
    check_formats();
    check_flushes(b);
    pim_buf_free(b);
    pim_finalize();
    printf("layout std %d policy %d intlv %lld contig %d\n", PIM_LAYOUT_STD,
           PIM_LAYOUT_POLICY, (long long)PIM_LAYOUT_INTLV_BYTES,
           PIM_LAYOUT_CONTIG_BYTES);
    return pim_test_summary("test_copy");
}
