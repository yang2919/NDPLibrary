/*
 * L1: the PIM address encoder and decoder (pim_addr.h).
 *
 * 1. Round trip over every coordinate of small organizations, for every
 *    policy, channel count and interleave: decode(encode(c)) == c, every
 *    offset reached once, address-contiguous pieces, row slices.
 * 2. The layout this program is built for (PIM_STD): the folded
 *    pim_encode() equals pim_geom_encode() and the header constants, and
 *    every coordinate of sampled rows round-trips.
 * 3. Optional: the vectors of ref_decode.py (gem5's and Ramulator 2.1's
 *    decoders, transcribed), for 4 standards x 6 policies x interleaves.
 *
 * Usage: test_addr [vectors file]
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gem5/pim/pim_addr.h>
#include <gem5/pim/pim_layout.h>

#include "pim_test.h"

static int
coord_eq(const pim_coord_t *a, const pim_coord_t *b)
{
    return a->ch == b->ch && a->unit == b->unit && a->sid == b->sid &&
           a->bg == b->bg && a->bank == b->bank && a->row == b->row &&
           a->col == b->col;
}

static void
print_geom(const pim_geom_t *g, char *buf, size_t n)
{
    snprintf(buf, n,
             "policy %u ch %u intlv %llu units %u sids %u bgs %u banks %u "
             "rows %u cols %u col_bytes %u",
             g->policy, g->channels, (unsigned long long)g->intlv_bytes,
             g->units, g->sids, g->bank_groups, g->banks_per_group, g->rows,
             g->cols, g->col_bytes);
}

/* Visit every coordinate of a geometry. */
#define FOR_EACH_COORD(g, c)                                                  \
    for ((c).ch = 0; (c).ch < (g)->channels; (c).ch++)                        \
        for ((c).unit = 0; (c).unit < (g)->units; (c).unit++)                 \
            for ((c).sid = 0; (c).sid < (g)->sids; (c).sid++)                 \
                for ((c).bg = 0; (c).bg < (g)->bank_groups; (c).bg++)         \
                    for ((c).bank = 0; (c).bank < (g)->banks_per_group;       \
                         (c).bank++)                                          \
                        for ((c).row = 0; (c).row < (g)->rows; (c).row++)     \
                            for ((c).col = 0; (c).col < (g)->cols; (c).col++)

/* Every coordinate, every offset, pieces and slices of a small geometry. */
static void
check_exhaustive(const pim_geom_t *g)
{
    char name[256];
    print_geom(g, name, sizeof(name));
    const uint64_t size = pim_geom_size(g);
    const uint64_t ntx = size / g->col_bytes;
    unsigned char *seen = (unsigned char *)calloc(ntx, 1);
    const uint64_t contig = pim_geom_contig_bytes(g);
    const unsigned shift = pim_geom_row_shift(g);
    const int slice_contig = pim_geom_slice_contiguous(g);
    uint64_t *row_min = (uint64_t *)malloc(g->rows * sizeof(uint64_t));
    uint64_t *row_max = (uint64_t *)calloc(g->rows, sizeof(uint64_t));
    for (uint32_t r = 0; r < g->rows; r++) {
        row_min[r] = ~0ULL;
    }
    pim_coord_t c, d;
    long bad = 0;
    FOR_EACH_COORD(g, c)
    {
        const uint64_t off = pim_geom_encode(g, &c);
        if (off >= size || off % g->col_bytes) {
            bad++;
            continue;
        }
        if (seen[off / g->col_bytes]++) {
            bad++;
        }
        const uint32_t byte = pim_geom_decode(g, off + g->col_bytes - 1, &d);
        if (!coord_eq(&c, &d) || byte != g->col_bytes - 1) {
            bad++;
        }
        /* The piece holding this column is address-contiguous. */
        const uint32_t per_piece = (uint32_t)(contig / g->col_bytes);
        const uint64_t start = pim_geom_piece_offset(g, &c, c.col / per_piece);
        if (off != start + (uint64_t)(c.col % per_piece) * g->col_bytes ||
            start % contig) {
            bad++;
        }
        if (off < row_min[c.row]) {
            row_min[c.row] = off;
        }
        if (off > row_max[c.row]) {
            row_max[c.row] = off;
        }
    }
    CHECK(bad == 0, "%s: %ld coordinate errors", name, bad);
    uint64_t missing = 0;
    for (uint64_t i = 0; i < ntx; i++) {
        missing += seen[i] != 1;
    }
    CHECK(missing == 0, "%s: %llu offsets not reached once", name,
          (unsigned long long)missing);
    /* Row slices: contiguous exactly when pim_geom_slice_contiguous(). */
    int brute = 1;
    for (uint32_t r = 0; r < g->rows; r++) {
        const uint64_t span = row_max[r] + g->col_bytes - row_min[r];
        if (span != pim_geom_slice_bytes(g) ||
            row_min[r] != ((uint64_t)r << shift)) {
            brute = 0;
        }
    }
    CHECK(brute == slice_contig, "%s: slice contiguous %d, brute force %d",
          name, slice_contig, brute);
    if (slice_contig) {
        CHECK(pim_geom_slice_offset(g, 3 % g->rows) == row_min[3 % g->rows],
              "%s: slice offset", name);
    }
    free(seen);
    free(row_min);
    free(row_max);
}

static void
check_small(void)
{
    static const uint32_t cols_opts[] = {4, 8};
    long configs = 0;
    for (uint32_t policy = 0; policy <= 5; policy++) {
        for (uint32_t ch = 1; ch <= 4; ch *= 2) {
            for (uint32_t units = 1; units <= 2; units *= 2) {
                for (uint32_t sids = 1; sids <= 2; sids *= 2) {
                    for (uint32_t bgs = 1; bgs <= 2; bgs *= 2) {
                        for (uint32_t banks = 2; banks <= 4; banks *= 2) {
                            for (int ci = 0; ci < 2; ci++) {
                                const uint32_t cols = cols_opts[ci];
                                pim_geom_t g = {policy, ch,    0, units, sids,
                                                bgs,    banks, 8, cols,  32};
                                const uint64_t row = pim_geom_row_bytes(&g);
                                const uint64_t cap = pim_geom_size(&g) / ch;
                                for (uint64_t i = 32; i <= 2 * row; i *= 2) {
                                    g.intlv_bytes = i;
                                    if (policy == PIM_POLICY_RoRaBaChCo) {
                                        g.intlv_bytes = row;
                                    } else if (policy ==
                                               PIM_POLICY_ChRaBaRoCo) {
                                        g.intlv_bytes = cap;
                                    } else if ((policy ==
                                                    PIM_POLICY_RoRaBaCoCh ||
                                                policy ==
                                                    PIM_POLICY_RoCoRaBaCh) &&
                                               i > row) {
                                        continue;
                                    }
                                    check_exhaustive(&g);
                                    configs++;
                                    if (g.intlv_bytes != i) {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    printf("small organizations: %ld configurations\n", configs);
}

/* The included layout: constants, folded and run-time encoders. */
static void
check_default_layout(void)
{
    const pim_geom_t *g = pim_geom();
    /* A copy the compiler cannot fold, for the run-time path. */
    volatile pim_geom_t vg = *g;
    pim_geom_t rg;
    memcpy(&rg, (const void *)&vg, sizeof(rg));

    CHECK(pim_geom_size(g) == PIM_LAYOUT_WINDOW_SIZE, "size");
    CHECK(pim_geom_row_bytes(g) == PIM_LAYOUT_ROW_BYTES, "row bytes");
    CHECK(pim_geom_banks_per_unit(g) == PIM_LAYOUT_BANKS_PER_UNIT, "banks");
    CHECK(pim_geom_contig_bytes(&rg) == PIM_LAYOUT_CONTIG_BYTES, "contig %llu",
          (unsigned long long)pim_geom_contig_bytes(&rg));
    CHECK(pim_geom_row_shift(&rg) == PIM_LAYOUT_ROW_SHIFT, "row shift %u",
          pim_geom_row_shift(&rg));

    /* Each field's bit 0 is at its *_SHIFT (fields with bits only). */
    pim_coord_t one;
    memset(&one, 0, sizeof(one));
    one.col = 1;
    CHECK(pim_encode(&one) == 1ULL << PIM_LAYOUT_COL_SHIFT, "col shift");
    memset(&one, 0, sizeof(one));
    if (PIM_LAYOUT_CHANNELS > 1) {
        one.ch = 1;
        CHECK(pim_encode(&one) == 1ULL << PIM_LAYOUT_CH_SHIFT, "ch shift");
        memset(&one, 0, sizeof(one));
    }
    if (PIM_LAYOUT_UNITS > 1) {
        one.unit = 1;
        CHECK(pim_encode(&one) == 1ULL << PIM_LAYOUT_UNIT_SHIFT, "unit");
        memset(&one, 0, sizeof(one));
    }
    if (PIM_LAYOUT_SIDS > 1) {
        one.sid = 1;
        CHECK(pim_encode(&one) == 1ULL << PIM_LAYOUT_SID_SHIFT, "sid");
        memset(&one, 0, sizeof(one));
    }
    if (PIM_LAYOUT_BANK_GROUPS > 1) {
        one.bg = 1;
        CHECK(pim_encode(&one) == 1ULL << PIM_LAYOUT_BG_SHIFT, "bg");
        memset(&one, 0, sizeof(one));
    }
    one.bank = 1;
    CHECK(pim_encode(&one) == 1ULL << PIM_LAYOUT_BANK_SHIFT, "bank");

    /* Every coordinate of sampled rows. */
    const uint32_t rows[] = {0, 1, PIM_LAYOUT_ROWS / 2 + 3,
                             PIM_LAYOUT_REG_ROW0,
                             PIM_LAYOUT_REG_ROW0 + PIM_LAYOUT_NUM_REGS};
    long bad = 0, n = 0;
    pim_coord_t c, d;
    for (unsigned ri = 0; ri < sizeof(rows) / sizeof(rows[0]); ri++) {
        for (c.ch = 0; c.ch < g->channels; c.ch++) {
            for (c.unit = 0; c.unit < g->units; c.unit++) {
                for (c.sid = 0; c.sid < g->sids; c.sid++) {
                    for (c.bg = 0; c.bg < g->bank_groups; c.bg++) {
                        for (c.bank = 0; c.bank < g->banks_per_group;
                             c.bank++) {
                            c.row = rows[ri];
                            for (c.col = 0; c.col < g->cols; c.col++) {
                                const uint64_t off = pim_encode(&c);
                                n++;
                                if (off != pim_geom_encode(&rg, &c) ||
                                    off >= PIM_LAYOUT_WINDOW_SIZE) {
                                    bad++;
                                    continue;
                                }
                                if (pim_decode(off + 1, &d) != 1 ||
                                    !coord_eq(&c, &d)) {
                                    bad++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    /* Every row of every bank, first and last column. */
    for (c.row = 0; c.row < g->rows; c.row++) {
        for (c.ch = 0; c.ch < g->channels; c.ch++) {
            for (c.unit = 0; c.unit < g->units; c.unit++) {
                for (c.sid = 0; c.sid < g->sids; c.sid++) {
                    for (c.bg = 0; c.bg < g->bank_groups; c.bg++) {
                        for (c.bank = 0; c.bank < g->banks_per_group;
                             c.bank++) {
                            for (c.col = 0; c.col < g->cols;
                                 c.col += g->cols - 1) {
                                const uint64_t off = pim_encode(&c);
                                n++;
                                if (off >= PIM_LAYOUT_WINDOW_SIZE ||
                                    pim_decode(off, &d) != 0 ||
                                    !coord_eq(&c, &d)) {
                                    bad++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    CHECK(bad == 0, "%ld of %ld default-layout coordinates wrong", bad, n);
    printf("default layout (std %d, policy %d): %ld coordinates\n",
           PIM_LAYOUT_STD, PIM_LAYOUT_POLICY, n);
}

/* The vectors of ref_decode.py. */
static int
check_vectors(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        perror(path);
        return 1;
    }
    char line[512];
    pim_geom_t g;
    memset(&g, 0, sizeof(g));
    char name[256] = "";
    long configs = 0, vectors = 0, bad = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') {
            continue;
        }
        if (strncmp(line, "geom ", 5) == 0) {
            char std[16];
            unsigned long long intlv, contig;
            unsigned row_shift;
            int n =
                sscanf(line + 5,
                       "%15s %u %u %llu %u %u %u %u %u %u %u "
                       "%llu %u",
                       std, &g.policy, &g.channels, &intlv, &g.units, &g.sids,
                       &g.bank_groups, &g.banks_per_group, &g.rows, &g.cols,
                       &g.col_bytes, &contig, &row_shift);
            CHECK(n == 13, "bad geom line %s", line);
            g.intlv_bytes = intlv;
            snprintf(name, sizeof(name), "%s policy %u ch %u intlv %llu", std,
                     g.policy, g.channels, intlv);
            CHECK(pim_geom_contig_bytes(&g) == contig,
                  "%s: contig %llu, "
                  "generator %llu",
                  name, (unsigned long long)pim_geom_contig_bytes(&g), contig);
            CHECK(pim_geom_row_shift(&g) == row_shift,
                  "%s: row shift %u, generator %u", name,
                  pim_geom_row_shift(&g), row_shift);
            configs++;
            continue;
        }
        unsigned long long off;
        pim_coord_t e, d;
        unsigned byte;
        int n = sscanf(line, "%llx %u %u %u %u %u %u %u %u", &off, &e.ch,
                       &e.unit, &e.sid, &e.bg, &e.bank, &e.row, &e.col, &byte);
        if (n != 9) {
            continue;
        }
        vectors++;
        const uint32_t b = pim_geom_decode(&g, off, &d);
        const uint64_t enc = pim_geom_encode(&g, &e);
        if (!coord_eq(&d, &e) || b != byte || enc != off - byte) {
            if (bad++ < 10) {
                fprintf(stderr,
                        "%s: off %#llx: reference ch %u unit %u sid %u bg "
                        "%u bank %u row %u col %u byte %u; decoded ch %u "
                        "unit %u sid %u bg %u bank %u row %u col %u byte "
                        "%u; encoded %#llx\n",
                        name, off, e.ch, e.unit, e.sid, e.bg, e.bank, e.row,
                        e.col, byte, d.ch, d.unit, d.sid, d.bg, d.bank, d.row,
                        d.col, b, (unsigned long long)enc);
            }
        }
    }
    fclose(f);
    CHECK(bad == 0, "%ld of %ld reference vectors differ", bad, vectors);
    CHECK(configs > 0 && vectors > 0, "no vectors in %s", path);
    printf("reference vectors: %ld configurations, %ld vectors\n", configs,
           vectors);
    return 0;
}

int
main(int argc, char **argv)
{
    check_default_layout();
    if (argc > 1) {
        check_small();
        if (check_vectors(argv[1])) {
            return 1;
        }
    }
    return pim_test_summary("test_addr");
}
