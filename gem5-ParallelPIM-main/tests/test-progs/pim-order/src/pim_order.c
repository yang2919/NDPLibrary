/*
 * Program order of processing-in-memory (PIM) writes.
 *
 *   pim-order <warm|cold> [iterations]
 *
 * Each iteration writes the pattern g1 to the global buffer (GB) of
 * unit 0, fences, moves the GB to a bank row (pim.sb.mov.gbufbk), writes
 * another pattern g2 to the GB, fences, flushes and reads the row back:
 * the row must hold g1. "warm" loads the row's line with a normal load
 * before the move, so that the move evicts a valid copy and a cache
 * delays it behind the eviction; a later PIM write must still not
 * overtake it. "cold" flushes the line after each read. Prints
 * "PIM-ORDER <mode>: <wrong> of <n> wrong" and exits with 1 if any
 * iteration is wrong, 2 on a library error.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gem5/pim/pim.h>

int
main(int argc, char **argv)
{
    if (argc < 2 || (strcmp(argv[1], "warm") && strcmp(argv[1], "cold"))) {
        fprintf(stderr, "usage: pim-order <warm|cold> [iterations]\n");
        return 2;
    }
    const int warm = strcmp(argv[1], "warm") == 0;
    const int n = argc > 2 ? atoi(argv[2]) : 64;
    pim_status_t st = pim_init(NULL);
    if (st != PIM_OK) {
        fprintf(stderr, "pim_init: %s %s\n", pim_strerror(st),
                pim_error_detail());
        return 2;
    }
    pim_buf_t *buf;
    st = pim_buf_alloc(&buf, 1);
    if (st != PIM_OK) {
        fprintf(stderr, "pim_buf_alloc: %s\n", pim_strerror(st));
        return 2;
    }
    pim_coord_t c;
    memset(&c, 0, sizeof(c));
    volatile uint8_t *row = pim_buf_addr(buf, &c);
    void *gb = pim_gb_addr(0, 0, 0);
    enum { COL = PIM_LAYOUT_COL_BYTES };
    uint8_t g1[COL], g2[COL], out[COL];
    int wrong = 0;

    pim_flush_range((const void *)row, COL);
    pim_fence();
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < COL; k++) {
            g1[k] = (uint8_t)(i * 7 + k + 1);
            g2[k] = (uint8_t)~g1[k];
        }
        pim_wr_gb_mem(gb, g1, COL);
        pim_fence();
        if (warm)
            (void)*(volatile uint64_t *)row;
        pim_sb_mov_gbufbk((void *)row, COL);
        pim_wr_gb_mem(gb, g2, COL);
        pim_fence();
        pim_flush_range((const void *)row, COL);
        pim_fence();
        for (int k = 0; k < COL; k++)
            out[k] = row[k];
        wrong += memcmp(out, g1, COL) != 0;
        pim_flush_range((const void *)row, COL);
        pim_fence();
    }
    printf("PIM-ORDER %s: %d of %d wrong\n", argv[1], wrong, n);
    pim_buf_free(buf);
    pim_finalize();
    return wrong != 0;
}
