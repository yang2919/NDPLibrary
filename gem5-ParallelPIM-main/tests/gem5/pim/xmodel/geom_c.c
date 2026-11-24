/* Random geometries and offsets through pim_addr.h's pim_geom_*(): one line
 * per sample, "<geometry> <offset> <decoded coord> <byte> <re-encoded>". */
#include <stdio.h>
#include <stdlib.h>
#include <gem5/pim/pim_addr.h>
static uint32_t p2(unsigned lo, unsigned hi) { return 1u << (lo + rand() % (hi - lo + 1)); }
int main(void) {
    srand(7);
    for (int n = 0; n < 4000; n++) {
        pim_geom_t g;
        g.policy = rand() % 6; g.channels = p2(0, 3); g.units = p2(0, 2); g.sids = p2(0, 2);
        g.bank_groups = p2(0, 3); g.banks_per_group = p2(1, 3); g.rows = p2(4, 16);
        g.cols = p2(g.policy == 5 ? 2 : 0, 7); g.col_bytes = p2(4, 6);
        uint64_t row = (uint64_t)g.cols * g.col_bytes;
        g.intlv_bytes = (uint64_t)g.col_bytes << (rand() % 12);
        if (g.intlv_bytes > row * 4) g.intlv_bytes = row;
        uint64_t size = pim_geom_size(&g);
        for (int k = 0; k < 20; k++) {
            uint64_t off = (((uint64_t)rand() << 31) ^ rand()) % size;
            pim_coord_t c; uint32_t b = pim_geom_decode(&g, off, &c);
            printf("%u %u %llu %u %u %u %u %u %u %u %llu %u %u %u %u %u %u %u %u %llu\n", g.policy, g.channels,
                   (unsigned long long)g.intlv_bytes, g.units, g.sids, g.bank_groups, g.banks_per_group, g.rows,
                   g.cols, g.col_bytes, (unsigned long long)off, c.ch, c.unit, c.sid, c.bg, c.bank, c.row, c.col, b,
                   (unsigned long long)pim_geom_encode(&g, &c));
        }
    }
    return 0;
}
