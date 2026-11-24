/*
 * pim_addr.h as gem5's C++ would use it, compiled with
 * gem5's warning flags and -Werror: the geometry functions alone, with
 * values gem5 knows at run time, and the folded functions of a layout.
 */

#include <cstdio>

#include <gem5/pim/pim_addr.h>
#include <gem5/pim/pim_layout.h>

int
main()
{
    // LPDDR6 x 8 under RoRaBaChCo, as gem5 would fill it in.
    pim_geom_t g = {PIM_POLICY_RoRaBaChCo, 8, 2048, 1, 1, 4, 4, 65536, 64, 32};
    pim_coord_t c = {5, 0, 0, 3, 2, 1234, 17};
    const uint64_t off = pim_geom_encode(&g, &c);
    pim_coord_t d;
    const uint32_t byte = pim_geom_decode(&g, off + 7, &d);
    const bool ok = byte == 7 && d.ch == 5 && d.bg == 3 && d.bank == 2 &&
                    d.row == 1234 && d.col == 17 &&
                    pim_encode(&c) == pim_geom_encode(pim_geom(), &c);
    std::printf("C++: off %#llx %s\n", (unsigned long long)off,
                ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
