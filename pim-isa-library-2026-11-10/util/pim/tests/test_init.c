/*
 * L4: pim_init() on the guest (RISC-V), for a manual run in gem5:
 *   test_init [se|fs]
 * prints the status and its detail. Until the emulated PIM driver
 * exists (P3 step R7), SE mode passes the mode guard and stops at
 * /dev/mem (PIM_E_DEVICE); FS mode is refused by the node-name guard
 * (PIM_E_MODE).
 */

#include <stdio.h>
#include <string.h>

#include <gem5/pim/pim.h>

int
main(int argc, char **argv)
{
    pim_config_t cfg = PIM_CONFIG_INIT;
    if (argc > 1) {
        cfg.mode = strcmp(argv[1], "fs") == 0 ? PIM_MODE_FS : PIM_MODE_SE;
    }
    const pim_status_t s = pim_init(&cfg);
    printf("pim_init: %d (%s): %s\n", (int)s, pim_strerror(s),
           pim_error_detail());
    if (s == PIM_OK) {
        printf("region %p, %llu bytes, physical base %#llx\n",
               pim_region_base(), (unsigned long long)pim_region_size(),
               (unsigned long long)pim_region_phys());
        pim_finalize();
    }
    return 0;
}
