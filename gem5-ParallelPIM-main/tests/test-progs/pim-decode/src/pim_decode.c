/*
 * Driver of the processing-in-memory (PIM) decode test.
 *
 *   pim-decode scalar|vstore|vload|all
 *   pim-decode bad <n>      one reserved encoding (0-15)
 *   pim-decode guard <n>    an address guard (0: misaligned scalar,
 *                           1: vector page crossing)
 *
 * The PIM addresses point into pim_buf, a 4 KiB-aligned .bss buffer that
 * the program never touches with normal loads or stores, so no cached
 * copy exists. Before a PIM memory exists the requests reach ordinary
 * memory as uncacheable reads and writes, so nothing checks contents.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pim_insn.h"

void pim_scalar(void *buf, unsigned long desc);
void pim_vstore(void *buf, void *reg);
void pim_vmulti(void *buf);
void pim_vsplit(void *addr);
void pim_vload(void *buf, void *reg);
void pim_bad(long n, void *buf);
void pim_guard(long n, void *buf, unsigned long desc);

static unsigned char pim_buf[3 * 4096] __attribute__((aligned(4096)));

static void
marker(const char *group, const char *what)
{
    printf("PIM-DECODE %s %s\n", what, group);
    fflush(stdout);
}

#define GROUP(name, call)                                                     \
    do {                                                                      \
        marker(name, "begin");                                                \
        call;                                                                 \
        marker(name, "end");                                                  \
    } while (0)

int
main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s scalar|vstore|vload|all|bad n|guard n\n",
                argv[0]);
        return 2;
    }
    const char *mode = argv[1];
    printf("PIM-DECODE pim_buf %p\n", (void *)pim_buf);
    const int all = !strcmp(mode, "all");
    if (all || !strcmp(mode, "scalar"))
        GROUP("scalar", pim_scalar(pim_buf, PIM_DESC(4)));
    if (all || !strcmp(mode, "vstore")) {
        GROUP("vstore", pim_vstore(pim_buf, pim_buf));
        GROUP("vmulti", pim_vmulti(pim_buf));
        GROUP("vsplit", pim_vsplit(pim_buf + 48));
    }
    if (all || !strcmp(mode, "vload"))
        GROUP("vload", pim_vload(pim_buf, pim_buf));
    if (!strcmp(mode, "bad") && argc > 2)
        GROUP("bad", pim_bad(atol(argv[2]), pim_buf));
    if (!strcmp(mode, "guard") && argc > 2)
        GROUP("guard", pim_guard(atol(argv[2]), pim_buf, PIM_DESC(1)));
    printf("PIM-DECODE done %s\n", mode);
    return 0;
}
