/* See lto_a.c. */
#include <stddef.h>
#include <stdint.h>

#include <gem5/pim/pim_isa.h>

size_t lto_move(void *p);

int
main(void)
{
    static char buf[64];
    pim_ab_mac(buf, 32);
    return (int)lto_move(buf);
}
