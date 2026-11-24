// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include "../common/dtype.h"
#include "../common/kio.h"
#include "../common/bench_print.h"

#define HB     ((volatile uint32_t *)0x40100000)
#define VERIFY ((volatile uint32_t *)0x40100100)

#define LEN 1024

real_t V[LEN] __attribute__((aligned(32)));

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;
    uart_puts("\n[vfredsum " DTYPE_NAME "] N=1024 reduction sum\n");

    for (int i = 0; i < LEN; i++) V[i] = (real_t)(((float)((i % 31) - 15)) * 0.015625f);
    HB[0] = 0xCAFE0003;

    uint64_t t0 = rdcycle();
    real_t sum = r_redsum(V, LEN);
    uint64_t t1 = rdcycle();
    HB[0] = 0xCAFE0004;

    double refd = 0; for (int i = 0; i < LEN; i++) refd += (double)V[i];
    HB[0] = 0xCAFE0005;

    int pass = r_ok(sum, refd);
    uint32_t ulp = r_ulpd(sum, refd);

    VERIFY[0] = 0xA1A1B1B1;
    VERIFY[1] = pass ? 1 : 0;
    VERIFY[2] = (uint32_t)(t1 - t0);
    VERIFY[3] = RBITS(sum);
    VERIFY[4] = RBITS(r_round(refd));
    VERIFY[5] = ulp;
    VERIFY[6] = 0;
    VERIFY[7] = 0;

    bp_thru(LEN, t1 - t0, "add");
    uart_puts("sum got= "); print_hexdec(RBITS(sum), (double)sum);     uart_puts("\n");
    uart_puts("ref exp= "); print_hexdec(RBITS(r_round(refd)), refd);  uart_puts("\n");
    uart_puts("ulp    = "); print_u64(ulp);                    uart_puts("\n");
    uart_puts("verify = "); uart_puts(pass ? "PASS\n" : "FAIL\n");

    HB[0] = 0xCAFE000F;
    while (1) {}
}
