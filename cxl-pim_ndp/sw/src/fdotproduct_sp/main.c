// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include "../common/dtype.h"
#include "../common/uart_apb.h"
#include "../common/kio.h"
#include "../common/bench_print.h"

#define HB     ((volatile uint32_t *)0x40100000)
#define VERIFY ((volatile uint32_t *)0x40100100)

#define LEN 128

real_t A[LEN] __attribute__((aligned(32)));
real_t B[LEN] __attribute__((aligned(32)));

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;
    uart_puts("\n[fdotproduct " DTYPE_NAME "] N=128 vfmacc + vfredusum\n");

    for (int i = 0; i < LEN; i++) A[i] = (real_t)(((float)((i % 13) - 6)) * 0.0625f);
    for (int i = 0; i < LEN; i++) B[i] = (real_t)(((float)((i % 11) - 5)) * 0.0625f);
    HB[0] = 0xCAFE0003;

    uint64_t t0 = rdcycle();
    real_t dot = r_dotp(A, B, LEN);
    uint64_t t1 = rdcycle();
    HB[0] = 0xCAFE0004;

    double refd = 0; for (int i = 0; i < LEN; i++) refd += (double)A[i] * (double)B[i];
    real_t ref = r_round(refd);
    HB[0] = 0xCAFE0005;

    uint32_t ulp = RULP(dot, ref);
    double diff = (double)dot - refd;
    double rel  = refd != 0.0 ? diff / refd : diff;
    if (rel < 0) rel = -rel;
    int pass = (dot == dot) && (rel < R_PASS_REL);

    VERIFY[0] = 0xA1A1B1B1;
    VERIFY[1] = pass ? 1 : 0;
    VERIFY[2] = (uint32_t)(t1 - t0);
    VERIFY[3] = RBITS(dot);
    VERIFY[4] = RBITS(ref);
    VERIFY[5] = ulp;
    VERIFY[6] = (uint32_t)(rel * 1e9);
    VERIFY[7] = 0;

    bp_perf(2ull*LEN, t1-t0, R_PEAK_FMA);
    uart_puts("dot got= "); print_hexdec(RBITS(dot), (double)dot); uart_puts("\n");
    uart_puts("ref exp= "); print_hexdec(RBITS(ref), refd);        uart_puts("\n");
    uart_puts("ulp    = "); print_u64(ulp);         uart_puts("\n");
    uart_puts("verify = "); uart_puts(pass ? "PASS\n" : "FAIL\n");

    HB[0] = 0xCAFE000F;
    while (1) {}
}
