// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include "../common/dtype.h"
#include "../common/exp_hp.h"
#include "../common/kio.h"
#include "../common/bench_print.h"

#define HB     ((volatile uint32_t *)0x40100000)
#define VERIFY ((volatile uint32_t *)0x40100100)

#define LEN 128

real_t in_buf  [LEN] __attribute__((aligned(32)));
real_t shift   [LEN] __attribute__((aligned(32)));
real_t out_buf [LEN] __attribute__((aligned(32)));

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;
    uart_puts("\n[softmax " DTYPE_NAME "] N=128 max-sub-exp-sum-div\n");

    for (int i = 0; i < LEN; i++) in_buf[i] = (real_t)(-2.0f + 4.0f * ((float)i / (float)(LEN - 1)));
    HB[0] = 0xCAFE0003;

    uint64_t t0 = rdcycle();
    real_t mx = r_redmax(in_buf, LEN);
    real_t s = r_subexp_sum(out_buf, in_buf, mx, LEN);
    r_scale(out_buf, out_buf, (real_t)((real_t)1 / s), LEN);
    uint64_t t1 = rdcycle();
    HB[0] = 0xCAFE0004;

    double mxd = (double)in_buf[0];
    for (int i = 1; i < LEN; i++) if ((double)in_buf[i] > mxd) mxd = (double)in_buf[i];
    double ed[LEN], sd = 0;
    for (int i = 0; i < LEN; i++) { ed[i] = exp_ref((double)in_buf[i] - mxd); sd += ed[i]; }
    uint32_t maxulp = 0; double sumd = 0;
    for (int i = 0; i < LEN; i++) {
        double od = ed[i] / sd;
        uint32_t u = RULP(out_buf[i], r_round(od));
        if (u > maxulp) maxulp = u;
        sumd += (double)out_buf[i];
    }
    HB[0] = 0xCAFE0005;

    int pass = (mx == mx) && (maxulp <= R_MAXULP);

    VERIFY[0] = 0xA1A1B1B1;
    VERIFY[1] = pass ? 1 : 0;
    VERIFY[2] = (uint32_t)(t1 - t0);
    VERIFY[3] = RBITS(r_round(sumd));
    VERIFY[4] = RBITS(out_buf[0]);
    VERIFY[5] = maxulp;
    VERIFY[6] = RBITS(out_buf[LEN/2]);
    VERIFY[7] = RBITS(out_buf[LEN-1]);

    bp_thru(LEN, t1 - t0, "elem");
    uart_puts("sum(out) got= "); print_hexdec(RBITS(r_round(sumd)), sumd);
    uart_puts(" exp= "); print_hexdec(f2u(1.0f), 1.0); uart_puts("\n");
    uart_puts("out[0]   got= "); print_hexdec(RBITS(out_buf[0]), (double)out_buf[0]);
    uart_puts(" exp= "); { double r = ed[0]/sd; print_hexdec(RBITS(r_round(r)), r); } uart_puts("\n");
    uart_puts("out[end] got= "); print_hexdec(RBITS(out_buf[LEN-1]), (double)out_buf[LEN-1]);
    uart_puts(" exp= "); { double r = ed[LEN-1]/sd; print_hexdec(RBITS(r_round(r)), r); } uart_puts("\n");
    uart_puts("max ulp   = "); print_u64(maxulp);               uart_puts("\n");
    uart_puts("verify    = "); uart_puts(pass ? "PASS\n" : "FAIL\n");

    HB[0] = 0xCAFE000F;
    while (1) {}
}
