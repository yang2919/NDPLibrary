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
real_t neg_arg [LEN] __attribute__((aligned(32)));
real_t denom   [LEN] __attribute__((aligned(32)));
real_t out_buf [LEN] __attribute__((aligned(32)));

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;
    uart_puts("\n[gelu " DTYPE_NAME "] N=128 x/(1+exp(-1.702x))\n");

    for (int i = 0; i < LEN; i++) in_buf[i] = (real_t)(-3.0f + 6.0f * ((float)i / (float)(LEN - 1)));
    HB[0] = 0xCAFE0003;

    uint64_t t0 = rdcycle();
    r_gelu(out_buf, in_buf, denom, LEN);
    uint64_t t1 = rdcycle();
    HB[0] = 0xCAFE0004;

    uint32_t maxulp = 0;
    for (int i = 0; i < LEN; i++) {
        double od = (double)in_buf[i] / (1.0 + exp_ref(-1.702 * (double)in_buf[i]));
        uint32_t u = RULP(out_buf[i], r_round(od));
        if (u > maxulp) maxulp = u;
    }
    HB[0] = 0xCAFE0005;

    int pass = (out_buf[0] == out_buf[0]) && (maxulp <= R_MAXULP);

    VERIFY[0] = 0xA1A1B1B1;
    VERIFY[1] = pass ? 1 : 0;
    VERIFY[2] = (uint32_t)(t1 - t0);
    VERIFY[3] = RBITS(out_buf[0]);
    VERIFY[4] = RBITS(out_buf[LEN/4]);
    VERIFY[5] = maxulp;
    VERIFY[6] = RBITS(out_buf[LEN/2]);
    VERIFY[7] = RBITS(out_buf[LEN-1]);

    bp_thru(LEN, t1 - t0, "elem");
    { int sp[3] = { 0, LEN/2, LEN-1 }; const char *xl[3] = { " (x=-3)", " (x~0)", " (x=3)" };
      for (int t = 0; t < 3; t++) {
        double xi = (double)in_buf[sp[t]]; double ref = xi / (1.0 + exp_ref(-1.702 * xi));
        uart_puts("out["); print_u64((uint64_t)sp[t]); uart_puts("] got= ");
        print_hexdec(RBITS(out_buf[sp[t]]), (double)out_buf[sp[t]]);
        uart_puts(" exp= "); print_hexdec(RBITS(r_round(ref)), ref); uart_puts(xl[t]); uart_puts("\n");
      } }
    uart_puts("max ulp   = "); print_u64(maxulp);                 uart_puts("\n");
    uart_puts("verify    = "); uart_puts(pass ? "PASS\n" : "FAIL\n");

    HB[0] = 0xCAFE000F;
    while (1) {}
}
