// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include "../common/dtype.h"
#include "../common/kio.h"
#include "../common/bench_print.h"

#define HB     ((volatile uint32_t *)0x40100000)
#define VERIFY ((volatile uint32_t *)0x40100100)

#define LEN 128
#define EPS 1e-3

real_t in_buf [LEN] __attribute__((aligned(32)));
real_t center [LEN] __attribute__((aligned(32)));
real_t sqr    [LEN] __attribute__((aligned(32)));
real_t out_buf[LEN] __attribute__((aligned(32)));

int64_t event_trigger;

static double sqrt_d(double x) { if (x <= 0.0) return 0.0; double y = x; for (int i = 0; i < 40; i++) y = 0.5 * (y + x / y); return y; }

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;
    uart_puts("\n[layernorm " DTYPE_NAME "] N=128 mean/var/invstd/normalize\n");

    for (int i = 0; i < LEN; i++) in_buf[i] = (real_t)(2.0f + 0.1f * (float)i);
    HB[0] = 0xCAFE0003;

    uint64_t t0 = rdcycle();
    real_t mean = r_redsum(in_buf, LEN) / (real_t)LEN;
    r_subs(center, in_buf, mean, LEN);
    r_vmul(sqr, center, center, LEN);
    real_t var = r_redsum(sqr, LEN) / (real_t)LEN;
    real_t invstd = (real_t)1 / sqrt_r(var + (real_t)EPS);
    r_scale(out_buf, center, invstd, LEN);
    uint64_t t1 = rdcycle();
    HB[0] = 0xCAFE0004;

    double meand = 0; for (int i = 0; i < LEN; i++) meand += (double)in_buf[i]; meand /= LEN;
    double vard = 0; for (int i = 0; i < LEN; i++) { double c = (double)in_buf[i] - meand; vard += c * c; } vard /= LEN;
    double invd = 1.0 / sqrt_d(vard + EPS);

    const real_t ulp_valid_above = (real_t)0.05;
    uint32_t maxulp = 0, gated_ulp = 0;
    int elem_ok = 1;
    double s = 0, s2 = 0;
    for (int i = 0; i < LEN; i++) {
        double od = ((double)in_buf[i] - meand) * invd;
        real_t  or_ = r_round(od);
        uint32_t u = RULP(out_buf[i], or_);
        if (u > maxulp) maxulp = u;
        if (r_fabs(or_) >= ulp_valid_above) { if (u > gated_ulp) gated_ulp = u; }
        else if (!r_ok(out_buf[i], od))     { elem_ok = 0; }
        s += (double)out_buf[i]; s2 += (double)out_buf[i] * (double)out_buf[i];
    }
    HB[0] = 0xCAFE0005;

    int pass = (mean == mean) && elem_ok && (gated_ulp <= R_MAXULP);

    VERIFY[0] = 0xA1A1B1B1;
    VERIFY[1] = pass ? 1 : 0;
    VERIFY[2] = (uint32_t)(t1 - t0);
    VERIFY[3] = RBITS(mean);
    VERIFY[4] = RBITS(var);
    VERIFY[5] = maxulp;
    VERIFY[6] = RBITS(r_round(s));
    VERIFY[7] = RBITS(r_round(s2));

    bp_thru(LEN, t1 - t0, "elem");
    uart_puts("mean     got= "); print_hexdec(RBITS(mean), (double)mean); uart_puts("\n");
    uart_puts("var      got= "); print_hexdec(RBITS(var), (double)var);   uart_puts("\n");
    uart_puts("out maxulp= "); print_u64(maxulp);         uart_puts("\n");
    uart_puts("sum(out)  got= "); print_hexdec(RBITS(r_round(s)), s);
    uart_puts(" exp= "); print_hexdec(f2u(0.0f), 0.0); uart_puts("\n");
    uart_puts("sum(out^2)got= "); print_hexdec(RBITS(r_round(s2)), s2);
    uart_puts(" exp= "); print_hexdec(f2u((float)LEN), (double)LEN); uart_puts("\n");
    uart_puts("verify    = "); uart_puts(pass ? "PASS\n" : "FAIL\n");

    HB[0] = 0xCAFE000F;
    while (1) {}
}
