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

real_t in_buf [LEN] __attribute__((aligned(32)));
real_t out_buf[LEN] __attribute__((aligned(32)));

#ifdef USE_VTRANS

real_t out_cephes[LEN] __attribute__((aligned(32)));
real_t rin   [LEN] __attribute__((aligned(32)));
real_t rout  [LEN] __attribute__((aligned(32)));
#endif

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;
#ifdef USE_VTRANS
    uart_puts("\n[exp " DTYPE_NAME " vtrans] N=128 vfexp/vfrcp FU vs Cephes\n");
#else
    uart_puts("\n[exp " DTYPE_NAME "] N=128 vector exp (Cephes 6-coeff)\n");
#endif

    for (int i = 0; i < LEN; i++) in_buf[i] = (real_t)(-1.5f + 3.0f * ((float)i / (float)(LEN - 1)));
    for (int i = 0; i < LEN; i++) out_buf[i] = (real_t)0;
    HB[0] = 0xCAFE0003;

#ifdef USE_VTRANS
    uint64_t s0 = rdcycle();
    r_vexp_cephes(out_cephes, in_buf, LEN);
    uint64_t s1 = rdcycle();
    uint64_t t0 = rdcycle();
    r_vexp_vt(out_buf, in_buf, LEN);
    uint64_t t1 = rdcycle();
#else
    uint64_t t0 = rdcycle();
    r_vexp(out_buf, in_buf, LEN);
    uint64_t t1 = rdcycle();
#endif
    HB[0] = 0xCAFE0004;

    uint32_t maxulp = 0;
    for (int i = 0; i < LEN; i++) {
        double od = exp_ref((double)in_buf[i]);
        uint32_t u = RULP(out_buf[i], r_round(od));
        if (u > maxulp) maxulp = u;
    }
    HB[0] = 0xCAFE0005;

#ifdef USE_VTRANS
    for (int i = 0; i < LEN; i++) rin[i] = (real_t)(0.5f + 3.5f * ((float)i / (float)(LEN - 1)));
    uint64_t r0 = rdcycle();
    r_vrcp_vt(rout, rin, LEN);
    uint64_t r1 = rdcycle();
    uint32_t rcp_maxulp = 0;
    for (int i = 0; i < LEN; i++) {
        uint32_t u = RULP(rout[i], r_round(1.0 / (double)rin[i]));
        if (u > rcp_maxulp) rcp_maxulp = u;
    }
    HB[0] = 0xCAFE0006;

    int pass = (out_buf[0] == out_buf[0]) && (maxulp <= R_MAXULP) && (rcp_maxulp <= R_MAXULP);
#else
    int pass = (out_buf[0] == out_buf[0]) && (maxulp <= R_MAXULP);
#endif

    VERIFY[0] = 0xA1A1B1B1;
    VERIFY[1] = pass ? 1 : 0;
    VERIFY[2] = (uint32_t)(t1 - t0);
    VERIFY[3] = RBITS(out_buf[0]);
    VERIFY[4] = RBITS(out_buf[LEN/2]);
    VERIFY[5] = maxulp;
    VERIFY[6] = RBITS(r_round(exp_ref((double)in_buf[0])));
#ifdef USE_VTRANS
    VERIFY[7] = rcp_maxulp;
#else
    VERIFY[7] = RBITS(out_buf[LEN-1]);
#endif

    bp_thru(LEN, t1 - t0, "elem");
#ifdef USE_VTRANS
    { uint64_t cc = s1 - s0, vc = (t1 - t0) ? (t1 - t0) : 1, rc = (r1 - r0) ? (r1 - r0) : 1;
      bp_puts("exp via cephes polynomial : "); bp_u64(cc); bp_puts(" cycles, ");
      bp_frac(LEN * 1000 / cc, 3); bp_puts(" elements/cycle\n");
      bp_puts("exp via vtrans unit       : "); bp_u64(t1 - t0); bp_puts(" cycles, ");
      bp_frac(LEN * 1000 / vc, 3); bp_puts(" elements/cycle, ");
      bp_frac(cc * 100 / vc, 2); bp_puts(" times faster\n");
      bp_puts("reciprocal via vtrans unit: "); bp_u64(r1 - r0); bp_puts(" cycles, ");
      bp_frac(LEN * 1000 / rc, 3); bp_puts(" elements/cycle, max ulp ");
      bp_u64(rcp_maxulp); bp_puts("\n"); }
#endif
    int sp[3] = { 0, LEN/2, LEN-1 };
    for (int t = 0; t < 3; t++) {
        double ref = exp_ref((double)in_buf[sp[t]]);
        uart_puts("out["); print_u64((uint64_t)sp[t]); uart_puts("] got= ");
        print_hexdec(RBITS(out_buf[sp[t]]), (double)out_buf[sp[t]]);
        uart_puts("  exp= "); print_hexdec(RBITS(r_round(ref)), ref); uart_puts("\n");
    }
    uart_puts("max ulp   = "); print_u64(maxulp);                 uart_puts("\n");
    uart_puts("verify    = "); uart_puts(pass ? "PASS\n" : "FAIL\n");

    HB[0] = 0xCAFE000F;
    while (1) {}
}
