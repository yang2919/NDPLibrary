// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include "../common/dtype.h"
#include "../common/exp_hp.h"
#include "../common/kio.h"
#ifdef HP
#include "hp-fmatmul.h"
#define MATMUL(c,a,b,M,N,P) hp_fmatmul((c),(a),(b),(M),(N),(P))
#else
#include "sp-fmatmul.h"
#define MATMUL(c,a,b,M,N,P) sp_fmatmul((c),(a),(b),(M),(N),(P))
#endif

#define HB     ((volatile uint32_t *)0x40100000)
#define VERIFY ((volatile uint32_t *)0x40100100)

#define SEQ     128
#define D_MODEL 768
#define D_HEAD  64

#ifdef HP
#define ROWSUM_TOL 0.1
#else
#define ROWSUM_TOL 0.01
#endif

real_t X [SEQ*D_MODEL] __attribute__((aligned(32)));
real_t Wq[D_MODEL*D_HEAD] __attribute__((aligned(32)));
real_t Wk[D_MODEL*D_HEAD] __attribute__((aligned(32)));
real_t Wv[D_MODEL*D_HEAD] __attribute__((aligned(32)));
real_t Q [SEQ*D_HEAD] __attribute__((aligned(32)));
real_t K [SEQ*D_HEAD] __attribute__((aligned(32)));
real_t V [SEQ*D_HEAD] __attribute__((aligned(32)));
real_t Kt[D_HEAD*SEQ] __attribute__((aligned(32)));
real_t S [SEQ*SEQ] __attribute__((aligned(32)));
real_t P [SEQ*SEQ] __attribute__((aligned(32)));
real_t P_in[SEQ] __attribute__((aligned(32)));
real_t O [SEQ*D_HEAD] __attribute__((aligned(32)));

static void stage_stat(const char *nm, const real_t *a, int n) {
    int nc = 0; double mx = 0;
    for (int i = 0; i < n; i++) { double v = (double)a[i];
        if (v != v) nc++; else { double av = v < 0 ? -v : v; if (av > mx) mx = av; } }
    uart_puts("  "); uart_puts(nm); uart_puts(" nan="); print_u64((uint64_t)nc);
    uart_puts(" max="); print_rdec(mx); uart_puts("\n");
}

static double o_golden(int idx) {
    int i = idx / D_HEAD, j = idx % D_HEAD; double s = 0;
    for (int k = 0; k < SEQ; k++) s += (double)P[i*SEQ + k] * (double)V[k*D_HEAD + j];
    return s;
}

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;
    uart_puts("\n[attention " DTYPE_NAME "] gpt2-small head: seq=128 d_model=768 d_head=64\n");

    for (int i = 0; i < SEQ*D_MODEL;  i++) X[i]  = (real_t)(((float)((i % 13) - 6)) * 0.0625f);
    for (int i = 0; i < D_MODEL*D_HEAD; i++) Wq[i] = (real_t)(((float)((i % 11) - 5)) * 0.03125f);
    for (int i = 0; i < D_MODEL*D_HEAD; i++) Wk[i] = (real_t)(((float)((i %  7) - 3)) * 0.03125f);
    for (int i = 0; i < D_MODEL*D_HEAD; i++) Wv[i] = (real_t)(((float)((i %  9) - 4)) * 0.03125f);
    HB[0] = 0xCAFE0003;

    uint64_t t0 = rdcycle();
    MATMUL(Q, X, Wq, SEQ, D_MODEL, D_HEAD);
    MATMUL(K, X, Wk, SEQ, D_MODEL, D_HEAD);
    MATMUL(V, X, Wv, SEQ, D_MODEL, D_HEAD);
    HB[0] = 0xCAFE0004;

    const real_t inv_sqrt = (real_t)0.125f;
#ifdef USE_SWOPT

    r_scale(Q, Q, inv_sqrt, SEQ*D_HEAD);
    r_transpose(Kt, K, SEQ, D_HEAD);
    MATMUL(S, Q, Kt, SEQ, D_HEAD, SEQ);
#else
    for (int i = 0; i < SEQ; i++)
        for (int j = 0; j < SEQ; j++)
            S[i*SEQ + j] = (real_t)(r_dotp(Q + i*D_HEAD, K + j*D_HEAD, D_HEAD) * inv_sqrt);
#endif
    HB[0] = 0xCAFE0005;

    double row_sum_min = 1e9, row_sum_max = -1e9;
    int rows_ok = 1;
    for (int i = 0; i < SEQ; i++) {
        real_t *rs = S + i*SEQ, *rp = P + i*SEQ;
        real_t m = r_redmax(rs, SEQ);
        real_t s = r_subexp_sum(rp, rs, m, SEQ);
        r_scale(rp, rp, (real_t)((real_t)1 / s), SEQ);
        double chk = 0; for (int j = 0; j < SEQ; j++) chk += (double)rp[j];
        if (chk < row_sum_min) row_sum_min = chk;
        if (chk > row_sum_max) row_sum_max = chk;
        if (chk < 1.0 - ROWSUM_TOL || chk > 1.0 + ROWSUM_TOL) rows_ok = 0;
    }
    HB[0] = 0xCAFE0006;

    MATMUL(O, P, V, SEQ, SEQ, D_HEAD);
    uint64_t t1 = rdcycle();
    HB[0] = 0xCAFE0007;

    int spot[3] = { 0, SEQ*D_HEAD/2, SEQ*D_HEAD-1 };
#ifdef HP
    const double O_TOL = 0.05, O_ABS = 1e-3;

#else
    const double O_TOL = 0.01, O_ABS = 1e-5;
#endif
    double max_rel = 0, max_abs = 0; int o_fin = 1, o_ok = 1;
    for (int s = 0; s < 3; s++) {
        double got = (double)O[spot[s]], ref = o_golden(spot[s]);
        if (got != got) { o_fin = 0; continue; }
        double ae = got - ref; if (ae < 0) ae = -ae;
        double r = (ref != 0.0) ? ae / (ref < 0 ? -ref : ref) : ae;
        if (r > max_rel) max_rel = r;
        if (ae > max_abs) max_abs = ae;
        if (ae > O_ABS && r > O_TOL) o_ok = 0;
    }
    int pass = rows_ok && o_fin && o_ok;

    VERIFY[0] = 0xA1A1B1B1;
    VERIFY[1] = pass ? 1 : 0;
    VERIFY[2] = (uint32_t)(t1 - t0);
    VERIFY[3] = f2u((float)row_sum_max);
    VERIFY[4] = RBITS(O[0]);
    VERIFY[5] = RBITS(r_round(o_golden(0)));
    VERIFY[6] = RBITS(O[SEQ*D_HEAD-1]);
    VERIFY[7] = RBITS(r_round(o_golden(SEQ*D_HEAD-1)));

    uart_puts("cycles       = "); print_u64(t1 - t0); uart_puts("\n");
    uart_puts(" stage nan/max:\n");
    stage_stat("Q", Q, SEQ*D_HEAD); stage_stat("V", V, SEQ*D_HEAD);
    stage_stat("P", P, SEQ*SEQ);    stage_stat("O", O, SEQ*D_HEAD);
    uart_puts("P row sum min= "); print_rdec(row_sum_min);
    uart_puts("  max= ");         print_rdec(row_sum_max); uart_puts("   (expect 1.0)\n");
    for (int s = 0; s < 3; s++) {
        double ref = o_golden(spot[s]);
        uart_puts("O["); print_u64((uint64_t)spot[s]); uart_puts("] got= ");
        print_hexdec(RBITS(O[spot[s]]), (double)O[spot[s]]);
        uart_puts("  exp= "); print_hexdec(RBITS(r_round(ref)), ref); uart_puts("\n");
    }
    uart_puts("max O rel err= "); print_rdec(max_rel);
    uart_puts("  abs= ");         print_rdec(max_abs); uart_puts("\n");
    uart_puts("verify       = "); uart_puts(pass ? "PASS\n" : "FAIL\n");

    HB[0] = 0xCAFE000F;
    while (1) {}
}
