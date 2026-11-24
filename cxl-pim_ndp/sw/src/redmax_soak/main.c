// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include "../common/dtype.h"
#include "../common/kio.h"
#include "data.h"

#define HB     ((volatile uint32_t *)0x40100000)
#define VERIFY ((volatile uint32_t *)0x40100100)

#ifndef SOAK_ITER1
#define SOAK_ITER1 1000000u
#endif
#ifndef SOAK_ITER2
#define SOAK_ITER2 8000u
#endif
#ifndef SOAK_ITER3
#define SOAK_ITER3 2000u
#endif
#define N       128
#define ROW_LEN 107

static real_t buf [N] __attribute__((aligned(256)));
static real_t rot [N] __attribute__((aligned(256)));

static uint64_t g_calls, g_miss, g_miss_p1;
static uint32_t g_first_bad, g_first_gold;

static real_t scalar_max(const real_t *v, size_t n) {
    real_t m = v[0];
    for (size_t i = 1; i < n; i++) if (v[i] > m) m = v[i];
    return m;
}

static real_t redmax_seed(const real_t *v, size_t n, real_t seed) {
    size_t vl0 = VSETVL(n);
    vreal_t acc = VLE(v, vl0);
    vreal_t s   = VFMV_V_F(seed, vl0);
    return VFMV_F_S(VFREDMAX(acc, s, vl0));
}

#ifdef HP
#define R_EW "16"
#define NEG_INF_BITS 0xFC00u
static real_t neg_inf(void) { union { uint16_t u; _Float16 h; } c = { .u = 0xFC00u }; return c.h; }
#else
#define R_EW "32"
#define NEG_INF_BITS 0xFF800000u
static real_t neg_inf(void) { union { uint32_t u; float f; } c = { .u = 0xFF800000u }; return c.f; }
#endif

static real_t probe_buf[N] __attribute__((aligned(256)));

static void soak(const real_t *v, size_t n, uint32_t iters, int phase1) {
    const real_t g = scalar_max(v, n);
    const uint32_t gb = (uint32_t)RBITS(g);
    for (uint32_t it = 0; it < iters; it++) {
        real_t m = r_redmax(v, n);
        g_calls++;
        if ((uint32_t)RBITS(m) != gb) {
            if (!g_miss) { g_first_bad = (uint32_t)RBITS(m); g_first_gold = gb; }
            g_miss++;
            if (phase1) g_miss_p1++;
        }
    }
}

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;
    uart_puts("\n[redmax soak " DTYPE_NAME "] SLDU compare path vs scalar FPU\n");

    for (int i = 0; i < N; i++) {
#ifdef HP
        union { uint16_t u; _Float16 h; } c = { .u = SROW_BITS[i] };
        buf[i] = c.h;
#else
        union { uint16_t u; _Float16 h; } c = { .u = SROW_BITS[i] };
        buf[i] = (float)c.h;
#endif
    }
    HB[0] = 0xCAFE0003;

    uint64_t t0 = rdcycle();
    soak(buf, ROW_LEN, SOAK_ITER1, 1);
    HB[0] = 0xCAFE0004;

    for (int r = 1; r < N; r++) {
        for (int i = 0; i < N; i++) rot[i] = buf[(i + r) & (N - 1)];
        soak(rot, ROW_LEN, SOAK_ITER2, 0);
    }
    HB[0] = 0xCAFE0005;

    for (size_t len = 1; len <= (size_t)N; len++)
        soak(buf, len, SOAK_ITER3, 0);
    uint64_t t1 = rdcycle();
    HB[0] = 0xCAFE0006;

    const real_t ninf = neg_inf();
    const uint32_t probeA = (uint32_t)RBITS(redmax_seed(buf, ROW_LEN, ninf));

    uint32_t reach[4] = {0, 0, 0, 0};
    uint32_t reach_cnt = 0;
    const real_t one = (real_t)1;
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < N; i++) probe_buf[i] = ninf;
        probe_buf[k] = one;
        real_t r = redmax_seed(probe_buf, N, ninf);
        if (RBITS(r) == RBITS(one)) { reach[k >> 5] |= (1u << (k & 31)); reach_cnt++; }
    }
    HB[0] = 0xCAFE0007;

    uint32_t probeC = 0;
    {
        size_t vl;
        int32_t  iv[64] __attribute__((aligned(256)));
        for (int i = 0; i < 64; i++) iv[i] = (i == 37) ? 12345 : -1000 - i;
        int32_t igot;
        asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(64));
        asm volatile("vle32.v v8, (%0)" :: "r"(iv) : "memory");
        asm volatile("vmv.v.x v9, %0" :: "r"(-2000000000));
        asm volatile("vredmax.vs v8, v8, v9");
        asm volatile("vmv.x.s %0, v8" : "=r"(igot));
        if (igot == 12345) probeC |= 1u;

        real_t fv[64] __attribute__((aligned(256)));
        for (int i = 0; i < 64; i++) fv[i] = (real_t)((i == 37) ? -1 : 5);
        real_t fgot;

        union { uint32_t u; float f; } pinf32 = { .u = 0x7F800000u };
#ifdef HP
        union { uint16_t u; _Float16 h; } pinf = { .u = 0x7C00u };
        real_t pinfv = pinf.h;
#else
        real_t pinfv = pinf32.f;
#endif
        asm volatile("vsetvli %0, %1, e" R_EW ", m1, ta, ma" : "=r"(vl) : "r"(64));
        asm volatile("vle" R_EW ".v v8, (%0)" :: "r"(fv) : "memory");
        asm volatile("vfmv.v.f v9, %0" :: "f"(pinfv));
        asm volatile("vfredmin.vs v8, v8, v9");
        asm volatile("vfmv.f.s %0, v8" : "=f"(fgot));
        if (fgot == (real_t)-1) probeC |= 2u;

        for (int i = 0; i < 64; i++) fv[i] = (real_t)1;
        asm volatile("vsetvli %0, %1, e" R_EW ", m1, ta, ma" : "=r"(vl) : "r"(64));
        asm volatile("vle" R_EW ".v v8, (%0)" :: "r"(fv) : "memory");
        asm volatile("vfmv.v.f v9, %0" :: "f"((real_t)0));
        asm volatile("vfredusum.vs v8, v8, v9");
        asm volatile("vfmv.f.s %0, v8" : "=f"(fgot));
        if (fgot == (real_t)64) probeC |= 4u;
        (void)vl;
    }

    int pass = (g_miss == 0);
    VERIFY[0] = 0xA1A1B1B1;
    VERIFY[1] = (uint32_t)pass;
    VERIFY[2] = (uint32_t)(t1 - t0);
    VERIFY[3] = (uint32_t)g_miss;
    VERIFY[4] = (uint32_t)g_miss_p1;
    VERIFY[5] = g_first_bad;
    VERIFY[6] = g_first_gold;
    VERIFY[7] = (uint32_t)g_calls;
    VERIFY[8]  = probeA;
    VERIFY[9]  = (uint32_t)RBITS(scalar_max(buf, ROW_LEN));
    VERIFY[10] = NEG_INF_BITS;
    VERIFY[11] = (uint32_t)RBITS(buf[0]);
    VERIFY[12] = reach[0]; VERIFY[13] = reach[1];
    VERIFY[14] = reach[2]; VERIFY[15] = reach[3];
    VERIFY[16] = reach_cnt;
    VERIFY[17] = probeC;

    uart_puts("redmax calls      = "); print_u64(g_calls); uart_puts("\n");
    uart_puts("cycles            = "); print_u64(t1 - t0); uart_puts("\n");
    uart_puts("mismatches        = "); print_u64(g_miss);
    uart_puts("   (exact-pattern phase: "); print_u64(g_miss_p1); uart_puts(")\n");
    if (g_miss) {
        uart_puts("first bad result  = "); print_rhex(g_first_bad);
        uart_puts("   expected ");         print_rhex(g_first_gold); uart_puts("\n");
    }
    uart_puts("\nprobe A: seed = -inf over the row (three modes give three answers)\n");
    uart_puts("  got            = "); print_rhex(probeA);
    if      (probeA == (uint32_t)RBITS(scalar_max(buf, ROW_LEN))) uart_puts("   FOLDS CORRECTLY\n");
    else if (probeA == (uint32_t)NEG_INF_BITS)                    uart_puts("   returns the SEED vs1[0]\n");
    else if (probeA == (uint32_t)RBITS(buf[0]))                   uart_puts("   returns ELEMENT 0 vs2[0]\n");
    else                                                          uart_puts("   UNEXPECTED\n");
    uart_puts("  row max        = "); print_rhex((uint32_t)RBITS(scalar_max(buf, ROW_LEN))); uart_puts("\n");
    uart_puts("  element 0      = "); print_rhex((uint32_t)RBITS(buf[0])); uart_puts("\n");

    uart_puts("probe B: element positions that reach the result (all -inf but one)\n");
    uart_puts("  positions      = "); print_u64(reach_cnt); uart_puts(" of "); print_u64(N); uart_puts("\n");
    uart_puts("  map            = ");
    for (int w = 3; w >= 0; w--) { print_rhex(reach[w]); uart_putc(' '); }
    uart_puts("\n  first 16 k     = ");
    for (int k = 0; k < 16; k++) uart_putc((reach[0] >> k) & 1 ? '1' : '.');
    uart_puts("\n\n");
    uart_puts("probe C: which reduction class folds?\n");
    uart_puts("  int vredmax (VALU)   = "); uart_puts((probeC & 1u) ? "folds\n" : "BROKEN\n");
    uart_puts("  vfredmin (NONCOMP)   = "); uart_puts((probeC & 2u) ? "folds\n" : "BROKEN\n");
    uart_puts("  vfredusum (ADDMUL)   = "); uart_puts((probeC & 4u) ? "folds\n" : "BROKEN\n");
    uart_puts("\nverify = "); uart_puts(pass ? "PASS\n" : "FAIL\n");
    HB[0] = 0xCAFE000F;
    while (1) {}
}
