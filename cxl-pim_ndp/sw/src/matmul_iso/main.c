// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include "../common/dtype.h"
#include "../common/kio.h"
#ifdef HP
#include "hp-fmatmul.h"
#define MATMUL(c, a, b, M, N, P) hp_fmatmul((c), (a), (b), (M), (N), (P))
#else
#include "sp-fmatmul.h"
#define MATMUL(c, a, b, M, N, P) sp_fmatmul((c), (a), (b), (M), (N), (P))
#endif

#define HB     ((volatile uint32_t *)0x40100000)
#define VERIFY ((volatile uint32_t *)0x40100100)

#define MAXM 128
#define MAXN 768
#define P    64

real_t X[MAXM * MAXN] __attribute__((aligned(32)));
real_t W[MAXN * P]    __attribute__((aligned(32)));
real_t Q[MAXM * P]    __attribute__((aligned(32)));

static double run_case(const char *tag, int M, int N, int *garbage) {
    for (int i = 0; i < M * N; i++) X[i] = (real_t)(((float)((i % 13) - 6)) * 0.0625f);
    for (int i = 0; i < N * P; i++) W[i] = (real_t)(((float)((i % 11) - 5)) * 0.03125f);
    for (int i = 0; i < M * P; i++) Q[i] = (real_t)0;

    MATMUL(Q, X, W, M, N, P);

    double mx = 0; int nan = 0;
    for (int i = 0; i < M * P; i++) { double v = (double)Q[i];
        if (v != v) nan++; else { double a = v < 0 ? -v : v; if (a > mx) mx = a; } }
    *garbage = (mx > 1.0) || nan;

    uart_puts(tag);
    uart_puts(" Qmax="); print_hexdec(RBITS(r_round(mx)), mx);
    uart_puts(" nan=");  print_u64((uint64_t)nan);
    uart_puts(*garbage ? "  <-- GARBAGE\n" : "  ok\n");
    return mx;
}

int main(void) {
    HB[0] = 0xCAFE0001; uart_init(UART_DIVISOR); HB[0] = 0xCAFE0002;
    uart_puts("\n[matmul_iso " DTYPE_NAME "] sp_fmatmul path sweep @ N=768\n");
    uart_puts(" dispatch: M=8/128 -> 8x8/m2 ; M=16/64 -> 16x16/m1 ; correct Qmax ~0.1..0.4\n");

    double q[5]; int g[5];
    q[0] = run_case(" 8x8/m2   M=8   N=768", 8,   768, &g[0]); HB[0] = 0xCAFE0003;
    q[1] = run_case(" 16x16/m1 M=16  N=768", 16,  768, &g[1]); HB[0] = 0xCAFE0004;
    q[2] = run_case(" 16x16/m1 M=64  N=768", 64,  768, &g[2]); HB[0] = 0xCAFE0005;
    q[3] = run_case(" 8x8/m2   M=128 N=768", 128, 768, &g[3]); HB[0] = 0xCAFE0006;
    q[4] = run_case(" 8x8/m2   M=8   N=64 ", 8,   64,  &g[4]); HB[0] = 0xCAFE0007;

    int pattern = g[0] && g[3] && !g[1] && !g[2];

    VERIFY[0] = 0xA1A1B1B1;
    VERIFY[1] = pattern ? 1 : 0;
    VERIFY[2] = RBITS(r_round(q[0]));
    VERIFY[3] = RBITS(r_round(q[1]));
    VERIFY[4] = RBITS(r_round(q[2]));
    VERIFY[5] = RBITS(r_round(q[3]));
    VERIFY[6] = RBITS(r_round(q[4]));
    VERIFY[7] = (uint32_t)((g[0]) | (g[1] << 1) | (g[2] << 2) | (g[3] << 3) | (g[4] << 4));

    uart_puts("board-physical signature (8x8/m2 garbage & 16x16/m1 ok) = ");
    uart_puts(pattern ? "CONFIRMED\n" : "NO\n");

    HB[0] = 0xCAFE000F;
    while (1) {}
}
