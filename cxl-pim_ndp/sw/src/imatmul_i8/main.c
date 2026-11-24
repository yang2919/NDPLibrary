// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>

#define RHEX_NIB 2
#include "../common/kio.h"

#include "bp-imatmul.h"

#define HB     ((volatile uint32_t *)0x40100000)
#define VERIFY ((volatile uint32_t *)0x40100100)

#define MAXM 128
#define N    64
#define P    64

#ifndef SOAK_ITER
#define SOAK_ITER 5000u
#endif
#define SOAK_M 8

static int8_t A[MAXM * N] __attribute__((aligned(32)));
static int8_t B[N * P]    __attribute__((aligned(32)));
static int8_t C[MAXM * P] __attribute__((aligned(32)));
static int8_t G[MAXM * P] __attribute__((aligned(32)));

static uint32_t g_miss, g_first_idx = 0xFFFFFFFFu, g_first_pair;

static void golden(unsigned M) {
    for (unsigned m = 0; m < M; m++)
        for (unsigned p = 0; p < P; p++) {
            int32_t acc = 0;
            for (unsigned n = 0; n < N; n++)
                acc += (int32_t)A[m * N + n] * (int32_t)B[n * P + p];
            G[m * P + p] = (int8_t)acc;
        }
}

static uint32_t run_case(const char *tag, unsigned M) {
    for (unsigned i = 0; i < M * N; i++) A[i] = (int8_t)(((i * 37u) % 251u) - 125u);
    for (unsigned i = 0; i < N * P; i++) B[i] = (int8_t)(((i * 61u) % 241u) - 120u);
    for (unsigned i = 0; i < M * P; i++) C[i] = 0;

    golden(M);

    uint64_t t0 = rdcycle();
    bp_imatmul(C, A, B, M, N, P);
    uint64_t t1 = rdcycle();

    uint32_t bad = 0;
    for (unsigned i = 0; i < M * P; i++)
        if (C[i] != G[i]) {
            if (g_first_idx == 0xFFFFFFFFu) {
                g_first_idx  = i;
                g_first_pair = ((uint32_t)(uint8_t)C[i] << 8) | (uint32_t)(uint8_t)G[i];
            }
            bad++;
        }
    g_miss += bad;

    uart_puts(tag);
    uart_puts("  cycles="); print_u64(t1 - t0);
    uart_puts("  mismatch="); print_u64(bad);
    uart_puts(bad ? "   <-- FAIL\n" : "   ok\n");
    return (uint32_t)(t1 - t0);
}

int main(void) {
    HB[0] = 0xCAFE0001; uart_init(UART_DIVISOR); HB[0] = 0xCAFE0002;
    uart_puts("\n[imatmul_i8] int8 matmul, N=" "64" " P=" "64" "\n");
    uart_puts(" vmacc.vx at SEW=8 vs a scalar int32 dot product truncated to int8\n");

    uint32_t c4, c8, c128;
    c4   = run_case(" M=4   4x4 e8/m4", 4);   HB[0] = 0xCAFE0003;
    c8   = run_case(" M=8   8x8 e8/m2", 8);   HB[0] = 0xCAFE0004;
    c128 = run_case(" M=128 8x8 e8/m2", 128); HB[0] = 0xCAFE0005;

    uint32_t save_miss = g_miss, save_idx = g_first_idx, save_pair = g_first_pair;
    uint32_t c16 = run_case(" M=16  16x16 e32/m1 PROBE", 16);
    uint32_t probe_miss = g_miss - save_miss;
    g_miss = save_miss; g_first_idx = save_idx; g_first_pair = save_pair;

    for (unsigned i = 0; i < SOAK_M * N; i++) A[i] = (int8_t)(((i * 37u) % 251u) - 125u);
    for (unsigned i = 0; i < N * P;      i++) B[i] = (int8_t)(((i * 61u) % 241u) - 120u);
    golden(SOAK_M);
    uint32_t soak_bad = 0;
    uint64_t s0 = rdcycle();
    for (uint32_t it = 0; it < SOAK_ITER; it++) {
        for (unsigned i = 0; i < SOAK_M * P; i++) C[i] = 0;
        bp_imatmul(C, A, B, SOAK_M, N, P);
        for (unsigned i = 0; i < SOAK_M * P; i++)
            if (C[i] != G[i]) { soak_bad++; break; }
    }
    uint64_t s1 = rdcycle();
    HB[0] = 0xCAFE0006;
    uart_puts(" soak M=8 x "); print_u64(SOAK_ITER);
    uart_puts("  cycles=");    print_u64(s1 - s0);
    uart_puts("  deviations="); print_u64(soak_bad);
    uart_puts(soak_bad ? "   <-- FAIL\n" : "   ok\n");

    int pass = (g_miss == 0) && (soak_bad == 0);

    VERIFY[0] = 0xA1A1B1B1;
    VERIFY[1] = pass ? 1u : 0u;
    VERIFY[2] = c4;
    VERIFY[3] = c8;
    VERIFY[4] = c128;
    VERIFY[5] = g_miss;
    VERIFY[6] = g_first_idx;
    VERIFY[7] = g_first_pair;
    VERIFY[8] = SOAK_ITER;
    VERIFY[9] = soak_bad;
    VERIFY[10] = c16;
    VERIFY[11] = probe_miss;

    if (!pass) {
        uart_puts("first bad idx = "); print_u64(g_first_idx);
        uart_puts("  got ");           print_rhex(g_first_pair >> 8);
        uart_puts("  expected ");      print_rhex(g_first_pair & 0xFFu);
        uart_puts("\n");
    }
    uart_puts("verify = "); uart_puts(pass ? "PASS\n" : "FAIL\n");

    HB[0] = 0xCAFE000F;
    while (1) {}
}
