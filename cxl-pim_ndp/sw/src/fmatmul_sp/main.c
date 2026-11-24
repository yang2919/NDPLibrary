// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "../common/uart_apb.h"
#include "../common/bench_print.h"
#include "sp-fmatmul.h"

#define HB     ((volatile uint32_t *)0x40100000)
#define VERIFY ((volatile uint32_t *)0x40100100)

#define M 64
#define N 64
#define K 64

float A[M*N] __attribute__((aligned(32)));
float B[N*K] __attribute__((aligned(32)));
float C[M*K] __attribute__((aligned(32)));

int64_t event_trigger;

static inline uint64_t rdcycle(void) {
    uint64_t c; asm volatile ("rdcycle %0" : "=r"(c)); return c;
}

static void uart_puts(const char *s) {
    while (*s) { if (*s == '\n') uart_putc('\r'); uart_putc(*s++); }
}

static void print_u64(uint64_t v) {
    char b[21]; int i = 20; b[20] = 0;
    if (!v) { uart_putc('0'); return; }
    while (v) { b[--i] = '0' + (v % 10); v /= 10; }
    while (b[i]) uart_putc(b[i++]);
}

static void print_hex32(uint32_t v) {
    uart_puts("0x");
    for (int i = 7; i >= 0; i--) {
        uint32_t n = (v >> (i*4)) & 0xF;
        uart_putc(n < 10 ? '0' + n : 'a' + (n - 10));
    }
}

static inline uint32_t f2u(float f) { union { float f; uint32_t u; } x = { .f = f }; return x.u; }

static float fabs_f(float x) { return x < 0 ? -x : x; }
static int near(float a, float b, float tol) { return fabs_f(a - b) <= tol; }

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;

    uart_puts("\n[fmatmul_sp] 64x64 fp32 ara vector matmul\n");

    for (int i = 0; i < M*N; i++) A[i] = ((float)((i % 23) - 11)) * 0.0625f;
    for (int i = 0; i < N*K; i++) B[i] = ((float)((i % 17) - 8))  * 0.0625f;
    for (int i = 0; i < M*K; i++) C[i] = 0.0f;
    HB[0] = 0xCAFE0003;

    uint64_t t0 = rdcycle();
    sp_fmatmul(C, A, B, M, N, K);
    uint64_t t1 = rdcycle();
    HB[0] = 0xCAFE0004;

    float ref0 = 0, ref_mid = 0, ref_last = 0;
    int mid_i = (M*K)/2 / K, mid_j = (M*K)/2 % K;
    int last_i = M-1, last_j = K-1;
    for (int n = 0; n < N; n++) {
        ref0     += A[0*N + n]      * B[n*K + 0];
        ref_mid  += A[mid_i*N + n]  * B[n*K + mid_j];
        ref_last += A[last_i*N + n] * B[n*K + last_j];
    }
    HB[0] = 0xCAFE0005;

    int pass = near(C[0],         ref0,     1e-3f)
            && near(C[(M*K)/2],   ref_mid,  1e-3f)
            && near(C[M*K-1],     ref_last, 1e-3f);

    VERIFY[0] = 0xA1A1B1B1;
    VERIFY[1] = pass ? 1 : 0;
    VERIFY[2] = (uint32_t)(t1 - t0);
    VERIFY[3] = f2u(C[0]);
    VERIFY[4] = f2u(C[(M*K)/2]);
    VERIFY[5] = f2u(C[M*K-1]);
    VERIFY[6] = f2u(ref0);
    VERIFY[7] = f2u(ref_last);

    bp_perf(2ull*M*N*K, t1-t0, 16);
    uart_puts("c[0]      = ");        print_hex32(f2u(C[0]));      uart_puts("\n");
    uart_puts("c[mid]    = ");        print_hex32(f2u(C[(M*K)/2])); uart_puts("\n");
    uart_puts("c[last]   = ");        print_hex32(f2u(C[M*K-1]));  uart_puts("\n");
    uart_puts("ref[0]    = ");        print_hex32(f2u(ref0));      uart_puts("\n");
    uart_puts("ref[last] = ");        print_hex32(f2u(ref_last));  uart_puts("\n");
    uart_puts("verify    = ");        uart_puts(pass ? "PASS\n" : "FAIL\n");

    HB[0] = 0xCAFE000F;
    while (1) {}
}
