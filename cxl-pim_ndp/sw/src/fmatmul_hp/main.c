// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "../common/uart_apb.h"
#include "../common/bench_print.h"
#include "hp-fmatmul.h"

#define HB     ((volatile uint32_t *)0x40100000)
#define VERIFY ((volatile uint32_t *)0x40100100)

#define M 64
#define N 64
#define K 64

_Float16 A[M*N] __attribute__((aligned(32)));
_Float16 B[N*K] __attribute__((aligned(32)));
_Float16 C[M*K] __attribute__((aligned(32)));

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

static void print_hex16(uint16_t v) {
    uart_puts("0x");
    for (int i = 3; i >= 0; i--) {
        uint32_t n = (v >> (i*4)) & 0xF;
        uart_putc(n < 10 ? '0' + n : 'a' + (n - 10));
    }
}

static inline uint16_t h2u(_Float16 f) { union { _Float16 f; uint16_t u; } x = { .f = f }; return x.u; }

static _Float16 fabs_h(_Float16 x) { return x < (_Float16)0 ? -x : x; }
static int near_h(_Float16 a, _Float16 b, _Float16 tol) { return fabs_h(a - b) <= tol; }

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;

    uart_puts("\n[fmatmul_hp] 64x64 fp16 ara vector matmul\n");

    for (int i = 0; i < M*N; i++) A[i] = (_Float16)0.0f;
    for (int i = 0; i < M; i++)   A[i*N + i] = (_Float16)1.0f;
    for (int i = 0; i < N*K; i++) B[i] = (_Float16)1.0f;
    for (int i = 0; i < M*K; i++) C[i] = (_Float16)0.0f;
    HB[0] = 0xCAFE0003;

    uint64_t t0 = rdcycle();
    hp_fmatmul(C, A, B, M, N, K);
    uint64_t t1 = rdcycle();
    HB[0] = 0xCAFE0004;

    _Float16 ref0 = 0, ref_mid = 0, ref_last = 0;
    int mid_i = (M*K)/2 / K, mid_j = (M*K)/2 % K;
    int last_i = M-1, last_j = K-1;
    for (int n = 0; n < N; n++) {
        ref0     += A[0*N + n]      * B[n*K + 0];
        ref_mid  += A[mid_i*N + n]  * B[n*K + mid_j];
        ref_last += A[last_i*N + n] * B[n*K + last_j];
    }
    HB[0] = 0xCAFE0005;

    int pass = near_h(C[0],         ref0,     (_Float16)0.05)
            && near_h(C[(M*K)/2],   ref_mid,  (_Float16)0.05)
            && near_h(C[M*K-1],     ref_last, (_Float16)0.05);

    VERIFY[0] = 0xA1A1B1B1;
    VERIFY[1] = pass ? 1 : 0;
    VERIFY[2] = (uint32_t)(t1 - t0);

    VERIFY[3] = ((uint32_t)h2u(A[0])     << 16) | (uint32_t)h2u(B[0]);
    VERIFY[4] = ((uint32_t)h2u(A[N+1])   << 16) | (uint32_t)h2u(B[N]);
    VERIFY[5] = ((uint32_t)h2u(C[0])     << 16) | (uint32_t)h2u(C[M*K-1]);
    VERIFY[6] = ((uint32_t)h2u(ref0)     << 16) | (uint32_t)h2u(ref_last);
    VERIFY[7] = ((uint32_t)h2u(ref_mid)  << 16);

    bp_perf(2ull * M * N * K, t1 - t0, 32);
    uart_puts("c[0]      = ");        print_hex16(h2u(C[0]));      uart_puts("\n");
    uart_puts("c[mid]    = ");        print_hex16(h2u(C[(M*K)/2])); uart_puts("\n");
    uart_puts("c[last]   = ");        print_hex16(h2u(C[M*K-1]));  uart_puts("\n");
    uart_puts("ref[0]    = ");        print_hex16(h2u(ref0));      uart_puts("\n");
    uart_puts("ref[last] = ");        print_hex16(h2u(ref_last));  uart_puts("\n");
    uart_puts("verify    = ");        uart_puts(pass ? "PASS\n" : "FAIL\n");

    HB[0] = 0xCAFE000F;
    while (1) {}
}
