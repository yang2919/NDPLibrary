// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>

#include "../common/platform.h"
#include "../common/uart_apb.h"
#define RHEX_NIB 8
#include "../common/kio.h"
#include "../common/verify_block.h"
#include "../common/fp8_asm.h"

#define RES VB_RES
#define RES_MAGIC 0xF8080001u

#define N 128

static uint8_t A[N] __attribute__((aligned(64)));
static uint8_t B[N] __attribute__((aligned(64)));
static uint8_t C[N] __attribute__((aligned(64)));
static uint8_t O[N] __attribute__((aligned(64)));
static uint8_t R[N] __attribute__((aligned(64)));

static const uint8_t ADD_A[8] = {0x3c, 0x40, 0x44, 0x38, 0x3c, 0x40, 0x44, 0x48};
static const uint8_t ADD_B[8] = {0x3c, 0x40, 0x40, 0x38, 0x40, 0x44, 0x3c, 0x44};
static const uint8_t MUL_A[8] = {0x3c, 0x40, 0x44, 0x38, 0x3e, 0x40, 0xbc, 0x48};
static const uint8_t MUL_B[8] = {0x40, 0x40, 0x38, 0x38, 0x40, 0xc0, 0x44, 0x34};
static const uint8_t MAC_A[8] = {0x3c, 0x40, 0x3c, 0x40, 0x3c, 0x38, 0x44, 0x40};
static const uint8_t MAC_B[8] = {0x3c, 0x40, 0x40, 0x3c, 0x40, 0x38, 0x40, 0x40};
static const uint8_t MAC_C[8] = {0x3c, 0x00, 0x40, 0x3c, 0x44, 0x38, 0x00, 0x40};

static const uint8_t TIE_A[8] = {0x3c, 0x3d, 0x3e, 0x3f, 0xbc, 0xbd, 0xbe, 0xbf};
static const uint8_t TIE_B[8] = {0x30, 0x30, 0x30, 0x30, 0xb0, 0xb0, 0xb0, 0xb0};
static const uint8_t TIE_R[8] = {0x3c, 0x3e, 0x3e, 0x40, 0xbc, 0xbe, 0xbe, 0xc0};

static uint32_t g_inexact = 0;

static uint8_t golden(float v) {
    uint8_t r;
    if (!fp8_from_f32_exact(v, &r)) { g_inexact++; return 0xFFu; }
    return r;
}
static uint32_t compare(const uint8_t *got, const uint8_t *want, int n) {
    uint32_t bad = 0;
    for (int i = 0; i < n; i++) if (got[i] != want[i]) bad++;
    return bad;
}
static void pack16(uint32_t slot, const uint8_t *v) {
    for (int w = 0; w < 4; w++)
        RES[slot + (uint32_t)w] = (uint32_t)v[w * 4] | ((uint32_t)v[w * 4 + 1] << 8)
                                | ((uint32_t)v[w * 4 + 2] << 16) | ((uint32_t)v[w * 4 + 3] << 24);
}
static void report(const char *name, uint32_t bad, const uint8_t *o) {
    uart_puts(name); uart_puts(" bad="); print_u64(bad);
    uart_puts("  out[0..3]=");
    for (int i = 0; i < 4; i++) {
        uint32_t b = o[i];
        uart_putc(' ');
        uart_putc((char)((b >> 4) < 10 ? '0' + (b >> 4) : 'a' + (b >> 4) - 10));
        uart_putc((char)((b & 15) < 10 ? '0' + (b & 15) : 'a' + (b & 15) - 10));
    }
    uart_puts("\n");
}

int main(void) {
    VB_HB[0] = 0xCAFE0001;
    uart_init(ARA_UART_DIVISOR);
    VB_HB[0] = 0xCAFE0002;
    uart_puts("\n[fp8_diag] fp8 (E5M2) arithmetic at vsew=EW8\n");

    RES[0] = RES_MAGIC;
    RES[1] = N;
    for (uint32_t k = 2; k < 32u; k++) RES[k] = 0xFFFFFFFFu;
    vb_arm();
    uint64_t t0 = rdcycle();

    VB_HB[0] = 0xCAFE0010;
    for (int i = 0; i < N; i++) {
        A[i] = ADD_A[i & 7]; B[i] = ADD_B[i & 7];
        R[i] = golden(fp8_to_f32(A[i]) + fp8_to_f32(B[i]));
        O[i] = 0;
    }
    fp8_add(A, B, O, N);
    uint32_t bad_add = compare(O, R, N);
    VB_VER[3] = bad_add;
    pack16(2, O);
    report("add ", bad_add, O);

    VB_HB[0] = 0xCAFE0011;
    for (int i = 0; i < N; i++) {
        A[i] = MUL_A[i & 7]; B[i] = MUL_B[i & 7];
        R[i] = golden(fp8_to_f32(A[i]) * fp8_to_f32(B[i]));
        O[i] = 0;
    }
    fp8_mul(A, B, O, N);
    uint32_t bad_mul = compare(O, R, N);
    VB_VER[4] = bad_mul;
    pack16(6, O);
    report("mul ", bad_mul, O);

    VB_HB[0] = 0xCAFE0012;
    for (int i = 0; i < N; i++) {
        A[i] = MAC_A[i & 7]; B[i] = MAC_B[i & 7]; C[i] = MAC_C[i & 7];
        R[i] = golden(fp8_to_f32(C[i]) + fp8_to_f32(A[i]) * fp8_to_f32(B[i]));
        O[i] = 0;
    }
    fp8_macc(A, B, C, O, N);
    uint32_t bad_macc = compare(O, R, N);
    VB_VER[5] = bad_macc;
    pack16(10, O);
    report("macc", bad_macc, O);

    VB_HB[0] = 0xCAFE0013;
    for (int i = 0; i < N; i++) {
        A[i] = TIE_A[i & 7]; B[i] = TIE_B[i & 7]; R[i] = TIE_R[i & 7];
        O[i] = 0;
    }
    fp8_add(A, B, O, N);
    uint32_t bad_tie = compare(O, R, N);
    VB_VER[6] = bad_tie;
    pack16(14, O);
    report("tie ", bad_tie, O);

    VB_VER[7] = g_inexact;
    uart_puts("golden_inexact="); print_u64(g_inexact); uart_puts("\n");

    vb_publish((bad_add == 0 && bad_mul == 0 && bad_macc == 0 && bad_tie == 0
                && g_inexact == 0) ? 1u : 0u,
               (uint32_t)(rdcycle() - t0),
               bad_add, bad_mul, bad_macc, bad_tie, g_inexact);
    uart_puts("fp8_diag done\n");
    VB_HB[0] = 0xCAFE000F;
    while (1) {}
}
