// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

#include "../common/platform.h"
#include "../common/uart_apb.h"
#define RHEX_NIB 8
#include "../common/kio.h"
#include "../common/verify_block.h"

#define RES VB_RES
#define RES_MAGIC 0xBF160001u

#define N 128

static uint16_t A16[N] __attribute__((aligned(64)));
static uint16_t B16[N] __attribute__((aligned(64)));
static float    ACC[N] __attribute__((aligned(64)));
static float    OUT[N] __attribute__((aligned(64)));
static float    REF[N] __attribute__((aligned(64)));

static inline uint16_t bf16_of_f32(float f) {
    union { float f; uint32_t u; } x = { .f = f };
    return (uint16_t)(x.u >> 16);
}
static inline float f32_of_bf16(uint16_t b) {
    union { uint32_t u; float f; } x = { .u = (uint32_t)b << 16 };
    return x.f;
}

static inline __bf16 bf16_scalar(uint16_t b) {
    union { uint16_t u; __bf16 b; } x = { .u = b };
    return x.b;
}

int main(void) {
    VB_HB[0] = 0xCAFE0001;
    uart_init(ARA_UART_DIVISOR);
    VB_HB[0] = 0xCAFE0002;
    uart_puts("\n[bf16_diag] vfwmaccbf16 on silicon\n");

    RES[0] = RES_MAGIC;
    RES[1] = N;
    for (uint32_t k = 2; k < 32u; k++) RES[k] = 0xFFFFFFFFu;
    vb_arm();
    uint64_t t0 = rdcycle();

    for (int i = 0; i < N; i++) {
        float a = (float)((i % 8) + 1);
        float b = 0.5f * (float)((i % 7) + 1);
        float c = (float)(i % 5);
        A16[i] = bf16_of_f32(a);
        B16[i] = bf16_of_f32(b);
        ACC[i] = c;
        REF[i] = c + a * b;
        OUT[i] = 0.0f;
    }

    VB_HB[0] = 0xCAFE0010;
    for (int i = 0; i < N; ) {
        size_t vl = __riscv_vsetvl_e16m1((size_t)(N - i));
        vfloat32m2_t acc = __riscv_vle32_v_f32m2(ACC + i, vl);
        vbfloat16m1_t va = __riscv_vle16_v_bf16m1((const __bf16 *)(A16 + i), vl);
        vbfloat16m1_t vb = __riscv_vle16_v_bf16m1((const __bf16 *)(B16 + i), vl);
        acc = __riscv_vfwmaccbf16_vv_f32m2(acc, va, vb, vl);
        __riscv_vse32_v_f32m2(OUT + i, acc, vl);
        i += (int)vl;
    }
    uint32_t bad_vv = 0;
    for (int i = 0; i < N; i++) if (OUT[i] != REF[i]) bad_vv++;
    VB_VER[3] = bad_vv;
    for (int i = 0; i < 8; i++) { union { float f; uint32_t u; } x = { .f = OUT[i] }; RES[2 + i] = x.u; }
    VB_HB[0] = 0xCAFE0011;
    uart_puts("vv  bad="); print_u64(bad_vv);
    uart_puts("  out[0]="); print_rhex(RES[2]);
    uart_puts(" out[1]="); print_rhex(RES[3]); uart_puts("\n");

    const uint16_t sb = bf16_of_f32(3.0f);
    const __bf16 s = bf16_scalar(sb);
    for (int i = 0; i < N; i++) {
        ACC[i] = (float)(i % 5);
        REF[i] = (float)(i % 5) + 3.0f * f32_of_bf16(B16[i]);
        OUT[i] = 0.0f;
    }
    VB_HB[0] = 0xCAFE0020;
    for (int i = 0; i < N; ) {
        size_t vl = __riscv_vsetvl_e16m1((size_t)(N - i));
        vfloat32m2_t acc = __riscv_vle32_v_f32m2(ACC + i, vl);
        vbfloat16m1_t vb = __riscv_vle16_v_bf16m1((const __bf16 *)(B16 + i), vl);
        acc = __riscv_vfwmaccbf16_vf_f32m2(acc, s, vb, vl);
        __riscv_vse32_v_f32m2(OUT + i, acc, vl);
        i += (int)vl;
    }
    uint32_t bad_vf = 0;
    for (int i = 0; i < N; i++) if (OUT[i] != REF[i]) bad_vf++;
    VB_VER[4] = bad_vf;
    for (int i = 0; i < 8; i++) { union { float f; uint32_t u; } x = { .f = OUT[i] }; RES[10 + i] = x.u; }
    VB_HB[0] = 0xCAFE0021;
    uart_puts("vf  bad="); print_u64(bad_vf);
    uart_puts("  out[0]="); print_rhex(RES[10]);
    uart_puts(" out[1]="); print_rhex(RES[11]); uart_puts("\n");

    vb_publish((bad_vv == 0 && bad_vf == 0) ? 1u : 0u,
               (uint32_t)(rdcycle() - t0), bad_vv, bad_vf, 0, 0, 0);
    uart_puts("bf16_diag done\n");
    VB_HB[0] = 0xCAFE000F;
    while (1) {}
}
