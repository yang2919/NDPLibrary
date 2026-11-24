// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#ifndef ARA_VHK158_BF16_MATMUL_H
#define ARA_VHK158_BF16_MATMUL_H

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

static inline float bf16_to_f32(uint16_t b) {
    union { uint32_t u; float f; } x = { .u = (uint32_t)b << 16 };
    return x.f;
}
static inline uint16_t bf16_from_f32_trunc(float f) {
    union { float f; uint32_t u; } x = { .f = f };
    return (uint16_t)(x.u >> 16);
}

static inline __bf16 bf16_scalar_of_bits(uint16_t b) {
    union { uint16_t u; __bf16 b; } x = { .u = b };
    return x.b;
}

static void bf16_fmatmul(float *c, const uint16_t *a, const uint16_t *b,
                         unsigned M, unsigned N, unsigned P) {
    for (unsigned p = 0; p < P; ) {
        size_t vl = __riscv_vsetvl_e16m1((size_t)(P - p));
        for (unsigned m = 0; m < M; m++) {
            vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            for (unsigned n = 0; n < N; n++) {
                __bf16 s = bf16_scalar_of_bits(a[(size_t)m * N + n]);
                vbfloat16m1_t vb =
                    __riscv_vle16_v_bf16m1((const __bf16 *)(b + (size_t)n * P + p), vl);
                acc = __riscv_vfwmaccbf16_vf_f32m2(acc, s, vb, vl);
            }
            __riscv_vse32_v_f32m2(c + (size_t)m * P + p, acc, vl);
        }
        p += (unsigned)vl;
    }
}

static float bf16_dotp(const uint16_t *a, const uint16_t *b, size_t n) {
    size_t vl0 = __riscv_vsetvl_e16m1(n);
    vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl0);
    for (size_t i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e16m1(n - i);
        vbfloat16m1_t va = __riscv_vle16_v_bf16m1((const __bf16 *)(a + i), vl);
        vbfloat16m1_t vb = __riscv_vle16_v_bf16m1((const __bf16 *)(b + i), vl);
        acc = __riscv_vfwmaccbf16_vv_f32m2(acc, va, vb, vl);
        i += vl;
    }

    vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, vl0);
    return __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(acc, z, vl0));
}

#endif
