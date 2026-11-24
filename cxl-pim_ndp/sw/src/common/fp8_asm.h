// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#ifndef ARA_VHK158_FP8_ASM_H
#define ARA_VHK158_FP8_ASM_H

#include <stdint.h>
#include <stddef.h>

static __attribute__((noinline))
void fp8_add(const uint8_t *a, const uint8_t *b, uint8_t *o, int n) {
    size_t vl;
    while (n > 0) {
        asm volatile("vsetvli %0, %1, e8, m1, ta, ma" : "=r"(vl) : "r"((size_t)n));
        asm volatile("vle8.v v2, (%0)" :: "r"(a) : "memory");
        asm volatile("vle8.v v4, (%0)" :: "r"(b) : "memory");
        asm volatile("vfadd.vv v6, v2, v4");
        asm volatile("vse8.v v6, (%0)" :: "r"(o) : "memory");
        a += vl; b += vl; o += vl; n -= (int)vl;
    }
}

static __attribute__((noinline))
void fp8_mul(const uint8_t *a, const uint8_t *b, uint8_t *o, int n) {
    size_t vl;
    while (n > 0) {
        asm volatile("vsetvli %0, %1, e8, m1, ta, ma" : "=r"(vl) : "r"((size_t)n));
        asm volatile("vle8.v v2, (%0)" :: "r"(a) : "memory");
        asm volatile("vle8.v v4, (%0)" :: "r"(b) : "memory");
        asm volatile("vfmul.vv v6, v2, v4");
        asm volatile("vse8.v v6, (%0)" :: "r"(o) : "memory");
        a += vl; b += vl; o += vl; n -= (int)vl;
    }
}

static __attribute__((noinline))
void fp8_macc(const uint8_t *a, const uint8_t *b, const uint8_t *c,
              uint8_t *o, int n) {
    size_t vl;
    while (n > 0) {
        asm volatile("vsetvli %0, %1, e8, m1, ta, ma" : "=r"(vl) : "r"((size_t)n));
        asm volatile("vle8.v v2, (%0)" :: "r"(a) : "memory");
        asm volatile("vle8.v v4, (%0)" :: "r"(b) : "memory");
        asm volatile("vle8.v v6, (%0)" :: "r"(c) : "memory");
        asm volatile("vfmacc.vv v6, v2, v4");
        asm volatile("vse8.v v6, (%0)" :: "r"(o) : "memory");
        a += vl; b += vl; c += vl; o += vl; n -= (int)vl;
    }
}

static inline float fp8_to_f32(uint8_t b) {
    uint32_t s = (uint32_t)(b >> 7) & 1u;
    uint32_t e = (uint32_t)(b >> 2) & 0x1Fu;
    uint32_t m = (uint32_t)b & 3u;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s << 31;
        else {
            int32_t exp = -14;
            uint32_t man = m;
            while (!(man & 4u)) { man <<= 1; exp--; }
            u = (s << 31) | ((uint32_t)(exp + 127) << 23) | ((man & 3u) << 21);
        }
    } else if (e == 31) {
        u = (s << 31) | (255u << 23) | (m ? (1u << 22) : 0u);
    } else {
        u = (s << 31) | ((e - 15u + 127u) << 23) | (m << 21);
    }
    union { uint32_t u; float f; } x = { .u = u };
    return x.f;
}

static inline int fp8_from_f32_exact(float f, uint8_t *out) {
    union { float f; uint32_t u; } x = { .f = f };
    uint32_t s = x.u >> 31, e = (x.u >> 23) & 0xFFu, m = x.u & 0x7FFFFFu;
    if (e == 0 && m == 0) { *out = (uint8_t)(s << 7); return 1; }
    if (e == 0 || e == 255) return 0;
    if (m & 0x1FFFFFu) return 0;
    int32_t ne = (int32_t)e - 127 + 15;
    if (ne < 1 || ne > 30) return 0;
    *out = (uint8_t)((s << 7) | ((uint32_t)ne << 2) | (m >> 21));
    return 1;
}

#endif
