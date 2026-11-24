// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>

#include "../common/platform.h"
#include "../common/uart_apb.h"
#define RHEX_NIB 8
#include "../common/kio.h"
#include "../common/verify_block.h"

#define RES       VB_RES
#define RES_MAGIC 0xF8160001u

#define N 64

static const uint8_t IN[12] = {0x3c, 0x40, 0x44, 0x38, 0x3d, 0x3e, 0x3f, 0xbc,
                               0x01, 0x04, 0x7b, 0xb6};

static uint8_t  A[N]  __attribute__((aligned(64)));
static uint8_t  ONE[N] __attribute__((aligned(64)));
static uint16_t ACC[N] __attribute__((aligned(64)));
static uint16_t OUT_VV[N] __attribute__((aligned(64)));
static uint16_t OUT_VF[N] __attribute__((aligned(64)));

static __attribute__((noinline)) void widen_vv(const uint8_t *a, const uint8_t *b,
                                               const uint16_t *acc, uint16_t *o) {
    size_t vl;
    asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(vl) : "r"((size_t)N));
    asm volatile("vle16.v v4, (%0)" :: "r"(acc) : "memory");
    asm volatile("vsetvli %0, %1, e8, m1, ta, ma" : "=r"(vl) : "r"((size_t)N));
    asm volatile("vle8.v v2, (%0)" :: "r"(a) : "memory");
    asm volatile("vle8.v v6, (%0)" :: "r"(b) : "memory");
    asm volatile("vfwmacc.vv v4, v2, v6");
    asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(vl) : "r"((size_t)N));
    asm volatile("vse16.v v4, (%0)" :: "r"(o) : "memory");
}

static __attribute__((noinline)) void widen_vf(const uint8_t *a, uint64_t boxed,
                                               const uint16_t *acc, uint16_t *o) {
    size_t vl;
    double s;
    asm volatile("fmv.d.x %0, %1" : "=f"(s) : "r"(boxed));
    asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(vl) : "r"((size_t)N));
    asm volatile("vle16.v v4, (%0)" :: "r"(acc) : "memory");
    asm volatile("vsetvli %0, %1, e8, m1, ta, ma" : "=r"(vl) : "r"((size_t)N));
    asm volatile("vle8.v v2, (%0)" :: "r"(a) : "memory");
    asm volatile("vfwmacc.vf v4, %0, v2" :: "f"(s));
    asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(vl) : "r"((size_t)N));
    asm volatile("vse16.v v4, (%0)" :: "r"(o) : "memory");
}

static uint32_t check(const uint16_t *o, uint32_t *first_bad, const char *name) {
    uint32_t bad = 0;
    *first_bad = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < N; i++) {
        uint16_t want = (uint16_t)((uint16_t)A[i] << 8);
        if (o[i] != want) {
            if (bad == 0) *first_bad = i;
            bad++;
        }
    }
    uart_puts(name); uart_puts(" bad="); print_u64(bad);
    uart_puts(" of "); print_u64(N); uart_puts("\n");
    return bad;
}

static void pack16(uint32_t base, const uint16_t *o) {
    for (uint32_t w = 0; w < 8; w++)
        RES[base + w] = (uint32_t)o[2 * w] | ((uint32_t)o[2 * w + 1] << 16);
}

int main(void) {
    VB_HB[0] = 0xCAFE0001;
    uart_init(ARA_UART_DIVISOR);
    VB_HB[0] = 0xCAFE0002;
    uart_puts("\n[fp8_widen_diag] fp8 (E5M2) -> fp16 widening, .vv and .vf\n");

    RES[0] = RES_MAGIC;
    RES[1] = N;
    for (uint32_t k = 2; k < 32u; k++) RES[k] = 0xFFFFFFFFu;
    vb_arm();

    for (uint32_t i = 0; i < N; i++) {
        A[i]   = IN[i % 12];
        ONE[i] = 0x3c;
        ACC[i] = 0;
        OUT_VV[i] = 0xDEAD;
        OUT_VF[i] = 0xDEAD;
    }
    uint64_t t0 = rdcycle();

    VB_HB[0] = 0xCAFE0010;
    widen_vv(A, ONE, ACC, OUT_VV);
    uint32_t first_vv, bad_vv = check(OUT_VV, &first_vv, "vfwmacc.vv");

    VB_HB[0] = 0xCAFE0011;
    widen_vf(A, 0xFFFFFFFFFFFFFF3Cull, ACC, OUT_VF);
    uint32_t first_vf, bad_vf = check(OUT_VF, &first_vf, "vfwmacc.vf");

    uint64_t t1 = rdcycle();
    pack16(2, OUT_VV);
    pack16(10, OUT_VF);

    uint32_t pass = (bad_vv == 0 && bad_vf == 0);
    uart_puts(pass ? "[fp8_widen_diag] PASS\n" : "[fp8_widen_diag] FAIL\n");

    vb_publish(pass, (uint32_t)(t1 - t0), bad_vv, bad_vf, N, first_vv, first_vf);
    VB_HB[0] = 0xCAFE000F;
    while (1) {}
}
