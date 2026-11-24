// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

#include "../common/uart_apb.h"
#define RHEX_NIB 8
#include "../common/kio.h"
#include "../common/bench_print.h"

#define HB  ((volatile uint32_t *)0x40100000)
#define VER ((volatile uint32_t *)0x40100100)
#define RES ((volatile uint32_t *)0x40100200)
#define RES_MAGIC 0x5D1A6001u

#define SPM_BASE  0x30000000u
#define SPM_ELEMS (512u * 1024u / 4u)

static volatile uint32_t * const SPM   = (volatile uint32_t *)(uintptr_t)SPM_BASE;

static inline void vspm_poke(volatile uint32_t *a, uint32_t val) {
    size_t vl = __riscv_vsetvl_e32m1(1);
    __riscv_vse32_v_u32m1((uint32_t *)a, __riscv_vmv_v_x_u32m1(val, vl), vl);
}
static inline uint32_t vspm_peek(const volatile uint32_t *a) {
    size_t vl = __riscv_vsetvl_e32m1(1);
    return __riscv_vmv_x_s_u32m1_u32(__riscv_vle32_v_u32m1((const uint32_t *)a, vl));
}
static volatile uint64_t * const SPM64 = (volatile uint64_t *)(uintptr_t)SPM_BASE;

static uint32_t CTL[16]  __attribute__((aligned(64)));
static uint32_t CTL0[256] __attribute__((aligned(64)));

static const uint32_t WALK_BIT[8] = {0, 1, 2, 4, 8, 12, 14, 16};

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;
    uart_puts("\n[spm_diag] raw scratchpad read-back @ 0x30000000\n");

    RES[0] = RES_MAGIC;
    RES[1] = 0;

    for (uint32_t k = 2; k < 64u; k++) RES[k] = 0xFFFFFFFFu;

    {
        RES[56] = 0;
        asm volatile("vsetvli t0, %0, e32, m1, ta, ma" :: "r"(8) : "t0");
        asm volatile("vmv.v.i v1, 0");
        asm volatile("vmv.v.i v2, 0");
        asm volatile("vmv.v.i v8, 0");
        asm volatile("vmv.v.i v16, 0");
        asm volatile("vsetvli t0, %0, e32, m4, ta, ma" :: "r"(256) : "t0");
        asm volatile("vmv.v.i v4, 1");
        asm volatile("vredsum.vs v12, v4, v8");
        RES[56] = 1;
        asm volatile("vredsum.vs v12, v4, v4");
        RES[56] = 2;
        asm volatile("vredsum.vs v12, v4, v16");
        RES[56] = 3;
        asm volatile("vredsum.vs v12, v4, v2");
        RES[56] = 4;
        asm volatile("vredsum.vs v12, v4, v1");
        RES[56] = 5;
        uart_puts("vs1-align probe reached stage "); print_u64(RES[56]); uart_puts("\n");
    }

#ifndef NO_STEP0

    {
        HB[0] = 0xCAFE0005;
        for (uint32_t k = 0; k < 256u; k++) CTL0[k] = k;
        size_t vl = __riscv_vsetvl_e32m4(256);
        vuint32m4_t a = __riscv_vle32_v_u32m4(CTL0, vl);
        vuint32m1_t z = __riscv_vmv_v_x_u32m1(0, 1);
        HB[0] = 0xCAFE0006;
        uint32_t got = __riscv_vmv_x_s_u32m1_u32(
                           __riscv_vredsum_vs_u32m4_u32m1(a, z, vl));
        HB[0] = 0xCAFE0007;
        RES[60] = got;
        RES[61] = (got == (255u * 256u / 2u)) ? 0u : 1u;
        uart_puts("int vredsum e32m4 over DRAM: "); print_rhex(got);
        uart_puts(RES[61] == 0 ? "  OK\n" : "  WRONG\n");
    }
#endif

    for (uint32_t k = 0; k < 16u; k++) vspm_poke(&SPM[k], 0x10000000u + k);
    uint32_t pass32 = 1;
    for (uint32_t k = 0; k < 16u; k++) {
        uint32_t got = vspm_peek(&SPM[k]);
        RES[2 + k] = got;
        if (got != 0x10000000u + k) pass32 = 0;
    }
    VER[1] = pass32;
    HB[0] = 0xCAFE0010;

    for (int b = 0; b < 8; b++) vspm_poke(&SPM[1u << WALK_BIT[b]], 0xA0000000u + (uint32_t)b);
    uint32_t pass_walk = 1;
    for (int b = 0; b < 8; b++) {
        uint32_t got = vspm_peek(&SPM[1u << WALK_BIT[b]]);
        RES[18 + b] = got;
        if (got != 0xA0000000u + (uint32_t)b) pass_walk = 0;
    }
    VER[2] = pass_walk;
    HB[0] = 0xCAFE0011;

    {
        size_t vl64 = __riscv_vsetvl_e64m1(1);
        __riscv_vse64_v_u64m1((uint64_t *)&SPM64[64],
                              __riscv_vmv_v_x_u64m1(0xDEADBEEFCAFEF00DULL, vl64), vl64);
    }
    uint32_t lo = vspm_peek(&SPM[128]), hi = vspm_peek(&SPM[129]);
    RES[26] = lo; RES[27] = hi;
    uint32_t pass64 = (lo == 0xCAFEF00Du && hi == 0xDEADBEEFu);
    VER[3] = pass64;
    HB[0] = 0xCAFE0012;

    uint32_t pass_ctl = 1;
    for (uint32_t k = 0; k < 16u; k++) CTL[k] = 0x10000000u + k;
    for (uint32_t k = 0; k < 16u; k++) if (CTL[k] != 0x10000000u + k) pass_ctl = 0;
    VER[4] = pass_ctl;
    HB[0] = 0xCAFE0013;

    uart_puts("SPM[0..7] : ");
    for (int k = 0; k < 8; k++) { print_rhex(RES[2 + k]); uart_putc(' '); }
    uart_puts("\nwalk      : ");
    for (int b = 0; b < 8; b++) { print_rhex(RES[18 + b]); uart_putc(' '); }
    uart_puts("\n64-bit    : hi="); print_rhex(RES[27]);
    uart_puts(" lo="); print_rhex(RES[26]);
    uart_puts("\npass32="); print_u64(pass32);
    uart_puts(" walk="); print_u64(pass_walk);
    uart_puts(" p64="); print_u64(pass64);
    uart_puts(" dram_ctl="); print_u64(pass_ctl);
    uart_puts("\n");

    #define FENCE() asm volatile("" ::: "memory")
    {
        size_t vl = __riscv_vsetvl_e32m4(256);

        HB[0] = 0xCAFE0020; FENCE();
        vuint32m4_t v = __riscv_vid_v_u32m4(vl);
        v = __riscv_vadd_vx_u32m4(v, 0xB0000000u, vl);
        __riscv_vse32_v_u32m4((uint32_t *)SPM_BASE, v, vl);
        FENCE(); HB[0] = 0xCAFE0021; FENCE();
        uint32_t pass_vse = 1;
        for (uint32_t k = 0; k < 8u; k++) {
            uint32_t got = vspm_peek(&SPM[k]);
            RES[28 + k] = got;
            if (got != 0xB0000000u + k) pass_vse = 0;
        }
        VER[5] = pass_vse;
        FENCE(); HB[0] = 0xCAFE0022; FENCE();
        uart_puts("vse->vle   : ");
        for (int k = 0; k < 8; k++) { print_rhex(RES[28 + k]); uart_putc(' '); }
        uart_puts(pass_vse ? "  OK\n" : "  MISMATCH\n");

        for (uint32_t k = 0; k < 256u; k++) vspm_poke(&SPM[k], 0xC0000000u + k);
        FENCE(); HB[0] = 0xCAFE0023; FENCE();
        vuint32m4_t r = __riscv_vle32_v_u32m4((const uint32_t *)SPM_BASE, vl);
        FENCE(); HB[0] = 0xCAFE0024; FENCE();
        __riscv_vse32_v_u32m4(CTL, r, 8);
        uint32_t pass_vle = 1;
        for (uint32_t k = 0; k < 8u; k++) {
            RES[52 + k] = CTL[k];
            if (CTL[k] != 0xC0000000u + k) pass_vle = 0;
        }
        VER[6] = pass_vle;
        FENCE(); HB[0] = 0xCAFE0025; FENCE();
        uart_puts("scalar->vle: ");
        for (int k = 0; k < 8; k++) { print_rhex(RES[52 + k]); uart_putc(' '); }
        uart_puts(pass_vle ? "  OK\n" : "  MISMATCH\n");
    }

    {

        static uint32_t CTLBUF[256] __attribute__((aligned(64)));
        static const uint32_t LADDER[5] = {256u, 4096u, 32768u, 65536u, 131072u};
        for (int r = -1; r < 5; r++) {
            uint32_t n = (r < 0) ? 256u : LADDER[r];
            uint32_t *dst = (r < 0) ? CTLBUF : (uint32_t *)SPM_BASE;
            HB[0] = 0xCAFE0030u + (uint32_t)(r + 1);
            size_t vl = __riscv_vsetvl_e32m4(256);

            for (uint32_t i = 0; i < n; i += (uint32_t)vl) {
                vuint32m4_t v = __riscv_vid_v_u32m4(vl);
                v = __riscv_vadd_vx_u32m4(v, i, vl);
                __riscv_vse32_v_u32m4(dst + i, v, vl);
            }
            HB[0] = 0xCAFE0040u + (uint32_t)(r + 1);

            vuint32m4_t acc = __riscv_vmv_v_x_u32m4(0, vl);
            for (uint32_t i = 0; i < n; i += (uint32_t)vl)
                acc = __riscv_vadd_vv_u32m4(
                          acc, __riscv_vle32_v_u32m4(dst + i, vl), vl);
            vuint32m1_t z = __riscv_vmv_v_x_u32m1(0, 1);
            uint32_t got = __riscv_vmv_x_s_u32m1_u32(
                               __riscv_vredsum_vs_u32m4_u32m1(acc, z, vl));
            uint32_t want = (uint32_t)(((uint64_t)n * (n - 1)) / 2);
            RES[36 + (r + 1)] = got;
            RES[44 + (r + 1)] = (got == want) ? 0u : 1u;
            uart_puts(r < 0 ? "ctl(DRAM) n=" : "ladder n="); print_u64(n);
            uart_puts(got == want ? "  OK\n" : "  MISMATCH\n");
        }
    }

    {
        static uint32_t SBUF[512] __attribute__((aligned(64)));
        for (uint32_t k = 64; k < 80u; k++) RES[k] = 0xFFFFFFFFu;

        for (int r = 0; r < 4; r++) {
            uint32_t *dst = (r == 0) ? SBUF : (uint32_t *)SPM_BASE;
            uint32_t req = (r <= 1) ? 8u : (r == 2 ? 64u : 256u);
            int m4 = (r >= 2);
            HB[0] = 0xCAFE0050u + (uint32_t)r;

            size_t vl = m4 ? __riscv_vsetvl_e32m4(req) : __riscv_vsetvl_e32m1(req);
            RES[72 + r] = (uint32_t)vl;
            uint32_t n = (uint32_t)vl * 2u;

            for (uint32_t i = 0; i < n; i += (uint32_t)vl) {
                if (m4) __riscv_vse32_v_u32m4(dst + i, __riscv_vmv_v_x_u32m4(0, vl), vl);
                else    __riscv_vse32_v_u32m1(dst + i, __riscv_vmv_v_x_u32m1(0, vl), vl);
            }

            if (m4) {
                vuint32m4_t v = __riscv_vadd_vx_u32m4(__riscv_vid_v_u32m4(vl), 1u, vl);
                __riscv_vsse32_v_u32m4(dst + 1, 8, v, vl);
            } else {
                vuint32m1_t v = __riscv_vadd_vx_u32m1(__riscv_vid_v_u32m1(vl), 1u, vl);
                __riscv_vsse32_v_u32m1(dst + 1, 8, v, vl);
            }
            HB[0] = 0xCAFE0058u + (uint32_t)r;

            for (uint32_t i = 0; i < n; i += (uint32_t)vl) {
                if (m4) __riscv_vse32_v_u32m4(SBUF + i, __riscv_vle32_v_u32m4(dst + i, vl), vl);
                else    __riscv_vse32_v_u32m1(SBUF + i, __riscv_vle32_v_u32m1(dst + i, vl), vl);
            }
            uint32_t bad = 0;
            for (uint32_t k = 0; k < n; k++) {
                uint32_t want = (k & 1u) ? (k / 2u + 1u) : 0u;
                if (SBUF[k] != want) bad++;
            }
            RES[64 + r] = bad;
            uart_puts("strided r="); print_u64((uint64_t)r);
            uart_puts(" bad="); print_u64(bad); uart_puts("\n");
        }

    }

    RES[1] = 49;
    VER[0] = 0xA1A1B1B1;
    uart_puts("spm_diag done\n");
    HB[0] = 0xCAFE000F;
    while (1) {}
}
