// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
#include "../common/uart_apb.h"

#define HB     ((volatile uint32_t *)0x40100000)
#define VERIFY ((volatile uint32_t *)0x40100100)

#define VL 64

_Float16 src[VL] __attribute__((aligned(32)));
_Float16 dst1[VL] __attribute__((aligned(32)));
_Float16 dst2[VL] __attribute__((aligned(32)));
_Float16 dst3[VL] __attribute__((aligned(32)));

static void uart_puts(const char *s) { while (*s) { if (*s == '\n') uart_putc('\r'); uart_putc(*s++); } }
static void print_hex16(uint16_t v) { uart_puts("0x"); for (int i=3;i>=0;i--){uint32_t n=(v>>(i*4))&0xF; uart_putc(n<10?'0'+n:'a'+(n-10));} }
static inline uint16_t h2u(_Float16 f) { union { _Float16 f; uint16_t u; } x = { .f = f }; return x.u; }

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;
    uart_puts("\n[fp16_vec_lane0] split fp16 vector ops to localise element-0 bug\n");

    for (int i = 0; i < VL; i++) src[i]  = (_Float16)1.0f;
    for (int i = 0; i < VL; i++) dst1[i] = (_Float16)0.5f;
    for (int i = 0; i < VL; i++) dst2[i] = (_Float16)0.5f;
    for (int i = 0; i < VL; i++) dst3[i] = (_Float16)0.5f;
    HB[0] = 0xCAFE0003;

    size_t vl;
    asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"((size_t)VL));

    _Float16 two = (_Float16)2.0f;
    _Float16 zero = (_Float16)0.0f;

    HB[0] = 0xCAFE0011;
    asm volatile("vfmv.v.f v1, %0" :: "f"(two));
    asm volatile("vse16.v v1, (%0)" :: "r"(dst1));
    HB[0] = 0xCAFE0012;

    asm volatile("vle16.v v2, (%0)" :: "r"(src));
    asm volatile("vse16.v v2, (%0)" :: "r"(dst2));
    HB[0] = 0xCAFE0022;

    asm volatile("vfmv.v.f v3, %0" :: "f"(zero));
    asm volatile("vle16.v v4, (%0)" :: "r"(src));
    asm volatile("vfmacc.vf v3, %0, v4" :: "f"(two));
    asm volatile("vse16.v v3, (%0)" :: "r"(dst3));
    HB[0] = 0xCAFE0032;

    HB[0] = 0xCAFE00FF;

    uint16_t d10 = h2u(dst1[0]), d11 = h2u(dst1[1]), d1L = h2u(dst1[VL-1]);
    uint16_t d20 = h2u(dst2[0]), d21 = h2u(dst2[1]), d2L = h2u(dst2[VL-1]);
    uint16_t d30 = h2u(dst3[0]), d31 = h2u(dst3[1]), d3L = h2u(dst3[VL-1]);

    int t1_ok = (d10 == 0x4000) && (d11 == 0x4000) && (d1L == 0x4000);
    int t2_ok = (d20 == 0x3c00) && (d21 == 0x3c00) && (d2L == 0x3c00);
    int t3_ok = (d30 == 0x4000) && (d31 == 0x4000) && (d3L == 0x4000);

    VERIFY[0] = 0xA1A1B1B1;
    VERIFY[1] = (t1_ok && t2_ok && t3_ok) ? 1 : 0;
    VERIFY[2] = ((uint32_t)d10 << 16) | d11;
    VERIFY[3] = ((uint32_t)d1L << 16) | d20;
    VERIFY[4] = ((uint32_t)d21 << 16) | d2L;
    VERIFY[5] = ((uint32_t)d30 << 16) | d31;
    VERIFY[6] = ((uint32_t)d3L << 16);
    VERIFY[7] = (t1_ok<<0) | (t2_ok<<1) | (t3_ok<<2);

    uart_puts("T1 vfmv  dst[0/1/end] = "); print_hex16(d10); uart_puts("/"); print_hex16(d11); uart_puts("/"); print_hex16(d1L); uart_puts(" (expect 0x4000 each)\n");
    uart_puts("T2 vle   dst[0/1/end] = "); print_hex16(d20); uart_puts("/"); print_hex16(d21); uart_puts("/"); print_hex16(d2L); uart_puts(" (expect 0x3c00 each)\n");
    uart_puts("T3 vfmacc dst[0/1/end]= "); print_hex16(d30); uart_puts("/"); print_hex16(d31); uart_puts("/"); print_hex16(d3L); uart_puts(" (expect 0x4000 each)\n");
    uart_puts("ok_bits = "); print_hex16((uint16_t)VERIFY[7]); uart_puts(" (T1|T2<<1|T3<<2, expect 0x7)\n");

    HB[0] = 0xCAFE000F;
    while (1) {}
}
