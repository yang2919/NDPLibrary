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
_Float16 dst[VL] __attribute__((aligned(32)));
_Float16 sentinel_after[VL] __attribute__((aligned(32)));

static void uart_puts(const char *s) { while (*s) { if (*s == '\n') uart_putc('\r'); uart_putc(*s++); } }
static void print_hex16(uint16_t v) { uart_puts("0x"); for (int i=3;i>=0;i--){uint32_t n=(v>>(i*4))&0xF; uart_putc(n<10?'0'+n:'a'+(n-10));} }
static void print_hex32(uint32_t v) { uart_puts("0x"); for (int i=7;i>=0;i--){uint32_t n=(v>>(i*4))&0xF; uart_putc(n<10?'0'+n:'a'+(n-10));} }
static inline uint16_t h2u(_Float16 f) { union { _Float16 f; uint16_t u; } x = { .f = f }; return x.u; }

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;
    uart_puts("\n[fp16_vec_test] vle16/vfmacc/vse16 with adjacent src/dst\n");

    for (int i = 0; i < VL; i++) src[i] = (_Float16)1.0f;
    for (int i = 0; i < VL; i++) dst[i] = (_Float16)0.0f;
    for (int i = 0; i < VL; i++) sentinel_after[i] = (_Float16)9.0f;

    HB[0] = 0xCAFE0003;

    uint16_t src0_pre  = h2u(src[0]);
    uint16_t src1_pre  = h2u(src[1]);
    uint16_t srcL_pre  = h2u(src[VL-1]);
    uint16_t dst0_pre  = h2u(dst[0]);
    uint16_t sent_pre  = h2u(sentinel_after[0]);

    HB[0] = 0xCAFE0004;

    size_t vl;
    asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"((size_t)VL));

    _Float16 two = (_Float16)2.0f;

    asm volatile("vle16.v v0, (%0)" :: "r"(src));
    asm volatile("vfmv.v.f v1, %0" :: "f"(two));
    asm volatile("vfmacc.vf v1, %0, v0" :: "f"(two));
    asm volatile("vse16.v v1, (%0)" :: "r"(dst));

    HB[0] = 0xCAFE0005;

    uint16_t src0_post = h2u(src[0]);
    uint16_t src1_post = h2u(src[1]);
    uint16_t srcL_post = h2u(src[VL-1]);
    uint16_t dst0_post = h2u(dst[0]);
    uint16_t dst1_post = h2u(dst[1]);
    uint16_t dstL_post = h2u(dst[VL-1]);
    uint16_t sent_post = h2u(sentinel_after[0]);

    HB[0] = 0xCAFE00FF;

    int src_unchanged = (src0_post == 0x3c00) && (src1_post == 0x3c00) && (srcL_post == 0x3c00);
    int dst_correct   = (dst0_post == 0x4400) && (dst1_post == 0x4400) && (dstL_post == 0x4400);
    int sent_unchanged = (sent_post == 0x4880);
    int pass = src_unchanged && dst_correct && sent_unchanged;

    VERIFY[0] = 0xA1A1B1B1;
    VERIFY[1] = pass ? 1 : 0;
    VERIFY[2] = ((uint32_t)src0_pre << 16) | dst0_pre;
    VERIFY[3] = ((uint32_t)src0_post << 16) | src1_post;
    VERIFY[4] = ((uint32_t)srcL_post << 16) | dst0_post;
    VERIFY[5] = ((uint32_t)dst1_post << 16) | dstL_post;
    VERIFY[6] = ((uint32_t)sent_pre << 16) | sent_post;
    VERIFY[7] = (src_unchanged<<0) | (dst_correct<<1) | (sent_unchanged<<2);

    uart_puts("src[0] pre/post   = "); print_hex16(src0_pre);  uart_puts(" / "); print_hex16(src0_post);  uart_puts(" (expect 0x3c00 / 0x3c00)\n");
    uart_puts("src[1] post       = "); print_hex16(src1_post); uart_puts(" (expect 0x3c00)\n");
    uart_puts("src[end] post     = "); print_hex16(srcL_post); uart_puts(" (expect 0x3c00)\n");
    uart_puts("dst[0/1/end] post = "); print_hex16(dst0_post); uart_puts(" / "); print_hex16(dst1_post); uart_puts(" / "); print_hex16(dstL_post); uart_puts(" (expect 0x4400 each)\n");
    uart_puts("sentinel pre/post = "); print_hex16(sent_pre);  uart_puts(" / "); print_hex16(sent_post); uart_puts(" (expect 0x4880 / 0x4880)\n");
    uart_puts("pass_bits         = "); print_hex32(VERIFY[7]); uart_puts(" (src|dst<<1|sent<<2, expect 0x7)\n");
    uart_puts("verify            = "); uart_puts(pass ? "PASS\n" : "FAIL\n");

    HB[0] = 0xCAFE000F;
    while (1) {}
}
