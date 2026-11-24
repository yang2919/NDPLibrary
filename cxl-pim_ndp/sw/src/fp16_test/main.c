// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include "../common/uart_apb.h"

#define HB     ((volatile uint32_t *)0x40100000)
#define VERIFY ((volatile uint32_t *)0x40100100)

volatile _Float16 storage __attribute__((aligned(2)));

static void uart_puts(const char *s) { while (*s) { if (*s == '\n') uart_putc('\r'); uart_putc(*s++); } }
static void print_hex32(uint32_t v) { uart_puts("0x"); for (int i=7;i>=0;i--){uint32_t n=(v>>(i*4))&0xF; uart_putc(n<10?'0'+n:'a'+(n-10));} }
static void print_hex16(uint16_t v) { uart_puts("0x"); for (int i=3;i>=0;i--){uint32_t n=(v>>(i*4))&0xF; uart_putc(n<10?'0'+n:'a'+(n-10));} }
static inline uint16_t h2u(_Float16 f) { union { _Float16 f; uint16_t u; } x = { .f = f }; return x.u; }
static inline uint32_t f2u(float f) { union { float f; uint32_t u; } x = { .f = f }; return x.u; }

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;
    uart_puts("\n[fp16_test] isolated fp16 scalar ops\n");

    HB[0] = 0xCAFE0011;
    storage = (_Float16)2.5f;
    HB[0] = 0xCAFE0012;
    uint16_t t1_raw = h2u(storage);

    HB[0] = 0xCAFE0021;
    _Float16 loaded = storage;
    HB[0] = 0xCAFE0022;
    uint16_t t2_raw = h2u(loaded);

    HB[0] = 0xCAFE0031;
    _Float16 a = (_Float16)2.5f;
    _Float16 b = (_Float16)1.5f;
    _Float16 c_add = a + b;
    HB[0] = 0xCAFE0032;
    uint16_t t3_raw = h2u(c_add);

    HB[0] = 0xCAFE0041;
    _Float16 c_mul = a * b;
    HB[0] = 0xCAFE0042;
    uint16_t t4_raw = h2u(c_mul);

    HB[0] = 0xCAFE0051;
    float f32 = (float)c_add;
    HB[0] = 0xCAFE0052;
    uint32_t t5_raw = f2u(f32);

    HB[0] = 0xCAFE0061;
    _Float16 back_h = (_Float16)f32;
    HB[0] = 0xCAFE0062;
    uint16_t t6_raw = h2u(back_h);

    HB[0] = 0xCAFE00FF;

    int t1_ok = (t1_raw == 0x4100);
    int t2_ok = (t2_raw == 0x4100);
    int t3_ok = (t3_raw == 0x4400);
    int t4_ok = (t4_raw == 0x43c0);
    int t5_ok = (t5_raw == 0x40800000);
    int t6_ok = (t6_raw == 0x4400);

    VERIFY[0] = 0xA1A1B1B1;
    VERIFY[1] = (t1_ok && t2_ok && t3_ok && t4_ok && t5_ok && t6_ok) ? 1 : 0;
    VERIFY[2] = ((uint32_t)t1_raw << 16) | t2_raw;
    VERIFY[3] = ((uint32_t)t3_raw << 16) | t4_raw;
    VERIFY[4] = t5_raw;
    VERIFY[5] = (uint32_t)t6_raw;
    VERIFY[6] = (t1_ok<<0) | (t2_ok<<1) | (t3_ok<<2) | (t4_ok<<3) | (t5_ok<<4) | (t6_ok<<5);
    VERIFY[7] = 0;

    uart_puts("T1 store (2.5)  = "); print_hex16(t1_raw); uart_puts(" expect 0x4100\n");
    uart_puts("T2 load  (2.5)  = "); print_hex16(t2_raw); uart_puts(" expect 0x4100\n");
    uart_puts("T3 add (2.5+1.5)= "); print_hex16(t3_raw); uart_puts(" expect 0x4400\n");
    uart_puts("T4 mul (2.5*1.5)= "); print_hex16(t4_raw); uart_puts(" expect 0x43c0\n");
    uart_puts("T5 h->s (4.0)   = "); print_hex32(t5_raw); uart_puts(" expect 0x40800000\n");
    uart_puts("T6 s->h (4.0)   = "); print_hex16(t6_raw); uart_puts(" expect 0x4400\n");
    uart_puts("ok_bits = ");        print_hex32(VERIFY[6]); uart_puts(" (T1|T2<<1|T3<<2|T4<<3|T5<<4|T6<<5)\n");

    HB[0] = 0xCAFE000F;
    while (1) {}
}
