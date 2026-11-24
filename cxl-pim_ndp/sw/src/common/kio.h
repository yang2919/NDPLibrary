// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#ifndef ARA_VHK158_KIO_H
#define ARA_VHK158_KIO_H

#include <stdint.h>
#include "uart_apb.h"

static inline uint64_t rdcycle(void) { uint64_t c; asm volatile ("rdcycle %0" : "=r"(c)); return c; }
static inline void uart_puts(const char *s) { while (*s) { if (*s == '\n') uart_putc('\r'); uart_putc(*s++); } }
static inline void print_u64(uint64_t v) { char b[21]; int i = 20; b[20] = 0; if (!v) { uart_putc('0'); return; } while (v) { b[--i] = (char)('0' + (v % 10)); v /= 10; } while (b[i]) uart_putc(b[i++]); }
static inline void print_rhex(uint32_t v) { uart_puts("0x"); for (int i = RHEX_NIB - 1; i >= 0; i--) { uint32_t n = (v >> (i * 4)) & 0xF; uart_putc(n < 10 ? '0' + n : 'a' + (n - 10)); } }

static inline void print_rdec(double v) {
    if (v != v) { uart_puts("nan"); return; }
    if (v < 0) { uart_putc('-'); v = -v; }
    if (v > 1e18) { uart_puts("inf"); return; }
    uint64_t ip = (uint64_t)v; double f = v - (double)ip;
    print_u64(ip); uart_putc('.');
    for (int i = 0; i < 4; i++) { f *= 10.0; int d = (int)f; uart_putc((char)('0' + d)); f -= (double)d; }
}

static inline void print_hexdec(uint32_t bits, double v) {
    print_rhex(bits); uart_puts(" ("); print_rdec(v); uart_putc(')');
}

#endif
