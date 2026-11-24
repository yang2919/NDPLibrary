// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#ifndef ARA_VHK158_BENCH_PRINT_H
#define ARA_VHK158_BENCH_PRINT_H

#include <stdint.h>
#include "uart_apb.h"

static inline void bp_puts(const char *s) {
    while (*s) { if (*s == '\n') uart_putc('\r'); uart_putc(*s++); }
}

static inline void bp_u64(uint64_t v) {
    char b[21]; int i = 20; b[20] = 0;
    if (!v) { uart_putc('0'); return; }
    while (v) { b[--i] = (char)('0' + (v % 10)); v /= 10; }
    while (b[i]) uart_putc(b[i++]);
}

static inline void bp_frac(uint64_t scaled, int digits) {
    uint64_t div = 1;
    for (int i = 0; i < digits; i++) div *= 10;
    bp_u64(scaled / div);
    uart_putc('.');
    uint64_t frac = scaled % div;

    for (uint64_t p = div / 10; p > 1; p /= 10) {
        if (frac < p) uart_putc('0');
    }
    bp_u64(frac);
}

static inline int bp_ndigits(uint64_t v) {
    int n = 1; while (v >= 10) { v /= 10; n++; } return n;
}
static inline void bp_spaces(int n) { while (n-- > 0) uart_putc(' '); }

static inline void bp_col(const char *s, int w) {
    int n = 0; while (s[n]) n++;
    bp_puts(s); bp_spaces(w - n);
}

static inline void bp_col_u64(uint64_t v, int w) {
    bp_u64(v); bp_spaces(w - bp_ndigits(v));
}

static inline void bp_col_frac(uint64_t scaled, int digits, int w) {
    uint64_t div = 1;
    for (int i = 0; i < digits; i++) div *= 10;
    bp_frac(scaled, digits);
    bp_spaces(w - (bp_ndigits(scaled / div) + 1 + digits));
}

static inline void bp_col_pct(uint64_t tenths, int w) {
    int len = bp_ndigits(tenths / 10) + 1 + 1 + 1;
    bp_spaces(w - len);
    bp_frac(tenths, 1); uart_putc('%');
}

static inline void bp_perf_u(uint64_t ops, uint64_t cycles, uint32_t peak,
                             const char *unit) {
    bp_puts("cycles    = "); bp_u64(cycles);  bp_puts("\n");
    if (!cycles) { bp_puts("perf      = n/a (0 cycles)\n"); return; }
    bp_puts("perf      = ");
    bp_frac(ops * 100 / cycles, 2);
    bp_puts(" "); bp_puts(unit); bp_puts("/cycle  (");
    bp_frac(ops * 1000 / ((uint64_t)cycles * peak), 1);
    bp_puts("% of "); bp_u64(peak); bp_puts(" peak)\n");
}

static inline void bp_perf(uint64_t flop, uint64_t cycles, uint32_t peak) {
    bp_perf_u(flop, cycles, peak, "FLOP");
}

static inline void bp_thru(uint64_t elems, uint64_t cycles, const char *unit) {
    bp_puts("cycles    = "); bp_u64(cycles);  bp_puts("\n");
    if (!cycles) { bp_puts("perf      = n/a (0 cycles)\n"); return; }
    bp_puts("perf      = ");
    bp_frac(elems * 1000 / cycles, 3);
    bp_puts(" "); bp_puts(unit); bp_puts("/cycle\n");
}

#endif
