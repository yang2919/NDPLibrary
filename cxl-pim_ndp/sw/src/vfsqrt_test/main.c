// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
#include "../common/uart_apb.h"

#define HB     ((volatile uint32_t *)0x40100000)
#define VERIFY ((volatile uint32_t *)0x40100100)

float arr[16] __attribute__((aligned(32)));

int64_t event_trigger;

static inline uint64_t rdcycle(void) { uint64_t c; asm volatile ("rdcycle %0" : "=r"(c)); return c; }
static void uart_puts(const char *s) { while (*s) { if (*s == '\n') uart_putc('\r'); uart_putc(*s++); } }
static void print_u64(uint64_t v) { char b[21]; int i=20; b[20]=0; if(!v){uart_putc('0'); return;} while(v){b[--i]='0'+(v%10); v/=10;} while(b[i])uart_putc(b[i++]); }
static void print_hex32(uint32_t v) { uart_puts("0x"); for (int i=7;i>=0;i--){uint32_t n=(v>>(i*4))&0xF; uart_putc(n<10?'0'+n:'a'+(n-10));} }
static inline uint32_t f2u(float f) { union { float f; uint32_t u; } x = { .f = f }; return x.u; }

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;
    uart_puts("\n[vfsqrt_test] vfsqrt.v vl={1,4,vlmax} isolation\n");

    uint64_t t0 = rdcycle();

    HB[0] = 0xCAFE0010;
    size_t vl1 = __riscv_vsetvl_e32m1(1);
    vfloat32m1_t v1 = __riscv_vfmv_v_f_f32m1(4.0f, vl1);
    HB[0] = 0xCAFE0011;
    v1 = __riscv_vfsqrt_v_f32m1(v1, vl1);
    HB[0] = 0xCAFE0012;
    float r1 = __riscv_vfmv_f_s_f32m1_f32(v1);
    HB[0] = 0xCAFE0013;

    HB[0] = 0xCAFE0020;
    arr[0] = 1.0f; arr[1] = 4.0f; arr[2] = 9.0f; arr[3] = 16.0f;
    size_t vl4 = __riscv_vsetvl_e32m1(4);
    vfloat32m1_t v4 = __riscv_vle32_v_f32m1(arr, vl4);
    HB[0] = 0xCAFE0021;
    v4 = __riscv_vfsqrt_v_f32m1(v4, vl4);
    HB[0] = 0xCAFE0022;
    __riscv_vse32_v_f32m1(arr, v4, vl4);
    HB[0] = 0xCAFE0023;

    HB[0] = 0xCAFE0030;
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vm = __riscv_vfmv_v_f_f32m1(4.0f, vlmax);
    HB[0] = 0xCAFE0031;
    vm = __riscv_vfsqrt_v_f32m1(vm, vlmax);
    HB[0] = 0xCAFE0032;
    float r3 = __riscv_vfmv_f_s_f32m1_f32(vm);
    HB[0] = 0xCAFE0033;

    uint64_t t1 = rdcycle();

    int t1_ok = (r1 > 1.99f && r1 < 2.01f);
    int t2_ok = (arr[0] > 0.99f && arr[0] < 1.01f) && (arr[3] > 3.99f && arr[3] < 4.01f);
    int t3_ok = (r3 > 1.99f && r3 < 2.01f);

    VERIFY[0] = 0xA1A1B1B1;
    VERIFY[1] = (t1_ok && t2_ok && t3_ok) ? 1 : 0;
    VERIFY[2] = (uint32_t)(t1 - t0);
    VERIFY[3] = f2u(r1);
    VERIFY[4] = f2u(arr[0]);
    VERIFY[5] = f2u(arr[3]);
    VERIFY[6] = f2u(r3);
    VERIFY[7] = (t1_ok ? 1 : 0) | (t2_ok ? 2 : 0) | (t3_ok ? 4 : 0);

    uart_puts("test1 vl=1   sqrt(4) = "); print_hex32(f2u(r1));      uart_puts("  (expect 0x40000000)\n");
    uart_puts("test2 vl=4   sqrt(1) = "); print_hex32(f2u(arr[0])); uart_puts("  (expect 0x3f800000)\n");
    uart_puts("test2 vl=4   sqrt(16)= "); print_hex32(f2u(arr[3])); uart_puts("  (expect 0x40800000)\n");
    uart_puts("test3 vlmax  sqrt(4) = "); print_hex32(f2u(r3));      uart_puts("  (expect 0x40000000)\n");
    uart_puts("passed bits = "); print_u64(VERIFY[7]);                uart_puts("  (t1|t2<<1|t3<<2)\n");
    uart_puts("cycles      = "); print_u64(t1 - t0);                  uart_puts("\n");

    HB[0] = 0xCAFE000F;
    while (1) {}
}
