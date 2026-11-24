// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include "../common/dtype.h"
#include "../common/kio.h"

#define HB     ((volatile uint32_t *)0x40100000)
#define VERIFY ((volatile uint32_t *)0x40100100)

#ifdef HP
#define B_PZERO 0x0000u
#define B_NZERO 0x8000u
#define B_PINF  0x7c00u
#define B_NINF  0xfc00u
#define B_NAN   0x7e00u
#define B_DEN   0x0001u
#define B_FMAX  0x7bffu
#define B_ONE   0x3c00u
#define B_TWO   0x4000u
#define B_FIVE  0x4500u
#define B_THREE 0x4200u
#else
#define B_PZERO 0x00000000u
#define B_NZERO 0x80000000u
#define B_PINF  0x7f800000u
#define B_NINF  0xff800000u
#define B_NAN   0x7fc00000u
#define B_DEN   0x00000001u
#define B_FMAX  0x7f7fffffu
#define B_ONE   0x3f800000u
#define B_TWO   0x40000000u
#define B_FIVE  0x40a00000u
#define B_THREE 0x40400000u
#endif

static real_t b2r(uint32_t bits) {
#ifdef HP
    union { uint16_t u; _Float16 f; } x = { .u = (uint16_t)bits }; return x.f;
#else
    union { uint32_t u; float f; } x = { .u = bits }; return x.f;
#endif
}
static int is_nan_bits(uint32_t b) {
#ifdef HP
    return ((b & 0x7c00u) == 0x7c00u) && (b & 0x03ffu);
#else
    return ((b & 0x7f800000u) == 0x7f800000u) && (b & 0x007fffffu);
#endif
}

enum { OP_ADD, OP_MUL, OP_MIN, OP_MAX };
static uint32_t binop(int op, uint32_t ab, uint32_t bb) {
    real_t a[1], b[1], o[1];
    a[0] = b2r(ab); b[0] = b2r(bb);
    size_t vl = VSETVL(1);
    vreal_t va = VLE(a, vl), vb = VLE(b, vl), vr = va;
    if (op == OP_ADD) vr = VFADD_VV(va, vb, vl);
    else if (op == OP_MUL) vr = VFMUL_VV(va, vb, vl);
    else if (op == OP_MIN) vr = VFMIN_VV(va, vb, vl);
    else if (op == OP_MAX) vr = VFMAX_VV(va, vb, vl);
    VSE(o, vr, vl);
    return RBITS(o[0]);
}

static int npass, nfail;
static uint32_t failmap;
static void report(const char *name, int ok) {
    if (npass + nfail < 32 && !ok) failmap |= (1u << (npass + nfail));
    if (ok) npass++; else nfail++;
    uart_puts("  "); uart_puts(name); uart_puts(ok ? " PASS\n" : " FAIL\n");
}

static void chk(const char *name, uint32_t got, uint32_t exp, int exp_nan) {
    report(name, exp_nan ? is_nan_bits(got) : (got == exp));
}

static void test_pow2k(int k, const char *name) {
    real_t kf[1]; kf[0] = (real_t)k; size_t vl = VSETVL(1); vreal_t kv = VLE(kf, vl); real_t o[1];
    int cvt;
#ifdef HP
    vint16m1_t iv = __riscv_vfcvt_x_f_v_i16m1(kv, vl);
    int16_t cb[1]; __riscv_vse16_v_i16m1(cb, iv, vl); cvt = cb[0];
    iv = __riscv_vadd_vv_i16m1(iv, __riscv_vmv_v_x_i16m1(15, vl), vl);
    iv = __riscv_vsll_vv_i16m1(iv, __riscv_vmv_v_x_u16m1(10, vl), vl);
    __riscv_vse16_v_f16m1((_Float16 *)o, __riscv_vreinterpret_v_i16m1_f16m1(iv), vl);
    uint32_t exp = (uint16_t)((k + 15) << 10);
#else
    vint32m1_t iv = __riscv_vfcvt_x_f_v_i32m1(kv, vl);
    int32_t cb[1]; __riscv_vse32_v_i32m1(cb, iv, vl); cvt = cb[0];
    iv = __riscv_vadd_vv_i32m1(iv, __riscv_vmv_v_x_i32m1(127, vl), vl);
    iv = __riscv_vsll_vv_i32m1(iv, __riscv_vmv_v_x_u32m1(23, vl), vl);
    __riscv_vse32_v_f32m1((float *)o, __riscv_vreinterpret_v_i32m1_f32m1(iv), vl);
    uint32_t exp = (uint32_t)((k + 127) << 23);
#endif
    uint32_t got = RBITS(o[0]); int ok = (got == exp);
    uart_puts("  "); uart_puts(name);
    uart_puts(" cvt="); print_u64((uint64_t)(int64_t)cvt);
    uart_puts(" got="); print_rhex(got); uart_puts(" exp="); print_rhex(exp);
    uart_puts(ok ? " PASS\n" : " FAIL\n");
    if (npass + nfail < 32 && !ok) failmap |= (1u << (npass + nfail));
    if (ok) npass++; else nfail++;
}

#ifdef HP
#define ASM_RED(LM) ({ _Float16 _r; size_t _n = 64; \
    asm volatile("vsetvli t0,%1,e16," #LM ",ta,ma\n\t" "vle16.v v8,(%2)\n\t" \
                 "vmv.s.x v16,zero\n\t" "vfredusum.vs v16,v8,v16\n\t" "vfmv.f.s %0,v16\n\t" \
        : "=f"(_r) : "r"(_n), "r"(vbuf) \
        : "t0","v8","v9","v10","v11","v12","v13","v14","v15","v16","memory"); (double)_r; })
#define LREDI(MUL) do { \
    size_t vl = __riscv_vsetvl_e16m##MUL(64); \
    vfloat16m##MUL##_t v = __riscv_vle16_v_f16m##MUL((const _Float16 *)vbuf, vl); \
    __riscv_vse16_v_f16m##MUL((_Float16 *)obuf, v, vl); \
    int ld = 1; for (size_t i = 0; i < vl; i++) if ((double)obuf[i] != 1.0) ld = 0; \
    vfloat16m1_t z = __riscv_vfmv_v_f_f16m1((_Float16)0, __riscv_vsetvlmax_e16m1()); \
    real_t s = __riscv_vfmv_f_s_f16m1_f16(__riscv_vfredusum_vs_f16m##MUL##_f16m1(v, z, vl)); \
    report("m" #MUL " gcc-intrinsic", ld && (double)s == 64.0); \
} while (0)
#else
#define ASM_RED(LM) ({ float _r; size_t _n = 64; \
    asm volatile("vsetvli t0,%1,e32," #LM ",ta,ma\n\t" "vle32.v v8,(%2)\n\t" \
                 "vmv.s.x v16,zero\n\t" "vfredusum.vs v16,v8,v16\n\t" "vfmv.f.s %0,v16\n\t" \
        : "=f"(_r) : "r"(_n), "r"(vbuf) \
        : "t0","v8","v9","v10","v11","v12","v13","v14","v15","v16","memory"); (double)_r; })
#define LREDI(MUL) do { \
    size_t vl = __riscv_vsetvl_e32m##MUL(64); \
    vfloat32m##MUL##_t v = __riscv_vle32_v_f32m##MUL((const float *)vbuf, vl); \
    __riscv_vse32_v_f32m##MUL((float *)obuf, v, vl); \
    int ld = 1; for (size_t i = 0; i < vl; i++) if ((double)obuf[i] != 1.0) ld = 0; \
    vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1()); \
    real_t s = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m##MUL##_f32m1(v, z, vl)); \
    report("m" #MUL " gcc-intrinsic", ld && (double)s == 64.0); \
} while (0)
#endif

#define LEN 4
real_t vbuf[8192] __attribute__((aligned(64)));
real_t obuf[256]  __attribute__((aligned(64)));

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;
    uart_puts("\n[corner " DTYPE_NAME "] FP-special + vl/LMUL\n");
    npass = nfail = 0; failmap = 0;

    uart_puts(" (A) FP special:\n");
    chk("inf+(-inf)=NaN", binop(OP_ADD, B_PINF, B_NINF), 0, 1);
    chk("inf+1=inf",      binop(OP_ADD, B_PINF, B_ONE), B_PINF, 0);
    chk("(+0)+(-0)=+0",   binop(OP_ADD, B_PZERO, B_NZERO), B_PZERO, 0);
    chk("1*inf=inf",      binop(OP_MUL, B_ONE, B_PINF), B_PINF, 0);
    chk("0*inf=NaN",      binop(OP_MUL, B_PZERO, B_PINF), 0, 1);
    chk("fmax+fmax=inf",  binop(OP_ADD, B_FMAX, B_FMAX), B_PINF, 0);
    chk("NaN+1=NaN",      binop(OP_ADD, B_NAN, B_ONE), 0, 1);
    chk("min(NaN,1)=1",   binop(OP_MIN, B_NAN, B_ONE), B_ONE, 0);
    chk("max(NaN,1)=1",   binop(OP_MAX, B_NAN, B_ONE), B_ONE, 0);
    chk("min(-0,+0)=-0",  binop(OP_MIN, B_NZERO, B_PZERO), B_NZERO, 0);
    chk("den*2=2den",     binop(OP_MUL, B_DEN, B_TWO), B_DEN * 2, 0);
    HB[0] = 0xCAFE0003;

    {
        real_t v[3] = { b2r(B_ONE), b2r(B_PINF), b2r(B_NINF) };
        size_t vl = VSETVL(3);
        vreal_t acc = VLE(v, vl), z = VFMV_V_F(0.0, VSETVLMAX());
        uint32_t r = RBITS(VFMV_F_S(VFREDUSUM(acc, z, vl)));
        chk("redusum[1,inf,-inf]=NaN", r, 0, 1);
    }
    {

        real_t v[4] = { b2r(B_ONE), b2r(B_FIVE), b2r(B_THREE), b2r(B_TWO) };
        size_t vl = VSETVL(4);
        vreal_t acc = VLE(v, vl), s = VFMV_V_F(b2r(B_NINF), VSETVLMAX());
        uint32_t r = RBITS(VFMV_F_S(VFREDMAX(acc, s, vl)));
        chk("redmax[1,5,3,2]=5", r, B_FIVE, 0);
    }
    HB[0] = 0xCAFE0004;

    uart_puts(" (B) vl boundary (m1):\n");
    for (unsigned i = 0; i < 8192; i++) vbuf[i] = (real_t)1;
    size_t vlmax = VSETVLMAX();
    unsigned vls[3] = { 1u, 7u, (unsigned)vlmax };
    for (int t = 0; t < 3; t++) {
        size_t vl = VSETVL(vls[t]);
        vreal_t z = VFMV_V_F(0.0, vlmax);
        real_t sum = VFMV_F_S(VFREDUSUM(VLE(vbuf, vl), z, vl));
        char nm[24]; int p = 0; const char *pre = "vl="; while (pre[p]) { nm[p] = pre[p]; p++; }
        unsigned v = vls[t]; char tmp[8]; int q = 0; if (!v) tmp[q++] = '0'; while (v) { tmp[q++] = (char)('0' + v % 10); v /= 10; }
        while (q) { nm[p++] = tmp[--q]; }
        nm[p] = 0;
        report(nm, (double)sum == (double)vls[t]);
    }
    HB[0] = 0xCAFE0005;

    HB[0] = 0xCAFE0006;

    uart_puts(" (C) exp 2^k (vfcvt.x.f + vadd + vsll):\n");
    test_pow2k(-7, "2^-7"); test_pow2k(-2, "2^-2"); test_pow2k(0, "2^0");
    test_pow2k(2, "2^2");   test_pow2k(5, "2^5");   test_pow2k(7, "2^7");
    HB[0] = 0xCAFE0007;

    uart_puts(" (D) LMUL redusum m2/m4/m8 (asm proper-vtype, then gcc-intrinsic):\n");
    HB[0] = 0xCAFE0B20; report("m2 proper-vtype", ASM_RED(m2) == 64.0);
    HB[0] = 0xCAFE0B21; asm volatile("" ::: "memory"); LREDI(2);
    HB[0] = 0xCAFE0B40; report("m4 proper-vtype", ASM_RED(m4) == 64.0);
    HB[0] = 0xCAFE0B41; asm volatile("" ::: "memory"); LREDI(4);
    HB[0] = 0xCAFE0B80; report("m8 proper-vtype", ASM_RED(m8) == 64.0);
    HB[0] = 0xCAFE0B81; asm volatile("" ::: "memory"); LREDI(8);
    HB[0] = 0xCAFE0008;

    int total = npass + nfail;
    int pass = (nfail == 0);
    VERIFY[0] = 0xA1A1B1B1;
    VERIFY[1] = pass ? 1 : 0;
    VERIFY[2] = (uint32_t)total;
    VERIFY[3] = (uint32_t)npass;
    VERIFY[4] = (uint32_t)nfail;
    VERIFY[5] = failmap;
    VERIFY[6] = 0;
    VERIFY[7] = 0;

    uart_puts("==== "); print_u64(npass); uart_puts(" / "); print_u64(total);
    uart_puts(pass ? " PASS ====\n" : " (some FAIL) ====\n");

    HB[0] = 0xCAFE000F;
    while (1) {}
}
