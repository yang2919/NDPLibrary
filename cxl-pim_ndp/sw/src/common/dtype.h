// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#ifndef ARA_VHK158_DTYPE_H
#define ARA_VHK158_DTYPE_H

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
#include "platform.h"
#include "vtrans_asm.h"

static inline uint32_t f2u(float f)    { union { float f; uint32_t u; } x = { .f = f }; return x.u; }
static inline uint16_t h2u(_Float16 f) { union { _Float16 f; uint16_t u; } x = { .f = f }; return x.u; }

static inline uint32_t ulp32(float a, float b) {
    int64_t ia = (int32_t)f2u(a), ib = (int32_t)f2u(b);
    if (ia < 0) ia = (int64_t)0x80000000 - ia;
    if (ib < 0) ib = (int64_t)0x80000000 - ib;
    int64_t d = ia - ib; return (uint32_t)(d < 0 ? -d : d);
}
static inline uint32_t ulp16(_Float16 a, _Float16 b) {
    int32_t ia = (int16_t)h2u(a), ib = (int16_t)h2u(b);
    if (ia < 0) ia = 0x8000 - ia;
    if (ib < 0) ib = 0x8000 - ib;
    int32_t d = ia - ib; return (uint32_t)(d < 0 ? -d : d);
}

#ifdef HP
typedef _Float16     real_t;
typedef vfloat16m1_t vreal_t;
#define DTYPE_NAME          "fp16"
#define RHEX_NIB            4
#define RBITS(f)            ((uint32_t)h2u(f))
#define RULP(a,b)           ulp16((a),(b))
#define R_PEAK_FMA          32
#define R_PEAK_ELEM         16
#define R_PASS_REL          5e-2
#define R_ABS               5e-2
#define R_MAXULP            8192
#define VSETVL(n)           __riscv_vsetvl_e16m1((n))
#define VSETVLMAX()         __riscv_vsetvlmax_e16m1()
#define VLE(p,vl)           __riscv_vle16_v_f16m1((p),(vl))
#define VSE(p,v,vl)         __riscv_vse16_v_f16m1((p),(v),(vl))
#define VFMV_V_F(f,vl)      __riscv_vfmv_v_f_f16m1((real_t)(f),(vl))
#define VFMV_F_S(v)         __riscv_vfmv_f_s_f16m1_f16((v))
#define VFMACC_VV_TU(a,x,y,vl) __riscv_vfmacc_vv_f16m1_tu((a),(x),(y),(vl))
#define VFADD_VV(a,b,vl)       __riscv_vfadd_vv_f16m1((a),(b),(vl))
#define VFADD_VV_TU(a,x,y,vl)  __riscv_vfadd_vv_f16m1_tu((a),(x),(y),(vl))
#define VFADD_VF(a,f,vl)       __riscv_vfadd_vf_f16m1((a),(real_t)(f),(vl))
#define VFMUL_VV(a,b,vl)       __riscv_vfmul_vv_f16m1((a),(b),(vl))
#define VFMUL_VF(a,f,vl)       __riscv_vfmul_vf_f16m1((a),(real_t)(f),(vl))
#define VFSUB_VV(a,b,vl)       __riscv_vfsub_vv_f16m1((a),(b),(vl))
#define VFSUB_VF(a,f,vl)       __riscv_vfsub_vf_f16m1((a),(real_t)(f),(vl))
#define VFDIV_VV(a,b,vl)       __riscv_vfdiv_vv_f16m1((a),(b),(vl))
#define VFREC7(a,vl)           __riscv_vfrec7_v_f16m1((a),(vl))
#define VFNMSUB_VV(a,b,c,vl)   __riscv_vfnmsub_vv_f16m1((a),(b),(c),(vl))
#define VFMAX_VV(a,b,vl)       __riscv_vfmax_vv_f16m1((a),(b),(vl))
#define VFMAX_VV_TU(a,x,y,vl)  __riscv_vfmax_vv_f16m1_tu((a),(x),(y),(vl))
#define VFMIN_VV(a,b,vl)       __riscv_vfmin_vv_f16m1((a),(b),(vl))
#define VFREDUSUM(v,s,vl)      __riscv_vfredusum_vs_f16m1_f16m1((v),(s),(vl))
#define VFREDMAX(v,s,vl)       __riscv_vfredmax_vs_f16m1_f16m1((v),(s),(vl))

typedef vint16m1_t   vint_t;
typedef int16_t      rint_t;
typedef uint16_t     ruint_t;
#define R_MAG_MASK          ((rint_t)0x7FFF)
#define R_SIGN_BIT          ((ruint_t)0x8000u)
#define R_INT_MIN           ((rint_t)0x8000)
#define VREINT_F2I(v)       __riscv_vreinterpret_v_f16m1_i16m1((v))
#define VAND_VX(v,x,vl)     __riscv_vand_vx_i16m1((v),(x),(vl))
#define VRSUB_VX(v,x,vl)    __riscv_vrsub_vx_i16m1((v),(x),(vl))
#define VMSLT_VX(v,x,vl)    __riscv_vmslt_vx_i16m1_b16((v),(x),(vl))
#define VMERGE_VVM(a,b,m,vl)__riscv_vmerge_vvm_i16m1((a),(b),(m),(vl))
#define VMV_V_X(x,vl)       __riscv_vmv_v_x_i16m1((x),(vl))
#define VREDMAX_I(v,s,vl)   __riscv_vredmax_vs_i16m1_i16m1((v),(s),(vl))
#define VMV_X_S(v)          __riscv_vmv_x_s_i16m1_i16((v))
static inline real_t u2r(ruint_t u) { union { ruint_t u; real_t r; } x = { .u = u }; return x.r; }
#else
typedef float        real_t;
typedef vfloat32m1_t vreal_t;
#define DTYPE_NAME          "fp32"
#define RHEX_NIB            8
#define RBITS(f)            (f2u(f))
#define RULP(a,b)           ulp32((a),(b))
#define R_PEAK_FMA          16
#define R_PEAK_ELEM         8
#define R_PASS_REL          1e-4
#define R_ABS               1e-4
#define R_MAXULP            64
#define VSETVL(n)           __riscv_vsetvl_e32m1((n))
#define VSETVLMAX()         __riscv_vsetvlmax_e32m1()
#define VLE(p,vl)           __riscv_vle32_v_f32m1((p),(vl))
#define VSE(p,v,vl)         __riscv_vse32_v_f32m1((p),(v),(vl))
#define VFMV_V_F(f,vl)      __riscv_vfmv_v_f_f32m1((real_t)(f),(vl))
#define VFMV_F_S(v)         __riscv_vfmv_f_s_f32m1_f32((v))
#define VFMACC_VV_TU(a,x,y,vl) __riscv_vfmacc_vv_f32m1_tu((a),(x),(y),(vl))
#define VFADD_VV(a,b,vl)       __riscv_vfadd_vv_f32m1((a),(b),(vl))
#define VFADD_VV_TU(a,x,y,vl)  __riscv_vfadd_vv_f32m1_tu((a),(x),(y),(vl))
#define VFADD_VF(a,f,vl)       __riscv_vfadd_vf_f32m1((a),(real_t)(f),(vl))
#define VFMUL_VV(a,b,vl)       __riscv_vfmul_vv_f32m1((a),(b),(vl))
#define VFMUL_VF(a,f,vl)       __riscv_vfmul_vf_f32m1((a),(real_t)(f),(vl))
#define VFSUB_VV(a,b,vl)       __riscv_vfsub_vv_f32m1((a),(b),(vl))
#define VFSUB_VF(a,f,vl)       __riscv_vfsub_vf_f32m1((a),(real_t)(f),(vl))
#define VFDIV_VV(a,b,vl)       __riscv_vfdiv_vv_f32m1((a),(b),(vl))
#define VFREC7(a,vl)           __riscv_vfrec7_v_f32m1((a),(vl))
#define VFNMSUB_VV(a,b,c,vl)   __riscv_vfnmsub_vv_f32m1((a),(b),(c),(vl))
#define VFMAX_VV(a,b,vl)       __riscv_vfmax_vv_f32m1((a),(b),(vl))
#define VFMAX_VV_TU(a,x,y,vl)  __riscv_vfmax_vv_f32m1_tu((a),(x),(y),(vl))
#define VFMIN_VV(a,b,vl)       __riscv_vfmin_vv_f32m1((a),(b),(vl))
#define VFREDUSUM(v,s,vl)      __riscv_vfredusum_vs_f32m1_f32m1((v),(s),(vl))
#define VFREDMAX(v,s,vl)       __riscv_vfredmax_vs_f32m1_f32m1((v),(s),(vl))

typedef vint32m1_t   vint_t;
typedef int32_t      rint_t;
typedef uint32_t     ruint_t;
#define R_MAG_MASK          ((rint_t)0x7FFFFFFF)
#define R_SIGN_BIT          ((ruint_t)0x80000000u)
#define R_INT_MIN           ((rint_t)0x80000000)
#define VREINT_F2I(v)       __riscv_vreinterpret_v_f32m1_i32m1((v))
#define VAND_VX(v,x,vl)     __riscv_vand_vx_i32m1((v),(x),(vl))
#define VRSUB_VX(v,x,vl)    __riscv_vrsub_vx_i32m1((v),(x),(vl))
#define VMSLT_VX(v,x,vl)    __riscv_vmslt_vx_i32m1_b32((v),(x),(vl))
#define VMERGE_VVM(a,b,m,vl)__riscv_vmerge_vvm_i32m1((a),(b),(m),(vl))
#define VMV_V_X(x,vl)       __riscv_vmv_v_x_i32m1((x),(vl))
#define VREDMAX_I(v,s,vl)   __riscv_vredmax_vs_i32m1_i32m1((v),(s),(vl))
#define VMV_X_S(v)          __riscv_vmv_x_s_i32m1_i32((v))
static inline real_t u2r(ruint_t u) { union { ruint_t u; real_t r; } x = { .u = u }; return x.r; }
#endif

static inline real_t r_dotp(const real_t *a, const real_t *b, size_t n) {
    size_t vl0 = VSETVL(n);
    if (n < 8 * vl0) {
        vreal_t acc = VFMUL_VV(VLE(a, vl0), VLE(b, vl0), vl0);
        for (size_t i = vl0; i < n; ) { size_t vl = VSETVL(n - i);
            acc = VFMACC_VV_TU(acc, VLE(a + i, vl), VLE(b + i, vl), vl); i += vl; }
        vreal_t z = VFMV_V_F(0.0, vl0);
        return VFMV_F_S(VFREDUSUM(acc, z, vl0));
    }
    vreal_t a0 = VFMV_V_F(0.0, vl0), a1 = a0, a2 = a0, a3 = a0;
    size_t i = 0;
    for (; i + 4 * vl0 <= n; i += 4 * vl0) {
        a0 = VFMACC_VV_TU(a0, VLE(a + i,           vl0), VLE(b + i,           vl0), vl0);
        a1 = VFMACC_VV_TU(a1, VLE(a + i + vl0,     vl0), VLE(b + i + vl0,     vl0), vl0);
        a2 = VFMACC_VV_TU(a2, VLE(a + i + 2 * vl0, vl0), VLE(b + i + 2 * vl0, vl0), vl0);
        a3 = VFMACC_VV_TU(a3, VLE(a + i + 3 * vl0, vl0), VLE(b + i + 3 * vl0, vl0), vl0);
    }
    for (; i < n; ) {
        size_t vl = VSETVL(n - i);
        a0 = VFMACC_VV_TU(a0, VLE(a + i, vl), VLE(b + i, vl), vl);
        i += vl;
    }
    a0 = VFADD_VV(a0, a1, vl0); a2 = VFADD_VV(a2, a3, vl0); a0 = VFADD_VV(a0, a2, vl0);
    vreal_t z = VFMV_V_F(0.0, vl0);
    return VFMV_F_S(VFREDUSUM(a0, z, vl0));
}
static inline real_t r_redsum(const real_t *v, size_t n) {
    size_t vl0 = VSETVL(n);
    if (n < 8 * vl0) {
        vreal_t acc = VLE(v, vl0);
        for (size_t i = vl0; i < n; ) { size_t vl = VSETVL(n - i);
            acc = VFADD_VV_TU(acc, acc, VLE(v + i, vl), vl); i += vl; }
        vreal_t z = VFMV_V_F(0.0, vl0);
        return VFMV_F_S(VFREDUSUM(acc, z, vl0));
    }
    vreal_t a0 = VFMV_V_F(0.0, vl0), a1 = a0, a2 = a0, a3 = a0;
    size_t i = 0;
    for (; i + 4 * vl0 <= n; i += 4 * vl0) {
        a0 = VFADD_VV_TU(a0, a0, VLE(v + i,           vl0), vl0);
        a1 = VFADD_VV_TU(a1, a1, VLE(v + i + vl0,     vl0), vl0);
        a2 = VFADD_VV_TU(a2, a2, VLE(v + i + 2 * vl0, vl0), vl0);
        a3 = VFADD_VV_TU(a3, a3, VLE(v + i + 3 * vl0, vl0), vl0);
    }
    for (; i < n; ) {
        size_t vl = VSETVL(n - i);
        a0 = VFADD_VV_TU(a0, a0, VLE(v + i, vl), vl);
        i += vl;
    }
    a0 = VFADD_VV(a0, a1, vl0); a2 = VFADD_VV(a2, a3, vl0); a0 = VFADD_VV(a0, a2, vl0);
    vreal_t z = VFMV_V_F(0.0, vl0);
    return VFMV_F_S(VFREDUSUM(a0, z, vl0));
}

static inline real_t r_vfold_max(vreal_t acc, size_t vl0) {
#if !ARA_ERRATUM_FP_MINMAX_REDUCTION

    return VFMV_F_S(VFREDMAX(acc, VFMV_V_F(-1.0e30, vl0), vl0));
#else
    vint_t k   = VREINT_F2I(acc);
    vint_t neg = VRSUB_VX(VAND_VX(k, R_MAG_MASK, vl0), 0, vl0);
    k = VMERGE_VVM(k, neg, VMSLT_VX(k, 0, vl0), vl0);
    rint_t best = VMV_X_S(VREDMAX_I(k, VMV_V_X(R_INT_MIN, vl0), vl0));
    return u2r(best < 0 ? (R_SIGN_BIT | (ruint_t)(-best)) : (ruint_t)best);
#endif
}

static inline real_t r_redmax(const real_t *v, size_t n) {
    size_t vl0 = VSETVL(n);
    if (n < 8 * vl0) {
        vreal_t acc = VLE(v, vl0);
        for (size_t i = vl0; i < n; ) { size_t vl = VSETVL(n - i);
            acc = VFMAX_VV_TU(acc, acc, VLE(v + i, vl), vl); i += vl; }
        return r_vfold_max(acc, vl0);
    }
    vreal_t a0 = VFMV_V_F(v[0], vl0), a1 = a0, a2 = a0, a3 = a0;
    size_t i = 0;
    for (; i + 4 * vl0 <= n; i += 4 * vl0) {
        a0 = VFMAX_VV_TU(a0, a0, VLE(v + i,           vl0), vl0);
        a1 = VFMAX_VV_TU(a1, a1, VLE(v + i + vl0,     vl0), vl0);
        a2 = VFMAX_VV_TU(a2, a2, VLE(v + i + 2 * vl0, vl0), vl0);
        a3 = VFMAX_VV_TU(a3, a3, VLE(v + i + 3 * vl0, vl0), vl0);
    }
    for (; i < n; ) {
        size_t vl = VSETVL(n - i);
        a0 = VFMAX_VV_TU(a0, a0, VLE(v + i, vl), vl);
        i += vl;
    }
    a0 = VFMAX_VV(a0, a1, vl0); a2 = VFMAX_VV(a2, a3, vl0); a0 = VFMAX_VV(a0, a2, vl0);
    return r_vfold_max(a0, vl0);
}
static inline void r_subs(real_t *o, const real_t *in, real_t s, size_t n) {
    size_t vl;
    for (size_t i = 0; i < n; i += vl) { vl = VSETVL(n - i); VSE(o + i, VFSUB_VF(VLE(in + i, vl), s, vl), vl); }
}
static inline void r_scale(real_t *o, const real_t *in, real_t s, size_t n) {
    size_t vl;
    for (size_t i = 0; i < n; i += vl) { vl = VSETVL(n - i); VSE(o + i, VFMUL_VF(VLE(in + i, vl), s, vl), vl); }
}
static inline void r_vadds(real_t *o, const real_t *in, real_t s, size_t n) {
    size_t vl;
    for (size_t i = 0; i < n; i += vl) { vl = VSETVL(n - i); VSE(o + i, VFADD_VF(VLE(in + i, vl), s, vl), vl); }
}
static inline void r_vmul(real_t *o, const real_t *x, const real_t *y, size_t n) {
    size_t vl;
    for (size_t i = 0; i < n; i += vl) { vl = VSETVL(n - i); VSE(o + i, VFMUL_VV(VLE(x + i, vl), VLE(y + i, vl), vl), vl); }
}

static inline __attribute__((noinline))
void r_vexp_vt(real_t *o, const real_t *in, size_t n) {
    size_t vl, vlm;
#ifdef HP
    asm volatile("vsetvli %0, zero, e16, m1, ta, ma" : "=r"(vlm));
#ifdef USE_SWOPT

    while (n >= 2 * vlm) {
        asm volatile("vle16.v v2, (%0)" :: "r"(in)       : "memory");
        asm volatile("vle16.v v4, (%0)" :: "r"(in + vlm) : "memory");
        vt_exp_v1v2();
        vt_exp_v3v4();
        asm volatile("vse16.v v1, (%0)" :: "r"(o)       : "memory");
        asm volatile("vse16.v v3, (%0)" :: "r"(o + vlm) : "memory");
        in += 2 * vlm; o += 2 * vlm; n -= 2 * vlm;
    }
#endif
    while (n > 0) {
        asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle16.v v2, (%0)" :: "r"(in) : "memory");
        vt_exp_v1v2();
        asm volatile("vse16.v v1, (%0)" :: "r"(o) : "memory");
        in += vl; o += vl; n -= vl;
    }
#else
    asm volatile("vsetvli %0, zero, e32, m1, ta, ma" : "=r"(vlm));
#ifdef USE_SWOPT
    while (n >= 2 * vlm) {
        asm volatile("vle32.v v2, (%0)" :: "r"(in)       : "memory");
        asm volatile("vle32.v v4, (%0)" :: "r"(in + vlm) : "memory");
        vt_exp_v1v2();
        vt_exp_v3v4();
        asm volatile("vse32.v v1, (%0)" :: "r"(o)       : "memory");
        asm volatile("vse32.v v3, (%0)" :: "r"(o + vlm) : "memory");
        in += 2 * vlm; o += 2 * vlm; n -= 2 * vlm;
    }
#endif
    while (n > 0) {
        asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle32.v v2, (%0)" :: "r"(in) : "memory");
        vt_exp_v1v2();
        asm volatile("vse32.v v1, (%0)" :: "r"(o) : "memory");
        in += vl; o += vl; n -= vl;
    }
#endif
    (void)vlm;
}
static inline __attribute__((noinline))
void r_vrcp_vt(real_t *o, const real_t *in, size_t n) {
    size_t vl, vlm;
#ifdef HP
    asm volatile("vsetvli %0, zero, e16, m1, ta, ma" : "=r"(vlm));
#ifdef USE_SWOPT
    while (n >= 2 * vlm) {
        asm volatile("vle16.v v2, (%0)" :: "r"(in)       : "memory");
        asm volatile("vle16.v v4, (%0)" :: "r"(in + vlm) : "memory");
        vt_rcp_v1v2();
        vt_rcp_v3v4();
        asm volatile("vse16.v v1, (%0)" :: "r"(o)       : "memory");
        asm volatile("vse16.v v3, (%0)" :: "r"(o + vlm) : "memory");
        in += 2 * vlm; o += 2 * vlm; n -= 2 * vlm;
    }
#endif
    while (n > 0) {
        asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle16.v v2, (%0)" :: "r"(in) : "memory");
        vt_rcp_v1v2();
        asm volatile("vse16.v v1, (%0)" :: "r"(o) : "memory");
        in += vl; o += vl; n -= vl;
    }
#else
    asm volatile("vsetvli %0, zero, e32, m1, ta, ma" : "=r"(vlm));
#ifdef USE_SWOPT
    while (n >= 2 * vlm) {
        asm volatile("vle32.v v2, (%0)" :: "r"(in)       : "memory");
        asm volatile("vle32.v v4, (%0)" :: "r"(in + vlm) : "memory");
        vt_rcp_v1v2();
        vt_rcp_v3v4();
        asm volatile("vse32.v v1, (%0)" :: "r"(o)       : "memory");
        asm volatile("vse32.v v3, (%0)" :: "r"(o + vlm) : "memory");
        in += 2 * vlm; o += 2 * vlm; n -= 2 * vlm;
    }
#endif
    while (n > 0) {
        asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle32.v v2, (%0)" :: "r"(in) : "memory");
        vt_rcp_v1v2();
        asm volatile("vse32.v v1, (%0)" :: "r"(o) : "memory");
        in += vl; o += vl; n -= vl;
    }
#endif
    (void)vlm;
}
static inline __attribute__((noinline))
void r_vdiv_vt(real_t *o, const real_t *x, const real_t *y, size_t n) {
    size_t vl;
    while (n > 0) {
#ifdef HP
        asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle16.v v2, (%0)" :: "r"(y) : "memory");
        vt_rcp_v1v2();
        asm volatile("vle16.v v2, (%0)" :: "r"(x) : "memory");
        asm volatile("vfmul.vv v1, v1, v2");
        asm volatile("vse16.v v1, (%0)" :: "r"(o) : "memory");
#else
        asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle32.v v2, (%0)" :: "r"(y) : "memory");
        vt_rcp_v1v2();
        asm volatile("vle32.v v2, (%0)" :: "r"(x) : "memory");
        asm volatile("vfmul.vv v1, v1, v2");
        asm volatile("vse32.v v1, (%0)" :: "r"(o) : "memory");
#endif
        x += vl; y += vl; o += vl; n -= vl;
    }
}

static inline vreal_t r_vrcp_nr(vreal_t x, size_t vl) {
    vreal_t two = VFMV_V_F(2.0, vl);
    vreal_t y   = VFREC7(x, vl);
    y = VFMUL_VV(y, VFNMSUB_VV(y, x, two, vl), vl);
#ifndef HP
    y = VFMUL_VV(y, VFNMSUB_VV(y, x, two, vl), vl);
#endif
    return y;
}

#ifdef USE_REC7
#define R_DIV_VV(a,b,vl)  VFMUL_VV((a), r_vrcp_nr((b),(vl)), (vl))
#else
#define R_DIV_VV(a,b,vl)  VFDIV_VV((a),(b),(vl))
#endif

static inline void r_vdiv(real_t *o, const real_t *x, const real_t *y, size_t n) {
#ifdef USE_VTRANS
    r_vdiv_vt(o, x, y, n);
#else
    size_t vl;
    for (size_t i = 0; i < n; i += vl) { vl = VSETVL(n - i); VSE(o + i, R_DIV_VV(VLE(x + i, vl), VLE(y + i, vl), vl), vl); }
#endif
}

#if defined(USE_VTRANS) && defined(USE_SWOPT)

#ifdef HP
#define VT_E "e16"
#define VT_L "vle16.v"
#define VT_S "vse16.v"
#else
#define VT_E "e32"
#define VT_L "vle32.v"
#define VT_S "vse32.v"
#endif
static inline __attribute__((noinline))
real_t r_subexp_sum_vt(real_t *o, const real_t *in, real_t mx, size_t n) {
    size_t vl, vlm;
    real_t s;
    asm volatile("vsetvli %0, zero, " VT_E ", m1, ta, ma" : "=r"(vlm));
    asm volatile("vmv.v.i v6, 0");
    asm volatile("vmv.v.i v7, 0");
    while (n >= 2 * vlm) {
        asm volatile(VT_L " v2, (%0)" :: "r"(in)       : "memory");
        asm volatile(VT_L " v4, (%0)" :: "r"(in + vlm) : "memory");
        asm volatile("vfsub.vf v2, v2, %0" :: "f"(mx));
        asm volatile("vfsub.vf v4, v4, %0" :: "f"(mx));
        vt_exp_v1v2();
        vt_exp_v3v4();
        asm volatile("vfadd.vv v6, v6, v1");
        asm volatile("vfadd.vv v7, v7, v3");
        asm volatile(VT_S " v1, (%0)" :: "r"(o)       : "memory");
        asm volatile(VT_S " v3, (%0)" :: "r"(o + vlm) : "memory");
        in += 2 * vlm; o += 2 * vlm; n -= 2 * vlm;
    }
    while (n > 0) {
        asm volatile("vsetvli %0, %1, " VT_E ", m1, tu, ma" : "=r"(vl) : "r"(n));
        asm volatile(VT_L " v2, (%0)" :: "r"(in) : "memory");
        asm volatile("vfsub.vf v2, v2, %0" :: "f"(mx));
        vt_exp_v1v2();
        asm volatile("vfadd.vv v6, v6, v1");
        asm volatile(VT_S " v1, (%0)" :: "r"(o) : "memory");
        in += vl; o += vl; n -= vl;
    }
    asm volatile("vsetvli zero, %0, " VT_E ", m1, ta, ma" :: "r"(vlm));
    asm volatile("vfadd.vv v6, v6, v7");
    asm volatile("vmv.v.i v8, 0");
    asm volatile("vfredusum.vs v8, v6, v8");
    asm volatile("vfmv.f.s %0, v8" : "=f"(s));
    return s;
}

static inline __attribute__((noinline))
void r_gelu_vt(real_t *o, const real_t *in, size_t n) {
    size_t vl, vlm;
    const real_t c = (real_t)-1.702f, one = (real_t)1;
    asm volatile("vsetvli %0, zero, " VT_E ", m1, ta, ma" : "=r"(vlm));
    while (n >= 2 * vlm) {
        asm volatile(VT_L " v5, (%0)" :: "r"(in)       : "memory");
        asm volatile(VT_L " v6, (%0)" :: "r"(in + vlm) : "memory");
        asm volatile("vfmul.vf v2, v5, %0" :: "f"(c));
        asm volatile("vfmul.vf v4, v6, %0" :: "f"(c));
        vt_exp_v1v2();
        vt_exp_v3v4();
        asm volatile("vfadd.vf v2, v1, %0" :: "f"(one));
        asm volatile("vfadd.vf v4, v3, %0" :: "f"(one));
        vt_rcp_v1v2();
        vt_rcp_v3v4();
        asm volatile("vfmul.vv v1, v1, v5");
        asm volatile("vfmul.vv v3, v3, v6");
        asm volatile(VT_S " v1, (%0)" :: "r"(o)       : "memory");
        asm volatile(VT_S " v3, (%0)" :: "r"(o + vlm) : "memory");
        in += 2 * vlm; o += 2 * vlm; n -= 2 * vlm;
    }
    while (n > 0) {
        asm volatile("vsetvli %0, %1, " VT_E ", m1, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile(VT_L " v5, (%0)" :: "r"(in) : "memory");
        asm volatile("vfmul.vf v2, v5, %0" :: "f"(c));
        vt_exp_v1v2();
        asm volatile("vfadd.vf v2, v1, %0" :: "f"(one));
        vt_rcp_v1v2();
        asm volatile("vfmul.vv v1, v1, v5");
        asm volatile(VT_S " v1, (%0)" :: "r"(o) : "memory");
        in += vl; o += vl; n -= vl;
    }
}
#undef VT_E
#undef VT_L
#undef VT_S
#endif

static inline void r_transpose(real_t *dst, const real_t *src, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            dst[j * rows + i] = src[i * cols + j];
}

static inline real_t r_fabs(real_t x) { return x < (real_t)0 ? -x : x; }

static inline real_t sqrt_r(real_t x) {
    if (x <= (real_t)0) return (real_t)0;
    real_t y = x;
    for (int i = 0; i < 12; i++) y = (real_t)0.5 * (y + x / y);
    return y;
}
static inline real_t r_round(double v) { return (real_t)v; }

static inline uint32_t r_ulpd(real_t got, double refd) { return RULP(got, r_round(refd)); }
static inline int r_ok(real_t got, double refd) {
    double d = (double)got - refd; if (d < 0) d = -d;
    double ar = refd < 0 ? -refd : refd;
    return (got == got) && (d < ar * R_PASS_REL + R_ABS);
}

#endif
