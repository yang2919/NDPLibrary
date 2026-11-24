// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#ifndef ARA_VHK158_EXP_HP_H
#define ARA_VHK158_EXP_HP_H

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

static inline double exp_ref(double x) {
    if (x >  709.0) return 1e308 * 1e308;
    if (x < -745.0) return 0.0;
    const double LN2 = 0.69314718055994530942, LOG2E = 1.44269504088896340736;
    int k = (int)(x * LOG2E + (x >= 0.0 ? 0.5 : -0.5));
    double r = x - (double)k * LN2;
    double term = 1.0, sum = 1.0;
    for (int n = 1; n <= 12; n++) { term *= r / (double)n; sum += term; }
    union { uint64_t u; double d; } p;
    p.u = (uint64_t)((int64_t)k + 1023) << 52;
    return sum * p.d;
}

static inline vfloat32m1_t exp32(vfloat32m1_t x, size_t gvl) {
    vfloat32m1_t L2E = __riscv_vfmv_v_f_f32m1(1.44269504088896341f, gvl);
    vfloat32m1_t C1 = __riscv_vfmv_v_f_f32m1(0.693359375f, gvl);
    vfloat32m1_t C2 = __riscv_vfmv_v_f_f32m1(-2.12194440e-4f, gvl);
    vfloat32m1_t p0 = __riscv_vfmv_v_f_f32m1(1.9875691500E-4f, gvl);
    vfloat32m1_t p1 = __riscv_vfmv_v_f_f32m1(1.3981999507E-3f, gvl);
    vfloat32m1_t p2 = __riscv_vfmv_v_f_f32m1(8.3334519073E-3f, gvl);
    vfloat32m1_t p3 = __riscv_vfmv_v_f_f32m1(4.1665795894E-2f, gvl);
    vfloat32m1_t p4 = __riscv_vfmv_v_f_f32m1(1.6666665459E-1f, gvl);
    vfloat32m1_t p5 = __riscv_vfmv_v_f_f32m1(5.0000001201E-1f, gvl);
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, gvl), zero = __riscv_vfmv_v_f_f32m1(0.0f, gvl);
    vfloat32m1_t tmp, tmp2, tmp4, fx, z, y; vbool32_t mask; vint32m1_t imm0, tmp3;
    x = __riscv_vfmin_vv_f32m1(x, __riscv_vfmv_v_f_f32m1(88.376263f, gvl), gvl);
    x = __riscv_vfmax_vv_f32m1(x, __riscv_vfmv_v_f_f32m1(-88.376263f, gvl), gvl);
    fx = __riscv_vfmacc_vv_f32m1(__riscv_vfmv_v_f_f32m1(0.5f, gvl), x, L2E, gvl);
    tmp3 = __riscv_vfcvt_x_f_v_i32m1(fx, gvl);
    tmp = __riscv_vfcvt_f_x_v_f32m1(tmp3, gvl);
    mask = __riscv_vmflt_vv_f32m1_b32(fx, tmp, gvl);
    tmp2 = __riscv_vmerge_vvm_f32m1(zero, one, mask, gvl);
    fx = __riscv_vfsub_vv_f32m1(tmp, tmp2, gvl);
    x = __riscv_vfsub_vv_f32m1(x, __riscv_vfmul_vv_f32m1(fx, C1, gvl), gvl);
    x = __riscv_vfsub_vv_f32m1(x, __riscv_vfmul_vv_f32m1(fx, C2, gvl), gvl);
    z = __riscv_vfmul_vv_f32m1(x, x, gvl);
    y = p0;
    y = __riscv_vfmadd_vv_f32m1(y, x, p1, gvl); y = __riscv_vfmadd_vv_f32m1(y, x, p2, gvl);
    y = __riscv_vfmadd_vv_f32m1(y, x, p3, gvl); y = __riscv_vfmadd_vv_f32m1(y, x, p4, gvl);
    y = __riscv_vfmadd_vv_f32m1(y, x, p5, gvl); y = __riscv_vfmadd_vv_f32m1(y, z, x, gvl);
    y = __riscv_vfadd_vv_f32m1(y, one, gvl);
    imm0 = __riscv_vfcvt_x_f_v_i32m1(fx, gvl);
    imm0 = __riscv_vadd_vv_i32m1(imm0, __riscv_vmv_v_x_i32m1(0x7f, gvl), gvl);
    imm0 = __riscv_vsll_vv_i32m1(imm0, __riscv_vmv_v_x_u32m1(23, gvl), gvl);
    tmp4 = __riscv_vreinterpret_v_i32m1_f32m1(imm0);
    return __riscv_vfmul_vv_f32m1(y, tmp4, gvl);
}

static inline vfloat16m1_t exp16(vfloat16m1_t x, size_t gvl) {
    vfloat16m1_t L2E = __riscv_vfmv_v_f_f16m1((_Float16)1.44269504f, gvl);
    vfloat16m1_t C1 = __riscv_vfmv_v_f_f16m1((_Float16)0.693359375f, gvl);
    vfloat16m1_t C2 = __riscv_vfmv_v_f_f16m1((_Float16)-2.12194440e-4f, gvl);
    vfloat16m1_t p0 = __riscv_vfmv_v_f_f16m1((_Float16)1.9875691500E-4f, gvl);
    vfloat16m1_t p1 = __riscv_vfmv_v_f_f16m1((_Float16)1.3981999507E-3f, gvl);
    vfloat16m1_t p2 = __riscv_vfmv_v_f_f16m1((_Float16)8.3334519073E-3f, gvl);
    vfloat16m1_t p3 = __riscv_vfmv_v_f_f16m1((_Float16)4.1665795894E-2f, gvl);
    vfloat16m1_t p4 = __riscv_vfmv_v_f_f16m1((_Float16)1.6666665459E-1f, gvl);
    vfloat16m1_t p5 = __riscv_vfmv_v_f_f16m1((_Float16)5.0000001201E-1f, gvl);
    vfloat16m1_t one = __riscv_vfmv_v_f_f16m1((_Float16)1, gvl), zero = __riscv_vfmv_v_f_f16m1((_Float16)0, gvl);
    vfloat16m1_t tmp, tmp2, tmp4, fx, z, y; vbool16_t mask; vint16m1_t imm0, tmp3;

    x = __riscv_vfmin_vv_f16m1(x, __riscv_vfmv_v_f_f16m1((_Float16)10.5f, gvl), gvl);
    x = __riscv_vfmax_vv_f16m1(x, __riscv_vfmv_v_f_f16m1((_Float16)-10.5f, gvl), gvl);
    fx = __riscv_vfmacc_vv_f16m1(__riscv_vfmv_v_f_f16m1((_Float16)0.5f, gvl), x, L2E, gvl);
    tmp3 = __riscv_vfcvt_x_f_v_i16m1(fx, gvl);
    tmp = __riscv_vfcvt_f_x_v_f16m1(tmp3, gvl);
    mask = __riscv_vmflt_vv_f16m1_b16(fx, tmp, gvl);
    tmp2 = __riscv_vmerge_vvm_f16m1(zero, one, mask, gvl);
    fx = __riscv_vfsub_vv_f16m1(tmp, tmp2, gvl);
    x = __riscv_vfsub_vv_f16m1(x, __riscv_vfmul_vv_f16m1(fx, C1, gvl), gvl);
    x = __riscv_vfsub_vv_f16m1(x, __riscv_vfmul_vv_f16m1(fx, C2, gvl), gvl);
    z = __riscv_vfmul_vv_f16m1(x, x, gvl);
    y = p0;
    y = __riscv_vfmadd_vv_f16m1(y, x, p1, gvl); y = __riscv_vfmadd_vv_f16m1(y, x, p2, gvl);
    y = __riscv_vfmadd_vv_f16m1(y, x, p3, gvl); y = __riscv_vfmadd_vv_f16m1(y, x, p4, gvl);
    y = __riscv_vfmadd_vv_f16m1(y, x, p5, gvl); y = __riscv_vfmadd_vv_f16m1(y, z, x, gvl);
    y = __riscv_vfadd_vv_f16m1(y, one, gvl);
    imm0 = __riscv_vfcvt_x_f_v_i16m1(fx, gvl);
    imm0 = __riscv_vadd_vv_i16m1(imm0, __riscv_vmv_v_x_i16m1(15, gvl), gvl);
    imm0 = __riscv_vsll_vv_i16m1(imm0, __riscv_vmv_v_x_u16m1(10, gvl), gvl);
    tmp4 = __riscv_vreinterpret_v_i16m1_f16m1(imm0);
    return __riscv_vfmul_vv_f16m1(y, tmp4, gvl);
}

static inline void r_vexp_cephes(real_t *o, const real_t *in, size_t n) {
    size_t i = 0;
#ifdef HP
    size_t vlm = __riscv_vsetvlmax_e16m1();
    for (; i + 2 * vlm <= n; i += 2 * vlm) {
        vfloat16m1_t a = __riscv_vle16_v_f16m1((const _Float16 *)in + i, vlm);
        vfloat16m1_t b = __riscv_vle16_v_f16m1((const _Float16 *)in + i + vlm, vlm);
        a = exp16(a, vlm);
        b = exp16(b, vlm);
        __riscv_vse16_v_f16m1((_Float16 *)o + i, a, vlm);
        __riscv_vse16_v_f16m1((_Float16 *)o + i + vlm, b, vlm);
    }
    for (size_t vl; i < n; i += vl) { vl = __riscv_vsetvl_e16m1(n - i);
        __riscv_vse16_v_f16m1((_Float16 *)o + i, exp16(__riscv_vle16_v_f16m1((const _Float16 *)in + i, vl), vl), vl); }
#else
    size_t vlm = __riscv_vsetvlmax_e32m1();
    for (; i + 2 * vlm <= n; i += 2 * vlm) {
        vfloat32m1_t a = __riscv_vle32_v_f32m1((const float *)in + i, vlm);
        vfloat32m1_t b = __riscv_vle32_v_f32m1((const float *)in + i + vlm, vlm);
        a = exp32(a, vlm);
        b = exp32(b, vlm);
        __riscv_vse32_v_f32m1((float *)o + i, a, vlm);
        __riscv_vse32_v_f32m1((float *)o + i + vlm, b, vlm);
    }
    for (size_t vl; i < n; i += vl) { vl = __riscv_vsetvl_e32m1(n - i);
        __riscv_vse32_v_f32m1((float *)o + i, exp32(__riscv_vle32_v_f32m1((const float *)in + i, vl), vl), vl); }
#endif
}

static inline void r_vexp(real_t *o, const real_t *in, size_t n) {
#ifdef USE_VTRANS
    r_vexp_vt(o, in, n);
#else
    r_vexp_cephes(o, in, n);
#endif
}

#ifdef HP
#define EXPV(x,vl)  exp16((x),(vl))
#else
#define EXPV(x,vl)  exp32((x),(vl))
#endif

static inline real_t r_subexp_sum_sw(real_t *o, const real_t *in, real_t mx, size_t n) {
    size_t vlm = VSETVLMAX(), i = 0;
    vreal_t acc = VFMV_V_F(0.0, vlm);
    for (; i + vlm <= n; i += vlm) {
        vreal_t e = EXPV(VFSUB_VF(VLE(in + i, vlm), mx, vlm), vlm);
        acc = VFADD_VV(acc, e, vlm);
        VSE(o + i, e, vlm);
    }
    for (size_t vl; i < n; i += vl) {
        vl = VSETVL(n - i);
        vreal_t e = EXPV(VFSUB_VF(VLE(in + i, vl), mx, vl), vl);
        acc = VFADD_VV_TU(acc, acc, e, vl);
        VSE(o + i, e, vl);
    }
    vreal_t z = VFMV_V_F(0.0, vlm);
    return VFMV_F_S(VFREDUSUM(acc, z, vlm));
}

static inline void r_gelu_sw(real_t *o, const real_t *in, size_t n) {
    for (size_t i = 0, vl; i < n; i += vl) {
        vl = VSETVL(n - i);
        vreal_t x = VLE(in + i, vl);
        vreal_t d = VFADD_VF(EXPV(VFMUL_VF(x, (real_t)-1.702f, vl), vl), (real_t)1, vl);
        VSE(o + i, R_DIV_VV(x, d, vl), vl);
    }
}

static inline real_t r_subexp_sum(real_t *o, const real_t *in, real_t mx, size_t n) {
#if defined(USE_VTRANS) && defined(USE_SWOPT)
    return r_subexp_sum_vt(o, in, mx, n);
#elif defined(USE_SWOPT)
    return r_subexp_sum_sw(o, in, mx, n);
#else
    r_subs(o, in, mx, n);
    r_vexp(o, o, n);
    return r_redsum(o, n);
#endif
}

static inline void r_gelu(real_t *o, const real_t *in, real_t *tmp, size_t n) {
#if defined(USE_VTRANS) && defined(USE_SWOPT)
    (void)tmp; r_gelu_vt(o, in, n);
#elif defined(USE_SWOPT)
    (void)tmp; r_gelu_sw(o, in, n);
#else
    r_scale(tmp, in, (real_t)-1.702f, n);
    r_vexp(tmp, tmp, n);
    r_vadds(tmp, tmp, (real_t)1, n);
    r_vdiv(o, in, tmp, n);
#endif
}

#endif
