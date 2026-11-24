/*
 * Kernels, synthetic data and the worker pool of the LLM examples
 * (llm.h). Each kernel has an RVV version and a plain C version for the
 * host reference build; both use the same algorithm (for example the same
 * exponential), so their results differ only by rounding order.
 */

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>

#endif

/* ---- synthetic data ---------------------------------------------------- */

#define H0 0x9e3779b9u
#define H1 0x85ebca6bu
#define H2 0xc2b2ae35u

static inline uint32_t
hash32(uint32_t s0, uint32_t i)
{
    uint32_t h = (i + s0) * H1;
    h ^= h >> 13;
    h *= H2;
    h ^= h >> 16;
    return h;
}

/* (float)((int)(h >> 8) - 2^23) * (scale / 2^23): exact integer, then one
 * rounded multiply, the same on RVV and on the host. */
static inline float
hash_float(uint32_t h, float k)
{
    return (float)((int32_t)(h >> 8) - 0x800000) * k;
}

void
llm_fill_bf16(bf16_t *dst, size_t first, size_t n, uint32_t seed,
              float scale)
{
    const uint32_t s0 = seed * H0;
    const float k = scale / 8388608.0f;
#if defined(__riscv_vector)
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        vuint32m8_t u = __riscv_vadd_vx_u32m8(
            __riscv_vid_v_u32m8(vl), (uint32_t)(first + i) + s0, vl);
        u = __riscv_vmul_vx_u32m8(u, H1, vl);
        u = __riscv_vxor_vv_u32m8(u, __riscv_vsrl_vx_u32m8(u, 13, vl), vl);
        u = __riscv_vmul_vx_u32m8(u, H2, vl);
        u = __riscv_vxor_vv_u32m8(u, __riscv_vsrl_vx_u32m8(u, 16, vl), vl);
        vint32m8_t v = __riscv_vsub_vx_i32m8(
            __riscv_vreinterpret_v_u32m8_i32m8(
                __riscv_vsrl_vx_u32m8(u, 8, vl)),
            0x800000, vl);
        vfloat32m8_t f =
            __riscv_vfmul_vf_f32m8(__riscv_vfcvt_f_x_v_f32m8(v, vl), k, vl);
        vuint16m4_t b = __riscv_vnsrl_wx_u16m4(
            __riscv_vreinterpret_v_f32m8_u32m8(f), 16, vl);
        __riscv_vse16_v_u16m4(dst + i, b, vl);
        i += vl;
    }
#else
    for (size_t i = 0; i < n; i++) {
        float f = hash_float(hash32(s0, (uint32_t)(first + i)), k);
        uint32_t bits;
        memcpy(&bits, &f, 4);
        dst[i] = (bf16_t)(bits >> 16);
    }
#endif
}

void
llm_fill_f32(float *dst, size_t first, size_t n, uint32_t seed, float scale,
             float offset)
{
    const uint32_t s0 = seed * H0;
    const float k = scale / 8388608.0f;
    for (size_t i = 0; i < n; i++)
        dst[i] = offset + hash_float(hash32(s0, (uint32_t)(first + i)), k);
}

/* ---- exponential: 2^n * p(r), r = x - n ln 2, |r| <= ln 2 / 2 ---------- */

#define EXP_LO -87.0f
#define EXP_HI 88.0f
#define LOG2E 1.44269504f
#define LN2_HI 0.693145752f
#define LN2_LO 1.42860677e-06f
#define C2 0.5f
#define C3 0.166666672f
#define C4 0.0416666679f
#define C5 0.00833333377f
#define C6 0.00138888892f

static inline float
exp_scalar(float x)
{
    x = x < EXP_LO ? EXP_LO : (x > EXP_HI ? EXP_HI : x);
    const float n = rintf(x * LOG2E);
    float r = x - n * LN2_HI;
    r = r - n * LN2_LO;
    float p = C6;
    p = p * r + C5;
    p = p * r + C4;
    p = p * r + C3;
    p = p * r + C2;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    const int32_t e = ((int32_t)n + 127) << 23;
    float s;
    memcpy(&s, &e, 4);
    return p * s;
}

#if defined(__riscv_vector)
static inline vfloat32m8_t
exp_rvv(vfloat32m8_t x, size_t vl)
{
    x = __riscv_vfmax_vf_f32m8(__riscv_vfmin_vf_f32m8(x, EXP_HI, vl),
                               EXP_LO, vl);
    vint32m8_t ni = __riscv_vfcvt_x_f_v_i32m8(
        __riscv_vfmul_vf_f32m8(x, LOG2E, vl), vl); /* round to nearest */
    vfloat32m8_t n = __riscv_vfcvt_f_x_v_f32m8(ni, vl);
    vfloat32m8_t r = __riscv_vfsub_vv_f32m8(
        x, __riscv_vfmul_vf_f32m8(n, LN2_HI, vl), vl);
    r = __riscv_vfsub_vv_f32m8(r, __riscv_vfmul_vf_f32m8(n, LN2_LO, vl), vl);
    vfloat32m8_t p = __riscv_vfmv_v_f_f32m8(C6, vl);
    p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, r, vl), C5, vl);
    p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, r, vl), C4, vl);
    p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, r, vl), C3, vl);
    p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, r, vl), C2, vl);
    p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, r, vl), 1.0f, vl);
    p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, r, vl), 1.0f, vl);
    vint32m8_t e = __riscv_vsll_vx_i32m8(
        __riscv_vadd_vx_i32m8(ni, 127, vl), 23, vl);
    return __riscv_vfmul_vv_f32m8(p, __riscv_vreinterpret_v_i32m8_f32m8(e),
                                  vl);
}

/*
 * Accumulating loops use LMUL 1: on gem5's O3 CPU, a loop that accumulates
 * into an LMUL 8 register group (vfmacc or vfadd, tail undisturbed) gives
 * wrong sums, while LMUL 1 and the Atomic CPU are exact (found 2026-11-12,
 * build_logs/llm/ktest2 in the demo worktree).
 */
static inline float
sum_m1(vfloat32m1_t v, size_t vlmax)
{
    vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    return __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m1_f32m1(v, z, vlmax));
}
#endif

/* ---- kernels ----------------------------------------------------------- */

#if defined(__riscv_vector)
/* bf16 weights widened to fp32: zero-extend to 32 bits, shift by 16. */
static inline vfloat32m1_t
widen_bf16(const bf16_t *p, size_t vl)
{
    return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vsll_vx_u32m1(
        __riscv_vzext_vf2_u32m1(__riscv_vle16_v_u16mf2(p, vl), vl), 16, vl));
}

/*
 * The register tile: rows r (weights wr0) and, if two, r + 1 (wr1), over
 * columns [k0, k1), for nb <= 8 batch elements. Each weight chunk is loaded
 * and widened once and used nb times; each x chunk is used for both rows.
 * 2 x nb accumulators (LMUL 1, see sum_m1) + 2 weight + 1 x registers fit
 * the 32 vector registers. Partial sums are added to y. nb and two are
 * constants at every call site (gemv_tile_n), so the unused accumulators
 * and branches compile away.
 */
static inline __attribute__((always_inline)) void
gemv_tile(const int nb, const int two, float *y, size_t ldy,
          const bf16_t *wr0, const bf16_t *wr1, const float *x, size_t ldx,
          size_t k0, size_t k1, size_t r)
{
    const size_t vlmax = __riscv_vsetvlmax_e32m1();
    const vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    vfloat32m1_t a00 = z, a01 = z, a02 = z, a03 = z, a04 = z, a05 = z,
                 a06 = z, a07 = z, a10 = z, a11 = z, a12 = z, a13 = z,
                 a14 = z, a15 = z, a16 = z, a17 = z;
    for (size_t j = k0; j < k1;) {
        const size_t vl = __riscv_vsetvl_e32m1(k1 - j);
        const vfloat32m1_t w0 = widen_bf16(wr0 + j, vl);
        const vfloat32m1_t w1 = two ? widen_bf16(wr1 + j, vl) : z;
#define GEMV_STEP(i)                                                       \
    if (nb > i) {                                                          \
        const vfloat32m1_t xv = __riscv_vle32_v_f32m1(x + i * ldx + j, vl); \
        a0##i = __riscv_vfmacc_vv_f32m1_tu(a0##i, w0, xv, vl);             \
        if (two)                                                           \
            a1##i = __riscv_vfmacc_vv_f32m1_tu(a1##i, w1, xv, vl);         \
    }
        GEMV_STEP(0)
        GEMV_STEP(1)
        GEMV_STEP(2)
        GEMV_STEP(3)
        GEMV_STEP(4)
        GEMV_STEP(5)
        GEMV_STEP(6)
        GEMV_STEP(7)
#undef GEMV_STEP
        j += vl;
    }
#define GEMV_OUT(i)                                                        \
    if (nb > i) {                                                          \
        y[i * ldy + r] += sum_m1(a0##i, vlmax);                            \
        if (two)                                                           \
            y[i * ldy + r + 1] += sum_m1(a1##i, vlmax);                    \
    }
    GEMV_OUT(0)
    GEMV_OUT(1)
    GEMV_OUT(2)
    GEMV_OUT(3)
    GEMV_OUT(4)
    GEMV_OUT(5)
    GEMV_OUT(6)
    GEMV_OUT(7)
#undef GEMV_OUT
}

/* gemv_tile with nb and two as constants. */
static void
gemv_tile_n(int nb, int two, float *y, size_t ldy, const bf16_t *wr0,
            const bf16_t *wr1, const float *x, size_t ldx, size_t k0,
            size_t k1, size_t r)
{
#define GEMV_CASE(n)                                                       \
    case n:                                                                \
        if (two)                                                           \
            gemv_tile(n, 1, y, ldy, wr0, wr1, x, ldx, k0, k1, r);          \
        else                                                               \
            gemv_tile(n, 0, y, ldy, wr0, wr1, x, ldx, k0, k1, r);          \
        break;
    switch (nb) {
        GEMV_CASE(1)
        GEMV_CASE(2)
        GEMV_CASE(3)
        GEMV_CASE(4)
        GEMV_CASE(5)
        GEMV_CASE(6)
        GEMV_CASE(7)
        GEMV_CASE(8)
    }
#undef GEMV_CASE
}
#endif

/* Bytes of x (fp32, all batch elements of a group) kept in the L1 cache
 * per k tile: half of the 32 KiB L1, the rest for the streaming weights. */
#define GEMV_X_TILE_BYTES 16384

void
llm_gemv_bf16(float *y, size_t ldy, const bf16_t *w, const float *x,
              size_t ldx, size_t nb, size_t r0, size_t r1, size_t k)
{
#if defined(__riscv_vector)
    /*
     * Batch elements in groups of at most 8 (the register tile); per group,
     * k in tiles whose x fits GEMV_X_TILE_BYTES, so x stays in the L1
     * cache while every weight of rows [r0, r1) is read once from memory:
     * 2 x nb flops per 2-byte weight, growing with the batch.
     */
    const size_t vlmax = __riscv_vsetvlmax_e32m1();
    for (size_t b0 = 0; b0 < nb; b0 += 8) {
        const int g = nb - b0 < 8 ? (int)(nb - b0) : 8;
        float *yg = y + b0 * ldy;
        const float *xg = x + b0 * ldx;
        for (int i = 0; i < g; i++)
            for (size_t r = r0; r < r1; r++)
                yg[i * ldy + r] = 0.0f;
        size_t kt = GEMV_X_TILE_BYTES / (sizeof(float) * (size_t)g);
        kt = kt / vlmax * vlmax;
        if (kt < vlmax)
            kt = vlmax;
        for (size_t k0 = 0; k0 < k; k0 += kt) {
            const size_t k1 = k0 + kt < k ? k0 + kt : k;
            for (size_t r = r0; r < r1; r += 2) {
                const int two = r + 1 < r1;
                gemv_tile_n(g, two, yg, ldy, w + r * k,
                            w + (two ? r + 1 : r) * k, xg, ldx, k0, k1, r);
            }
        }
    }
#else
    for (size_t r = r0; r < r1; r++) {
        const bf16_t *wr = w + r * k;
        for (size_t b = 0; b < nb; b++) {
            const float *xb = x + b * ldx;
            float s = 0.0f;
            for (size_t j = 0; j < k; j++) {
                uint32_t bits = (uint32_t)wr[j] << 16;
                float wf;
                memcpy(&wf, &bits, 4);
                s += wf * xb[j];
            }
            y[b * ldy + r] = s;
        }
    }
#endif
}

void
llm_rmsnorm(float *y, const float *x, const float *g, size_t n, float eps)
{
    float ss = 0.0f;
#if defined(__riscv_vector)
    const size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(x + i, vl);
        acc = __riscv_vfmacc_vv_f32m1_tu(acc, v, v, vl);
        i += vl;
    }
    ss = sum_m1(acc, vlmax);
    const float s = 1.0f / sqrtf(ss / (float)n + eps);
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t v = __riscv_vfmul_vf_f32m8(
            __riscv_vle32_v_f32m8(x + i, vl), s, vl);
        __riscv_vse32_v_f32m8(
            y + i,
            __riscv_vfmul_vv_f32m8(v, __riscv_vle32_v_f32m8(g + i, vl), vl),
            vl);
        i += vl;
    }
#else
    for (size_t i = 0; i < n; i++)
        ss += x[i] * x[i];
    const float s = 1.0f / sqrtf(ss / (float)n + eps);
    for (size_t i = 0; i < n; i++)
        y[i] = (x[i] * s) * g[i];
#endif
}

void
llm_rope(float *v, size_t head_dim, size_t pos, float theta)
{
    const size_t half = head_dim / 2;
    for (size_t i = 0; i < half; i++) {
        const double f = pow((double)theta, -2.0 * (double)i / head_dim);
        const double a = (double)pos * f;
        const float c = (float)cos(a), s = (float)sin(a);
        const float x1 = v[i], x2 = v[i + half];
        v[i] = x1 * c - x2 * s;
        v[i + half] = x2 * c + x1 * s;
    }
}

void
llm_softmax(float *p, const float *s, size_t n, float scale)
{
#if defined(__riscv_vector)
    const size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t mx = __riscv_vfmv_v_f_f32m1(-INFINITY, vlmax);
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m1(n - i);
        mx = __riscv_vfmax_vv_f32m1_tu(
            mx, mx, __riscv_vle32_v_f32m1(s + i, vl), vl);
        i += vl;
    }
    vfloat32m1_t lo = __riscv_vfmv_v_f_f32m1(-INFINITY, 1);
    const float m = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmax_vs_f32m1_f32m1(mx, lo, vlmax));
    /* the exponentials at LMUL 8 (no accumulation), the sum at LMUL 1 */
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t e = exp_rvv(
            __riscv_vfmul_vf_f32m8(
                __riscv_vfsub_vf_f32m8(__riscv_vle32_v_f32m8(s + i, vl), m,
                                       vl),
                scale, vl),
            vl);
        __riscv_vse32_v_f32m8(p + i, e, vl);
        i += vl;
    }
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m1(n - i);
        acc = __riscv_vfadd_vv_f32m1_tu(acc, acc,
                                        __riscv_vle32_v_f32m1(p + i, vl), vl);
        i += vl;
    }
    const float inv = 1.0f / sum_m1(acc, vlmax);
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        __riscv_vse32_v_f32m8(
            p + i,
            __riscv_vfmul_vf_f32m8(__riscv_vle32_v_f32m8(p + i, vl), inv, vl),
            vl);
        i += vl;
    }
#else
    float m = -INFINITY, sum = 0.0f;
    for (size_t i = 0; i < n; i++)
        m = s[i] > m ? s[i] : m;
    for (size_t i = 0; i < n; i++) {
        p[i] = exp_scalar((s[i] - m) * scale);
        sum += p[i];
    }
    const float inv = 1.0f / sum;
    for (size_t i = 0; i < n; i++)
        p[i] *= inv;
#endif
}

void
llm_silu_mul(float *y, const float *g, const float *u, size_t n)
{
#if defined(__riscv_vector)
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t gv = __riscv_vle32_v_f32m8(g + i, vl);
        vfloat32m8_t e = exp_rvv(__riscv_vfneg_v_f32m8(gv, vl), vl);
        vfloat32m8_t sg = __riscv_vfdiv_vv_f32m8(
            gv, __riscv_vfadd_vf_f32m8(e, 1.0f, vl), vl);
        __riscv_vse32_v_f32m8(
            y + i,
            __riscv_vfmul_vv_f32m8(sg, __riscv_vle32_v_f32m8(u + i, vl), vl),
            vl);
        i += vl;
    }
#else
    for (size_t i = 0; i < n; i++)
        y[i] = (g[i] / (exp_scalar(-g[i]) + 1.0f)) * u[i];
#endif
}

void
llm_add(float *y, const float *x, size_t n)
{
#if defined(__riscv_vector)
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        __riscv_vse32_v_f32m8(
            y + i,
            __riscv_vfadd_vv_f32m8(__riscv_vle32_v_f32m8(y + i, vl),
                                   __riscv_vle32_v_f32m8(x + i, vl), vl),
            vl);
        i += vl;
    }
#else
    for (size_t i = 0; i < n; i++)
        y[i] += x[i];
#endif
}

void
llm_axpy(float *y, float a, const float *x, size_t n)
{
#if defined(__riscv_vector)
    /* LMUL 1, as every multiply-add here (see sum_m1) */
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m1(n - i);
        __riscv_vse32_v_f32m1(
            y + i,
            __riscv_vfmacc_vf_f32m1(__riscv_vle32_v_f32m1(y + i, vl), a,
                                    __riscv_vle32_v_f32m1(x + i, vl), vl),
            vl);
        i += vl;
    }
#else
    for (size_t i = 0; i < n; i++)
        y[i] += a * x[i];
#endif
}

void
llm_topk(const float *v, size_t n, size_t k, int *idx)
{
    for (size_t r = 0; r < k; r++) {
        int best = -1;
        for (size_t i = 0; i < n; i++) {
            int taken = 0;
            for (size_t q = 0; q < r && !taken; q++)
                taken = idx[q] == (int)i;
            if (!taken && (best < 0 || v[i] > v[best]))
                best = (int)i;
        }
        idx[r] = best;
    }
}

float
llm_round_pim(float x)
{
    uint8_t e[4];
    pim_f32_to_elem(e, x);
    return pim_elem_to_f32(e);
}

/* ---- worker pool ------------------------------------------------------- */

#define MAX_WORKERS 64

static struct {
    int n;
    pthread_t th[MAX_WORKERS];
    _Atomic unsigned gen;
    _Atomic int done;
    _Atomic int stop;
    llm_task_fn fn;
    void *arg;
    size_t total;
} pool;

static void *
worker_main(void *p)
{
    const int w = (int)(intptr_t)p;
    unsigned seen = 0;
    for (;;) {
        unsigned g;
        while ((g = atomic_load_explicit(&pool.gen, memory_order_acquire)) ==
               seen)
            ;
        seen = g;
        if (atomic_load_explicit(&pool.stop, memory_order_acquire))
            return NULL;
        const size_t lo = pool.total * (size_t)w / (size_t)pool.n;
        const size_t hi = pool.total * (size_t)(w + 1) / (size_t)pool.n;
        if (lo < hi)
            pool.fn(pool.arg, lo, hi, w);
        atomic_fetch_add_explicit(&pool.done, 1, memory_order_release);
    }
}

void
llm_pool_start(int nworkers)
{
    if (nworkers < 1 || nworkers > MAX_WORKERS) {
        fprintf(stderr, "llm: 1 to %d workers\n", MAX_WORKERS);
        exit(2);
    }
    pool.n = nworkers;
    for (int w = 0; w < nworkers; w++) {
        if (pthread_create(&pool.th[w], NULL, worker_main,
                           (void *)(intptr_t)w)) {
            fprintf(stderr, "llm: pthread_create failed\n");
            exit(2);
        }
    }
}

int
llm_pool_size(void)
{
    return pool.n;
}

void
llm_pool_launch(llm_task_fn fn, void *arg, size_t n)
{
    pool.fn = fn;
    pool.arg = arg;
    pool.total = n;
    atomic_store_explicit(&pool.done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&pool.gen, 1, memory_order_release);
}

void
llm_pool_wait(void)
{
    while (atomic_load_explicit(&pool.done, memory_order_acquire) < pool.n)
        ;
}

void
llm_pool_run(llm_task_fn fn, void *arg, size_t n)
{
    llm_pool_launch(fn, arg, n);
    llm_pool_wait();
}

void
llm_pool_stop(void)
{
    atomic_store_explicit(&pool.stop, 1, memory_order_release);
    atomic_fetch_add_explicit(&pool.gen, 1, memory_order_release);
    for (int w = 0; w < pool.n; w++)
        pthread_join(pool.th[w], NULL);
}
