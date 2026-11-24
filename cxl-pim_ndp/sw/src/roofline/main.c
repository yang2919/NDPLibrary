// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>

#ifdef I8

#include <riscv_vector.h>
typedef int8_t real_t;
#define DTYPE_NAME "int8"
#define RHEX_NIB 2
#include "bp-imatmul.h"
#else
#include "../common/dtype.h"
#include "../common/exp_hp.h"
#endif

#include "../common/uart_apb.h"
#include "../common/kio.h"
#include "../common/bench_print.h"

#ifndef I8
#ifdef HP
#include "hp-fmatmul.h"
#define FMATMUL hp_fmatmul
#else
#include "sp-fmatmul.h"
#define FMATMUL sp_fmatmul
#endif
#endif

#define HB  ((volatile uint32_t *)0x40100000)
#define RES ((volatile uint32_t *)0x40100200)
#define RES_MAGIC 0x2F00F11Eu

static const int MM_N[]  = {16, 32, 64, 128, 256};
static const int EW_N[]  = {256, 1024, 4096, 16384};
#ifndef I8
static const int ATT_S[] = {64, 128};
#endif
#define N_MM  ((int)(sizeof(MM_N) / sizeof(MM_N[0])))
#define N_EW  ((int)(sizeof(EW_N) / sizeof(EW_N[0])))
#define N_ATT ((int)(sizeof(ATT_S) / sizeof(ATT_S[0])))
#define D_HEAD 64
#define LN_EPS ((real_t)1e-3)

#define MAXN 256
static real_t A[MAXN * MAXN] __attribute__((aligned(32)));
static real_t B[MAXN * MAXN] __attribute__((aligned(32)));
static real_t C[MAXN * MAXN] __attribute__((aligned(32)));
#ifndef I8
static real_t D[MAXN * MAXN] __attribute__((aligned(32)));
#endif

#ifdef I8

#include "../common/dtype_i8.h"
#endif

static int g_r;
static uint32_t g_pass;

#ifdef RL_SPM

#if defined(I8) || defined(USE_SWOPT)
#error "RL_SPM covers the base fp build only: the SWOPT attention transposes K with a scalar loop (dtype.h r_transpose)"
#endif
#define SPM_BASE      0x30000000ul
#define SPM_BYTES     0x80000ul
#define RES_SPM       ((volatile uint32_t *)0x40100400)
#define RES_SPM_MAGIC 0x5B00F11Eu
#define SPM_WORDS     7
#define SPM_ROUNDS    5
#define EW_MAX        16384
#define SPM_EW(k)     ((real_t *)SPM_BASE + (size_t)(k) * EW_MAX)
_Static_assert(4ul * EW_MAX * sizeof(real_t) <= SPM_BYTES, "element-wise slots exceed the scratchpad");
_Static_assert(2ul * MAXN * MAXN * sizeof(real_t) <= SPM_BYTES, "matmul B + C exceed the scratchpad");

static real_t E[MAXN * MAXN] __attribute__((aligned(32)));
static volatile uint32_t g_sync;

static inline void st_drain(void) {
    uint32_t x;
    asm volatile("lw %0, 0(%1)" : "=r"(x) : "r"(&g_sync) : "memory");
    (void)x;
}

#define T_SYNC(stmt) ({ uint64_t t_a_ = rdcycle(); stmt; st_drain(); rdcycle() - t_a_; })

static void vcopy(void *dst, const void *src, size_t bytes) {
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;
    size_t vlm = __riscv_vsetvlmax_e8m8(), i = 0;
    for (; i + vlm <= bytes; i += vlm)
        __riscv_vse8_v_u8m8(d + i, __riscv_vle8_v_u8m8(s + i, vlm), vlm);
    if (i < bytes) {
        size_t vl = __riscv_vsetvl_e8m8(bytes - i);
        __riscv_vse8_v_u8m8(d + i, __riscv_vle8_v_u8m8(s + i, vl), vl);
    }
}

static int same_bytes(const void *x, const void *y, size_t bytes) {
    const uint8_t *a = (const uint8_t *)x, *b = (const uint8_t *)y;
    for (size_t i = 0; i < bytes; i++)
        if (a[i] != b[i]) return 0;
    return 1;
}

static uint64_t median_rounds(uint64_t *v) {
    for (int i = 1; i < SPM_ROUNDS; i++)
        for (int j = i; j > 0 && v[j - 1] > v[j]; j--) { uint64_t t = v[j]; v[j] = v[j - 1]; v[j - 1] = t; }
    return v[SPM_ROUNDS / 2];
}

#define ROUNDS(dram_stmt, spm_stmt, md, ms) do {                                   \
        uint64_t d_[SPM_ROUNDS], s_[SPM_ROUNDS];                                    \
        for (int k_ = 0; k_ < SPM_ROUNDS; k_++) {                                   \
            d_[k_] = T_SYNC(dram_stmt);                                             \
            s_[k_] = T_SYNC(spm_stmt);                                              \
        }                                                                           \
        md = median_rounds(d_); ms = median_rounds(s_);                             \
    } while (0)

static __attribute__((noinline)) real_t k_dp(const real_t *a, const real_t *b, size_t n) { return r_dotp(a, b, n); }
static __attribute__((noinline)) real_t k_red(const real_t *a, size_t n) { return r_redsum(a, n); }
static __attribute__((noinline)) void k_exp(real_t *o, const real_t *a, size_t n) { r_vexp(o, a, n); }
static __attribute__((noinline)) void k_gelu(real_t *o, const real_t *a, real_t *tmp, size_t n) { r_gelu(o, a, tmp, n); }
static __attribute__((noinline)) real_t k_sfx(real_t *c, const real_t *a, size_t n) {
    real_t mx = r_redmax(a, n);
    real_t sm = r_subexp_sum(c, a, mx, n);
    r_scale(c, c, (real_t)((real_t)1 / sm), n);
    return sm;
}
static __attribute__((noinline)) real_t k_lyn(real_t *b, real_t *c, real_t *d, const real_t *a, size_t n) {
    real_t mean = r_redsum(a, n) / (real_t)n;
    r_subs(b, a, mean, n); r_vmul(c, b, b, n);
    real_t var = r_redsum(c, n) / (real_t)n;
    real_t invstd = (real_t)1 / sqrt_r(var + LN_EPS);
    r_scale(d, b, invstd, n);
    return invstd;
}
static __attribute__((noinline)) void k_attn(const real_t *Q, const real_t *K, const real_t *V, real_t *S,
                                             real_t *P, real_t *out, int seq, real_t inv_sqrt) {
    for (int i = 0; i < seq; i++)
        for (int j = 0; j < seq; j++)
            S[i * seq + j] = (real_t)(r_dotp(Q + i * D_HEAD, K + j * D_HEAD, D_HEAD) * inv_sqrt);
    for (int i = 0; i < seq; i++) {
        real_t m = r_redmax(S + i * seq, (size_t)seq);
        real_t sm = r_subexp_sum(S + i * seq, S + i * seq, m, (size_t)seq);
        r_scale(P + i * seq, S + i * seq, (real_t)((real_t)1 / sm), (size_t)seq);
    }
    FMATMUL(out, P, V, (unsigned)seq, (unsigned)seq, (unsigned)D_HEAD);
}

static void rec_spm(uint64_t ctrl, uint64_t ctrl_sync, uint64_t dram_med, uint64_t spm_med,
                    uint64_t stage_in, uint64_t copy_out, int equal) {
    volatile uint32_t *r = RES_SPM + 2 + (g_r - 1) * SPM_WORDS;
    r[0] = (uint32_t)ctrl;     r[1] = (uint32_t)ctrl_sync;
    r[2] = (uint32_t)dram_med; r[3] = (uint32_t)spm_med;
    r[4] = (uint32_t)stage_in; r[5] = (uint32_t)copy_out;
    r[6] = (uint32_t)equal;
}
#endif

#ifndef RL_MEM_BPC
#define RL_MEM_BPC   8ull
#endif
#define RL_FMA_PEAK  (2u * (64u / (8u * (unsigned)sizeof(real_t))) * 2u)

#define RL_LANE_ELEM (RL_FMA_PEAK / 2u)

static uint32_t rec_ops_per_elem(const char *tag) {
#if defined(USE_VTRANS)
    const uint32_t EXP_OPS = 1;
#else
    const uint32_t EXP_OPS = 21;

#endif
    switch (tag[0]) {
        case 'r': return 1;
        case 'e': return EXP_OPS;
        case 'l': return 5;
#ifdef USE_SWOPT
        case 'g': return EXP_OPS + 4;
        case 's': return EXP_OPS + 4;
#else
        case 'g': return EXP_OPS + 3;
        case 's': return EXP_OPS + 4;
#endif
        default : return 1;
    }
}

static uint32_t rec_bytes_per_elem(const char *tag) {
    const uint32_t E = (uint32_t)sizeof(real_t);
    switch (tag[0]) {
        case 'r': return 1u * E;
        case 'e': return 2u * E;
        case 'l': return 9u * E;
#ifdef USE_SWOPT
        case 'g': return 2u * E;
        case 's': return 5u * E;
#else
        case 'g': return 9u * E;
        case 's': return 8u * E;
#endif
        default : return 0u;
    }
}

static uint64_t rec_flop(const char *tag, uint64_t n) {
    if (tag[0] == 'm' && tag[1] == 'm')   return 2ull * n * n * n;
    if (tag[0] == 'd' && tag[1] == 'p')   return 2ull * n;
    if (tag[0] == 'a')                    return 4ull * n * n * D_HEAD;
    return 0;
}

static void rec(uint64_t cyc, int finite, const char *tag, int n) {
    RES[2 + g_r] = (uint32_t)cyc;
    if (finite) g_pass |= (1u << g_r);
    if (g_r == 0) {
        bp_col("kernel", 7); bp_col("N", 7); bp_col("cycles", 11);
        bp_col("rate", 10); bp_col("unit", 11); bp_col("ideal", 10);
        bp_col("% of ideal", 11); bp_puts("ideal from\n");
        bp_puts("------ ------ ---------- --------- ---------- --------- ---------- ----------\n");
    }

    uint64_t f = rec_flop(tag, (uint64_t)n);
    uint64_t rate, ideal;
    const char *unit, *ref;
    if (f) {
        rate = cyc ? f * 1000 / cyc : 0;
        unit = "FLOP/cycle";
        if (tag[0] == 'd') {
            uint64_t comp = (uint64_t)RL_FMA_PEAK * 1000;
            uint64_t mem  = RL_MEM_BPC * 1000 / (2 * sizeof(real_t)) * 2;
            ideal = comp < mem ? comp : mem;
            ref   = comp < mem ? "compute" : "memory";
        } else {
            ideal = (uint64_t)RL_FMA_PEAK * 1000;
            ref   = "compute";
        }
    } else {
        rate  = cyc ? (uint64_t)n * 1000 / cyc : 0;
        unit  = "elem/cycle";
        uint64_t comp = (uint64_t)RL_LANE_ELEM * 1000 / rec_ops_per_elem(tag);
        uint64_t mem  = RL_MEM_BPC * 1000 / rec_bytes_per_elem(tag);
        ideal = comp < mem ? comp : mem;
        ref   = comp < mem ? "compute" : "memory";
    }

    bp_col(tag, 7); bp_col_u64((uint64_t)n, 7); bp_col_u64(cyc, 11);
    bp_col_frac(rate, 3, 10); bp_col(unit, 11); bp_col_frac(ideal, 3, 10);
    bp_col_pct(ideal ? rate * 1000 / ideal : 0, 10); bp_puts("  ");
    bp_puts(ref); bp_puts("\n");
    g_r++;
}

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;

#if defined(USE_VTRANS) && defined(USE_SWOPT)
#define BUILD_TAG "hw+sw"
#elif defined(USE_VTRANS)
#define BUILD_TAG "hw"
#elif defined(USE_SWOPT)
#define BUILD_TAG "sw"
#else
#define BUILD_TAG "base"
#endif
    uart_puts("\n[roofline " DTYPE_NAME " " BUILD_TAG "] full-suite size sweep\n");

    RES[0] = RES_MAGIC;
    g_r = 0; g_pass = 0;
#ifdef RL_SPM
    RES_SPM[0] = 0;
    uart_puts("[roofline] RL_SPM: every record again with its operands in the scratchpad\n");
#endif

#ifdef I8

    for (int s = 0; s < N_MM; s++) {
        int n = MM_N[s];
        for (int i = 0; i < n * n; i++) {
            A[i] = (int8_t)((i % 23) - 11);
            B[i] = (int8_t)((i % 17) - 8);
            C[i] = 0;
        }
        uint64_t t0 = rdcycle();
        bp_imatmul_8x8(C, A, B, (unsigned)n, (unsigned)n, (unsigned)n);
        uint64_t t1 = rdcycle();
        rec(t1 - t0, 1, "mm", n);
    }

    for (int s = 0; s < N_MM; s++) {
        int n = MM_N[s];
        for (int i = 0; i < n * n; i++) {
            A[i] = (int8_t)((i % 23) - 11);
            B[i] = (int8_t)((i % 17) - 8);
            C[i] = 0;
        }
        uint64_t t0 = rdcycle();
        bp_imatmul_16x16(C, A, B, (unsigned)n, (unsigned)n, (unsigned)n);
        uint64_t t1 = rdcycle();
        rec(t1 - t0, 1, "mm16", n);
    }
    for (int s = 0; s < N_EW; s++) {
        int n = EW_N[s];
        for (int i = 0; i < n; i++) {
            A[i] = (int8_t)((i % 13) - 6);
            B[i] = (int8_t)((i % 11) - 5);
        }
        { uint64_t t0 = rdcycle(); volatile int8_t d = i8_dotp(A, B, (size_t)n); uint64_t t1 = rdcycle();
          (void)d; rec(t1 - t0, 1, "dp", n); }
        { uint64_t t0 = rdcycle(); volatile int8_t d = i8_redsum(A, (size_t)n); uint64_t t1 = rdcycle();
          (void)d; rec(t1 - t0, 1, "red", n); }
    }
#else

    for (int s = 0; s < N_MM; s++) {
        int n = MM_N[s];
        for (int i = 0; i < n * n; i++) {
            A[i] = (real_t)(((float)((i % 23) - 11)) * 0.0625f);
            B[i] = (real_t)(((float)((i % 17) - 8)) * 0.0625f);
            C[i] = (real_t)0;
        }
        uint64_t t0 = rdcycle();
        FMATMUL(C, A, B, (unsigned)n, (unsigned)n, (unsigned)n);
        uint64_t t1 = rdcycle();
#ifdef RL_SPM
        st_drain(); uint64_t t2 = rdcycle();
#endif
        rec(t1 - t0, C[0] == C[0] && C[n * n - 1] == C[n * n - 1], "mm", n);
#ifdef RL_SPM
        {
            size_t by = (size_t)n * n * sizeof(real_t);
            real_t *sB = (real_t *)SPM_BASE, *sC = sB + (size_t)n * n;
            uint64_t stg = T_SYNC(vcopy(sB, B, by)), md, ms;
            ROUNDS(FMATMUL(C, A, B, (unsigned)n, (unsigned)n, (unsigned)n),
                   FMATMUL(sC, A, sB, (unsigned)n, (unsigned)n, (unsigned)n), md, ms);
            uint64_t cpo = T_SYNC(vcopy(E, sC, by));
            rec_spm(t1 - t0, t2 - t0, md, ms, stg, cpo, same_bytes(E, C, by));
        }
#endif
    }

    for (int s = 0; s < N_EW; s++) {
        int n = EW_N[s];
        for (int i = 0; i < n; i++) {
            A[i] = (real_t)(((float)((i % 13) - 6)) * 0.0625f);
            B[i] = (real_t)(((float)((i % 11) - 5)) * 0.0625f);
        }

        { uint64_t t0 = rdcycle(); volatile real_t d = r_dotp(A, B, (size_t)n); uint64_t t1 = rdcycle();
#ifdef RL_SPM
          st_drain(); uint64_t t2 = rdcycle();
#endif
          rec(t1 - t0, d == d, "dp", n);
#ifdef RL_SPM
          size_t by = (size_t)n * sizeof(real_t);
          uint64_t stg = T_SYNC((vcopy(SPM_EW(0), A, by), vcopy(SPM_EW(1), B, by))), md, ms;
          volatile real_t vd = 0, vs = 0;
          ROUNDS(vd = k_dp(A, B, (size_t)n), vs = k_dp(SPM_EW(0), SPM_EW(1), (size_t)n), md, ms);
          real_t d0 = d, d1 = vd, s1 = vs;
          rec_spm(t1 - t0, t2 - t0, md, ms, stg, 0,
                  same_bytes(&s1, &d1, sizeof(real_t)) && same_bytes(&d1, &d0, sizeof(real_t)));
#endif
        }

        { uint64_t t0 = rdcycle(); volatile real_t d = r_redsum(A, (size_t)n); uint64_t t1 = rdcycle();
#ifdef RL_SPM
          st_drain(); uint64_t t2 = rdcycle();
#endif
          rec(t1 - t0, d == d, "red", n);
#ifdef RL_SPM
          size_t by = (size_t)n * sizeof(real_t);
          uint64_t stg = T_SYNC(vcopy(SPM_EW(0), A, by)), md, ms;
          volatile real_t vd = 0, vs = 0;
          ROUNDS(vd = k_red(A, (size_t)n), vs = k_red(SPM_EW(0), (size_t)n), md, ms);
          real_t d0 = d, d1 = vd, s1 = vs;
          rec_spm(t1 - t0, t2 - t0, md, ms, stg, 0,
                  same_bytes(&s1, &d1, sizeof(real_t)) && same_bytes(&d1, &d0, sizeof(real_t)));
#endif
        }

        { uint64_t t0 = rdcycle(); r_vexp(C, A, (size_t)n); uint64_t t1 = rdcycle();
#ifdef RL_SPM
          st_drain(); uint64_t t2 = rdcycle();
#endif
          rec(t1 - t0, C[0] == C[0] && C[n - 1] == C[n - 1], "exp", n);
#ifdef RL_SPM
          size_t by = (size_t)n * sizeof(real_t);
          uint64_t stg = T_SYNC(vcopy(SPM_EW(0), A, by)), md, ms;
          ROUNDS(k_exp(C, A, (size_t)n), k_exp(SPM_EW(2), SPM_EW(0), (size_t)n), md, ms);
          uint64_t cpo = T_SYNC(vcopy(E, SPM_EW(2), by));
          rec_spm(t1 - t0, t2 - t0, md, ms, stg, cpo, same_bytes(E, C, by));
#endif
        }

        { uint64_t t0 = rdcycle();
          r_gelu(B, A, C, (size_t)n);
          uint64_t t1 = rdcycle();
#ifdef RL_SPM
          st_drain(); uint64_t t2 = rdcycle();
#endif
          rec(t1 - t0, B[0] == B[0] && B[n - 1] == B[n - 1], "gelu", n);
#ifdef RL_SPM
          size_t by = (size_t)n * sizeof(real_t);
          uint64_t stg = T_SYNC(vcopy(SPM_EW(0), A, by)), md, ms;
          ROUNDS(k_gelu(B, A, C, (size_t)n), k_gelu(SPM_EW(1), SPM_EW(0), SPM_EW(2), (size_t)n), md, ms);
          uint64_t cpo = T_SYNC(vcopy(E, SPM_EW(1), by));
          rec_spm(t1 - t0, t2 - t0, md, ms, stg, cpo, same_bytes(E, B, by));
#endif
        }

        { for (int i = 0; i < n; i++) A[i] = (real_t)(((float)((i % 13) - 6)) * 0.0625f);
          uint64_t t0 = rdcycle();
          real_t mx = r_redmax(A, (size_t)n);
          real_t sm = r_subexp_sum(C, A, mx, (size_t)n);
          r_scale(C, C, (real_t)((real_t)1 / sm), (size_t)n);
          uint64_t t1 = rdcycle();
#ifdef RL_SPM
          st_drain(); uint64_t t2 = rdcycle();
#endif
          rec(t1 - t0, C[0] == C[0] && sm == sm, "sfx", n);
#ifdef RL_SPM
          size_t by = (size_t)n * sizeof(real_t);
          uint64_t stg = T_SYNC(vcopy(SPM_EW(0), A, by)), md, ms;
          real_t smd = 0, sms = 0;
          ROUNDS(smd = k_sfx(C, A, (size_t)n), sms = k_sfx(SPM_EW(2), SPM_EW(0), (size_t)n), md, ms);
          uint64_t cpo = T_SYNC(vcopy(E, SPM_EW(2), by));
          rec_spm(t1 - t0, t2 - t0, md, ms, stg, cpo,
                  same_bytes(E, C, by) && same_bytes(&sms, &smd, sizeof(real_t)) && same_bytes(&smd, &sm, sizeof(real_t)));
#endif
        }

        { uint64_t t0 = rdcycle();
          real_t mean = r_redsum(A, (size_t)n) / (real_t)n;
          r_subs(B, A, mean, (size_t)n); r_vmul(C, B, B, (size_t)n);
          real_t var = r_redsum(C, (size_t)n) / (real_t)n;
          real_t invstd = (real_t)1 / sqrt_r(var + LN_EPS);
          r_scale(D, B, invstd, (size_t)n);
          uint64_t t1 = rdcycle();
#ifdef RL_SPM
          st_drain(); uint64_t t2 = rdcycle();
#endif
          rec(t1 - t0, D[0] == D[0] && invstd == invstd, "lyn", n);
#ifdef RL_SPM
          size_t by = (size_t)n * sizeof(real_t);
          uint64_t stg = T_SYNC(vcopy(SPM_EW(0), A, by)), md, ms;
          real_t ivd = 0, ivs = 0;
          ROUNDS(ivd = k_lyn(B, C, D, A, (size_t)n),
                 ivs = k_lyn(SPM_EW(1), SPM_EW(2), SPM_EW(3), SPM_EW(0), (size_t)n), md, ms);
          uint64_t cpo = T_SYNC(vcopy(E, SPM_EW(3), by));
          rec_spm(t1 - t0, t2 - t0, md, ms, stg, cpo,
                  same_bytes(E, D, by) && same_bytes(&ivs, &ivd, sizeof(real_t)) && same_bytes(&ivd, &invstd, sizeof(real_t)));
#endif
        }
    }

    for (int s = 0; s < N_ATT; s++) {
        int seq = ATT_S[s];
        real_t *Q = A, *K = A + seq * D_HEAD, *V = B, *S = C, *P = D, *outp = B + seq * D_HEAD;
        for (int i = 0; i < seq * D_HEAD; i++) {
            Q[i] = (real_t)(((float)((i % 23) - 11)) * 0.03125f);
            K[i] = (real_t)(((float)((i % 17) - 8)) * 0.03125f);
            V[i] = (real_t)(((float)((i % 13) - 6)) * 0.03125f);
        }
        real_t inv_sqrt = (real_t)1 / sqrt_r((real_t)D_HEAD);
        uint64_t t0 = rdcycle();
#ifdef USE_SWOPT

        real_t *Kt = A + 2 * seq * D_HEAD;
        r_scale(Q, Q, inv_sqrt, (size_t)(seq * D_HEAD));
        r_transpose(Kt, K, (size_t)seq, (size_t)D_HEAD);
        FMATMUL(S, Q, Kt, (unsigned)seq, (unsigned)D_HEAD, (unsigned)seq);
#else

        for (int i = 0; i < seq; i++)
            for (int j = 0; j < seq; j++)
                S[i * seq + j] = (real_t)(r_dotp(Q + i * D_HEAD, K + j * D_HEAD, D_HEAD) * inv_sqrt);
#endif
        for (int i = 0; i < seq; i++) {
            real_t m = r_redmax(S + i * seq, (size_t)seq);
            real_t sm = r_subexp_sum(S + i * seq, S + i * seq, m, (size_t)seq);
            r_scale(P + i * seq, S + i * seq, (real_t)((real_t)1 / sm), (size_t)seq);
        }
        FMATMUL(outp, P, V, (unsigned)seq, (unsigned)seq, (unsigned)D_HEAD);
        uint64_t t1 = rdcycle();
#ifdef RL_SPM
        st_drain(); uint64_t t2 = rdcycle();
#endif
        rec(t1 - t0, outp[0] == outp[0] && outp[seq * D_HEAD - 1] == outp[seq * D_HEAD - 1], "attn", seq);
#ifdef RL_SPM
        {

            size_t by = (size_t)seq * D_HEAD * sizeof(real_t);
            real_t *sQ = SPM_EW(0), *sK = sQ + (size_t)seq * D_HEAD;
            real_t *sV = SPM_EW(1), *sO = sV + (size_t)seq * D_HEAD;
            uint64_t stg = T_SYNC((vcopy(sQ, Q, by), vcopy(sK, K, by), vcopy(sV, V, by))), md, ms;
            ROUNDS(k_attn(Q, K, V, S, P, outp, seq, inv_sqrt), k_attn(sQ, sK, sV, S, P, sO, seq, inv_sqrt), md, ms);
            uint64_t cpo = T_SYNC(vcopy(E, sO, by));
            rec_spm(t1 - t0, t2 - t0, md, ms, stg, cpo, same_bytes(E, outp, by));
        }
#endif
    }
#endif

#ifdef RL_SPM
    RES_SPM[1] = (uint32_t)g_r;
    RES_SPM[0] = RES_SPM_MAGIC;
#endif
    RES[1] = (uint32_t)g_r;
    RES[2 + g_r] = g_pass;
    HB[0] = 0xCAFE000F;
    uart_puts("roofline done\n");
    while (1) {}
}
