// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include "../common/dtype.h"
#include "../common/exp_hp.h"
#include "../common/kio.h"
#include "../common/bench_print.h"
#ifdef HP
#include "hp-fmatmul.h"
#define MATMUL(c,a,b,M,N,P) hp_fmatmul((c),(a),(b),(M),(N),(P))
#else
#include "sp-fmatmul.h"
#define MATMUL(c,a,b,M,N,P) sp_fmatmul((c),(a),(b),(M),(N),(P))
#endif

#define HB     ((volatile uint32_t *)0x40100000)
#define VERIFY ((volatile uint32_t *)0x40100100)

#define DIAG_M ((volatile uint32_t *)0x40110000)
#define DIAG_S ((volatile uint32_t *)0x40112000)

#ifndef GQA_SEQ
#define GQA_SEQ       128
#endif
#ifndef GQA_D_MODEL
#define GQA_D_MODEL   4096
#endif
#ifndef GQA_D_HEAD
#define GQA_D_HEAD    128
#endif
#ifndef GQA_GROUP
#define GQA_GROUP     4
#endif
#ifndef GQA_KV_HEADS
#define GQA_KV_HEADS  2
#endif

#define SEQ      GQA_SEQ
#define D_MODEL  GQA_D_MODEL
#define D_HEAD   GQA_D_HEAD
#define G        GQA_GROUP
#define NKV      GQA_KV_HEADS
#define NQ       (NKV * G)

#define MODEL_KV_HEADS 8
#define MODEL_Q_HEADS  32

#define INV_SQRT_DH  ((real_t)0.08838834764831845f)

#ifdef HP
#define ROWSUM_TOL 0.1
#define S_REL_TOL  0.15
#else
#define ROWSUM_TOL 0.01
#define S_REL_TOL  0.02
#endif

real_t X   [SEQ * D_MODEL]      __attribute__((aligned(32)));
real_t Wq  [D_MODEL * NQ  * D_HEAD] __attribute__((aligned(32)));
real_t Wk  [D_MODEL * NKV * D_HEAD] __attribute__((aligned(32)));
real_t Wv  [D_MODEL * NKV * D_HEAD] __attribute__((aligned(32)));
real_t Qall[SEQ * NQ  * D_HEAD] __attribute__((aligned(32)));
real_t Kall[SEQ * NKV * D_HEAD] __attribute__((aligned(32)));
real_t Vall[SEQ * NKV * D_HEAD] __attribute__((aligned(32)));
real_t Oall[SEQ * NQ  * D_HEAD] __attribute__((aligned(32)));
real_t Qb  [SEQ * D_HEAD] __attribute__((aligned(32)));
real_t Kb  [SEQ * D_HEAD] __attribute__((aligned(32)));
real_t Vb  [SEQ * D_HEAD] __attribute__((aligned(32)));
real_t Ob  [SEQ * D_HEAD] __attribute__((aligned(32)));
real_t Kt  [D_HEAD * SEQ] __attribute__((aligned(32)));
real_t Smat[SEQ * SEQ] __attribute__((aligned(32)));
real_t Pmat[SEQ * SEQ] __attribute__((aligned(32)));
#ifndef GQA_NO_MHA_ARM
real_t KVm [SEQ * NQ * D_HEAD] __attribute__((aligned(32)));
#endif

static void fill_pattern(real_t *a, size_t n, uint32_t seed, float step) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        a[i] = (real_t)(((float)(int32_t)((s >> 24) & 0xF) - 7.5f) * step);
    }
}

static inline void r_zero(real_t *o, size_t n) {
    size_t vl;
    for (size_t i = 0; i < n; i += vl) { vl = VSETVL(n - i); VSE(o + i, VFMV_V_F(0, vl), vl); }
}

static void head_gather(real_t *dst, const real_t *src, int h, int nh) {
    for (int i = 0; i < SEQ; i++)
        for (int d = 0; d < D_HEAD; d++)
            dst[i * D_HEAD + d] = src[i * nh * D_HEAD + h * D_HEAD + d];
}

static void head_scatter(real_t *dst, const real_t *src, int h, int nh) {
    for (int i = 0; i < SEQ; i++)
        for (int d = 0; d < D_HEAD; d++)
            dst[i * nh * D_HEAD + h * D_HEAD + d] = src[i * D_HEAD + d];
}

static void stage_stat(const char *nm, const real_t *a, int n) {
    int nc = 0; double mx = 0;
    for (int i = 0; i < n; i++) { double v = (double)a[i];
        if (v != v) nc++; else { double av = v < 0 ? -v : v; if (av > mx) mx = av; } }
    uart_puts("  "); uart_puts(nm); uart_puts(" nan="); print_u64((uint64_t)nc);
    uart_puts(" max="); print_rdec(mx); uart_puts("\n");
}

static double s00_golden(int h) {
    int kv = h / G;
    double s = 0;
    for (int d = 0; d < D_HEAD; d++)
        s += (double)Qall[h * D_HEAD + d] * (double)Kall[kv * D_HEAD + d];
    return s * (double)INV_SQRT_DH;
}

static double o_golden(int idx) {
    int i = idx / D_HEAD, j = idx % D_HEAD; double s = 0;
    for (int k = 0; k < SEQ; k++) s += (double)Pmat[i * SEQ + k] * (double)Vb[k * D_HEAD + j];
    return s;
}

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;

    uart_puts("\n[gqa attention " DTYPE_NAME "] llama-3-8b block\n");
    uart_puts("  d_model = ");        print_u64(D_MODEL);
    uart_puts("   d_head = ");        print_u64(D_HEAD);
    uart_puts("   seq = ");           print_u64(SEQ); uart_puts("\n");
    uart_puts("  group size = ");     print_u64(G);
    uart_puts(" query heads per KV head\n");
    uart_puts("  running ");          print_u64(NKV);
    uart_puts(" of ");                print_u64(MODEL_KV_HEADS);
    uart_puts(" KV heads (");         print_u64(NQ);
    uart_puts(" of ");                print_u64(MODEL_Q_HEADS);
    uart_puts(" query heads)\n\n");

    fill_pattern(X,  (size_t)SEQ * D_MODEL,          0x2026u, 0.03125f);
    fill_pattern(Wq, (size_t)D_MODEL * NQ  * D_HEAD, 0x51a7u, 0.03125f);
    fill_pattern(Wk, (size_t)D_MODEL * NKV * D_HEAD, 0x9c31u, 0.03125f);
    fill_pattern(Wv, (size_t)D_MODEL * NKV * D_HEAD, 0xb4e9u, 0.03125f);
    HB[0] = 0xCAFE0003;

    uint64_t t0 = rdcycle();
    MATMUL(Qall, X, Wq, SEQ, D_MODEL, NQ * D_HEAD);
    uint64_t t_q = rdcycle();
    MATMUL(Kall, X, Wk, SEQ, D_MODEL, NKV * D_HEAD);
    MATMUL(Vall, X, Wv, SEQ, D_MODEL, NKV * D_HEAD);
    uint64_t t_kv = rdcycle();
    HB[0] = 0xCAFE0004;

    double row_sum_min = 1e9, row_sum_max = -1e9;
    double s00_max_rel = 0;
    int rows_ok = 1, map_ok = 1;
    uint64_t c_pack = 0, c_score = 0, c_soft = 0, c_pv = 0;

    uint32_t rows_fail_loc = 0xFFFFFFFFu;
    uint32_t rows_fail_chk = 0;
    uint32_t rows_fail_s   = 0;
    uint32_t rows_fail_cnt = 0;
    uint32_t map_fail_head = 0xFFFFFFFFu;
    uint32_t map_fail_bits = 0;

    for (int h = 0; h < NQ; h++) {
        const int kv = h / G;

        uint64_t a = rdcycle();
        head_gather(Qb, Qall, h,  NQ);
        head_gather(Kb, Kall, kv, NKV);
        head_gather(Vb, Vall, kv, NKV);
        uint64_t b = rdcycle(); c_pack += b - a;

#ifdef USE_SWOPT

        r_scale(Qb, Qb, INV_SQRT_DH, (size_t)SEQ * D_HEAD);
        r_transpose(Kt, Kb, SEQ, D_HEAD);
        MATMUL(Smat, Qb, Kt, SEQ, D_HEAD, SEQ);
#else
        for (int i = 0; i < SEQ; i++)
            for (int j = 0; j < SEQ; j++)
                Smat[i * SEQ + j] =
                    (real_t)(r_dotp(Qb + i * D_HEAD, Kb + j * D_HEAD, D_HEAD) * INV_SQRT_DH);
#endif
        a = rdcycle(); c_score += a - b;

        {
            double ref = s00_golden(h), got = (double)Smat[0];
            double ae = got - ref; if (ae < 0) ae = -ae;
            double ar = ref < 0 ? -ref : ref;
            double rel = ar > 1e-6 ? ae / ar : ae;
            if (rel > s00_max_rel) s00_max_rel = rel;
            if (!(rel <= S_REL_TOL)) {
                if (map_ok) { map_fail_head = (uint32_t)h;
                              map_fail_bits = (uint32_t)RBITS(Smat[0]); }
                map_ok = 0;
            }
        }

        b = rdcycle();
        for (int i = 0; i < SEQ; i++) {
            real_t *rs = Smat + i * SEQ, *rp = Pmat + i * SEQ;
            size_t len = (size_t)i + 1;
            real_t m = r_redmax(rs, len);
            real_t s = r_subexp_sum(rp, rs, m, len);

            DIAG_M[h * SEQ + i] = (uint32_t)RBITS(m);
            DIAG_S[h * SEQ + i] = (uint32_t)RBITS(s);
            r_scale(rp, rp, (real_t)((real_t)1 / s), len);
            if (len < (size_t)SEQ) r_zero(rp + len, (size_t)SEQ - len);
            double chk = 0; for (size_t j = 0; j < len; j++) chk += (double)rp[j];
            if (chk < row_sum_min) row_sum_min = chk;
            if (chk > row_sum_max) row_sum_max = chk;
            if (!(chk >= 1.0 - ROWSUM_TOL && chk <= 1.0 + ROWSUM_TOL)) {
                if (rows_ok) { rows_fail_loc = ((uint32_t)h << 8) | (uint32_t)i;
                               rows_fail_chk = f2u((float)chk);
                               rows_fail_s   = (uint32_t)RBITS(s); }
                rows_fail_cnt++;
                rows_ok = 0;
            }
        }
        a = rdcycle(); c_soft += a - b;

        MATMUL(Ob, Pmat, Vb, SEQ, SEQ, D_HEAD);
        b = rdcycle(); c_pv += b - a;
        head_scatter(Oall, Ob, h, NQ);
        c_pack += rdcycle() - b;
    }
    uint64_t t1 = rdcycle();
    HB[0] = 0xCAFE0005;

    uint64_t c_kv_mha = 0;
#ifndef GQA_NO_MHA_ARM
    {
        uint64_t a = rdcycle();
        MATMUL(KVm, X, Wq, SEQ, D_MODEL, NQ * D_HEAD);
        MATMUL(KVm, X, Wq, SEQ, D_MODEL, NQ * D_HEAD);
        c_kv_mha = rdcycle() - a;
    }
#endif
    HB[0] = 0xCAFE0006;

#ifdef HP
    const double O_TOL = 0.08;
#else
    const double O_TOL = 1e-5;
#endif
    real_t o_absmax_r = (real_t)0;
    for (int i = 0; i < SEQ * D_HEAD; i++) {
        real_t a = r_fabs(Ob[i]); if (a > o_absmax_r) o_absmax_r = a;
    }
    const double o_absmax = (double)o_absmax_r;
    const double o_allow  = O_TOL * o_absmax;
    int spot[3] = { 0, SEQ * D_HEAD / 2, SEQ * D_HEAD - 1 };
    double o_max_rel = 0, o_max_abs = 0; int o_ok = 1;
    for (int s = 0; s < 3; s++) {
        double got = (double)Ob[spot[s]], ref = o_golden(spot[s]);
        if (got != got) { o_ok = 0; continue; }
        double ae = got - ref; if (ae < 0) ae = -ae;
        double rel = (ref != 0.0) ? ae / (ref < 0 ? -ref : ref) : ae;
        if (rel > o_max_rel) o_max_rel = rel;
        if (ae > o_max_abs) o_max_abs = ae;
        if (!(ae <= o_allow)) o_ok = 0;
    }
    int pass = rows_ok && map_ok && o_ok;

    const uint64_t cyc_q    = t_q - t0;
    const uint64_t cyc_kv   = t_kv - t_q;
    const uint64_t cyc_attn = t1 - t_kv;
    const uint64_t cyc_tot  = t1 - t0;
    const uint64_t flop_q   = 2ull * SEQ * D_MODEL * NQ  * D_HEAD;
    const uint64_t flop_kv  = 2ull * 2 * SEQ * D_MODEL * NKV * D_HEAD;
    const uint64_t flop_att = (uint64_t)NQ * 4ull * SEQ * SEQ * D_HEAD;
    const uint64_t flop_tot = flop_q + flop_kv + flop_att;

    VERIFY[0] = 0xA1A1B1B1;
    VERIFY[1] = (uint32_t)pass;
    VERIFY[2] = (uint32_t)cyc_tot;
    VERIFY[3] = (uint32_t)cyc_kv;
    VERIFY[4] = (uint32_t)c_kv_mha;
    VERIFY[5] = f2u((float)row_sum_max);
    VERIFY[6] = f2u((float)s00_max_rel);
    VERIFY[7] = RBITS(Ob[0]);
    VERIFY[8]  = (uint32_t)(rows_ok | (map_ok << 1) | (o_ok << 2));
    VERIFY[9]  = f2u((float)row_sum_min);
    VERIFY[10] = rows_fail_loc;
    VERIFY[11] = rows_fail_chk;
    VERIFY[12] = rows_fail_s;
    VERIFY[13] = rows_fail_cnt;
    VERIFY[14] = map_fail_head;
    VERIFY[15] = map_fail_bits;

    uart_puts("stage cycles\n");
    uart_puts("  Q projection      = "); print_u64(cyc_q);   uart_puts("\n");
    uart_puts("  KV projection     = "); print_u64(cyc_kv);  uart_puts("\n");
    uart_puts("  attention heads   = "); print_u64(cyc_attn); uart_puts("\n");
    uart_puts("    head gather     = "); print_u64(c_pack);  uart_puts("\n");
    uart_puts("    scores  Q K^T   = "); print_u64(c_score); uart_puts("\n");
    uart_puts("    causal softmax  = "); print_u64(c_soft);  uart_puts("\n");
    uart_puts("    output  P V     = "); print_u64(c_pv);    uart_puts("\n");
    uart_puts("  total             = "); print_u64(cyc_tot); uart_puts("\n\n");

    uart_puts("throughput\n");
    uart_puts("  total FLOP        = "); print_u64(flop_tot); uart_puts("\n");
    uart_puts("  rate              = "); bp_frac(flop_tot * 100 / cyc_tot, 2);
    uart_puts(" FLOP/cycle of ");        print_u64(R_PEAK_FMA / 2);
    uart_puts(" peak\n");
    uart_puts("  projection rate   = ");
    bp_frac((flop_q + flop_kv) * 100 / (cyc_q + cyc_kv), 2); uart_puts(" FLOP/cycle\n\n");

#ifndef GQA_NO_MHA_ARM
    uart_puts("grouped-query saving (KV projection, measured)\n");
    uart_puts("  GQA  ");  print_u64(NKV); uart_puts(" KV heads   = ");
    print_u64(cyc_kv); uart_puts(" cycles\n");
    uart_puts("  MHA  ");  print_u64(NQ);  uart_puts(" KV heads   = ");
    print_u64(c_kv_mha); uart_puts(" cycles\n");
    uart_puts("  cycles removed    = ");
    print_u64(c_kv_mha > cyc_kv ? c_kv_mha - cyc_kv : 0); uart_puts("\n");
    uart_puts("  ratio MHA / GQA   = ");
    bp_frac(c_kv_mha * 100 / (cyc_kv ? cyc_kv : 1), 2);
    uart_puts("   (expect ");  print_u64(G); uart_puts(".00)\n\n");
#endif

    uart_puts("stage nan / max magnitude\n");
    stage_stat("Q", Qall, SEQ * NQ  * D_HEAD);
    stage_stat("K", Kall, SEQ * NKV * D_HEAD);
    stage_stat("V", Vall, SEQ * NKV * D_HEAD);
    stage_stat("P", Pmat, SEQ * SEQ);
    stage_stat("O", Oall, SEQ * NQ  * D_HEAD);
    uart_puts("\n");

    uart_puts("checks\n");
    uart_puts("  causal row sum    = "); print_rdec(row_sum_min);
    uart_puts(" .. ");                   print_rdec(row_sum_max);
    uart_puts("   (expect 1.0)\n");
    uart_puts("  head -> KV map    = "); print_rdec(s00_max_rel);
    uart_puts(" max relative error on S[0][0]");
    uart_puts(map_ok ? "   ok\n" : "   MISMATCH\n");
    for (int s = 0; s < 3; s++) {
        double ref = o_golden(spot[s]);
        uart_puts("  O["); print_u64((uint64_t)spot[s]); uart_puts("] got ");
        print_hexdec(RBITS(Ob[spot[s]]), (double)Ob[spot[s]]);
        uart_puts(" expect "); print_hexdec(RBITS(r_round(ref)), ref); uart_puts("\n");
    }
    uart_puts("  max |O|           = "); print_rdec(o_absmax); uart_puts("\n");
    uart_puts("  O max abs error   = "); print_rdec(o_max_abs);
    uart_puts("   allowed ");            print_rdec(o_allow);
    uart_puts(o_ok ? "   ok\n" : "   OVER\n");
    uart_puts("  O max rel error   = "); print_rdec(o_max_rel);
    uart_puts("   not a verdict: meaningless where |O| is near zero\n\n");
    uart_puts("verify = "); uart_puts(pass ? "PASS\n" : "FAIL\n");

    HB[0] = 0xCAFE000F;
    while (1) {}
}
