// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

#include "../common/uart_apb.h"
#define RHEX_NIB 8
#include "../common/kio.h"
#include "../common/perf.h"
#include "../common/bench_print.h"

#define HB  ((volatile uint32_t *)0x40100000)
#define VER ((volatile uint32_t *)0x40100100)
#define RES ((volatile uint32_t *)0x40100200)
#define RES_MAGIC 0x6A7B0001u

#define HASH_K 2654435761u
#define DATA_ELEMS (1u << 23)
#define N_STREAM   (1u << 21)
#define N_G        65536u
#define VLM4       256u
#define NREC       32u
#define GB_MASK_SLOT 80u

#define GB_WIN_SLOT  96u
#define GB_TAP_SLOT  112u
#define GB_VINSN_SLOT  148u

static void res_put64(uint32_t slot, uint64_t v) {
    RES[slot]     = (uint32_t)v;
    RES[slot + 1] = (uint32_t)(v >> 32);
}

static void res_win(uint32_t w, int tap, const perf_snap_t *a, uint64_t cyc) {
    uint32_t s = GB_WIN_SLOT + w * 8u;
    res_put64(s + 0, perf_rd_count(tap)   - a->rd_count);
    res_put64(s + 2, perf_rd_lat_sum(tap) - a->rd_lat_sum);
    RES[s + 4] = (uint32_t)perf_rd_outst_max(tap);
    res_put64(s + 5, cyc);
    RES[s + 7] = (uint32_t)tap;
}

#define SPM_BASE   0x30000000u
#define SPM_ELEMS  (512u * 1024u / 4u)
#define SPM_G_N    16384u
static uint32_t * const SPM = (uint32_t *)(uintptr_t)SPM_BASE;

static uint32_t DATA[DATA_ELEMS] __attribute__((aligned(32)));
static uint32_t IDX[N_G] __attribute__((aligned(32)));

static uint8_t  CODES[N_G] __attribute__((aligned(64)));
static uint32_t SINK [N_G] __attribute__((aligned(64)));

static int g_r;
static uint32_t g_pass;

static const char *g_tag[NREC];
static uint32_t g_span[NREC];
static uint32_t g_elems[NREC];
static uint32_t g_cyc[NREC];

static uint32_t gold_range_sum(uint32_t n) {
    uint64_t tri = ((uint64_t)n * (n - 1)) / 2;
    return (uint32_t)(HASH_K * (uint32_t)tri);
}

static void rec(const char *tag, uint32_t span_kib, uint32_t elems,
                uint64_t cyc, int pass) {
    g_tag[g_r] = tag; g_span[g_r] = span_kib;
    g_elems[g_r] = elems; g_cyc[g_r] = (uint32_t)cyc;
    RES[2 + g_r] = (uint32_t)cyc;
    if (pass) g_pass |= (1u << g_r);
    RES[GB_MASK_SLOT] = g_pass;
    g_r++;

    RES[1] = (uint32_t)g_r;
}

static uint64_t arm_stream(const uint32_t *base, uint32_t n, uint32_t *sum) {
    size_t vl = __riscv_vsetvl_e32m4(VLM4);
    vuint32m4_t acc = __riscv_vmv_v_x_u32m4(0, vl);
    uint64_t t0 = rdcycle();
    for (uint32_t i = 0; i < n; i += (uint32_t)vl) {
        acc = __riscv_vadd_vv_u32m4(acc, __riscv_vle32_v_u32m4(base + i, vl), vl);

        if ((i & 0x1FFFFu) == 0) HB[0] = 0xCAFE01E0u + (i >> 17);
    }
    vuint32m1_t z = __riscv_vmv_v_x_u32m1(0, 1);
    *sum = __riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(acc, z, vl));
    return rdcycle() - t0;
}

static uint64_t arm_gather_resident(const uint32_t *base, uint32_t reps, uint32_t *sum) {
    size_t vl = __riscv_vsetvl_e32m4(VLM4);
    vuint32m4_t vi = __riscv_vle32_v_u32m4(IDX, vl);
    vuint32m4_t acc = __riscv_vmv_v_x_u32m4(0, vl);
    uint64_t t0 = rdcycle();
    for (uint32_t r = 0; r < reps; r++) {
        acc = __riscv_vadd_vv_u32m4(acc,
                  __riscv_vluxei32_v_u32m4(base, vi, vl), vl);

        asm volatile("" ::: "memory");
    }
    vuint32m1_t z = __riscv_vmv_v_x_u32m1(0, 1);
    *sum = __riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(acc, z, vl));
    return rdcycle() - t0;
}

static uint64_t arm_gather_streamed(const uint32_t *base, uint32_t n, uint32_t *sum) {
    size_t vl = __riscv_vsetvl_e32m4(VLM4);
    vuint32m4_t acc = __riscv_vmv_v_x_u32m4(0, vl);
    uint64_t t0 = rdcycle();
    for (uint32_t i = 0; i < n; i += (uint32_t)vl) {
        vuint32m4_t vi = __riscv_vle32_v_u32m4(IDX + i, vl);
        acc = __riscv_vadd_vv_u32m4(acc,
                  __riscv_vluxei32_v_u32m4(base, vi, vl), vl);
    }
    vuint32m1_t z = __riscv_vmv_v_x_u32m1(0, 1);
    *sum = __riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(acc, z, vl));
    return rdcycle() - t0;
}

static uint64_t arm_strided(const uint32_t *base, uint32_t stride_b, uint32_t n, uint32_t *sum) {
    size_t vl = __riscv_vsetvl_e32m4(VLM4);
    vuint32m4_t acc = __riscv_vmv_v_x_u32m4(0, vl);
    uint64_t t0 = rdcycle();
    for (uint32_t i = 0; i < n; i += (uint32_t)vl) {
        const uint32_t *p =
            (const uint32_t *)((const uint8_t *)base + (uint64_t)i * stride_b);
        acc = __riscv_vadd_vv_u32m4(acc,
                  __riscv_vlse32_v_u32m4(p, (ptrdiff_t)stride_b, vl), vl);
    }
    vuint32m1_t z = __riscv_vmv_v_x_u32m1(0, 1);
    *sum = __riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(acc, z, vl));
    return rdcycle() - t0;
}

static uint64_t arm_two_stream_vl(const uint32_t *, uint32_t, uint32_t, uint32_t *);

static uint64_t arm_two_stream(const uint32_t *gbase, uint32_t n, uint32_t *sum) {
    return arm_two_stream_vl(gbase, n, VLM4, sum);
}

static uint64_t arm_two_stream_vl(const uint32_t *gbase, uint32_t n,
                                  uint32_t vl_req, uint32_t *sum) {
    size_t vl = __riscv_vsetvl_e32m4(vl_req);
    vuint32m4_t acc = __riscv_vmv_v_x_u32m4(0, vl);
    uint64_t t0 = rdcycle();
    for (uint32_t i = 0; i < n; i += (uint32_t)vl) {

        vuint8m1_t  c8  = __riscv_vle8_v_u8m1(CODES + i, vl);
        vuint32m4_t idx = __riscv_vzext_vf4_u32m4(c8, vl);
        idx = __riscv_vsll_vx_u32m4(idx, 2, vl);
        acc = __riscv_vadd_vv_u32m4(acc,
                  __riscv_vluxei32_v_u32m4(gbase, idx, vl), vl);
        __riscv_vse32_v_u32m4(SINK + i, acc, vl);
    }
    vuint32m1_t z = __riscv_vmv_v_x_u32m1(0, 1);
    *sum = __riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(acc, z, vl));
    return rdcycle() - t0;
}

static uint32_t lcg_next(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

static uint32_t idx_fill_rand(uint32_t cnt, uint32_t span_elems, uint32_t seed) {
    uint32_t s = seed, gold = 0;
    for (uint32_t i = 0; i < cnt; i++) {
        uint32_t e = (lcg_next(&s) >> 5) & (span_elems - 1);
        IDX[i] = e * 4u;
        gold += HASH_K * e;
    }
    return gold;
}

static void vinit_data(uint32_t *base, uint32_t lo, uint32_t hi) {
    size_t vl = __riscv_vsetvl_e32m4(VLM4);
    for (uint32_t i = lo; i < hi; i += (uint32_t)vl) {
        vuint32m4_t v = __riscv_vid_v_u32m4(vl);
        v = __riscv_vadd_vx_u32m4(v, i, vl);
        v = __riscv_vmul_vx_u32m4(v, HASH_K, vl);
        __riscv_vse32_v_u32m4(base + i, v, vl);
    }
}

static uint32_t RB[4096] __attribute__((aligned(64)));

static void vspm_readback(uint32_t *dst, const uint32_t *spm, uint32_t n) {
    for (uint32_t i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e32m4(n - i);
        __riscv_vse32_v_u32m4(dst + i, __riscv_vle32_v_u32m4(spm + i, vl), vl);
        i += (uint32_t)vl;
    }
}

int main(void) {
    HB[0] = 0xCAFE0001;
    uart_init(UART_DIVISOR);
    HB[0] = 0xCAFE0002;

    RES[0] = RES_MAGIC;
    RES[1] = 0;

    RES[90] = (uint32_t)(perf_magic() >> 32);
    RES[91] = (uint32_t)perf_magic();
    g_r = 0; g_pass = 0;

    uint32_t sum;
    uint64_t cyc;

    {
        HB[0] = 0xCAFE0001u;
        uint32_t bad = 0;
        uint64_t t0 = rdcycle();
        vinit_data(SPM, 0, 1024u);
        vspm_readback(RB, SPM, 1024u);
        cyc = rdcycle() - t0;
        for (uint32_t k = 0; k < 1024u; k++) if (RB[k] != HASH_K * k) bad++;
        rec("spm_vrt", 4, 1024u, cyc, bad == 0);
    }

    {
        HB[0] = 0xCAFE0002u;
        vinit_data(SPM, 0, SPM_ELEMS);
        cyc = arm_stream(SPM, SPM_ELEMS, &sum);
        rec("spm_full", 512, SPM_ELEMS, cyc, sum == gold_range_sum(SPM_ELEMS));
    }

    {
        HB[0] = 0xCAFE0003u;
        uint32_t bad = 0;
        uint64_t t0 = rdcycle();

        size_t vl = __riscv_vsetvl_e32m4(VLM4);
        for (uint32_t k = 1; k < 4096u; k += (uint32_t)vl * 2u) {
            vuint32m4_t v = __riscv_vid_v_u32m4(vl);
            v = __riscv_vmul_vx_u32m4(v, 2u, vl);
            v = __riscv_vadd_vx_u32m4(v, k, vl);
            v = __riscv_vmul_vx_u32m4(v, HASH_K, vl);
            v = __riscv_vnot_v_u32m4(v, vl);
            __riscv_vsse32_v_u32m4(SPM + k, 8, v, vl);
        }
        vspm_readback(RB, SPM, 4096u);
        cyc = rdcycle() - t0;
        for (uint32_t k = 0; k < 4096u; k++) {
            uint32_t want = (k & 1u) ? ~(HASH_K * k) : (HASH_K * k);
            if (RB[k] != want) bad++;
        }
        rec("spm_be", 16, 4096u, cyc, bad == 0);
        vinit_data(SPM, 0, 4096u);
    }

    HB[0] = 0xCAFE0004u;
    cyc = arm_stream(SPM, SPM_ELEMS, &sum);
    rec("spm_strm", 512, SPM_ELEMS, cyc, sum == gold_range_sum(SPM_ELEMS));

    {
        static const uint32_t SPAN_KIB_S[3] = {8, 64, 512};
        for (int a = 0; a < 3; a++) {
            HB[0] = 0xCAFE0005u + (uint32_t)a;
            uint32_t span_elems = SPAN_KIB_S[a] * 1024u / 4u;
            uint32_t gold1 = idx_fill_rand(VLM4, span_elems, 0x5C4A0000u + (uint32_t)a);
            uint32_t reps = SPM_G_N / VLM4;
            perf_snap_t p0; perf_snap(&p0, PERF_TAP_SPM);
            cyc = arm_gather_resident(SPM, reps, &sum);
            if (SPAN_KIB_S[a] == 512) res_win(0, PERF_TAP_SPM, &p0, cyc);
            rec("spm_gr", SPAN_KIB_S[a], SPM_G_N, cyc,
                sum == (uint32_t)(gold1 * reps));
        }
    }

    {
        HB[0] = 0xCAFE0008u;
        uint32_t gold = idx_fill_rand(SPM_G_N, 512u * 1024u / 4u, 0xB16B00B5u);
        cyc = arm_gather_streamed(SPM, SPM_G_N, &sum);
        rec("spm_g+ix", 512, SPM_G_N, cyc, sum == gold);
    }

    {
        HB[0] = 0xCAFE0009u;
        uint32_t n = 8192u;
        uint32_t gold = 0;
        for (uint32_t i = 0; i < n; i++) gold += HASH_K * (i * 16u);
        cyc = arm_strided(SPM, 64u, n, &sum);
        rec("spm_s64B", 512, n, cyc, sum == gold);
    }

    for (uint32_t k = 0; k < 16384u; k++) DATA[k] = HASH_K * k;
    HB[0] = 0xCAFE0011;

    HB[0] = 0xCAFE0012;
    cyc = arm_stream(DATA, 16384u, &sum);
    rec("s64cold", 64, 16384u, cyc, sum == gold_range_sum(16384u));

    HB[0] = 0xCAFE0013;
    vinit_data(DATA, 0, 16384u);

    HB[0] = 0xCAFE0014;
    cyc = arm_stream(DATA, 16384u, &sum);
    rec("s64post", 64, 16384u, cyc, sum == gold_range_sum(16384u));

    HB[0] = 0xCAFE0015;
    vinit_data(DATA, 0, DATA_ELEMS);

    static const uint32_t SN[3] = {16384u, 262144u, N_STREAM};
    for (int a = 0; a < 3; a++) {
        HB[0] = 0xCAFE0100 + (uint32_t)a;
        cyc = arm_stream(DATA, SN[a], &sum);
        rec("stream", (SN[a] * 4u) >> 10, SN[a], cyc,
            sum == gold_range_sum(SN[a]));
    }

    {
        HB[0] = 0xCAFE0120;
        uint32_t gold1 = 0;
        for (uint32_t i = 0; i < VLM4; i++) { IDX[i] = i * 4u; gold1 += HASH_K * i; }
        uint32_t reps = N_G / VLM4;
        cyc = arm_gather_resident(DATA, reps, &sum);
        rec("gseq", 1, N_G, cyc, sum == (uint32_t)(gold1 * reps));
    }

    static const uint32_t SPAN_KIB[5] = {8, 64, 512, 4096, 32768};
    uint64_t cyc_512k = 0, rate_512k = 0, rate_32m = 0, infl_32m = 0;
    for (int a = 0; a < 5; a++) {
        HB[0] = 0xCAFE0121 + (uint32_t)a;
        uint32_t span_elems = SPAN_KIB[a] * 1024u / 4u;
        uint32_t gold1 = idx_fill_rand(VLM4, span_elems, 0xC0FFEE00u + (uint32_t)a);
        uint32_t reps = N_G / VLM4;
        perf_snap_t p0; perf_snap(&p0, PERF_TAP_DRAM_ARA);
        cyc = arm_gather_resident(DATA, reps, &sum);
        if (SPAN_KIB[a] == 512) res_win(1, PERF_TAP_DRAM_ARA, &p0, cyc);
        int ok = (sum == (uint32_t)(gold1 * reps));
        rec("grand", SPAN_KIB[a], N_G, cyc, ok);
        uint64_t rate_m = cyc ? (uint64_t)N_G * 1000 / cyc : 0;
        if (SPAN_KIB[a] == 512)   { cyc_512k = cyc; rate_512k = rate_m; }
        if (SPAN_KIB[a] == 32768) { rate_32m = rate_m; infl_32m = cyc * 40 / N_G; }
    }

    {
        HB[0] = 0xCAFE0129;
        uint32_t gold = idx_fill_rand(N_G, 512u * 1024u / 4u, 0xB16B00B5u);
        cyc = arm_gather_streamed(DATA, N_G, &sum);
        rec("g+ix", 512, N_G, cyc, sum == gold);
    }

    {
        HB[0] = 0xCAFE012A;
        uint32_t gold = 0;
        for (uint32_t i = 0; i < N_G; i++) gold += HASH_K * (i * 16u);
        cyc = arm_strided(DATA, 64u, N_G, &sum);
        rec("s64B", (N_G * 64u) >> 10, N_G, cyc, sum == gold);

        HB[0] = 0xCAFE012B;
        uint32_t n4k = 8192u;
        gold = 0;
        for (uint32_t i = 0; i < n4k; i++) gold += HASH_K * (i * 1024u);
        cyc = arm_strided(DATA, 4096u, n4k, &sum);
        rec("s4096B", 32768, n4k, cyc, sum == gold);
    }

    {

        for (uint32_t i = 0; i < N_G; i++) CODES[i] = (uint8_t)(i * 37u);
        uint32_t n2 = 16384u;
        uint32_t g_2s = 0;
        for (uint32_t i = 0; i < n2; i++) g_2s += HASH_K * (uint32_t)CODES[i];
        uint32_t g_1s_one = 0;
        for (uint32_t i = 0; i < VLM4; i++) g_1s_one += HASH_K * ((i * 37u) & 0xFFu);
        uint32_t g_1s = (uint32_t)(g_1s_one * (n2 / VLM4));
        HB[0] = 0xCAFE0130;
        cyc = arm_two_stream(DATA, n2, &sum);
        rec("2s_dram", 1, n2, cyc, sum == g_2s);
        HB[0] = 0xCAFE0131;
        cyc = arm_two_stream(SPM, n2, &sum);
        rec("2s_spm", 1, n2, cyc, sum == g_2s);

        for (uint32_t i = 0; i < VLM4; i++) IDX[i] = (uint32_t)((i * 37u) & 0xFFu) * 4u;
        uint32_t reps = n2 / VLM4;
        HB[0] = 0xCAFE0132;
        cyc = arm_gather_resident(DATA, reps, &sum);
        rec("1s_dram", 1, n2, cyc, sum == g_1s);
        HB[0] = 0xCAFE0133;
        cyc = arm_gather_resident(SPM, reps, &sum);
        rec("1s_spm", 1, n2, cyc, sum == g_1s);

        HB[0] = 0xCAFE0134;
        cyc = arm_two_stream_vl(DATA, n2, 64u, &sum);
        rec("2s64_dr", 1, n2, cyc, sum == g_2s);
        HB[0] = 0xCAFE0135;
        cyc = arm_two_stream_vl(SPM, n2, 64u, &sum);
        rec("2s64_sp", 1, n2, cyc, sum == g_2s);
        HB[0] = 0xCAFE0136;
        cyc = arm_two_stream_vl(DATA, n2, 16u, &sum);
        rec("2s16_dr", 1, n2, cyc, sum == g_2s);
        HB[0] = 0xCAFE0137;
        cyc = arm_two_stream_vl(SPM, n2, 16u, &sum);
        rec("2s16_sp", 1, n2, cyc, sum == g_2s);
    }

    int all = (g_pass == ((1u << g_r) - 1u));
    RES[GB_MASK_SLOT] = g_pass;
    VER[1] = (uint32_t)all;
    VER[2] = (uint32_t)cyc_512k;
    VER[3] = (uint32_t)(N_STREAM * 1000ull / (g_cyc[4] ? g_cyc[4] : 1));
    VER[4] = (uint32_t)((uint64_t)N_G * 1000 / (g_cyc[5] ? g_cyc[5] : 1));
    VER[5] = (uint32_t)rate_512k;
    VER[6] = (uint32_t)rate_32m;
    VER[7] = (uint32_t)infl_32m;
    VER[0] = 0xA1A1B1B1;

    for (int t = 0; t < PERF_NUM_TAPS; t++) {
        uint32_t s = GB_TAP_SLOT + (uint32_t)t * 12u;
        res_put64(s + 0, perf_rd_count(t));
        res_put64(s + 2, perf_rd_lat_sum(t));
        RES[s + 4] = (uint32_t)perf_rd_outst_max(t);
        res_put64(s + 5, perf_wr_count(t));
        res_put64(s + 7, perf_wr_lat_sum(t));
        RES[s + 9] = (uint32_t)perf_wr_outst_max(t);
    }
    res_put64(GB_VINSN_SLOT, perf_vinsn_active());
    RES[GB_VINSN_SLOT + 2] = (uint32_t)(perf_magic() >> 32);

    HB[0] = 0xCAFE000E;
    uart_puts("\n[gather_bench] VLSU per-element rate vs stream (SPM go/no-go)\n");
    bp_col("record", 8); bp_col("spanKiB", 9); bp_col("elems", 9);
    bp_col("cycles", 11); bp_col("elem/cyc", 10); bp_col("B/cyc", 8);
    bp_col("inferL", 8); bp_puts("check\n");
    for (int k = 0; k < g_r; k++) {
        uint64_t c = g_cyc[k];
        uint64_t rate_m = c ? (uint64_t)g_elems[k] * 1000 / c : 0;
        bp_col(g_tag[k], 8); bp_col_u64(g_span[k], 9); bp_col_u64(g_elems[k], 9);
        bp_col_u64(c, 11); bp_col_frac(rate_m, 3, 10);
        bp_col_frac(c ? (uint64_t)g_elems[k] * 4000 / c : 0, 3, 8);
        if (rate_m && rate_m < 1000) bp_col_frac(c * 40 / g_elems[k], 1, 8);
        else                         bp_col("-", 8);
        bp_puts((g_pass >> k) & 1 ? "PASS\n" : "FAIL\n");
    }
    uart_puts("gather_bench done, pass_mask=");
    print_rhex(g_pass); uart_puts("\n");
    HB[0] = 0xCAFE000F;

    {
        uart_puts("\n---- hw counters (magic ");
        print_rhex((uint32_t)(perf_magic() >> 32));
        uart_puts(") ----\n");
        static const char *TAPN[PERF_NUM_TAPS] = {"spm     ", "dram_ara", "dram_all"};
        for (int t = 0; t < PERF_NUM_TAPS; t++) {
            uart_puts(TAPN[t]);
            uart_puts("  rd n=");      print_u64(perf_rd_count(t));
            uart_puts("  meanL(x1e3)="); print_u64(perf_rd_mean_lat_milli(t));
            uart_puts("  outst_max="); print_u64(perf_rd_outst_max(t));
            uart_puts("  wr n=");      print_u64(perf_wr_count(t));
            uart_puts("\n");
        }
        uart_puts("vinsn_active="); print_u64(perf_vinsn_active()); uart_puts("\n");
    }

    while (1) {}
}
