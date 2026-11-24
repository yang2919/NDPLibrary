/*
 * gqa: one decode step of one Qwen3-14B layer (grouped-query attention) on
 * the RISC-V PIM system, for a batch of sequences.
 *
 *   gqa [batch [ctx [rvv_workers]]]
 *       batch        sequences decoded together (default 4; the example
 *                    targets 4 and 8)
 *       ctx          tokens per sequence including the new one: a number,
 *                    or short (1,024), middle (4,096), max (config.h)
 *       rvv_workers  RVV threads (default 1); the gem5 configuration must
 *                    have rvv_workers + 1 cores
 *
 * Core 0 (the main thread) is the PIM core: it alone calls the PIM
 * library. Cores 1 .. rvv_workers run every other operation with RVV.
 * Weights are bf16 in the OS memory; every layer's weights are allocated
 * and the simulated layer's are written. Every layer's KV cache is
 * allocated in the PIM memory, the simulated layer's past tokens written.
 *
 * Phases: initialization (on gem5's Atomic CPU), hypercall 4 (the
 * configuration switches to the detailed CPUs), then one statistics block
 * per region of the decode step:
 *   attn-norm-qkv  RMSNorm, QKV projection                       (RVV)
 *   qk-norm-rope   per-head q/k RMSNorm, rotary embedding         (RVV)
 *   kv-append      the new k, v into the PIM memory: normal stores and
 *                  cache flushes                                 (PIM core)
 *   attention      scores K q and weighted sum V^T p with pim_gemv on the
 *                  PIM core, softmax on RVV, pipelined over (sequence,
 *                  query head) jobs
 *   o-proj         output projection, residual                   (RVV)
 *   ffn            RMSNorm, gate/up, SiLU, down, residual         (RVV)
 * Then checks: the attention output of two jobs against double precision,
 * and a digest of the layer output per sequence, which the host reference
 * build (bench.py build --host) prints too.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "config.h"
#include "llm.h"

#define QDIM (Q_HEADS * HEAD_DIM)
#define KVDIM (KV_HEADS * HEAD_DIM)
#define QKV_ROWS (QDIM + 2 * KVDIM)
#define K_OFF QDIM
#define V_OFF (QDIM + KVDIM)

typedef struct
{
    bf16_t *wqkv;  /* QKV_ROWS x HIDDEN: W_q, W_k, W_v stacked */
    bf16_t *wo;    /* HIDDEN x QDIM */
    bf16_t *wgu;   /* 2 FFN x HIDDEN: W_gate, W_up stacked */
    bf16_t *wdown; /* HIDDEN x FFN */
    float *attn_norm, *ffn_norm, *q_norm, *k_norm;
} layer_t;

static layer_t layer[LAYERS];
static size_t batch = BATCH_DEFAULT, ctx = CTX_SHORT;
static float *qkv;

static void *
xalloc(size_t bytes)
{
    void *p = malloc(bytes);
    if (!p) {
        fprintf(stderr, "gqa: cannot allocate %zu bytes\n", bytes);
        exit(2);
    }
    return p;
}

/* ---- tasks of the RVV workers ------------------------------------------ */

typedef struct
{
    float *y;
    size_t ldy;
    const bf16_t *w;
    const float *x;
    size_t ldx, k;
} gemv_arg_t;

static void
gemv_task(void *a, size_t lo, size_t hi, int w)
{
    (void)w;
    const gemv_arg_t *g = a;
    llm_gemv_bf16(g->y, g->ldy, g->w, g->x, g->ldx, batch, lo, hi, g->k);
}

/* y[b] = W x[b] for every sequence b; W rows split over the workers. */
static void
gemv(float *y, size_t ldy, const bf16_t *w, const float *x, size_t ldx,
     size_t rows, size_t k)
{
    gemv_arg_t g = {y, ldy, w, x, ldx, k};
    llm_pool_run(gemv_task, &g, rows);
}

typedef struct
{
    float *y;
    const float *x, *g;
    size_t n;
} vec_arg_t;

static void
norm_task(void *a, size_t lo, size_t hi, int w)
{
    (void)w;
    const vec_arg_t *v = a;
    for (size_t b = lo; b < hi; b++)
        llm_rmsnorm(v->y + b * v->n, v->x + b * v->n, v->g, v->n, RMS_EPS);
}

static void
add_task(void *a, size_t lo, size_t hi, int w)
{
    (void)w;
    const vec_arg_t *v = a;
    for (size_t b = lo; b < hi; b++)
        llm_add(v->y + b * v->n, v->x + b * v->n, v->n);
}

static void
silu_task(void *a, size_t lo, size_t hi, int w)
{
    (void)w;
    const vec_arg_t *v = a; /* x: gate and up, 2 FFN per sequence */
    for (size_t b = lo; b < hi; b++)
        llm_silu_mul(v->y + b * FFN, v->x + b * 2 * FFN,
                     v->x + b * 2 * FFN + FFN, FFN);
}

static void
qk_task(void *a, size_t lo, size_t hi, int w)
{
    (void)w;
    const layer_t *L = a;
    for (size_t i = lo; i < hi; i++) {
        const size_t b = i / (Q_HEADS + KV_HEADS);
        const size_t hh = i % (Q_HEADS + KV_HEADS);
        float *v = qkv + b * QKV_ROWS +
                   (hh < Q_HEADS ? hh * HEAD_DIM
                                 : K_OFF + (hh - Q_HEADS) * HEAD_DIM);
        llm_rmsnorm(v, v, hh < Q_HEADS ? L->q_norm : L->k_norm, HEAD_DIM,
                    RMS_EPS);
        llm_rope(v, HEAD_DIM, ctx - 1, ROPE_THETA);
    }
}

typedef struct
{
    bf16_t *dst;
    uint32_t seed;
    float scale;
} fill_arg_t;

static void
fill_task(void *a, size_t lo, size_t hi, int w)
{
    (void)w;
    const fill_arg_t *f = a;
    llm_fill_bf16(f->dst + lo, lo, hi - lo, f->seed, f->scale);
}

static void
fill(bf16_t *dst, size_t n, uint32_t seed, float scale)
{
    fill_arg_t f = {dst, seed, scale};
    llm_pool_run(fill_task, &f, n);
}

#ifdef LLM_DEBUG
static void
dbg(const char *what, const float *v, size_t n)
{
    double sum = 0, sq = 0;
    for (size_t i = 0; i < n; i++) {
        sum += v[i];
        sq += (double)v[i] * v[i];
    }
    printf("debug %-6s sum %.6e sumsq %.6e v0 %.6e\n", what, sum, sq, v[0]);
}
#define DBG(what, v, n) dbg(what, v, n)
#else
#define DBG(what, v, n) ((void)0)
#endif

/* ---- checks ------------------------------------------------------------ */

static float
rb(float x) /* rounded to the PIM operand format, as the PIM memory sees */
{
    uint8_t e[4];
    pim_f32_to_elem(e, x);
    return pim_elem_to_f32(e);
}

/* The attention output of job (b, h) against double precision; returns
 * the elements outside tolerance. */
static double worst; /* largest error / magnitude of the checks */

static size_t
check_attention(const llm_kv_t *kv, size_t b, size_t h, const float *attn)
{
    const size_t d = HEAD_DIM, g = h / (Q_HEADS / KV_HEADS);
    float *k = xalloc(ctx * d * sizeof(float));
    float *vt = xalloc(ctx * d * sizeof(float));
    double *p = xalloc(ctx * sizeof(double));
    llm_kv_past(kv, SIM_LAYER, b, g, SEED, k, vt);
    const float *q = qkv + b * QKV_ROWS + h * d;
    const float *kn = qkv + b * QKV_ROWS + K_OFF + g * d;
    const float *vn = qkv + b * QKV_ROWS + V_OFF + g * d;
    for (size_t j = 0; j < d; j++) {
        k[(ctx - 1) * d + j] = kn[j];
        vt[j * ctx + ctx - 1] = vn[j];
    }
    double m = -INFINITY, sum = 0;
    for (size_t t = 0; t < ctx; t++) {
        double s = 0;
        for (size_t j = 0; j < d; j++)
            s += (double)rb(k[t * d + j]) * rb(q[j]);
        p[t] = s / sqrt((double)d);
        m = p[t] > m ? p[t] : m;
    }
    for (size_t t = 0; t < ctx; t++)
        sum += p[t] = exp(p[t] - m);
    size_t bad = 0;
    for (size_t j = 0; j < d; j++) {
        double o = 0, mag = 0;
        for (size_t t = 0; t < ctx; t++) {
            const double pv =
                rb((float)(p[t] / sum)) * (double)rb(vt[j * ctx + t]);
            o += pv;
            mag += fabs(pv);
        }
        const double err = fabs(attn[b * QDIM + h * d + j] - o) / mag;
        worst = err > worst ? err : worst;
        bad += err > 1.0 / 256;
    }
    free(k);
    free(vt);
    free(p);
    return bad;
}

/* ---- main --------------------------------------------------------------- */

int
main(int argc, char **argv)
{
    int workers = 1;
    if (argc > 1)
        batch = mb_arg(argv[1]);
    if (argc > 2) {
        if (!strcmp(argv[2], "short"))
            ctx = CTX_SHORT;
        else if (!strcmp(argv[2], "middle"))
            ctx = CTX_MIDDLE;
        else if (!strcmp(argv[2], "max"))
            ctx = batch <= 4 ? CTX_MAX_B4 : CTX_MAX_B8;
        else
            ctx = mb_arg(argv[2]);
    }
    if (argc > 3)
        workers = (int)mb_arg(argv[3]);
    if (argc > 4 || ctx < 2 || ctx > MAX_POS) {
        fprintf(stderr, "usage: gqa [batch [ctx|short|middle|max "
                        "[rvv_workers]]], 2 <= ctx <= %d\n", MAX_POS);
        return 2;
    }
    printf("gqa model %s layers %d hidden %d heads %d/%d head %d ffn %d\n",
           MODEL_NAME, LAYERS, HIDDEN, Q_HEADS, KV_HEADS, HEAD_DIM, FFN);
    printf("gqa batch %zu ctx %zu rvv-workers %d simulated-layer %d\n",
           batch, ctx, workers, SIM_LAYER);

    /* Every layer's weights in the OS memory; only SIM_LAYER's are
     * written, so the others cost no simulation time. */
    size_t bytes = 2 * (size_t)VOCAB * HIDDEN * sizeof(bf16_t);
    bf16_t *embed = xalloc((size_t)VOCAB * HIDDEN * sizeof(bf16_t));
    bf16_t *lm_head = xalloc((size_t)VOCAB * HIDDEN * sizeof(bf16_t));
    for (int l = 0; l < LAYERS; l++) {
        layer_t *L = &layer[l];
        L->wqkv = xalloc((size_t)QKV_ROWS * HIDDEN * sizeof(bf16_t));
        L->wo = xalloc((size_t)HIDDEN * QDIM * sizeof(bf16_t));
        L->wgu = xalloc((size_t)2 * FFN * HIDDEN * sizeof(bf16_t));
        L->wdown = xalloc((size_t)HIDDEN * FFN * sizeof(bf16_t));
        L->attn_norm = xalloc(HIDDEN * sizeof(float));
        L->ffn_norm = xalloc(HIDDEN * sizeof(float));
        L->q_norm = xalloc(HEAD_DIM * sizeof(float));
        L->k_norm = xalloc(HEAD_DIM * sizeof(float));
        bytes += ((size_t)QKV_ROWS * HIDDEN + (size_t)HIDDEN * QDIM +
                  3 * (size_t)FFN * HIDDEN) * sizeof(bf16_t);
    }
    printf("gqa weights allocated: %.2f GiB (all layers)\n",
           bytes / 1073741824.0);
    (void)embed;
    (void)lm_head;

    llm_pool_start(workers);
    layer_t *L = &layer[SIM_LAYER];
    fill(L->wqkv, (size_t)QKV_ROWS * HIDDEN, SEED + 1, sqrtf(3.0f / HIDDEN));
    fill(L->wo, (size_t)HIDDEN * QDIM, SEED + 2, sqrtf(3.0f / QDIM));
    fill(L->wgu, (size_t)2 * FFN * HIDDEN, SEED + 3, sqrtf(3.0f / HIDDEN));
    fill(L->wdown, (size_t)HIDDEN * FFN, SEED + 4, sqrtf(3.0f / FFN));
    llm_fill_f32(L->attn_norm, 0, HIDDEN, SEED + 5, 0.1f, 1.0f);
    llm_fill_f32(L->ffn_norm, 0, HIDDEN, SEED + 6, 0.1f, 1.0f);
    llm_fill_f32(L->q_norm, 0, HEAD_DIM, SEED + 7, 0.1f, 1.0f);
    llm_fill_f32(L->k_norm, 0, HEAD_DIM, SEED + 8, 0.1f, 1.0f);

    mb_check(pim_init(NULL), "pim_init");
    llm_kv_t kv;
    pim_status_t s = llm_kv_alloc(&kv, LAYERS, batch, KV_HEADS, Q_HEADS,
                                  HEAD_DIM, ctx);
    if (s == PIM_E_NOMEM) {
        fprintf(stderr, "gqa: the KV cache of %d layers, batch %zu, ctx %zu "
                        "does not fit in the PIM memory (%u slices free)\n",
                LAYERS, batch, ctx, pim_free_slices());
        return 2;
    }
    mb_check(s, "llm_kv_alloc");
    printf("gqa KV cache allocated, %u row slices left\n", pim_free_slices());
    mb_check(llm_kv_fill_layer(&kv, SIM_LAYER, SEED), "llm_kv_fill_layer");

    float *x = xalloc(batch * HIDDEN * sizeof(float));
    float *h = xalloc(batch * HIDDEN * sizeof(float));
    float *attn = xalloc(batch * QDIM * sizeof(float));
    float *tmp = xalloc(batch * HIDDEN * sizeof(float));
    float *gu = xalloc(batch * 2 * FFN * sizeof(float));
    float *act = xalloc(batch * FFN * sizeof(float));
    qkv = xalloc(batch * QKV_ROWS * sizeof(float));
    llm_fill_f32(x, 0, batch * HIDDEN, SEED + 9, 1.0f, 0.0f);
    DBG("x", x, batch * HIDDEN);
    DBG("wqkv0", (const float *)L->wqkv, 64);
    printf("gqa initialized\n");
    fflush(stdout);

    mb_hypercall(4); /* gem5: switch to the detailed CPUs, reset stats */

    MB_REGION("attn-norm-qkv")
    {
        vec_arg_t n = {h, x, L->attn_norm, HIDDEN};
        llm_pool_run(norm_task, &n, batch);
        gemv(qkv, QKV_ROWS, L->wqkv, h, HIDDEN, QKV_ROWS, HIDDEN);
    }
    DBG("h", h, batch * HIDDEN); DBG("qkv", qkv, batch * QKV_ROWS);
    MB_REGION("qk-norm-rope")
    {
        llm_pool_run(qk_task, L, batch * (Q_HEADS + KV_HEADS));
    }
    DBG("qkvn", qkv, batch * QKV_ROWS);
    MB_REGION("kv-append")
    {
        for (size_t b = 0; b < batch; b++)
            for (size_t g = 0; g < KV_HEADS; g++)
                llm_kv_append(&kv, SIM_LAYER, b, g,
                              qkv + b * QKV_ROWS + K_OFF + g * HEAD_DIM,
                              qkv + b * QKV_ROWS + V_OFF + g * HEAD_DIM);
        llm_kv_sync();
    }
    MB_REGION("attention")
    {
        mb_check(llm_attention(&kv, SIM_LAYER, qkv, QKV_ROWS, attn, QDIM),
                 "llm_attention");
    }
    DBG("attn", attn, batch * QDIM);
#ifdef LLM_DEBUG
    for (size_t j = 0; j < batch * Q_HEADS; j++)
        printf("debug job %zu attn %.6e\n", j, attn[j * HEAD_DIM]);
#endif
    MB_REGION("o-proj")
    {
        gemv(tmp, HIDDEN, L->wo, attn, QDIM, HIDDEN, QDIM);
        vec_arg_t a = {x, tmp, NULL, HIDDEN};
        llm_pool_run(add_task, &a, batch);
    }
    DBG("x-o", x, batch * HIDDEN);
    MB_REGION("ffn")
    {
        vec_arg_t n = {h, x, L->ffn_norm, HIDDEN};
        llm_pool_run(norm_task, &n, batch);
        gemv(gu, 2 * FFN, L->wgu, h, HIDDEN, 2 * FFN, HIDDEN);
        vec_arg_t sg = {act, gu, NULL, FFN};
        llm_pool_run(silu_task, &sg, batch);
        gemv(tmp, HIDDEN, L->wdown, act, FFN, HIDDEN, FFN);
        vec_arg_t a = {x, tmp, NULL, HIDDEN};
        llm_pool_run(add_task, &a, batch);
    }
    llm_pool_stop();

    size_t bad = check_attention(&kv, 0, 0, attn);
    bad += check_attention(&kv, batch - 1, Q_HEADS - 1, attn);
    printf("attention max error / magnitude: %.3e\n", worst);
    printf("attention outside bf16 tolerance: %zu\n", bad);
    for (size_t b = 0; b < batch; b++) {
        double sum = 0, sq = 0;
        for (size_t i = 0; i < HIDDEN; i++) {
            sum += x[b * HIDDEN + i];
            sq += (double)x[b * HIDDEN + i] * x[b * HIDDEN + i];
        }
        printf("digest %zu sum %.6e sumsq %.6e x0 %.6e x1 %.6e attn0 %.6e\n",
               b, sum, sq, x[b * HIDDEN], x[b * HIDDEN + 1], attn[b * QDIM]);
    }
    pim_finalize();
    return bad ? 1 : 0;
}
