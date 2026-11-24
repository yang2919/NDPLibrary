/*
 * KV cache in the PIM memory and the attention core (llm.h).
 *
 * Placement is the PIM library's: K of one (layer, sequence, KV head) is
 * a PIM_MATRIX_AB tensor of ctx rows x head_dim columns (one token per
 * bank row), V^T one of head_dim rows x ctx columns, so that both
 * products are pim_gemv calls:
 *     scores = K q          (ctx results)
 *     out    = V^T p        (head_dim results)
 */

#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

#include "llm.h"

static size_t
idx(const llm_kv_t *kv, size_t l, size_t b, size_t g)
{
    return (l * kv->batch + b) * kv->kv_heads + g;
}

static uint32_t
kv_seed(uint32_t seed, size_t l, size_t b, size_t g, uint32_t which)
{
    return seed ^ (uint32_t)(((l * 131 + b) * 131 + g) * 2 + which) *
                      0x27d4eb2du;
}

pim_status_t
llm_kv_alloc(llm_kv_t *kv, size_t layers, size_t batch, size_t kv_heads,
             size_t q_heads, size_t head_dim, size_t ctx)
{
    const pim_dtype_t nf = (pim_dtype_t)PIM_LAYOUT_NUM_FMT;
    const pim_dtype_t af = (pim_dtype_t)PIM_LAYOUT_ACC_FMT;
    kv->layers = layers;
    kv->batch = batch;
    kv->kv_heads = kv_heads;
    kv->q_heads = q_heads;
    kv->head_dim = head_dim;
    kv->ctx = ctx;
    const size_t n = layers * batch * kv_heads;
    kv->k = calloc(n, sizeof(*kv->k));
    kv->vt = calloc(n, sizeof(*kv->vt));
    if (!kv->k || !kv->vt)
        return PIM_E_NOMEM;
    for (int sl = 0; sl < 2; sl++) {
        pim_status_t s;
        if ((s = pim_tensor_create(&kv->xq[sl], PIM_VECTOR_GB, 1, head_dim,
                                   nf)) != PIM_OK ||
            (s = pim_tensor_create(&kv->s[sl], PIM_VECTOR_SB, 1, ctx, af)) !=
                PIM_OK ||
            (s = pim_tensor_create(&kv->xp[sl], PIM_VECTOR_GB, 1, ctx, nf)) !=
                PIM_OK ||
            (s = pim_tensor_create(&kv->o[sl], PIM_VECTOR_SB, 1, head_dim,
                                   af)) != PIM_OK)
            return s;
        kv->scores[sl] = malloc(ctx * sizeof(float));
        kv->probs[sl] = malloc(ctx * sizeof(float));
        if (!kv->scores[sl] || !kv->probs[sl])
            return PIM_E_NOMEM;
    }
    for (size_t i = 0; i < n; i++) {
        pim_status_t s;
        if ((s = pim_tensor_create(&kv->k[i], PIM_MATRIX_AB, ctx, head_dim,
                                   nf)) != PIM_OK ||
            (s = pim_tensor_create(&kv->vt[i], PIM_MATRIX_AB, head_dim, ctx,
                                   nf)) != PIM_OK)
            return s;
    }
    return PIM_OK;
}

void
llm_kv_past(const llm_kv_t *kv, size_t layer, size_t b, size_t g,
            uint32_t seed, float *k, float *vt)
{
    const size_t d = kv->head_dim, ctx = kv->ctx;
    llm_fill_f32(k, 0, ctx * d, kv_seed(seed, layer, b, g, 0), 1.0f, 0.0f);
    llm_fill_f32(vt, 0, d * ctx, kv_seed(seed, layer, b, g, 1), 1.0f, 0.0f);
    for (size_t j = 0; j < d; j++) {
        k[(ctx - 1) * d + j] = 0.0f; /* the new token, appended later */
        vt[j * ctx + ctx - 1] = 0.0f;
    }
}

pim_status_t
llm_kv_fill_layer(llm_kv_t *kv, size_t layer, uint32_t seed)
{
    const size_t d = kv->head_dim, ctx = kv->ctx;
    float *k = malloc(ctx * d * sizeof(float));
    float *vt = malloc(ctx * d * sizeof(float));
    if (!k || !vt)
        return PIM_E_NOMEM;
    for (size_t b = 0; b < kv->batch; b++) {
        for (size_t g = 0; g < kv->kv_heads; g++) {
            llm_kv_past(kv, layer, b, g, seed, k, vt);
            pim_status_t s;
            if ((s = pim_copy_to(kv->k[idx(kv, layer, b, g)], k)) != PIM_OK ||
                (s = pim_copy_to(kv->vt[idx(kv, layer, b, g)], vt)) != PIM_OK)
                return s;
        }
    }
    free(k);
    free(vt);
    return PIM_OK;
}

/* One element with a normal store; returns its cache line. */
static uintptr_t
put(const pim_tensor_t *t, size_t r, size_t c, float v)
{
    pim_coord_t co;
    uint32_t byte;
    if (pim_tensor_locate(t, r, c, &co, &byte) != PIM_OK) {
        fprintf(stderr, "llm: locate (%zu, %zu) failed\n", r, c);
        exit(2);
    }
    uint8_t *p = (uint8_t *)pim_region_base() + pim_encode(&co) + byte;
    pim_f32_to_elem(p, v);
    return (uintptr_t)p & ~(uintptr_t)(PIM_CACHE_LINE - 1);
}

/* cbo.flush of every distinct line, after all the stores to it: a line
 * flushed before its last store would leave the rest dirty in the cache,
 * and the PIM command would read stale DRAM. */
static void
flush_lines(uintptr_t *lines, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        size_t j = 0;
        while (j < i && lines[j] != lines[i])
            j++;
        if (j == i)
            pim_cbo_flush((const void *)lines[i]);
    }
}

void
llm_kv_append(llm_kv_t *kv, size_t layer, size_t b, size_t g,
              const float *k, const float *v)
{
    const size_t d = kv->head_dim, t = kv->ctx - 1;
    const pim_tensor_t *kt = kv->k[idx(kv, layer, b, g)];
    const pim_tensor_t *vt = kv->vt[idx(kv, layer, b, g)];
    uintptr_t lines[512];
    if (d > 256) {
        fprintf(stderr, "llm: head_dim %zu > 256\n", d);
        exit(2);
    }
    for (size_t j = 0; j < d; j++)
        lines[j] = put(kt, t, j, k[j]); /* one bank row: a few lines */
    for (size_t j = 0; j < d; j++)
        lines[d + j] = put(vt, j, t, v[j]); /* one element per matrix row */
    flush_lines(lines, 2 * d);
}

void
llm_kv_sync(void)
{
    pim_fence();
}

size_t
llm_check_attention(const llm_kv_t *kv, size_t layer, uint32_t seed,
                    size_t b, size_t h, const float *q, const float *knew,
                    const float *vnew, const float *out, double *worst)
{
    const size_t d = kv->head_dim, ctx = kv->ctx;
    const size_t g = h / (kv->q_heads / kv->kv_heads);
    float *k = malloc(ctx * d * sizeof(float));
    float *vt = malloc(ctx * d * sizeof(float));
    double *p = malloc(ctx * sizeof(double));
    if (!k || !vt || !p) {
        fprintf(stderr, "llm: no memory for the attention check\n");
        exit(2);
    }
    llm_kv_past(kv, layer, b, g, seed, k, vt);
    for (size_t j = 0; j < d; j++) {
        k[(ctx - 1) * d + j] = knew[j];
        vt[j * ctx + ctx - 1] = vnew[j];
    }
    double m = -INFINITY, sum = 0;
    for (size_t t = 0; t < ctx; t++) {
        double s = 0;
        for (size_t j = 0; j < d; j++)
            s += (double)llm_round_pim(k[t * d + j]) * llm_round_pim(q[j]);
        p[t] = s / sqrt((double)d);
        m = p[t] > m ? p[t] : m;
    }
    for (size_t t = 0; t < ctx; t++)
        sum += p[t] = exp(p[t] - m);
    size_t bad = 0;
    for (size_t j = 0; j < d; j++) {
        double o = 0, mag = 0;
        for (size_t t = 0; t < ctx; t++) {
            const double pv = llm_round_pim((float)(p[t] / sum)) *
                              (double)llm_round_pim(vt[j * ctx + t]);
            o += pv;
            mag += fabs(pv);
        }
        const double err = fabs(out[j] - o) / mag;
        *worst = err > *worst ? err : *worst;
        bad += err > 1.0 / 256;
    }
    free(k);
    free(vt);
    free(p);
    return bad;
}

/* ---- attention: PIM on the calling thread, softmax on the workers ---- */

static struct {
    llm_kv_t *kv;
    _Atomic int *ready_s, *ready_p;
    float scale;
} att;

static void
softmax_task(void *arg, size_t lo, size_t hi, int worker)
{
    (void)arg;
    (void)worker;
    for (size_t j = lo; j < hi; j++) {
        while (!atomic_load_explicit(&att.ready_s[j], memory_order_acquire))
            ;
        const int sl = (int)(j & 1);
        llm_softmax(att.kv->probs[sl], att.kv->scores[sl], att.kv->ctx,
                    att.scale);
        atomic_store_explicit(&att.ready_p[j], 1, memory_order_release);
    }
}

static pim_status_t
weighted(llm_kv_t *kv, size_t layer, size_t j, float *out, size_t ldo)
{
    const size_t hq = kv->q_heads, d = kv->head_dim;
    const size_t b = j / hq, h = j % hq, g = h / (hq / kv->kv_heads);
    const int sl = (int)(j & 1);
    while (!atomic_load_explicit(&att.ready_p[j], memory_order_acquire))
        ;
    pim_status_t s;
    if ((s = pim_copy_to(kv->xp[sl], kv->probs[sl])) != PIM_OK ||
        (s = pim_gemv(kv->o[sl], kv->vt[idx(kv, layer, b, g)],
                      kv->xp[sl])) != PIM_OK ||
        (s = pim_copy_from(kv->o[sl], out + b * ldo + h * d)) != PIM_OK)
        return s;
    return PIM_OK;
}

pim_status_t
llm_attention(llm_kv_t *kv, size_t layer, const float *q, size_t ldq,
              float *out, size_t ldo)
{
    const size_t hq = kv->q_heads, d = kv->head_dim;
    const size_t jobs = kv->batch * hq;
    att.kv = kv;
    att.scale = 1.0f / sqrtf((float)d);
    att.ready_s = calloc(jobs, sizeof(*att.ready_s));
    att.ready_p = calloc(jobs, sizeof(*att.ready_p));
    if (!att.ready_s || !att.ready_p)
        return PIM_E_NOMEM;
    llm_pool_launch(softmax_task, NULL, jobs);
    pim_status_t s = PIM_OK;
    for (size_t j = 0; j < jobs && s == PIM_OK; j++) {
        const size_t b = j / hq, h = j % hq, g = h / (hq / kv->kv_heads);
        const int sl = (int)(j & 1);
        if ((s = pim_copy_to(kv->xq[sl], q + b * ldq + h * d)) != PIM_OK ||
            (s = pim_gemv(kv->s[sl], kv->k[idx(kv, layer, b, g)],
                          kv->xq[sl])) != PIM_OK ||
            (s = pim_copy_from(kv->s[sl], kv->scores[sl])) != PIM_OK)
            break;
#ifdef LLM_DEBUG
        {
            double ss = 0;
            for (size_t t = 0; t < kv->ctx; t++)
                ss += kv->scores[sl][t];
            printf("debug job %zu scores sum %.6e s0 %.6e slast %.6e\n", j,
                   ss, kv->scores[sl][0], kv->scores[sl][kv->ctx - 1]);
        }
#endif
        atomic_store_explicit(&att.ready_s[j], 1, memory_order_release);
        if (j > 0)
            s = weighted(kv, layer, j - 1, out, ldo);
    }
    if (s == PIM_OK)
        s = weighted(kv, layer, jobs - 1, out, ldo);
    llm_pool_wait();
    free((void *)att.ready_s);
    free((void *)att.ready_p);
    return s;
}
