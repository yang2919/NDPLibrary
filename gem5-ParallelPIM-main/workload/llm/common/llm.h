/*
 * Shared code of the LLM decode-block examples (workload/llm).
 *
 * Kernels run on the RISC-V Vector extension (RVV) when compiled for it,
 * and as plain C otherwise (the host reference build). Weights are bf16
 * (bfloat16) in the OS memory; activations are fp32.
 */

#ifndef LLM_H
#define LLM_H

#include <stddef.h>
#include <stdint.h>

#include <gem5/pim/pim.h>

typedef uint16_t bf16_t;

/* ---- synthetic data (identical on RVV and on the host) ---------------- */

/* dst[i] = bf16 of a hash of (seed, first + i), uniform in [-scale,
 * scale), for i < n; a tensor filled in chunks equals one filled whole. */
void llm_fill_bf16(bf16_t *dst, size_t first, size_t n, uint32_t seed,
                   float scale);
/* dst[i] = offset + a hash of (seed, first + i) in [-scale, scale). */
void llm_fill_f32(float *dst, size_t first, size_t n, uint32_t seed,
                  float scale, float offset);

/* ---- kernels ----------------------------------------------------------- */

/*
 * y[b * ldy + r] = sum_j w[r * k + j] * x[b * ldx + j] for rows [r0, r1)
 * and batch elements b < nb. RVV: register tiles of 2 rows x up to 8
 * batch elements, so each weight chunk is loaded and widened once per 8
 * batch elements and each x chunk serves 2 rows; k is tiled so that the
 * batch's x tiles stay in the L1 cache. Every weight is read once from
 * memory, so the flops per weight byte grow with nb.
 */
void llm_gemv_bf16(float *y, size_t ldy, const bf16_t *w, const float *x,
                   size_t ldx, size_t nb, size_t r0, size_t r1, size_t k);
/* y = x / rms(x) * g, n elements. */
void llm_rmsnorm(float *y, const float *x, const float *g, size_t n,
                 float eps);
/* Rotary position embedding of one head (rotate-half form), in place. */
void llm_rope(float *v, size_t head_dim, size_t pos, float theta);
/* p = softmax(scale * s), n elements. */
void llm_softmax(float *p, const float *s, size_t n, float scale);
/* y = silu(g) * u. */
void llm_silu_mul(float *y, const float *g, const float *u, size_t n);
/* y += x. */
void llm_add(float *y, const float *x, size_t n);
/* y += a x. */
void llm_axpy(float *y, float a, const float *x, size_t n);
/* idx[0 .. k) = the indices of the k largest of v[0 .. n), largest first;
 * of equal values the lower index first. */
void llm_topk(const float *v, size_t n, size_t k, int *idx);
/* x rounded to the PIM operand format (PIM_LAYOUT_NUM_FMT), as the PIM
 * memory sees a value written to it. */
float llm_round_pim(float x);

/* ---- worker pool: the RVV cores ---------------------------------------- */

/*
 * Threads land on cores in creation order in gem5's SE mode (no
 * sched_setaffinity): the main thread on core 0 (the PIM core), worker w on
 * core w + 1. Workers spin while idle.
 */
typedef void (*llm_task_fn)(void *arg, size_t lo, size_t hi, int worker);

void llm_pool_start(int nworkers);
int llm_pool_size(void);
/* Split [0, n) into one contiguous range per worker; returns at once. */
void llm_pool_launch(llm_task_fn fn, void *arg, size_t n);
/* Wait for the launched task. */
void llm_pool_wait(void);
/* launch + wait. */
void llm_pool_run(llm_task_fn fn, void *arg, size_t n);
void llm_pool_stop(void);

/* ---- KV cache in the PIM memory and the attention core ---------------- */

typedef struct llm_kv
{
    size_t layers, batch, kv_heads, q_heads, head_dim;
    size_t ctx;              /* tokens per sequence, the new one included */
    pim_tensor_t **k, **vt;  /* [layer][batch][kv head]: K ctx x d, V^T */
    pim_tensor_t *xq[2], *s[2], *xp[2], *o[2]; /* two job slots */
    float *scores[2], *probs[2];
} llm_kv_t;

/* Allocate every layer's K and V^T and the job slots. */
pim_status_t llm_kv_alloc(llm_kv_t *kv, size_t layers, size_t batch,
                          size_t kv_heads, size_t q_heads, size_t head_dim,
                          size_t ctx);
/* Synthetic keys and values of the past tokens 0 .. ctx - 2 of one layer;
 * the new token's row and column stay zero. */
pim_status_t llm_kv_fill_layer(llm_kv_t *kv, size_t layer, uint32_t seed);
/* The host values llm_kv_fill_layer placed for (layer, b, g):
 * k: ctx x d, vt: d x ctx (fp32, before rounding to the PIM format). */
void llm_kv_past(const llm_kv_t *kv, size_t layer, size_t b, size_t g,
                 uint32_t seed, float *k, float *vt);
/* Write the new token's k (row ctx - 1 of K) and v (column ctx - 1 of
 * V^T) with normal stores and flush them; call llm_kv_sync() after the
 * last one. */
void llm_kv_append(llm_kv_t *kv, size_t layer, size_t b, size_t g,
                   const float *k, const float *v);
void llm_kv_sync(void);
/*
 * out[b][h * d ...] = softmax(q K^T / sqrt(d)) V for every batch element
 * and query head of one layer (grouped-query: query head h uses KV head
 * h / (q_heads / kv_heads)). The calling thread (the PIM core) issues the
 * PIM work, scores and weighted sums, through pim_gemv; the pool's workers
 * compute the softmax, one job ahead.
 */
pim_status_t llm_attention(llm_kv_t *kv, size_t layer, const float *q,
                           size_t ldq, float *out, size_t ldo);
/*
 * Check of one attention job (sequence b, query head h of `layer`)
 * against double precision, with the operands rounded as the PIM memory
 * holds them: q the head's query, knew and vnew the new token's key and
 * value of its KV head (seed as given to llm_kv_fill_layer), out the
 * head's output. Returns the elements whose error exceeds 1/256 of the
 * sum of the magnitudes of their terms; *worst keeps the largest ratio.
 */
size_t llm_check_attention(const llm_kv_t *kv, size_t layer, uint32_t seed,
                           size_t b, size_t h, const float *q,
                           const float *knew, const float *vnew,
                           const float *out, double *worst);

#endif /* LLM_H */
