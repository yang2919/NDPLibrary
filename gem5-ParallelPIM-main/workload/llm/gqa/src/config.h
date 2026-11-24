/*
 * Configuration of the grouped-query attention (GQA) decode-block example.
 *
 * Model: Qwen3-14B, from its published config.json
 * (huggingface.co/Qwen/Qwen3-14B, fetched 2026-11-12): 40 layers, hidden
 * 5120, 40 query heads, 8 KV heads, head 128, SwiGLU 17408, per-head q/k
 * RMSNorm, RoPE theta 1e6, 40,960 positions, vocabulary 151,936, untied
 * embeddings: 14.77 B parameters, 27.5 GiB in bf16, which fits in the
 * 32 GiB OS memory with 4.5 GiB left.
 *
 * -DMODEL_TINY builds a small model of the same structure for quick checks.
 */

#ifndef GQA_CONFIG_H
#define GQA_CONFIG_H

#ifndef MODEL_TINY
#define MODEL_NAME "Qwen3-14B"
#define LAYERS 40
#define HIDDEN 5120
#define Q_HEADS 40
#define KV_HEADS 8
#define HEAD_DIM 128
#define FFN 17408
#define VOCAB 151936
#define MAX_POS 40960
#else
#define MODEL_NAME "tiny-gqa"
#define LAYERS 4
#define HIDDEN 512
#define Q_HEADS 8
#define KV_HEADS 2
#define HEAD_DIM 64
#define FFN 1024
#define VOCAB 1024
#define MAX_POS 40960
#endif

#define RMS_EPS 1e-6f
#define ROPE_THETA 1000000.0f

/* Batch sizes the example targets: 4 and 8, chosen at run time. */
#define BATCH_DEFAULT 4

/*
 * Context lengths: tokens per sequence including the one being decoded
 * (a short and a middle preset, and the maximum).
 *
 * Maximum with every layer's KV cache in the 32 GiB LPDDR6 PIM memory and
 * the library's placement: a row slice is one row index in every bank,
 * 16 channels x 16 banks x 2 KiB = 512 KiB; 65,534
 * slices are usable (two rows per bank hold the PU registers). Per layer,
 * sequence and KV head, K (ctx rows, one per bank row) takes ctx / 256
 * slices and V^T (128 rows) ctx / 1024, so 40 layers x 8 heads need
 * 40 x 8 x batch x 5 ctx / 1024 slices, plus a few for the job buffers:
 *     batch 4: ctx <= 10,240 (64,000 slices)    ideal packing: 40,960
 *     batch 8: ctx <= 5,120  (64,000 slices)    ideal packing: 26,213
 * ("ideal" = 160 KiB of bf16 K and V per token, the model limit 40,960.)
 */
#define CTX_SHORT 1024
#define CTX_MIDDLE 4096
#define CTX_MAX_B4 10240
#define CTX_MAX_B8 5120

/* The simulated layer; the others' weights and KV are allocated only. */
#define SIM_LAYER 0
#define SEED 20261112u

#endif /* GQA_CONFIG_H */
