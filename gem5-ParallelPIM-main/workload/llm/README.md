# LLM decode-block examples

One decode step of one layer of a large language model (LLM), for a batch
of sequences, on the RISC-V Processing-in-Memory (PIM) system.

| Example | Model |
|---|---|
| `gqa/` | Qwen3-14B, grouped-query attention (GQA): a full decode block |

## What one decode step does (`gqa/src/gqa.c`)

Core 0 runs the main thread and is the only core that issues PIM
instructions; cores 1 .. N run RISC-V Vector (RVV) worker threads.

| Region | Work | Where |
|---|---|---|
| `attn-norm-qkv` | RMSNorm, QKV projection from bf16 weights in the OS memory | RVV |
| `qk-norm-rope` | per-head q/k RMSNorm, rotary position embedding | RVV |
| `kv-append` | the new token's k (a K row) and v (a Vᵀ column) into the PIM memory: normal stores, then cache-line flushes | PIM core |
| `attention` | per (sequence, query head): scores = K q (`pim_gemv`), softmax (RVV), output = Vᵀ p (`pim_gemv`); the PIM core works one head ahead of the softmax | PIM core + RVV |
| `o-proj` | output projection, residual | RVV |
| `ffn` | RMSNorm, gate and up projections, SiLU, down projection, residual | RVV |

Before the decode step, on gem5's Atomic CPU: every layer's weights are
allocated (27.5 GiB of the 32 GiB OS memory) and the simulated layer's are
written with deterministic synthetic values; every layer's KV cache is
allocated in the 32 GiB PIM memory and the simulated layer's past tokens
are written. Hypercall 4 then switches all cores to the detailed CPU
(`configs/llm_se.py`).

## Sizes (`gqa/src/config.h`)

Qwen3-14B (published `config.json`): 40 layers, hidden 5120, 40 query and
8 KV heads of 128, feed-forward 17408, 40,960 positions; 27.5 GiB of bf16
weights. Context = tokens per sequence including the new one:

| Preset | Batch 4 | Batch 8 |
|---|---|---|
| short | 1,024 | 1,024 |
| middle | 4,096 | 4,096 |
| max (all 40 layers' KV in the PIM memory, library placement) | 10,240 | 5,120 |

With an ideally packed KV cache the maximum would be 40,960 (the model's
limit) and 26,213; the library stores one 128-element key per 2 KiB bank
row, which costs a factor 5; a packed layout is future work.

## Use

```sh
workload/llm/gqa/scripts/build.sh --host   # gem5 program + host reference
workload/llm/gqa/scripts/run.sh -j 6       # configs/runs.csv
workload/llm/gqa/scripts/collect.sh        # results/summary.md
workload/llm/gqa/scripts/check.sh          # results/check.md
```

Checks: the program compares the attention output of two (sequence, head)
jobs with double precision (error / magnitude below 1/256); `check.sh`
compares every run's layer output with the host reference build of the
same program (the PIM library's C model of the PIM memory and plain C
kernels).

A tiny model of the same structure (`-DMODEL_TINY`) runs in minutes for
quick tests.
