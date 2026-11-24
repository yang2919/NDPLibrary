# Qwen3 single-layer decode timing

Run on the existing Mac from the repository root with Docker running. Each
command starts one SE Timing simulation and reports its own result directory.
Start with MoE, whose eight active FFNs are smaller than the dense FFN.

```sh
STAGE=QWEN MODEL=moe CONTEXT=16 ./tools/gem5/pim_kernel/run.sh
STAGE=QWEN MODEL=dense CONTEXT=16 ./tools/gem5/pim_kernel/run.sh
```

CONTEXT counts KV entries **including the current token** (1 through 4096).
Batch size is one. The input to the layer is one synthetic hidden vector.
Prior K/V entries are initialized before measurement. No dataset, checkpoint,
tokenizer, Python client, or FS boot is required. ROI_TIMEOUT is an optional
wall-clock limit in seconds (default 0, unlimited).

| Official configuration | Hidden | Q / KV heads | Head dim | FFN |
|---|---:|---:|---:|---|
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json) | 5120 | 64 / 8 | 128 | 25600 |
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json) | 2048 | 32 / 4 | 128 | 128 experts of width 768, top 8 |

The benchmark uses the project's LPDDR6 16-channel FP16/FP32 accumulation
profile. It calls libpim directly inside the simulated guest. Projection
matrices occupy separate allocations in fresh zero-initialized PIM memory.
MoE reserves all experts, computes router logits, selects the eight highest
with index order as the tie breaker, and combines their outputs with normalized
weights. With zero matrices these are experts 0 through 7. This experiment
measures a fixed one-token access pattern. The simulated RISC-V CPU performs
normalization (unit scale), RoPE, attention, SiLU, routing and residual math in
float. Intermediate vectors are converted on libpim transfer. It measures this
kernel implementation, with no external PyTorch dispatch or separate PNM core.

## Read the result

Open roi.json in the printed run directory. phases.attention covers input RMSNorm,
Q/K/V projections, per-head Q/K RMSNorm, RoPE, KV append, grouped causal
attention, output projection and the first residual. phases.ffn covers the
second RMSNorm, dense SwiGLU or router and eight SwiGLUs, and final residual.
There are 7 projection GEMVs for dense and 29 for MoE (4 attention, 1 router,
24 expert). These launch counts differ from controller command counts.

Each phase reports simTicks, simFreq, simInsts, cpuCycles, and accepted PIM
requests by opcode. layer_ticks sums the two intervals. Divide by simFreq for
simulated seconds. For cache misses, transferred bytes and controller counters,
use the corresponding first and second dumps in sim/stats.txt. The third dump
includes teardown and is excluded. Initialization and allocations of weights
and KV cache precede the reset; temporary vector operations and conversions
inside the layer are timed. The printed checksum keeps the result observable.
No numerical acceptance threshold is applied. Compare runs with identical
CONTEXT and configuration; a single layer does not include embedding, sampling,
host dispatch or inter-layer scheduling.

## Measured results at context 16

Both user SE Timing runs completed with exit code 0. The clock for the statistics
was 1 trillion ticks per simulated second. Layer time is the sum of the two ROI
intervals and excludes startup and teardown.

| Model | Run directory | Attention (ms) | FFN including routing for MoE (ms) | Layer ticks | Total PIM requests |
|---|---|---:|---:|---:|---:|
| Qwen3-32B | `run-QWEN-f1K5O4Il` | 24.701 | 82.202 | 106,903,400,589 | 3,952,640 |
| Qwen3-30B-A3B | `run-QWEN-K5DvesQX` | 7.322 | 20.306 | 27,628,120,224 | 518,528 |

The dense/MoE layer time ratio was 3.87. FFN accounted for 76.9% of dense
time and 73.5% of MoE time. Both hidden width and FFN geometry differ between
the models. These figures describe the synthetic single-token benchmark above.
