# Workloads for the RISC-V Processing-in-Memory (PIM) system

Programs that run on the PIM-enabled gem5 + Ramulator 2.1 system in
system-call emulation (SE) mode, with their configurations and scripts. They need `build/RISCV_RAMULATOR2/gem5.opt` and the PIM library
(`libpim`), which every `build.sh` builds first.

| Directory | What |
|---|---|
| `microbench/gemv/` | y = A x: PIM against a scalar and a RISC-V Vector (RVV) CPU loop |
| `microbench/eltwise/` | element-wise add and multiply: PIM against CPU loops |
| `llm/gqa/` | one decode step of one Qwen3-14B layer (grouped-query attention, GQA), attention scores and weighted sums on PIM, everything else on RVV |
| `common/` | `bench.h` (measured regions, data, checks), `bench.py` (build, run, collect) |

Every example directory has the same layout:

```
src/        the program
configs/    bench.json (what to build, which gem5 configuration), runs.csv
scripts/    build.sh, run.sh, collect.sh (+ check.sh for llm/)
results/    written by the scripts: runs/<name>/ (cmd.txt, program.out,
            stats.txt, gem5.log), summary.csv, summary.md (+ check.md
            for llm/)
bin/        built programs, per memory layout
```

From the repository root, after `scons build/RISCV_RAMULATOR2/gem5.opt`:

```sh
workload/microbench/gemv/scripts/build.sh
workload/microbench/gemv/scripts/run.sh -j 8
workload/microbench/gemv/scripts/collect.sh
```

Each program brackets its measured regions with gem5's `m5_reset_stats`
and `m5_dump_stats`, so `stats.txt` has one block per region, and checks
its own results; `collect.sh` turns the blocks into `summary.md`.

`results/` and `bin/` are not versioned (`workload/.gitignore`).
