# PIM microbenchmarks

Small programs that time one Processing-in-Memory (PIM) kernel against the
same work on the host CPU, on the RISC-V gem5 + Ramulator 2.1 PIM system
(`configs/example/gem5_library/riscv-pim.py`, system-call emulation
(SE) mode).

| Benchmark | Kernel | Regions (one statistics block each) |
|---|---|---|
| `gemv/` | y = A x, A of M × N (default 2048 × 1024) | `pim` (`pim_gemv`), `cpu-scalar`, `cpu-rvv` |
| `eltwise/` | c = a + b, c = a × b on n elements (default 262,144) | `cpu-scalar-add`, `cpu-rvv-add`, `pim-add`, `pim-mul` |

PIM operands are bfloat16 (bf16) with single-precision (fp32)
accumulators; the CPU loops are fp32, a scalar loop and a RISC-V Vector
(RVV) loop. Each region sits between `m5_reset_stats` and `m5_dump_stats`
(`workload/common/bench.h`), so data placement and read-back are not
timed. Every program checks all results against double precision and exits with 1 if
one is wrong.

## Layout

```
../common/bench.h    regions (m5 statistics markers), random data, checks
../common/bench.py   build, run and collect for any example directory
<bench>/src/         the program
<bench>/configs/     bench.json (program, speedups), runs.csv (the runs)
<bench>/scripts/     build.sh, run.sh, collect.sh (wrappers of bench.py)
<bench>/results/     runs/<name>/ per run, summary.csv, summary.md (made
                     by the scripts, not versioned)
<bench>/bin/         binaries per layout (not versioned)
```

## Use

From the tree root, after `scons build/RISCV_RAMULATOR2/gem5.opt`:

```sh
workload/microbench/gemv/scripts/build.sh      # libpim + one binary per layout
workload/microbench/gemv/scripts/run.sh -j 8   # every run of configs/runs.csv
workload/microbench/gemv/scripts/collect.sh    # results/summary.{csv,md}
workload/microbench/gemv/scripts/run.sh --only lpddr6-o3
```

A run is one line of `configs/runs.csv`: `name, memory, cpu, ab_mode,
policy, args` (`memory`: `lpddr6`, `ddr5`, `gddr7`, `hbm4`; `cpu`: `atomic`,
`timing`, `minor`, `o3`; `ab_mode`: `baseline`, `ideal`, `aim`, `aespa`;
`policy`: empty for `RoRaBaChCo`, or another address-mapping policy, for
which `build.sh` generates a layout header with
`libpim/gen_pim_layout.py`). `GEM5=<binary>` selects another gem5.

Each `results/runs/<name>/` keeps `cmd.txt` (the exact command),
`program.out`, `stats.txt`, `gem5.log`, `rc` and `hosttime` (seconds).
In `summary.csv`, `pim_acts` and `pim_ab_acts` are per region: Ramulator
2.1's counters in `stats.txt` are cumulative, so `bench.py` takes
differences between blocks.

## Known issue

On the O3 CPU, gem5 crashes in register renaming
(`o3::SimpleRenameMap::rename`) when an RVV loop runs after `pim_add` in
the same program; `eltwise` therefore runs its CPU regions first.
