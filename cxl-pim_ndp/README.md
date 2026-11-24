# Ara RVV Vector Accelerator on AMD Versal VHK158

A RISC-V Vector (RVV 1.0) accelerator SoC for FPGA, built on the PULP Ara vector unit and the CVA6 scalar core, with a 512 KiB on-chip scratchpad.

| | |
|---|---|
| **Release** | v1.2, 2026-09-29 |
| **Status** | Verified on silicon, timing-clean |
| **Part** | `xcvh1582-vsva3697-2MP-e-S` (AMD Versal VHK158, 2 SLR) |
| **Config** | 2 lanes, VLEN 2048, 64-bit AXI, 120.000 MHz |
| **Timing** | WNS 0.000 ns, TNS 0.000 ns, 0 failing of 239,045 endpoints |
| **Utilization** | 219,423 LUT (12.5 %), 81,370 FF (2.3 %), 32 URAM, 60 BRAM, 104 DSP |
| **Bitstream** | [`pdi/ara_vhk158_2lane_120mhz_vlsu_spm_2026-09-29.pdi`](pdi/) |

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Documentation](#documentation)
- [Building from Source](#building-from-source)
- [Test Kernels](#test-kernels)
- [Identified Issues](#identified-issues)
- [Provenance](#provenance)

## Features

| Feature | Detail |
|---|---|
| RVV 1.0 vector unit | PULP Ara, 2 lanes, VLEN 2048, fp8 / fp16 / fp32 and int8 / int16 / int32 / int64 datapaths |
| On-chip scratchpad | 512 KiB at `0x3000_0000` inside the vector load/store unit, accessed with standard `vle` / `vse` / `vluxei`, 9.12x the random-gather throughput of DRAM |
| AXI transaction counters | Read-only at `0xD001_0000`, measuring memory round-trip latency and outstanding depth on the scratchpad and DRAM ports |
| Transcendental FU | `vfexp` and `vfrcp` from multipartite tables, 3-stage pipeline, custom-1 opcode |
| Trap reporting | Exceptions record `mcause`, `mepc`, `mtval`, `vtype`, `vl`, `vstart` at `0x4010_0300`, then halt |
| Software | 60 prebuilt kernels, kernel sources and a build script |

The scratchpad is attached behind an address decoder in `vlsu.sv` for a scratchpad transaction to bypass the interconnect and the NoC/DDR memory controller, while allowing a normal DRAM transaction.
No instructions have been added for the scratchpad, but transfers between DRAM and the scratchpad require vector instructions, since a CVA6 scalar load or store to `0x3000_0000` returns `DECERR`.

## Quick Start

The requirements are AMD Vivado 2025.2 for `xsdb` and a `hw_server` connected to the board.

```tcl
xsdb
  # 1. connect, program the bitstream, check DDR4 and load the helper scripts
  set HW_HOST localhost     ;# host running hw_server
  set PDI <this-repo>/pdi/ara_vhk158_2lane_120mhz_vlsu_spm_2026-09-29.pdi
  source <this-repo>/scripts/board_up.tcl

  # 2. once per power cycle
  mem_target                ;# select a JTAG target with memory access
  emio_init                 ;# enable the PMC EMIO line that drives pl_soft_rst

  # 3. run a kernel
  load_kernel fmatmul_sp    ;# load the image at the CVA6 boot address 0x4000_0000
  cva6_reset                ;# software reset through pl_soft_rst
  after 4000
  read_result               ;# PASS/FAIL and cycle count from the verify block
```

Expected output:

```
heartbeat = 0xcafe000f     magic = 0xa1a1b1b1
==== PASS   cycles = 0x000139a1 (80289)
```

`list_kernels` lists every image in `sw/bin`, and `read_trap` decodes the exception record when a kernel does not finish.

## Repository Layout

```
.
├── README.md                  overview and quick start
├── LICENSE                    Apache-2.0, for the files added by this project
├── docs/
│   ├── PERFORMANCE.md         roofline analysis result
│   └── isa_support.md         supported and unsupported RVV instructions
├── pdi/                       bitstream
├── rtl/
│   ├── ara/                   Ara and CVA6 RTL
│   └── board/                 board RTL, block design, constraints and build scripts
├── sw/
│   ├── bin/                   60 prebuilt kernels
│   └── src/                   kernel sources and build script
├── scripts/                   xsdb bring-up and kernel helpers
├── results/                   measured result tables
└── reports/                   Vivado timing, utilization, power and routing reports
```

## Documentation

| Document | Contents |
|---|---|
| [`docs/isa_support.md`](docs/isa_support.md) | Supported and unsupported RVV instructions, identified issues and verification |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | Roofline analysis result at 120 MHz |
| [`results/roofline_sweep_120mhz.md`](results/roofline_sweep_120mhz.md) | Roofline sweep at 120 MHz: cycles, FLOP/cycle and % of peak for every kernel and size |
| [`results/scratchpad_roofline_120mhz.md`](results/scratchpad_roofline_120mhz.md) | Roofline sweep at 120 MHz with the operands in DRAM and in the scratchpad |
| [`results/results_2026-08-14_int_verilator_2lane.md`](results/results_2026-08-14_int_verilator_2lane.md) | Integer ISA test results under Verilator |

## Building from Source

### Bitstream

```bash
source /path/to/Xilinx/2025.2/Vivado/settings64.sh
cd rtl/board
vivado -mode batch -source setup_project.tcl -log setup.log
vivado -mode batch -source run_synth.tcl     -log synth.log
vivado -mode batch -source run_impl.tcl      -log impl.log
```

`setup_project.tcl` creates the Vivado project, including the block design and the constraints.
`run_synth.tcl` and `run_impl.tcl` apply the synthesis and implementation settings of the released bitstream, and the PDI is written to `build/vivado/ara_vhk158.runs/impl_1/fpga_top_vhk158.pdi`.

### Ara RTL

[`rtl/ara/`](rtl/ara/) holds the 389 Ara and CVA6 source files and the include directories that `setup_project.tcl` reads.
The baseline is [PULP Ara](https://github.com/pulp-platform/ara) at `a0b02d7e`, and every file modified by KETI carries the line `Modified 2026 by KETI for the VHK158 support.` in its header.

### Software

Building the kernels requires RISC-V GCC 16 or newer with RVV 1.0 and `Zfh` vector intrinsics.
`build_all_kernels.sh` uses `riscv64-unknown-elf-` from `PATH`, or the prefix set in `G16`.

```bash
cd sw/src
./build_all_kernels.sh
```

## Test Kernels

Every image loads at `0x4000_0000` and writes a verify block at `0x4010_0100` with magic `0xA1A1B1B1`, and runs with `load_kernel <name>`, `cva6_reset` and `read_result`.

| Group | Kernels | Coverage |
|---|---|---|
| Dense linear algebra | `fmatmul_sp/hp`, `fdotproduct_sp/hp`, `imatmul_i8` | FMA throughput in fp32, fp16 and int8 |
| Element-wise and reduction | `exp`, `gelu`, `softmax`, `layernorm`, `vfredsum`, each `_sp`/`_hp`, with `_vt` variants | Memory-bound throughput and the `vtrans` FU |
| Attention | `attention_sp/hp`, `gqa_attention_sp/hp` | Multi-kernel pipelines and Llama-3-8B GQA shapes |
| Memory characterization | `gather_bench_sp/hp` | 31 records of streaming, gather and strided access to DRAM and the scratchpad |
| Scratchpad bring-up | `spm_diag_sp/hp` | Scalar access, address walk, 64-bit access, vector load and store, size sweep |
| Roofline | `roofline_sp/hp`, `roofline_spm_sp/hp` | Every kernel and size of [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md), in DRAM and in the scratchpad |
| FP formats | `fp8_diag`, `fp8_widen_diag`, `bf16_diag` | fp8 arithmetic, fp8 to fp16 widening, bf16 widening multiply-add |
| Scalar load/store | `lsu_bench` | CVA6 load and store latency and throughput |
| ISA corner cases | `corner_sp/hp`, `lmul_test_sp/hp`, `redmax_soak_sp/hp` | LMUL and `vtype` edges, FP special values |

`gather_bench_sp/hp` and `spm_diag_sp/hp` print result tables over UART, and `read_gather_bench` reads the `gather_bench` records over JTAG.

## Identified Issues

| Instruction | Behavior | Workaround |
|---|---|---|
| `vfredmax`, `vfredmin` | Returns the scalar seed | Chunked `vfmax` / `vfmin` with a scalar compare |
| `vrgather` | Wrong result | `vluxei` where the access pattern allows |
| `vsseg`, `vssseg` | Hangs | Indexed `vsuxseg` |
| `viota` directly after `vid` | Wrong result | Another vector instruction between the two |

`vsetvl(n)` returns `min(n, VLMAX)`, so a kernel loops until all `n` elements are processed.
[`docs/isa_support.md`](docs/isa_support.md) lists the platform each issue was observed on and the unsupported instructions.

## Provenance

| Component | Version |
|---|---|
| Upstream Ara | `pulp-platform/ara` at `a0b02d7e`, 2026-03-23 |
| Ara changes | 21 modified files, 3 new files |
| Upstream CVA6 | `pulp-platform/cva6` at `99eac9a64` |
| CVA6 changes | 2 modified files |
| Toolchain, bitstream | AMD Vivado 2025.2 |
| Toolchain, software | RISC-V GCC 16, `rv64gcv_zfh_zvfh`, `lp64d` |
| Simulation | Verilator at NrLanes 2 and 8 |

Ara and CVA6 are under the Solderpad Hardware License v0.51.
The UART (`apb_uart`) is under LGPL-2.1-or-later.
Files added by this project carry Apache-2.0 headers, with the license text in [`LICENSE`](LICENSE).
The license texts of Ara, CVA6 and their dependencies are kept beside their sources under `rtl/ara/`.
