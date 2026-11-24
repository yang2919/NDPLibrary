# gem5 PIM: a RISC-V Processing-in-Memory simulation framework

This repository is a fork of [gem5](https://www.gem5.org) for studying
Processing-in-Memory (PIM): DRAM devices whose banks hold small compute
units that run vector operations next to the data. It contains:

- a RISC-V-only gem5 (RV64GV: RV64IMAFD, Zicsr, Zifencei and the vector
  extension V); the other instruction sets, the AMD GPU model and KVM
  are removed;
- PIM instructions in the RISC-V decoder, executed through gem5's normal
  load/store path, the memory management unit (MMU) and the caches;
- a DRAM back end based on [Ramulator 2.1](https://github.com/CMU-SAFARI/ramulator2)
  (`ext/ramulator2`) with four PIM DRAM standards and a PIM command
  sequencer;
- `libpim`, a C library for writing PIM programs, with a host reference
  model;
- workloads (`workload/`): microbenchmarks and a large language model
  (LLM) decode block;
- tests: unit tests, TestLib suites and an acceptance script.

Based on gem5 `stable` (commit `f5c5a6e390`). The gem5 sources keep
their copyright notices and license (see [License](#license)).

gem5's own README is kept as [`README-gem5.md`](README-gem5.md); gem5's
documentation at <https://www.gem5.org/documentation> applies to
everything not described here.

## Contents

| Path | What |
|---|---|
| `src/arch/riscv/` | RISC-V ISA, including the PIM instructions (decoder, micro-ops), Sv48 paging (opt-in) and PMP fixes |
| `src/mem/ramulator2/` | gem5 wrapper of Ramulator 2.1: `Ramulator2VectorPorts` (a plain DRAM memory) and `PimRamulator2` (the PIM memory: PIM request queue, ordering, functional model of the PIM units) |
| `ext/ramulator2/` | Ramulator 2.1 with the PIM standards and the PIM command sequencer (`src/ramulator/pim/`); changes to upstream files are listed in `ext/ramulator2/README.gem5` |
| `src/python/gem5/components/memory/` | standard-library components: `Ramulator2Memory`, `PimMemory` and `PimSystemMemory` |
| `include/gem5/pim/` | the PIM instruction header and the address-layout headers, shared by gem5, `libpim` and programs |
| `libpim/` | the PIM library ([`libpim/README.md`](libpim/README.md)) |
| `workload/` | PIM workloads ([`workload/README.md`](workload/README.md)) |
| `configs/example/gem5_library/riscv-pim.py` | the main configuration: a RISC-V system with a PIM memory, in SE or FS mode |
| `tests/gem5/pim/`, `tests/test-progs/pim-*` | PIM tests: drivers, checkers, TestLib suites, test programs |
| `util/pim-kernel/` | build of the Linux kernel and disk image the full-system PIM tests use |

## The system model

A PIM system has two memories, both simulated by Ramulator 2.1:

- the **operating-system (OS) memory**, ordinary DRAM (`Ramulator2Memory`);
- the **PIM memory** (`PimMemory`), DRAM of a PIM standard whose units
  execute PIM commands.

`PimSystemMemory` combines them. Each default memory is 32 GiB:

| Standard | Device | Channels | PIM memory base (SE / FS) |
|---|---|---|---|
| LPDDR6 (default) | `LPDDR6_16Gb_x12` | 16 | `0x800000000` / `0x1000000000` |
| DDR5 | `DDR5_16Gb_x8`, 1 rank (2 with `--ranks 2`) | 4 | same |
| GDDR7 | `GDDR7_32Gb_x8` | 32 | same |
| HBM4 | `HBM4_32Gb_8Hi` | 32 | same |

One setting chooses the address-mapping policy of both memories:
gem5's `RoRaBaChCo` (default), `RoRaBaCoCh` and `RoCoRaBaCh`, ported to
Ramulator 2.1, or Ramulator 2.1's `RoBaRaCoCh`, `ChRaBaRoCo` and
`MOP4CLXOR`.

A program reaches the PIM memory in one of two ways:

- **Way 1** (SE and FS): the PIM memory is hidden from the OS; a program
  maps it linearly through `/dev/mem`.
- **Way 2** (FS): the PIM memory is a Linux NUMA node; a program
  allocates huge pages bound to that node.

PIM requests are never cached, but they go through the whole MMU path:
the TLB, the page-table walker, PMP and PMA checks, and faults. The
memory controller keeps them in their own queues and issues them as PIM
commands with their own timing; all-bank operations follow one of the
modes `baseline`, `ideal`, `aim` or `aespa` (`--ab-mode`), and PIM and
host traffic share each channel under an arbitration policy
(`--arb-policy`: `frfcfs`, `chopim`, `asyncdimm`).

## PIM instructions

Each instruction is one 32-bit word in a custom opcode space. Scalar
forms act on one bank (SB); vector forms on one bank or on every bank of
a unit (AB, all-bank); GB is a unit's global buffer, REG its registers.

| Instruction | Form | Opcode | Select field |
|---|---|---|---|
| `pim.sb.ewadd`, `pim.sb.ewmul`, `pim.sb.ewadd.gbuf`, `pim.sb.ewmul.gbuf`, `pim.sb.mac` | scalar, S-type: address `rs1`, descriptor `rs2` | custom-1 (`0x2b`) | `funct3` 0-4 |
| `pim.sb.mov.bkgbuf`, `pim.sb.mov.gbufbk`, `pim.ab.ewadd`, `pim.ab.ewmul`, `pim.ab.ewadd.gbuf`, `pim.ab.ewmul.gbuf`, `pim.ab.mac`, `pim.ab.mov.regbk`, `pim.ab.mov.bkreg`, `pim.wr.gb`, `pim.wr.reg` | vector store-like: address `rs1`, data `vs3`, length `vl` | custom-2 (`0x5b`) | `sumop` 0-10 |
| `pim.rd.gb`, `pim.rd.reg` | vector load-like: address `rs1`, destination `vd` | custom-0 (`0x0b`) | `lumop` 0-1 |

`include/gem5/pim/pim_isa.h` has a C wrapper and an assembler macro for
each; `libpim/tests/test_encoding.sh` checks every word.

## Building

Requirements: the build tools of gem5 (SCons, a C++17 compiler, Python 3;
see `README-gem5.md`), and for PIM programs the RISC-V cross compiler
`riscv64-linux-gnu-gcc` with the vector extension.

```sh
scons build/RISCV_RAMULATOR2/gem5.opt -j$(nproc)   # gem5 with Ramulator 2.1 and PIM
scons build/RISCV/gem5.opt -j$(nproc)              # gem5 without Ramulator 2.1
scons -C libpim riscv.CROSS_COMPILE=riscv64-linux-gnu- build/riscv
```

| Build | Contents |
|---|---|
| `build/RISCV_RAMULATOR2` | RISC-V with Ramulator 2.1 and the PIM memory (`build_opts/RISCV_RAMULATOR2`); needed for every PIM run |
| `build/RISCV` | RISC-V with gem5's native memory controllers only; PIM instructions decode and reach memory, but no device executes them |
| `build/ALL` | RISC-V with every kept Ruby protocol |

`yaml-cpp`, `fmt` and the other Ramulator 2.1 dependencies are vendored;
the build downloads nothing.

## Running

A PIM program in SE mode (the default program is `pim-gemv`, a
matrix-vector product checked against double precision):

```sh
make -C tests/test-progs/pim-gemv/src PIM_STD=lpddr6
build/RISCV_RAMULATOR2/gem5.opt configs/example/gem5_library/riscv-pim.py
```

`riscv-pim.py --help` lists the options: `--memory` (`lpddr6`, `ddr5`,
`gddr7`, `hbm4`), `--cpu` (`atomic`, `timing`, `minor`, `o3`),
`--cores`, `--policy`, `--channels`, `--ab-mode`, `--arb-policy`,
`--binary` and `--args` for another program.

In FS mode, `--mode fs` boots Ubuntu 24.04 (gem5 resources, downloaded on
first use) with Linux in Sv48 paging; `--fs-program <static binary>` runs
a program as root after boot, `--boot-cpu atomic` boots on the Atomic CPU
and switches to `--cpu`, and `--way 2` makes the PIM memory a NUMA node.
Way 2 and the library's cache-flush protocol in FS need the kernel and
disk image built by `util/pim-kernel/build.sh` (`util/pim-kernel/README.md`).

The PIM library and the workloads have their own READMEs:
[`libpim/README.md`](libpim/README.md),
[`workload/README.md`](workload/README.md).

## Testing

| What | How |
|---|---|
| Unit tests | `scons build/RISCV_RAMULATOR2/unittests.opt` and `scons build/ALL/unittests.opt` |
| Library host tests | `scons -C libpim build/native/test` (reference backend) |
| PIM test driver | `tests/gem5/pim/run_p4r.py --out <dir> --groups <groups>` (`--help` lists the groups; one PASS or FAIL line per case in `<dir>/summary.txt`) |
| TestLib | `cd tests && ./main.py run gem5/pim --isa RISCV_RAMULATOR2 --length long --skip-build` (also `--isa ALL`; the full-system suites are `--length very-long`) |
| Acceptance | `tests/gem5/pim/accept.sh quick\|full\|release <out>`: about 30 minutes, 1.5 hours and 13 hours |

The acceptance script's `release` profile and the full-system groups
need the kernel of `util/pim-kernel`, a few hundred GB of host memory for
the largest systems, and tens of GB of disk.

## Known limitations

- Sv48 covers `satp` only; with the hypervisor extension, `vsatp` and
  `hgatp` stay at Sv39 and Sv39x4. The device tree always describes the
  MMU as `riscv,sv48`.
- User-mode cache-block operations (`cbo.clean`, `cbo.flush`,
  `cbo.inval`) need `senvcfg` bits that Linux 6.8 does not set; the kernel
  of `util/pim-kernel` sets them.
- The O3 CPU matches stores to loads by virtual address, so two virtual
  mappings of one physical page can read stale data without a fence
  (upstream gem5 behaviour).
- An access to a physical address without memory stops the simulation
  instead of raising an access fault (upstream gem5 behaviour).
- The Minor CPU re-executes a vector instruction after a trap with a
  stale vector length.

## License

gem5 is under the BSD license (`LICENSE`, `COPYING`); Ramulator 2.1 is
under the MIT license (`ext/ramulator2/LICENSE`).
