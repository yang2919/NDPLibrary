# Processing-in-Memory (PIM) on RISC-V gem5: instruction set and library

Two draft documents and the source files they describe, copied from a
RISC-V-only gem5 tree (branch `riscv-only`, commit `6b242c1ce1`,
2026-11-10). The PIM memory is simulated with Ramulator 2.1, a dynamic
random-access memory (DRAM) simulator, behind a gem5 wrapper.

## Start here

| Document | What it covers |
|---|---|
| `docs/pim/pim-isa-spec.md` | The 18 PIM instructions: the PIM state in memory, encodings, operand addressing, semantics, faults and software errors, memory ordering and cache coherence |
| `docs/pim/pim-library-api.md` | The user-level C library that emits them: building and linking, layout headers, every public type and function, examples |
| `docs/pim/pim_background_and_spec.md` | The original PIM background and command specification both documents build on |

Both documents are drafts. They cite code as `path:line`; every path is
relative to this directory, and the files here are the exact versions
those citations were checked against.

## Contents

The files keep their place in the gem5 tree.

| Path | Part |
|---|---|
| `src/arch/riscv/isa/decoder.isa`, `isa/formats/pim.isa`, `insts/pim.hh` | Instruction decode, request flags, address guards |
| `src/arch/riscv/isa/templates/vector_mem.isa`, `faults.cc`, `tlb.cc` | The stock vector memory templates, faults and translation the PIM instructions reuse |
| `src/cpu/StaticInstFlags.py`, `src/cpu/static_inst.hh` | The `IsPim` instruction flag |
| `src/mem/pim_op.hh`, `src/mem/request.hh`, `src/mem/packet.{hh,cc}` | Operation numbers, the descriptor, request flags, packet commands |
| `src/mem/cache/{Cache.py,base.cc,cache.cc}` | Cache behaviour and diagnostics for PIM requests |
| `src/mem/ramulator2/` | The PIM memory: request decode and checks, the functional model, number formats, ordering, timing (with the plain Ramulator 2.1 wrapper it extends) |
| `ext/ramulator2/src/ramulator/pim/` | The per-channel PIM command sequencer inside Ramulator 2.1 |
| `src/python/gem5/components/memory/pim.py`, `configs/example/gem5_library/riscv-pim.py` | The gem5 memory component and the run script |
| `include/gem5/pim/` | The library headers (layout, address encoder, instruction wrappers, runtime, user API) and the four default layout headers |
| `util/pim/` | The library sources, layout generator, host reference model, tests and timing tables |
| `tests/test-progs/pim-gemv/`, `tests/test-progs/pim-eltwise/`, `tests/gem5/pim/ref/` | Two example programs and their reference outputs |

## What is not here

- **The rest of gem5.** This is a subset for reading; it does not build
  on its own. Building and running needs the full tree.
- **The internal planning notes.** The documents mention them as design
  background, for example `docs/pim/decisions.md` (decisions "D1" to
  "D13"), the plans "P2" to "P5b" (`docs/pim/02-isa-plan.md`,
  `03-request-path-plan.md`, `04-memctrl-plan.md`,
  `04r-ramulator-pim-plan.md`, `05b-architecture-timing-plan.md`) and
  `docs/pim/pim-library-plan.md` ("P-LIB"). Those files are not
  included. The documents are meant to stand without them: the rules
  they state are cited to the code included here, and a reference to a
  plan only says where a choice was made.
- **Build outputs**, such as the library archives under
  `util/pim/build/`.

## Licenses

gem5's files are under `LICENSE` and `COPYING`; Ramulator 2.1's files
under `ext/ramulator2/LICENSE`.
