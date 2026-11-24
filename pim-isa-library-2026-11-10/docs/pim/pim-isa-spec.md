# Processing-in-Memory (PIM) instruction set specification

**Status: draft, 2026-11-10.** Written from the tree at `96ab11514d`
(branch `riscv-only`). It states what is implemented, not what was
planned: where a plan and the code differ, the code wins and the plan is
named. Every `path:line` was checked in this tree on that date.

**Scope.** The 18 PIM instructions of the RISC-V 64-bit (RV64) custom
extension: their encodings, operands, semantics, faults and errors, and
their memory ordering. The C and assembly interface that software uses
to emit them is in the companion document
`docs/pim/pim-library-api.md`. The design record is
`docs/pim/decisions.md` (D1-D13), the instruction-set plan
`docs/pim/02-isa-plan.md` ("P2"), the request-path plan
`docs/pim/03-request-path-plan.md` ("P3"), the memory-controller plans
`docs/pim/04-memctrl-plan.md` ("P4") and
`docs/pim/04r-ramulator-pim-plan.md` ("P4R"), and the specification
`docs/pim/pim_background_and_spec.md`.

## Terms

- **Base instruction set architecture (ISA).** The target is RV64GV: the RISC-V general-purpose
  extensions (RVG: integer, multiply, atomic, single- and double-precision
  floating point, `Zicsr`, `Zifencei`) and the RISC-V Vector extension
  (RVV).
- **gem5 terms.** Central processing unit (CPU); the out-of-order CPU
  model (O3); Syscall Emulation (SE) and Full System (FS) modes. Vector terms: vector length (`vl`), vector type (`vtype`), the
  vector register length in bits (VLEN) and in bytes (VLENB = VLEN / 8),
  effective element width (EEW), register group multiplier (LMUL).
- **DRAM terms.** Dynamic random-access memory (DRAM). A **column** is
  one DRAM transaction (one burst): 32 bytes on Low-Power Double Data Rate
  6 (LPDDR6), Graphics Double Data Rate 7 (GDDR7) and High Bandwidth
  Memory 4 (HBM4), 64 bytes on Double Data Rate 5 (DDR5) in the default
  presets. A **bank row** is one row of one bank. A **row slice** is one
  row index in every bank of every channel.
- **Parallelism unit** ("unit"): the set of banks one all-bank command
  covers. It is a rank on DDR5, a pseudo channel on HBM4, and a channel
  on LPDDR6 and GDDR7. Units are numbered over the whole PIM memory:
  unit = channel × (units per channel) + rank or pseudo channel
  (`src/mem/ramulator2/pim/pim_layout.cc:207-210`).
- **Processing unit (PU):** the compute logic of one bank; there is one
  PU per bank (`include/gem5/pim/pim_layout.h:124`).
- **Global buffer (GB):** one row-sized buffer per unit, shared by its
  banks.
- **Command classes:** single-bank (SB), all-bank (AB) and non-bank;
  element-wise (EW) and multiply-accumulate (MAC) operations.
- **Request:** the memory packet a PIM instruction issues. One
  instruction may issue several requests (section 6).
- **Software error:** a request that is well formed for the core but
  violates a PIM rule; the PIM memory detects it (section 7.2).

## 1. Programming model

A PIM-integrated system has two DRAM memories: the operating-system (OS)
memory and the PIM memory, whose banks have PUs. Both are simulated by
Ramulator 2.1 and use the same physical-to-DRAM address-mapping policy.
Only a host core programs the PIM memory, and only through PIM
instructions, which it issues in program order.

Each PIM instruction names one operand address in `rs1`. That address is
a virtual address; it goes through the full memory management unit
(MMU) path (translation lookaside buffer (TLB), page-table walk,
physical memory protection (PMP) and physical memory attributes (PMA)
checks, faults), and the resulting physical address selects DRAM
coordinates (channel, unit, stack ID, bank group, bank, row, column)
under the configured policy. Software computes the address with the
library's encoder, the inverse of the controller's mapping.

The instructions follow the commands of the specification one to one
(D1.1): 7 single-bank, 7 all-bank and 4 non-bank. They reuse the
standard store and vector load/store execution paths, so the core
treats them as memory instructions; what makes them PIM is the request
they issue: `PIM | UNCACHEABLE | STRICT_ORDER` plus a 5-bit operation
number (section 3.4).

### 1.1 Architectural state outside the core

Per unit u (`src/mem/ramulator2/pim/pim_func.hh:1-23`):

| State | Size | Written by | Read by |
|---|---|---|---|
| GB | one row (`ROW_BYTES`) | `pim.wr.gb`, `pim.sb.mov.bkgbuf` | `pim.rd.gb`, `*.gbuf` EW, MAC, `pim.sb.mov.gbufbk` |
| Register j of each PU, j < `NUM_REGS` | one row each | EW (register 0), `pim.ab.mov.bkreg`, `pim.wr.reg` | `pim.ab.mov.regbk`, `pim.rd.reg` |
| Accumulator of each PU | one element of `acc_fmt` | MAC, `pim.wr.reg` | `pim.rd.reg` |

Bank data is the DRAM itself; normal loads and stores reach it too. The
state is in the memory, not in the core: it is not saved on a context
switch and is shared by every hart. In gem5 it is checkpointed with the
memory (`src/mem/ramulator2/pim_ramulator2.cc:1081-1089`).

### 1.2 The register window

The PU registers and accumulators are addressed as memory: the top
`NUM_REGS + 1` rows of every bank form the **register window**
(`src/mem/ramulator2/pim/pim_func.hh:13-17`). With
`REG_ROW0 = ROWS - NUM_REGS - 1`:

- row `REG_ROW0 + j` of bank b is register j of b's PU; column c holds
  register bytes [c × B, (c + 1) × B), B the column size;
- row `REG_ROW0 + NUM_REGS` is the accumulator row: its first
  `ACC_BYTES` bytes are the accumulator, the other bytes read as zero
  and writes to them are ignored
  (`src/mem/ramulator2/pim/pim_func.cc:182-218`).

The default is one register, so `REG_ROW0 = ROWS - 2`. Window rows are
not DRAM storage: bank operations on them and normal accesses to them
are software errors (section 7.2).

### 1.3 Geometry of the default configurations

From the committed layout headers `include/gem5/pim/pim_layout_<std>.h`
(all with policy `RoRaBaChCo`, one register, bf16 operands, fp32
accumulator):

| Standard | Channels | Unit | Units per channel | Banks per unit | Column | Row | Rows per bank | `REG_ROW0` | PIM memory |
|---|---|---|---|---|---|---|---|---|---|
| DDR5 (`DDR5_32Gb_x4`) | 2 | rank | 2 | 32 | 64 B | 8 KiB (128 columns) | 131072 | 131070 | 128 GiB |
| LPDDR6 (`LPDDR6_16Gb_x12`, default) | 8 | channel | 1 | 16 | 32 B | 2 KiB (64 columns) | 65536 | 65534 | 16 GiB |
| GDDR7 (`GDDR7_64Gb_x8`) | 8 | channel | 1 | 16 | 32 B | 2 KiB (64 columns) | 65536 | 65534 | 16 GiB |
| HBM4 (`HBM4_32Gb_16Hi`) | 8 | pseudo channel | 2 | 64 | 32 B | 1 KiB (32 columns) | 16384 | 16382 | 16 GiB |

The all-bank width (banks per unit) is an architecture parameter
(D8.5), so the same program does different amounts of work per AB
instruction on different standards.

### 1.4 Number formats

The PIM memory has two configured formats (`PimRamulator2.num_fmt`,
`acc_fmt`, `src/mem/ramulator2/PimRamulator2.py:32-35`):

- `num_fmt` (default bf16): operands in the banks and the GB, and EW
  results;
- `acc_fmt` (default fp32): the MAC accumulator.

Both are one of bf16, fp16, fp32, int8, int16, int32. Start-up refuses a
pair whose families differ (integer against floating), an `acc_fmt`
narrower than `num_fmt`, or a `num_fmt` whose size does not divide the
column size (`src/mem/ramulator2/pim/pim_numfmt.cc:102-114`). Elements
are little endian; a column of B bytes holds B / E elements of E bytes.

Arithmetic (`src/mem/ramulator2/pim/pim_numfmt.hh:1-25`):
- floating EW results are computed in fp32 with round to nearest even
  (RNE) and rounded to `num_fmt`; bf16 widens by a 16-bit shift and
  narrows with RNE (a not-a-number (NaN) value keeps its top 16 bits with
  the quiet bit set);
  fp16 converts through SoftFloat, so a NaN narrowed to fp16 is the
  canonical 0x7e00;
- integer formats wrap modulo 2^bits;
- MAC: per column, the products of the column's elements are summed in
  `acc_fmt` sequentially in ascending element order, and that sum is
  added to the accumulator once.

**The instruction's EEW is not the element format.** EEW only sizes the
byte range (`vl` × EEW / 8); the PIM memory interprets the bytes with
`num_fmt`. The library always uses EEW = 8.

## 2. Instruction summary

| # | Assembly name | Class | Form | Opcode | Select | `PimOp` |
|---|---|---|---|---|---|---|
| 1 | `pim.sb.ewadd` | SB compute | scalar store-like | custom-1 | `funct3` 0 | 1 |
| 2 | `pim.sb.ewmul` | SB compute | scalar store-like | custom-1 | `funct3` 1 | 2 |
| 3 | `pim.sb.ewadd.gbuf` | SB compute | scalar store-like | custom-1 | `funct3` 2 | 3 |
| 4 | `pim.sb.ewmul.gbuf` | SB compute | scalar store-like | custom-1 | `funct3` 3 | 4 |
| 5 | `pim.sb.mac` | SB compute | scalar store-like | custom-1 | `funct3` 4 | 5 |
| 6 | `pim.sb.mov.bkgbuf` | SB data movement | vector store-like | custom-2 | `sumop` 0x00 | 6 |
| 7 | `pim.sb.mov.gbufbk` | SB data movement | vector store-like | custom-2 | `sumop` 0x01 | 7 |
| 8 | `pim.ab.ewadd` | AB compute | vector store-like | custom-2 | `sumop` 0x02 | 8 |
| 9 | `pim.ab.ewmul` | AB compute | vector store-like | custom-2 | `sumop` 0x03 | 9 |
| 10 | `pim.ab.ewadd.gbuf` | AB compute | vector store-like | custom-2 | `sumop` 0x04 | 10 |
| 11 | `pim.ab.ewmul.gbuf` | AB compute | vector store-like | custom-2 | `sumop` 0x05 | 11 |
| 12 | `pim.ab.mac` | AB compute | vector store-like | custom-2 | `sumop` 0x06 | 12 |
| 13 | `pim.ab.mov.regbk` | AB data movement | vector store-like | custom-2 | `sumop` 0x07 | 13 |
| 14 | `pim.ab.mov.bkreg` | AB data movement | vector store-like | custom-2 | `sumop` 0x08 | 14 |
| 15 | `pim.rd.gb` | non-bank | vector load-like | custom-0 | `lumop` 0x00 | 15 |
| 16 | `pim.wr.gb` | non-bank | vector store-like | custom-2 | `sumop` 0x09 | 16 |
| 17 | `pim.rd.reg` | non-bank | vector load-like | custom-0 | `lumop` 0x01 | 17 |
| 18 | `pim.wr.reg` | non-bank | vector store-like | custom-2 | `sumop` 0x0a | 18 |

`PimOp` is the operation number carried by every request
(`src/mem/pim_op.hh:19-40`); 0 means "not PIM". In gem5 traces and
disassembly the names use `_` for `.` (`pim_sb_mac`), and vector
micro-ops print `<name>_micro` (P2 §1.2, §5).

## 3. Encoding

### 3.1 Opcode map

| Major opcode | Bits [6:0] | `OPCODE5` (bits [6:2]) | Form | Decoder |
|---|---|---|---|---|
| custom-0 | `0x0B` | `0x02` | RVV load-like (2 instructions) | `src/arch/riscv/isa/decoder.isa:1341-1359` |
| custom-1 | `0x2B` | `0x0a` | RVG store-like, S-type (5) | `src/arch/riscv/isa/decoder.isa:2094-2104` |
| custom-2 | `0x5B` | `0x16` | RVV store-like (11) | `src/arch/riscv/isa/decoder.isa:6063-6087` |
| custom-3 | `0x7B` | `0x1e` | not PIM: gem5's `m5op` pseudo-instructions | unchanged |

None of the three values was in use before; adding them only turned
encodings that decoded as `Unknown` into PIM instructions (P2 §1.6).

### 3.2 Scalar form (custom-1)

```
 31        25 24   20 19   15 14  12 11       7 6       0
+------------+-------+-------+------+----------+---------+
| imm[11:5]  |  rs2  |  rs1  |funct3| imm[4:0] | 0101011 |
+------------+-------+-------+------+----------+---------+
```

- `rs1` + sign-extended 12-bit `imm` is the PIM address EA.
- `rs2` holds the 64-bit **descriptor**:

  | Bits | Field | Meaning |
  |---|---|---|
  | [15:0] | `ncols` | Columns covered from EA's column; 0 = to the end of the row |
  | [63:16] | reserved | Must be zero; non-zero is a software error |

  Layout: `PimDesc` (`src/mem/pim_op.hh:69-74`); software side
  `PIM_DESC(ncols)` (`include/gem5/pim/pim_isa.h:43-47`).
- `funct3` 5-7 are reserved.

### 3.3 Vector forms (custom-0 and custom-2)

```
 31  29  28  27 26  25  24       20 19   15 14   12 11       7 6       0
+------+----+------+----+-----------+-------+-------+----------+---------+
|  nf  |mew | mop  | vm |lumop/sumop|  rs1  | width | vd / vs3 | opcode  |
|  000 | 0  |  00  | 1  | operation |       |  EEW  |          |         |
+------+----+------+----+-----------+-------+-------+----------+---------+
```

- The layout is the standard unit-stride vector load/store with another
  major opcode and the operation in `lumop`/`sumop`; each PIM vector
  word equals its `vle<eew>.v`/`vse<eew>.v` counterpart in every other
  bit (P2 §1.5).
- `width` ∈ {0, 5, 6, 7} selects EEW 8, 16, 32, 64. Widths 1-4 are
  reserved.
- `vm` must be 1 (unmasked), `mop`, `mew`, `nf` must be 0.
- `vs3` (store-like) supplies data only for `pim.wr.gb` and
  `pim.wr.reg`; the other nine store-like instructions ignore its
  content (D1.2 (a)). `vd` (load-like) receives the data.
- Reserved: `sumop` 0x0b-0x1f, `lumop` 0x02-0x1f.

### 3.4 Request attributes

Every request of a PIM instruction carries, fixed at decode
(`src/arch/riscv/isa/formats/pim.isa:11-13`):

| Attribute | Value | Definition |
|---|---|---|
| `Request::PIM` | bit 14 (`0x4000`) | `src/mem/request.hh:179` |
| `Request::UNCACHEABLE`, `STRICT_ORDER` | standard flags | — |
| Operation | `PimOp` in request bits 54-58 | `src/mem/request.hh:272-283` |

Static instructions carry the flag `IsPim` (`src/cpu/StaticInstFlags.py:103`,
`StaticInst::isPim()` at `src/cpu/static_inst.hh:153`). The OpClasses
are the standard `MemWrite`, `SimdUnitStrideStore` and
`SimdUnitStrideLoad`, so a PIM instruction has the core timing of `sd`,
`vse` or `vle` (P2 §3.5). No PIM instruction is serializing (D5).

### 3.5 Example machine words

With `rs1 = a1`, `rs2 = a2`, `imm = 0` (scalar) and `vd`/`vs3 = v8`,
`width = 7` (vector), checked with GNU binutils 2.42 (P2 §1.2) and by
`util/pim/tests/test_encoding.sh`:

| Instruction | Word | Instruction | Word |
|---|---|---|---|
| `pim.sb.ewadd a2, 0(a1)` | `0x00c5802b` | `pim.ab.ewmul.gbuf v8, (a1)` | `0x0255f45b` |
| `pim.sb.ewmul a2, 0(a1)` | `0x00c5902b` | `pim.ab.mac v8, (a1)` | `0x0265f45b` |
| `pim.sb.ewadd.gbuf a2, 0(a1)` | `0x00c5a02b` | `pim.ab.mov.regbk v8, (a1)` | `0x0275f45b` |
| `pim.sb.ewmul.gbuf a2, 0(a1)` | `0x00c5b02b` | `pim.ab.mov.bkreg v8, (a1)` | `0x0285f45b` |
| `pim.sb.mac a2, 0(a1)` | `0x00c5c02b` | `pim.rd.gb v8, (a1)` | `0x0205f40b` |
| `pim.sb.mov.bkgbuf v8, (a1)` | `0x0205f45b` | `pim.wr.gb v8, (a1)` | `0x0295f45b` |
| `pim.sb.mov.gbufbk v8, (a1)` | `0x0215f45b` | `pim.rd.reg v8, (a1)` | `0x0215f40b` |
| `pim.ab.ewadd v8, (a1)` | `0x0225f45b` | `pim.wr.reg v8, (a1)` | `0x02a5f45b` |
| `pim.ab.ewmul v8, (a1)` | `0x0235f45b` | | |
| `pim.ab.ewadd.gbuf v8, (a1)` | `0x0245f45b` | | |

The assembler has no mnemonics for them; use `.insn` (section 9).

## 4. Operand addressing

### 4.1 From address to coordinates

The PIM memory occupies one physical range (base a multiple of its size;
defaults in `pim_layout_<std>.h`: `WINDOW_BASE_SE`, `WINDOW_BASE_FS`).
The memory decodes `offset = PA − base` into coordinates under the
configured policy (`include/gem5/pim/pim_addr.h:216-288`): channel,
unit, stack ID (HBM4), bank group, bank, row, column, and the byte
within the column. Six policies exist: gem5's `RoRaBaChCo` (default),
`RoRaBaCoCh`, `RoCoRaBaCh`, and Ramulator 2.1's `RoBaRaCoCh`,
`ChRaBaRoCo`, `MOP4CLXOR`. Under the default LPDDR6 layout the offset
bits are, most significant first,
`row 33-18 | bank 17-16 | bg 15-14 | ch 13-11 | col 10-5 | byte 4-0`
(`include/gem5/pim/pim_layout_lpddr6.h:8-9`).

Banks are numbered within a unit as
`(sid × bank groups + bg) × banks per group + bank`
(`src/mem/ramulator2/pim/pim_layout.cc:212-216`), so the two banks of
an EW pair, 2k and 2k + 1, are in one bank group.

### 4.2 Which coordinates each class uses

| Class | Uses | Ignores |
|---|---|---|
| SB (1-7) | unit, bank, row, column (start) | — |
| AB (8-14) | unit, row, column (start); the bank only for the even-bank check of `pim.ab.ewadd`/`ewmul` | bank (the command covers every bank of the unit) |
| GB (15-16) | unit, column and byte (the GB offset) | stack ID, bank group, bank, row |
| Register (17-18) | every coordinate of every column touched (the window position) | — |

### 4.3 The byte range of a request

| Form | Range | Columns |
|---|---|---|
| Scalar (1-5) | One 8-byte request at EA, EA 8-byte aligned | From EA's column, `ncols` columns (0: to the row end); the byte bits of EA are ignored. Whole columns |
| Vector bank operations (6-14) | [addr, addr + size) of each request | The columns the bytes span; must be consecutive columns of one bank row. A column only partly covered is computed only on its covered elements or bytes |
| `pim.rd.gb`, `pim.wr.gb` | GB bytes [off, off + size), off = column × B + byte of the request's first byte | Byte exact; must end within the row |
| `pim.rd.reg`, `pim.wr.reg` | Every byte, decoded column by column through the window | Byte exact; may span banks (a gather or scatter) |

Source: `src/mem/ramulator2/pim/pim_layout.cc:265-355`.

The GB offset advances linearly inside a request, not along the encoder:
under a policy whose consecutive addresses leave the bank row within a
cache line, software issues one request per contiguous piece and
addresses each piece's GB column explicitly (library:
`pim_gb_addr()`).

## 5. Instruction semantics

Notation: `Row_k[c]` is column c of the addressed row of bank k of the
addressed unit u; b is the addressed bank; `reg_k[j]`, `acc_k` are the
registers and the accumulator of bank k's PU; `gb` is unit u's GB, and
`gb[c]` its bytes [c × B, (c + 1) × B); ⊕ is + for `*add*` and × for
`*mul*`, element by element in `num_fmt`. Every effect applies only to
the elements (compute) or bytes (copies) inside the request's range.
Source: `src/mem/ramulator2/pim/pim_func.cc:81-218`; P4 §9.4.

### 5.1 Single-bank instructions

Assembly: `pim.sb.<op> rs2, imm(rs1)` for 1-5, `pim.sb.mov.<dir> vs3,
(rs1)` for 6-7.

| Instruction | Effect for each column c covered | Notes |
|---|---|---|
| `pim.sb.ewadd`, `pim.sb.ewmul` | `reg_b[0][c] = Row_b[c] ⊕ Row_{b+1}[c]` | b must be even (the lower bank of the pair computes, D8.6) |
| `pim.sb.ewadd.gbuf`, `pim.sb.ewmul.gbuf` | `reg_b[0][c] = gb[c] ⊕ Row_b[c]` | |
| `pim.sb.mac` | `acc_b += Σ_i gb[c]_i × Row_b[c]_i` (one accumulator update per column, in `acc_fmt`) | The accumulator is not reset; write zeros with `pim.wr.reg` |
| `pim.sb.mov.bkgbuf` | `gb[c] = Row_b[c]` | |
| `pim.sb.mov.gbufbk` | `Row_b[c] = gb[c]` | Writes DRAM |

### 5.2 All-bank instructions

Assembly: `pim.ab.<op> vs3, (rs1)`. Each applies the SB operation to
every bank k of unit u at once, at the same row and columns:

| Instruction | Effect for each column c covered |
|---|---|
| `pim.ab.ewadd`, `pim.ab.ewmul` | For every even k: `reg_k[0][c] = Row_k[c] ⊕ Row_{k+1}[c]` (odd PUs unchanged) |
| `pim.ab.ewadd.gbuf`, `pim.ab.ewmul.gbuf` | For every k: `reg_k[0][c] = gb[c] ⊕ Row_k[c]` (the GB is shared) |
| `pim.ab.mac` | For every k: `acc_k += Σ_i gb[c]_i × Row_k[c]_i` |
| `pim.ab.mov.regbk` | For every k: `Row_k[c] = reg_k[0][c]` (writes DRAM) |
| `pim.ab.mov.bkreg` | For every k: `reg_k[0][c] = Row_k[c]` |

`pim.ab.mac` over one row computes, in one instruction, one partial dot
product per bank: with matrix row k in bank k and the vector in the GB,
the accumulators hold one element of y = A x each (the library's general
matrix-vector multiplication (GEMV),
`util/pim/src/ops.c:76-147`).

### 5.3 Non-bank instructions

| Instruction | Assembly | Effect |
|---|---|---|
| `pim.wr.gb` | `pim.wr.gb vs3, (rs1)` | `gb[off + i] = vs3` byte i |
| `pim.rd.gb` | `pim.rd.gb vd, (rs1)` | `vd` byte i = `gb[off + i]` |
| `pim.wr.reg` | `pim.wr.reg vs3, (rs1)` | The window bytes the range addresses = `vs3` bytes; any values, zeros reset registers and accumulators |
| `pim.rd.reg` | `pim.rd.reg vd, (rs1)` | `vd` bytes = the window bytes the range addresses (registers, accumulators, zeros past an accumulator) |

Reads have no side effects (D9.5). A load-like instruction writes `vd`
as a standard `vle<eew>.v` with `vl` elements would.

### 5.4 The register-0 rule

EW, `pim.ab.mov.regbk` and `pim.ab.mov.bkreg` use register 0 only and
require `NUM_REGS = 1`; with more registers they are software errors
(`src/mem/ramulator2/pim/pim_func.cc:74-75`). `pim.rd.reg` and
`pim.wr.reg` reach every register in either case.

## 6. Vector behaviour

- **Byte count.** A vector form covers `vl` × EEW / 8 bytes from `rs1`,
  unit stride. `vl` and `vtype` are read at decode as for every vector
  instruction; with `vl = 0` no request is issued
  (`src/arch/riscv/isa/templates/vector_mem.isa`, P2 §4.1).
- **`vill`.** If `vill` is set or the vector unit is off, the
  instruction raises an illegal-instruction exception (P2 §4.1).
- **`vstart`.** Treated as 0, as for every vector memory instruction in
  this tree. The page guard (section 7.1) makes a vector PIM instruction
  fault before its first request or not at all.
- **Requests.** One micro-op per vector register (`ceil(vl / (VLEN /
  EEW))` micro-ops, each at `rs1 + k × VLENB`); the core splits each
  micro-op again at cache-line boundaries (64 bytes in the standard
  configurations). Each piece is one request, and the PIM memory applies
  each request on its own. For example, `pim.ab.mac` at e8 and LMUL 8
  with VLENB 32 covers at most 256 bytes: 8 micro-ops, 8 requests of one
  LPDDR6 column each, so a 2 KiB row takes 8 instructions. Each request
  computes only on its own elements, so where the requests split a row
  changes the timing (a column touched by two requests costs two column
  commands) but not the result: MAC adds each product exactly once.
- **VLEN.** gem5's PIM configuration defaults to VLEN = 8 × the column
  bytes (VLENB = one column: 256 bits on LPDDR6, 512 on DDR5;
  `configs/example/gem5_library/riscv-pim.py:273-274`) and warns when
  VLENB is below a column (`src/python/gem5/components/memory/pim.py:585-596`).
  TimingSimpleCPU supports at most two pieces per access, so it needs
  VLENB ≤ the cache line (P2 §4.4).

## 7. Exceptions and software errors

### 7.1 Raised by the core (precise traps)

| Condition | Exception | Source |
|---|---|---|
| Reserved `funct3`, `sumop`, `lumop`; `vm = 0`; `mop`, `mew`, `nf` ≠ 0; `width` 1-4; RV32 | Illegal instruction (in SE mode gem5 panics `Unknown instruction`) | `src/arch/riscv/isa/decoder.isa:1336-1340`, `:2091-2093`, `:6058-6062`; `src/arch/riscv/faults.cc:290` |
| Scalar EA not 8-byte aligned | Store address misaligned | `src/arch/riscv/isa/formats/pim.isa:17-23` |
| Vector range [`rs1`, `rs1` + `vl` × EEW / 8) crosses a 4 KiB page | Store (or load) address misaligned, tval = `rs1`, before any request | `src/arch/riscv/isa/formats/pim.isa:27-33`; `src/arch/riscv/insts/pim.hh` |
| Translation, PMP or PMA fault | The standard page fault or access fault | MMU path, unchanged |

The two guards exist because gem5 restarts a vector memory instruction
from its first micro-op after a fault, and PIM commands are not
idempotent: a repeated MAC would accumulate twice (P2 §4.5).

### 7.2 Detected by the PIM memory (software errors)

| Rule | Instructions | Message rule (source) |
|---|---|---|
| Descriptor bits [63:16] not zero | scalar | `pim_layout.cc:286-288` |
| Start column + `ncols` past the row | scalar | `pim_layout.cc:292-294` |
| Range past the end of the row | vector bank, GB | `pim_layout.cc:333-334` |
| Range not consecutive columns of one bank row (crosses a row, bank or unit) | vector bank | `pim_layout.cc:335-347` |
| Bank operation on a register-window row | 1-14 | `pim_func.cc:70-71` |
| Odd bank for an EW pair | `pim.sb.ewadd`, `ewmul`, `pim.ab.ewadd`, `ewmul` | `pim_func.cc:72-73` |
| More than one register per PU | EW, `pim.ab.mov.regbk`, `pim.ab.mov.bkreg` | `pim_func.cc:74-75` |
| Compute range not element-aligned | EW, MAC | `pim_func.cc:76-77` |
| Range leaves the register window | `pim.rd.reg`, `pim.wr.reg` | `pim_layout.cc:310-312` |
| A normal load, store, fill, writeback or prefetch reaches a window row | non-PIM requests | `pim_ramulator2.cc:452-469` |

(Paths are under `src/mem/ramulator2/` and `src/mem/ramulator2/pim/`.)

Software errors are not traps: the core has already retired the
instruction. With `sw_error_fatal = True` (default) gem5 stops with a
message naming the operation, address, coordinates and rule; with
`False` it warns once, counts `pim.swErrors`, and completes the request
as a no-op (store-like acknowledged, load-like returns zeros)
(`src/mem/ramulator2/pim_ramulator2.cc:326-350`). A request outside the
PIM memory, or a scalar request that is not 8 aligned bytes, is a
simulator bug and panics.

## 8. Memory ordering and coherence

### 8.1 In the core

`STRICT_ORDER` keeps PIM requests in program order with each other and
with older stores: in O3 a strictly ordered load issues only at the
head of the reorder buffer, stores issue after commit, and neither takes
part in store-to-load forwarding; Minor issues them only at the head
(D5). PIM requests are never cached (`UNCACHEABLE`).

### 8.2 In the PIM memory

Requests take effect in acceptance order whenever their scopes conflict
(`src/mem/ramulator2/pim/pim_scope.hh:1-17`):

| Requester | Scope |
|---|---|
| Normal access (per column) | (unit, bank, row) |
| SB | (unit, bank, row); EW pair: banks b and b + 1 |
| AB | (unit, every bank, row) |
| Non-bank | the whole unit |

PIM requests of one unit stay in order; different units proceed in
parallel. Store-like PIM requests are **posted** by default: the memory
acknowledges them when it accepts them
(`src/mem/ramulator2/pim_ramulator2.cc:679-682`), so a `fence` after
them waits for acceptance, not for the DRAM commands; the ordering above
still holds for every later request.

### 8.3 Normal loads and stores to PIM memory

Bank rows (outside the window) are ordinary memory to normal loads and
stores, which are cacheable (D4.1 (ii)). Software keeps caches and PIM
consistent:

1. **Before** a PIM instruction reads or writes a row that normal
   stores may have touched: `cbo.flush` every cache line of that row,
   then `fence rw, rw`.
2. **After** PIM instructions wrote a row, before normal loads read it:
   `fence rw, rw`, `cbo.flush` the lines (dropping stale copies), then
   `fence rw, rw`.

What gem5 does when software forgets:

| Case | Result | Source |
|---|---|---|
| The issuing core's own cache holds the line | Evicted, a dirty line written back before the PIM request; counted in `pimValidBlk`, `pimDirtyBlk` | `src/mem/cache/cache.cc:178-192` |
| Another cache holds the line dirty, PIM write | `fatal` ("flush the line with cbo.flush and fence before PIM") | `src/mem/cache/base.cc:1018-1021` |
| Another cache holds the line dirty, PIM read | `fatal` by default; with `pim_dirty_read_fatal = False`, warn once and the copy answers | `src/mem/cache/base.cc:1022-1031`, `src/mem/cache/Cache.py:131-135` |
| A prefetcher fetches a PIM line | Prevented: `PimSystemMemory` puts the PIM range in every prefetcher's `excluded_ranges` | P4R.15 |

Use `cbo.flush`, not `cbo.inval`: gem5's `cbo.inval` always invalidates
without writing back (P3 R5).

## 9. Assembly and toolchain

Binutils has no PIM mnemonics. Emit the words with `.insn`:

```asm
.insn s 0x2b, 4, a2, 0(a1)          # pim.sb.mac a2, 0(a1)
.insn r 0x5b, 0, 1, x8, a1, x6      # pim.ab.mac v8, (a1), e8: x6 = sumop 6
.insn r 0x0b, 0, 1, x8, a1, x1      # pim.rd.reg v8, (a1), e8: x1 = lumop 1
```

In `.insn r`, `funct3` is `width`, `funct7 = 1` is {`nf`, `mew`, `mop`,
`vm`} = {0, 0, 0, 1}, `rd` is the vector register number and `rs2` the
operation number. `include/gem5/pim/pim_isa.h` wraps exactly this: C
inline functions, and under `__ASSEMBLER__` the macros `pim_sb_mac
rs2, off, rs1` and `pim_ab_mac vreg, rs1[, eew]`
(`include/gem5/pim/pim_isa.h:52-123`). Set `vl` with `vsetvli` before a
vector form, as for any vector memory instruction.

## 10. Implementation map

| Part | Where |
|---|---|
| Decode | `src/arch/riscv/isa/decoder.isa:1341`, `:2094`, `:6063` |
| Formats (flags, guards, bodies) | `src/arch/riscv/isa/formats/pim.isa` |
| Page-crossing helper | `src/arch/riscv/insts/pim.hh` |
| Operation numbers, descriptor | `src/mem/pim_op.hh` |
| Request flags | `src/mem/request.hh:176-179`, `:271-303`, `:1062-1069` |
| Packet commands (18 `Pim<Op>Req`/`Resp` pairs) | `src/mem/packet.hh`, `src/mem/packet.cc` (P3 R2) |
| TLB tracing (`--debug-flags=PIM`) | `src/arch/riscv/tlb.cc` (P3 R3) |
| Cache diagnostics | `src/mem/cache/cache.cc`, `src/mem/cache/base.cc` (P3 R4, DP-K) |
| Request decode and checks | `src/mem/ramulator2/pim/pim_layout.cc` |
| Functional model, formats, scopes | `src/mem/ramulator2/pim/pim_func.cc`, `pim_numfmt.cc`, `pim_scope.hh` |
| PIM memory (queues, ordering, timing) | `src/mem/ramulator2/pim_ramulator2.cc`; DRAM commands in `ext/ramulator2/src/ramulator/pim/` |
| Configuration | `src/python/gem5/components/memory/pim.py`, `configs/example/gem5_library/riscv-pim.py` |

The PIM memory is built only in `build/RISCV_RAMULATOR2`. The decode,
flags and cache diagnostics are in every build, so `build/RISCV`
decodes PIM instructions too, but only a PIM memory executes their
requests.

## 11. Not specified here, or open

- **Timing.** How the PIM memory turns requests into DRAM commands, the
  four all-bank activation modes (Ideal, AiM, Baseline, AESPA), the
  arbitration policies and the timing values are memory configuration,
  not instruction semantics: P4R §4 and
  `docs/pim/05b-architecture-timing-plan.md`.
- **Way 2** (PIM operands in OS huge pages on a non-uniform memory
  access (NUMA) node) is specified but not implemented; this document
  assumes the linear mapping of Way 1, under which virtual offsets equal
  physical offsets.
- **`vstart`** restart of a partly executed vector PIM instruction is
  not supported; the page guard replaces it.
- **More than one register per PU** is usable only through
  `pim.rd.reg`/`pim.wr.reg`; no compute instruction names a register.
- **MX formats** and the COSM arbitration policy are pending
  (`docs/pim/pim-mx-format-support.md`, D7.5).
