# The PIM library

The user-level library for the Processing-in-Memory (PIM) instructions of
this RISC-V-only gem5 tree. Its plan is `docs/pim/pim-library-plan.md`
("P-LIB"); this file says how to build and use it.

## Layers

| Layer | Header | Built as |
|---|---|---|
| Layout constants of one PIM memory configuration | `include/gem5/pim/pim_layout.h`, selecting a generated `pim_layout_<std>.h` | Header only |
| Address encoder and decoder, six policies | `include/gem5/pim/pim_addr.h` | Header only |
| PIM instruction wrappers, fence, cache flush | `include/gem5/pim/pim_isa.h` | Header only |
| Runtime: initialization, region, row allocator, registers, status codes, copy, coherence, number formats | `include/gem5/pim/pim_runtime.h` | `libpim_<std>.a` |
| User application programming interface (API): tensors, GEMV, add, multiply | `include/gem5/pim/pim.h` | `libpim_<std>.a` |

Running programs in gem5 (steps L7-L9 of P-LIB section 11) needs the
simulator's PIM support first.

## Layout headers

`gen_pim_layout.py` writes the constants of one configuration from the
Ramulator 2.1 organization in `ext/ramulator2/python` (it writes nothing
there). Every constant is a line `#define PIM_LAYOUT_<NAME> <number>`,
the form gem5's start-up check reads.

The four committed default headers, one per standard, are:

| Standard | Preset | Channels | PIM memory | Row | Base SE / FS |
|---|---|---|---|---|---|
| `ddr5` | `DDR5_32Gb_x4`, 2 ranks | 2 | 128 GiB | 8 KiB | `0x2000000000` / `0x4000000000` |
| `lpddr6` (default) | `LPDDR6_16Gb_x12` | 8 | 16 GiB | 2 KiB | `0x400000000` / `0x800000000` |
| `gddr7` | `GDDR7_64Gb_x8` | 8 | 16 GiB | 2 KiB | `0x400000000` / `0x800000000` |
| `hbm4` | `HBM4_32Gb_16Hi` | 8 | 16 GiB | 1 KiB | `0x400000000` / `0x800000000` |

All use the default policy `RoRaBaChCo` (the channel interleave is one
row), one register, bfloat16 operands and an fp32 accumulator, and an
operating-system (OS) memory of the same size as the PIM memory.

```sh
util/pim/gen_pim_layout.py --defaults     # rewrite the four headers
util/pim/gen_pim_layout.py --check        # exit 1 if one is out of date
util/pim/gen_pim_layout.py --standard lpddr6 --policy RoBaRaCoCh \
    --interleave 64 --describe            # print the address bit map
util/pim/gen_pim_layout.py --standard lpddr6 --policy MOP4CLXOR \
    -o my_layout.h                        # a non-default configuration
```

Options: `--standard`, `--channels`, `--policy` (`RoRaBaChCo`,
`RoRaBaCoCh`, `RoCoRaBaCh`, `RoBaRaCoCh`, `ChRaBaRoCo`, `MOP4CLXOR`),
`--interleave` (bytes; fixed by `RoRaBaChCo` and `ChRaBaRoCo`, one cache
line by default for the others), `--num-regs`, `--num-fmt`, `--acc-fmt`,
`--os-size`.

A program selects a header with `-DPIM_STD=<std>` (default `lpddr6`), or
a generated one with `-DPIM_LAYOUT_HEADER='"my_layout.h"'`, and includes
`<gem5/pim/pim.h>`. `pim_layout.h` checks the constants at compile time.

## Building

The build follows `util/m5`: SCons, one directory per variant under
`util/pim/build/`.

```sh
# RISC-V archives, one per default layout:
#   util/pim/build/riscv/out/libpim_<std>.a
scons -C util/pim riscv.CROSS_COMPILE=riscv64-linux-gnu- build/riscv

# Host build with the reference backend, and the host tests:
#   util/pim/build/native/libpim_ref_<std>.a, results in
#   util/pim/build/native/test/result/
scons -C util/pim build/native/test

# One more archive, libpim_custom.a, for a generated header:
scons -C util/pim riscv.CROSS_COMPILE=riscv64-linux-gnu- \
    PIM_LAYOUT=$PWD/my_layout.h build/riscv
```

The default `riscv.CROSS_COMPILE` is `riscv64-unknown-linux-gnu-`, as in
`util/m5`. The RISC-V objects are built with `-march=rv64gcv_zicbom
-mabi=lp64d`.

## Linking a program

Build the program with the same layout as the archive it links, and link
statically:

```sh
riscv64-linux-gnu-gcc -static -O2 -march=rv64gcv_zicbom -mabi=lp64d \
    -I$GEM5_ROOT/include -DPIM_STD=lpddr6 prog.c \
    -L$GEM5_ROOT/util/pim/build/riscv/out -lpim_lpddr6 -o prog
```

Each archive defines a symbol named after its layout (`GEM5_PIM_LAYOUT_ID`
in the layout header) that the library's entry points refer to, so a
program built for one layout does not link against another layout's
archive.

## Addresses

`pim_addr.h` turns DRAM coordinates (`pim_coord_t`: channel, unit = rank
or pseudo channel, stack ID, bank group, bank, row, column) into an
offset in the PIM memory and back, under the six policies. It is
arithmetic only, so gem5's C++ can include it too.

- `pim_geom_encode(g, c)`, `pim_geom_decode(g, off, c)` take a
  `pim_geom_t`; `pim_encode(c)`, `pim_decode(off, c)` use the included
  layout's constants, folded into shifts and masks.
- `pim_geom_contig_bytes(g)` (`PIM_LAYOUT_CONTIG_BYTES`) is the largest
  aligned piece of a bank row with consecutive columns;
  `pim_geom_piece_offset(g, c, i)` is the offset of piece i of the bank
  row of `c`.
- `pim_geom_slice_contiguous(g)` says whether row slice r (row r of
  every bank) is the one range `pim_geom_slice_offset(g, r)` +
  `pim_geom_slice_bytes(g)`.

The tests compare the encoder with reference decoders
(`tests/ref_decode.py`) that transcribe gem5's `decodePacket` and
Ramulator 2.1's mappers, for every standard, policy, channel count and
interleave.

## Instructions

`pim_isa.h` has one inline function per PIM instruction, named after the
assembly name with `_` for `.` (`pim.ab.mac` is `pim_ab_mac()`):

| Form | Functions | Signature |
|---|---|---|
| Scalar (custom-1) | `pim_sb_ewadd`, `pim_sb_ewmul`, `pim_sb_ewadd_gbuf`, `pim_sb_ewmul_gbuf`, `pim_sb_mac` | `void f(void *addr, uint64_t desc)`, `desc = PIM_DESC(ncols)` (0 = to the end of the row); `addr` 8-byte aligned |
| Vector, no data (custom-2) | `pim_sb_mov_bkgbuf`, `pim_sb_mov_gbufbk`, `pim_ab_ewadd`, `pim_ab_ewmul`, `pim_ab_ewadd_gbuf`, `pim_ab_ewmul_gbuf`, `pim_ab_mac`, `pim_ab_mov_regbk`, `pim_ab_mov_bkreg` | `size_t f(void *addr, size_t nbytes)` |
| Global buffer and registers, host memory | `pim_wr_gb_mem`, `pim_wr_reg_mem`, `pim_rd_gb_mem`, `pim_rd_reg_mem` | `size_t f(addr, buf, nbytes)` |
| Global buffer and registers, vector registers (RISC-V only) | `pim_wr_gb`, `pim_wr_reg`, `pim_rd_gb`, `pim_rd_reg` | `vuint8m8_t` data |

- Every vector form issues its own `vsetvli` at e8, m8 in the same
  inline assembly statement, sets `vl` = min(`nbytes`, VLMAX) and returns
  it; the caller loops over pieces and keeps each piece inside one
  4 KiB page and one `PIM_LAYOUT_CONTIG_BYTES` piece.
- `pim_fence()` is `fence rw, rw`; `pim_cbo_flush(p)` is `cbo.flush`
  (encoded with `.insn`, so `-march` needs no `zicbom`);
  `pim_flush_range(p, n)` flushes every line of a range.
- `-DPIM_DEBUG` adds checks that abort on misuse; `-DPIM_BACKEND_REF`
  replaces the instructions by calls into the host model.
- In `.S` files the same instructions are assembler macros, for example
  `pim_sb_mac a2, 0, a1` and `pim_ab_mac v8, a1, PIM_E64`.

`tests/test_encoding.sh` checks every word against the table of
`docs/pim/02-isa-plan.md` section 1.2.

## Reference backend

Built with `-DPIM_BACKEND_REF` (the `build/native` variant), the wrappers
call a C model of the PIM commands (`src/ref/`) instead of issuing
instructions: the PIM memory is a host byte array, and the model keeps
the global buffers, registers and accumulators
(`docs/pim/04-memctrl-plan.md` section 9). It gives golden results on
the host. `tests/vectors/` holds hand-computed cases in a text format
that the simulator side's Python reference will also read
(`tests/vectors/README.md`); `test_model --dump` prints the C model's
results for that cross-check.

## Runtime

```c
pim_status_t s = pim_init(NULL);      /* or a pim_config_t */
pim_buf_t *b;
s = pim_buf_alloc(&b, 8);             /* 8 row slices */
pim_coord_t c = {ch, unit, sid, bg, bank, slice, col};
void *p = pim_buf_addr(b, &c);        /* address of those coordinates */
pim_ab_mac(p, nbytes);
pim_buf_free(b);
pim_finalize();
```

- `pim_init()` maps the whole PIM memory linearly (Way 1): on RISC-V,
  `/dev/mem` at `WINDOW_BASE_SE` or `WINDOW_BASE_FS`; on the host, an
  anonymous mapping, or the file `PIM_REF_FILE`. The mode is
  `pim_config_t.mode`, else `PIM_MODE` (`se` or `fs`), else SE. It is
  refused (`PIM_E_MODE`) when the node name contradicts it (gem5's SE
  mode reports `sim.gem5.org`), and in FS when `/proc/iomem` shows System
  RAM in the PIM memory's range or hides the addresses (not root).
- A buffer is a set of whole row slices (one row index in every bank of
  every channel), allocated from rows 0 to `REG_ROW0` - 1; inside it a
  coordinate's row is the slice index. `pim_reg_addr()`,
  `pim_acc_addr()` and `pim_gb_addr()` address the register window and
  the global buffers.
- Every function returns a `pim_status_t`; `pim_strerror()` and
  `pim_error_detail()` explain it. `scons ... PIM_DEBUG=1` (and
  `-DPIM_DEBUG` in the program) adds misuse checks that abort: an
  address outside every buffer, a range across a contiguous piece or a
  4 KiB page, an unaligned scalar address, a register index of
  `NUM_REGS` or more, reserved descriptor bits.
- Host-only settings of the reference backend: `PIM_REF_FILE`,
  `PIM_REF_VLENB` (default: the column bytes), `PIM_REF_ERRORS=count`.

## Copy and coherence

- `pim_write_bank_row(b, c, off, src, n)` and `pim_read_bank_row()` copy
  bytes of one bank row with normal stores and loads, one
  address-contiguous piece (`PIM_LAYOUT_CONTIG_BYTES`) at a time, at the
  encoder's addresses; `pim_write_bank_row_f32()` and
  `pim_read_bank_row_f32()` convert fp32 to and from `NUM_FMT` (round to
  nearest even). The DRAM content, which is what the PIM units see, is
  the same under every policy.
- Before PIM commands, `pim_sync_to_device(b)` flushes every cache line
  of every row of the buffer and fences; after them,
  `pim_sync_from_device(b)` fences, flushes and fences again before the
  host reads results with normal loads. `pim_flush_rows()`,
  `pim_flush_unit_row()` and `pim_flush_bank_row()` flush less, without
  a fence, for code that batches its flushes. Each line is flushed once.
- The global buffer and the registers, like bank rows, are reached in
  pieces: the next piece of a unit's GB is at `pim_gb_addr(ch, unit,
  col)` for its first column, not at the next address.

## User API

```c
#include <gem5/pim/pim.h>

pim_init(NULL);
pim_tensor_t *A, *x, *y, *a, *b, *c;
pim_tensor_create(&A, PIM_MATRIX_AB, 1024, 1024, PIM_BF16);
pim_tensor_create(&x, PIM_VECTOR_GB, 1, 1024, PIM_BF16);
pim_tensor_create(&y, PIM_VECTOR_SB, 1, 1024, PIM_FP32);
pim_copy_to(A, a_host);                  /* converts, places, flushes */
pim_copy_to(x, x_host);
pim_gemv(y, A, x);                       /* y = A x */
pim_copy_from(y, y_host);

pim_tensor_create_pair(&a, &b, PIM_VECTOR_PAIRED, 1, n, PIM_BF16);
pim_tensor_create(&c, PIM_VECTOR_PAIRED, 1, n, PIM_BF16);
pim_copy_to(a, a_host);
pim_copy_to(b, b_host);
pim_add(c, a, b);                        /* or pim_mul */
pim_copy_from(c, c_host);
pim_finalize();
```

| Kind | Layout | Used by |
|---|---|---|
| `PIM_MATRIX_AB` | Matrix row m in bank m % B of unit (m % (U B)) / B (U units, B banks per unit), column tiles of one bank row, each tile of a row group in its own row slice | `pim_gemv` (`pim.ab.mac`) |
| `PIM_VECTOR_GB` | Held on the host; written into every unit's global buffer (GB) by the operation | `pim_gemv` |
| `PIM_VECTOR_PAIRED` | Bank rows in the paired banks of every unit: a single tensor in the even bank 2k, the two tensors of `pim_tensor_create_pair()` in banks 2k and 2k + 1 of the same rows | `pim_add`, `pim_mul` (`pim.ab.ewadd`, `pim.ab.ewmul`) |
| `PIM_VECTOR_SB` | Bank 0 of unit 0 (a pair: banks 0 and 1) | `pim_add`, `pim_mul` (`pim.sb.ewadd`, `pim.sb.ewmul`); the result of `pim_gemv` |

- `pim_add(c, a, b)` and `pim_mul(c, a, b)` need `a` and `b` from one
  `pim_tensor_create_pair()` (in that order) and `c` from
  `pim_tensor_create()`; the results go from the registers into `c`'s
  rows with `pim.ab.mov.regbk`, which writes every bank of `c`'s row
  slices.
- `pim_tensor_locate(t, row, col, &coord, &byte)` says where an element
  is: its DRAM coordinates (absolute row) and byte.
- Operands are of `PIM_LAYOUT_NUM_FMT` (bf16 by default); a GEMV result
  of `PIM_LAYOUT_ACC_FMT` (fp32). Host data are fp32.
- The library flushes and fences as the data move: `pim_copy_to()`
  flushes what it wrote, the operations flush rows the host wrote
  without flushing, and `pim_copy_from()` drops stale cached copies of
  rows that PIM commands wrote.

## Timing tables

`timing/pim_timing.py` resolves the PIM timing values of the four PIM
standards from gem5's derived Ramulator 2.1 standards
(`ext/ramulator2/python/ramulator/dram/pim.py` and `*_pim.py`) and
writes one committed table per standard, `timing/timing_<std>.json`, a
JavaScript Object Notation (JSON) file for host tools that cannot import
gem5 or Ramulator 2.1 (format 1, `docs/pim/05b-architecture-timing-plan.md`
section 4.3): the presets, clock, geometry, command cycles, every host
value, the host overrides with their reasons, the 18 PIM parameters with
their rules and sources, the model's per-standard terms and the
unloaded latency U per request class for one, two and a row of columns.

```sh
util/pim/timing/pim_timing.py --write        # after a descriptor change
util/pim/timing/pim_timing.py --check        # exit 1 if a table is stale
util/pim/timing/pim_timing.py --show lpddr6  # U per class and size
```

The same file is the analytic model of U (P5b section 5): from a
request's first command to its completion on an idle unit with every
bank closed, without refresh or host traffic, in DRAM clocks. gem5's U
(`src/mem/ramulator2/pim/pim_unloaded.cc`) adds `derived.issue_offset`,
the clocks from acceptance to the first command. From Python:

```python
import pim_timing as pt                    # sys.path: util/pim/timing
t = pt.load("LPDDR6")
pt.unloaded(t, "ab_rd", "baseline", 64)    # classes: pt.CLASSES
pt.act_sequence(t)                         # per-bank activations, flat order
pt.with_values(t, host={"nCCDL": 13})      # PIM defaults follow the host
```

`timing/sweep.py` writes the sensitivity report `timing/sensitivity.md`:
one host value at a time raised by 25%, the change of U for P5b section
8's requests (`--write`, `--check`). `timing/tests/test_timing.py` and
`timing/tests/test_sweep.py` run with the host tests, as does
`timing/tests/test_presets.py` (the final values of P5b section 3.5 on
the derived Ramulator 2.1 standards, and the stock standards unchanged).
