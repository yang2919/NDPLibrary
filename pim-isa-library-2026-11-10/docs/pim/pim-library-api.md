# Processing-in-Memory (PIM) library API reference

**Status: draft, 2026-11-10.** Written from the tree at `96ab11514d`
(branch `riscv-only`): the headers in `include/gem5/pim/` and the
sources in `util/pim/src/`. Where this text and a header comment
disagree, the code decides. Every `path:line` was checked in this tree
on that date.

**Scope.** The user-level library that programs the PIM memory from a
RISC-V program: building and linking it, its five layers, and every
public type, macro and function. The instructions it emits are specified
in the companion document `docs/pim/pim-isa-spec.md` ("the ISA spec";
ISA: instruction set architecture).
The design record is `docs/pim/pim-library-plan.md` ("P-LIB") and
decision D13 of `docs/pim/decisions.md`; build notes are in
`util/pim/README.md`.

## Terms

- Dynamic random-access memory (DRAM); operating system (OS); central
  processing unit (CPU); Syscall Emulation (SE) and Full System (FS)
  modes of gem5; application programming interface (API).
- RISC-V Vector extension terms: vector length (`vl`), the vector
  register length in bytes (VLENB), the largest `vl` of a setting
  (VLMAX), effective element width (EEW), register group multiplier
  (LMUL).
- **Column:** one DRAM transaction, `PIM_LAYOUT_COL_BYTES` bytes.
  **Bank row:** one row of one bank, `PIM_LAYOUT_ROW_BYTES` bytes.
  **Row slice:** one row index in every bank of every channel.
- **Unit:** the banks one all-bank (AB) command covers: a rank (Double
  Data Rate 5, DDR5), a pseudo channel (High Bandwidth Memory 4, HBM4)
  or a channel (Low-Power Double Data Rate 6, LPDDR6; Graphics Double
  Data Rate 7, GDDR7). A **processing unit (PU)** computes for one bank;
  each unit has one **global buffer (GB)**. Single-bank (SB),
  element-wise (EW), multiply-accumulate (MAC) and general
  matrix-vector multiplication (GEMV) are used as in the ISA spec.
- **Contiguous piece:** the largest aligned run of a bank row whose
  columns are at consecutive addresses, `PIM_LAYOUT_CONTIG_BYTES` (a
  whole row under the default policy).
- **Way 1:** the PIM memory, hidden from the OS, mapped whole and
  linearly into the program (virtual offset = physical offset). The only
  way implemented; Way 2 (OS huge pages on a non-uniform memory access
  (NUMA) node) is not.

## 1. Overview

| Layer | Header | Built as | Purpose |
|---|---|---|---|
| Layout | `include/gem5/pim/pim_layout.h` (selects one `pim_layout_<std>.h`) | header only | Constants of one PIM memory configuration, checked at compile time |
| Address | `include/gem5/pim/pim_addr.h` | header only | Encoder and decoder of DRAM coordinates, six policies |
| Instructions | `include/gem5/pim/pim_isa.h` | header only | One inline wrapper per PIM instruction; fence and cache flush |
| Runtime | `include/gem5/pim/pim_runtime.h` | `libpim_<std>.a` | Initialization, region, row-slice allocator, register and GB addresses, number formats, copies, coherence, status codes |
| User API | `include/gem5/pim/pim.h` | `libpim_<std>.a` | Tensors, `pim_gemv`, `pim_add`, `pim_mul` |

A program includes `<gem5/pim/pim.h>` (which includes the others) and
links the archive built for the same layout. Code that places data
itself uses the lower layers directly. All headers are usable from C11
and C++17 (`extern "C"`); `pim_addr.h` is also included by gem5's C++.

A layout is compiled in: every function is specialized for one
standard, channel count, policy, interleave, register count and pair of
number formats. A program built for one layout does not link against
another layout's archive (section 3.3).

## 2. Quick start

```sh
# 1. gem5 with Ramulator 2.1 (the PIM memory exists only in this build)
scons build/RISCV_RAMULATOR2/gem5.opt -j$(nproc)

# 2. The library: one RISC-V archive per default layout
scons -C util/pim riscv.CROSS_COMPILE=riscv64-linux-gnu- build/riscv
#    -> util/pim/build/riscv/out/libpim_{ddr5,lpddr6,gddr7,hbm4}.a

# 3. A program for the default LPDDR6 layout
riscv64-linux-gnu-gcc -static -O2 -march=rv64gcv_zicbom -mabi=lp64d \
    -ffp-contract=off -I include -DPIM_STD=lpddr6 prog.c \
    -L util/pim/build/riscv/out -lpim_lpddr6 -o prog

# 4. Run it in SE mode on an LPDDR6 PIM system memory
build/RISCV_RAMULATOR2/gem5.opt configs/example/gem5_library/riscv-pim.py \
    --memory lpddr6 --binary prog
```

`riscv-pim.py` writes the program's standard output to `program.out` in
the output directory and exits with the program's status
(`configs/example/gem5_library/riscv-pim.py:1-36`). Without `--binary`
it runs `tests/test-progs/pim-gemv` built for the chosen standard
(`make -C tests/test-progs/pim-gemv/src PIM_STD=lpddr6`).

A minimal `prog.c`, y = A x:

```c
#include <stdio.h>
#include <gem5/pim/pim.h>

enum { M = 37, N = 100 };
static float a[M * N], x[N], y[M];

int main(void)
{
    for (int i = 0; i < M * N; i++) a[i] = (i % 7) / 8.0f;
    for (int j = 0; j < N; j++) x[j] = (j % 5) / 4.0f;

    pim_status_t s = pim_init(NULL);
    if (s != PIM_OK) {
        fprintf(stderr, "%s: %s\n", pim_strerror(s), pim_error_detail());
        return 2;
    }
    pim_tensor_t *A, *X, *Y;
    pim_tensor_create(&A, PIM_MATRIX_AB, M, N, (pim_dtype_t)PIM_LAYOUT_NUM_FMT);
    pim_tensor_create(&X, PIM_VECTOR_GB, 1, N, (pim_dtype_t)PIM_LAYOUT_NUM_FMT);
    pim_tensor_create(&Y, PIM_VECTOR_SB, 1, M, (pim_dtype_t)PIM_LAYOUT_ACC_FMT);
    pim_copy_to(A, a);          /* fp32 -> bf16, placed, flushed */
    pim_copy_to(X, x);
    pim_gemv(Y, A, X);          /* pim.wr.reg, pim.wr.gb, pim.ab.mac, pim.rd.reg */
    pim_copy_from(Y, y);        /* fp32 accumulators -> host */
    printf("y[0] = %f\n", y[0]);
    pim_tensor_destroy(A); pim_tensor_destroy(X); pim_tensor_destroy(Y);
    pim_finalize();
    return 0;
}
```

(Check every status in real code; `tests/test-progs/pim-gemv/src/pim_gemv.c`
is the full, tested version.)

## 3. Building and linking

### 3.1 Choosing a layout

The program and the archive must be built with the same layout header.

| Selection | Compiler flag | Archive |
|---|---|---|
| A committed default (`ddr5`, `lpddr6`, `gddr7`, `hbm4`) | `-DPIM_STD=<std>` (default `lpddr6`) | `libpim_<std>.a` |
| A generated header | `-DPIM_LAYOUT_HEADER='"/abs/path/my_layout.h"'` | `libpim_custom.a`, built with `PIM_LAYOUT=/abs/path/my_layout.h` |

Source: `include/gem5/pim/pim_layout.h:7-44`. Generate a header with
`util/pim/gen_pim_layout.py` (options `--standard`, `--channels`,
`--policy`, `--interleave`, `--num-regs`, `--num-fmt`, `--acc-fmt`,
`--os-size`, `-o`; `--describe` prints the address bit map;
`util/pim/README.md`, "Layout headers").

The four defaults all use policy `RoRaBaChCo`, one register, bf16
operands and an fp32 accumulator:

| Standard | Channels | Banks per unit × units | Row | Column | PIM memory | Base SE / FS |
|---|---|---|---|---|---|---|
| `ddr5` | 2 | 32 × 2 ranks per channel | 8 KiB | 64 B | 128 GiB | `0x2000000000` / `0x4000000000` |
| `lpddr6` | 8 | 16 × 1 | 2 KiB | 32 B | 16 GiB | `0x400000000` / `0x800000000` |
| `gddr7` | 8 | 16 × 1 | 2 KiB | 32 B | 16 GiB | `0x400000000` / `0x800000000` |
| `hbm4` | 8 | 64 × 2 pseudo channels per channel | 1 KiB | 32 B | 16 GiB | `0x400000000` / `0x800000000` |

gem5 checks at start-up that the PIM memory it builds has exactly the
layout of the header the program was built with, when the configuration
passes the header (`riscv-pim.py --layout`, or the default header of
`--memory`); any differing constant stops the run naming it
(`PimRamulator2::checkExpectedLayout`,
`src/mem/ramulator2/pim_ramulator2.cc:275-310`, P4R.15).

### 3.2 Building the archives

```sh
scons -C util/pim riscv.CROSS_COMPILE=riscv64-linux-gnu- build/riscv
scons -C util/pim riscv.CROSS_COMPILE=riscv64-linux-gnu- \
    PIM_LAYOUT=$PWD/my_layout.h build/riscv          # adds libpim_custom.a
scons -C util/pim build/native/test                  # host reference build + tests
scons -C util/pim riscv.CROSS_COMPILE=... PIM_DEBUG=1 build/riscv   # debug checks
```

RISC-V objects use `-march=rv64gcv_zicbom -mabi=lp64d`; the default
`riscv.CROSS_COMPILE` is `riscv64-unknown-linux-gnu-`. Build programs
with `-ffp-contract=off` too, as the library and the test programs do,
so that host-side float code rounds as the reference does
(`tests/test-progs/pim-gemv/src/Makefile`).

### 3.3 The layout link guard

Each layout header defines `GEM5_PIM_LAYOUT_ID` as a symbol name
derived from its constants (for example `pim_layout_id_02fc103aa186`,
`include/gem5/pim/pim_layout_lpddr6.h:55`). The archive defines that
symbol; the inline `pim_init()` refers to it
(`include/gem5/pim/pim_runtime.h:199-220`). A program built for another
layout therefore fails to link, instead of computing with wrong
addresses.

### 3.4 Build-time macros

| Macro | Effect |
|---|---|
| `PIM_STD`, `PIM_LAYOUT_HEADER` | Layout selection (section 3.1) |
| `PIM_DEBUG` | The wrappers and the runtime check alignment, descriptors, 4 KiB pages, buffer rows, the register window and contiguous pieces, and `abort()` with a message on misuse (`include/gem5/pim/pim_isa.h:162-214`, `util/pim/src/regs.c:113-148`). Build the archive with `PIM_DEBUG=1` as well |
| `PIM_BACKEND_REF` | Host build: every wrapper calls the C reference model (`util/pim/src/ref/`) instead of issuing an instruction; link `libpim_ref_<std>.a` from `util/pim/build/native/` |
| `__ASSEMBLER__` (set by the compiler for `.S`) | `pim_isa.h` provides assembler macros instead of C functions |

### 3.5 Environment variables

| Variable | Read by | Meaning |
|---|---|---|
| `PIM_MODE` | `pim_init()` (`util/pim/src/init.c:194`) | `se` or `fs` (or upper case), when `pim_config_t.mode` is `PIM_MODE_DEFAULT` |
| `PIM_REF_FILE` | reference backend (`util/pim/src/region.c:31`) | Back the PIM memory with this file instead of anonymous memory |
| `PIM_REF_VLENB` | reference backend (`region.c:59`) | VLENB the model splits requests at (default: the column bytes) |
| `PIM_REF_ERRORS` | reference backend (`region.c:62`) | `count`: count software errors instead of aborting |
| `PIM_REF_COUNTS` | reference backend (`util/pim/src/ref/model.c:525`) | Write the predicted PIM request and column-command counts to this file |

## 4. Concepts

### 4.1 The region

`pim_init()` maps the whole PIM memory (`PIM_LAYOUT_WINDOW_SIZE` bytes)
with one `mmap` of `/dev/mem` at `PIM_LAYOUT_WINDOW_BASE_SE` or
`_FS`, read-write and shared (`util/pim/src/region.c:69-88`). In SE
mode gem5's `PimMemDriver` emulates `/dev/mem` for that window; in FS
mode the program needs root. Every PIM address is then
`pim_region_base() + pim_encode(&coord)`.

### 4.2 Coordinates

```c
typedef struct pim_coord {
    uint32_t ch;    /* channel */
    uint32_t unit;  /* rank or pseudo channel within the channel */
    uint32_t sid;   /* stack ID (HBM4), else 0 */
    uint32_t bg;    /* bank group, 0 if none */
    uint32_t bank;  /* bank within its bank group */
    uint32_t row;   /* row (absolute, or a slice index; see each function) */
    uint32_t col;   /* column */
} pim_coord_t;
```

(`include/gem5/pim/pim_addr.h:60-70`.) Initialize every field; the
positional initializer `{ch, unit, sid, bg, bank, row, col}` is the
order used throughout the library.

### 4.3 Buffers and row slices

PIM memory is allocated in whole row slices: a `pim_buf_t` owns n slice
indices, each one row index in every bank of every channel, from rows 0
to `PIM_LAYOUT_REG_ROW0 − 1`. The allocator takes the first run of n
consecutive free rows, else the lowest free rows
(`util/pim/src/alloc.c:34-67`). Inside a buffer, a coordinate's `row`
is the **slice index** (0 to n − 1); the library maps it to the
absolute row. The caller chooses channel, unit, bank and column, so one
buffer serves every placement and every policy.

### 4.4 Contiguous pieces and pages

Consecutive addresses stay in one bank row with consecutive columns
only within a contiguous piece (`PIM_LAYOUT_CONTIG_BYTES`, aligned).
Every vector wrapper covers at most `vl` = min(`nbytes`, VLMAX) bytes
(VLMAX = 8 × VLENB at e8, LMUL 8) and returns that count; the caller
loops, and must keep each call inside one contiguous piece and one
4 KiB page, or the PIM memory reports a software error or the core an
address-misaligned fault (ISA spec §7). The copy, flush and operation
functions of the runtime and user API do this for you.

### 4.5 Registers, accumulators and the GB

The PU registers are the register window, the top `NUM_REGS + 1` rows of
every bank (ISA spec §1.2): row `REG_ROW0 + j` is register j, row
`REG_ROW0 + NUM_REGS` the accumulator row, whose first `ACC_BYTES` bytes
are the accumulator. Only `pim.rd.reg` and `pim.wr.reg` may access these
rows; a normal load or store there is a software error. A unit's GB is
addressed by channel, unit and column; `pim_gb_addr()` gives one such
address.

### 4.6 Number formats

Host data are fp32. PIM operands are `PIM_LAYOUT_NUM_FMT`
(`PIM_LAYOUT_ELEM_BYTES` bytes, `PIM_LAYOUT_ELEMS_PER_COL` per column);
accumulators are `PIM_LAYOUT_ACC_FMT` (`PIM_LAYOUT_ACC_BYTES` bytes).
Conversions round to nearest even (RNE); integer formats round to the
nearest integer and wrap (`include/gem5/pim/pim_runtime.h:258-269`).

### 4.7 Coherence

Normal loads and stores to PIM memory are cached, PIM requests are not.
The rule (ISA spec §8.3): flush and fence before PIM commands touch rows
the host wrote; fence, flush and fence before the host reads rows PIM
commands wrote. The user API does this itself; code using the runtime
calls `pim_sync_to_device()` and `pim_sync_from_device()`, or the finer
`pim_flush_*()` plus `pim_fence()`. gem5 stops the run when a PIM
request meets a dirty line in another core's cache.

### 4.8 Status codes and threads

Every fallible function returns `pim_status_t` (section 5.2.1);
`pim_strerror()` names the code and `pim_error_detail()` describes the
last failure. The library keeps one global state and no locks: call it
from one thread, between one `pim_init()` and one `pim_finalize()`.

## 5. Reference

### 5.1 `pim.h`: tensors and operations

#### Types

```c
typedef enum pim_kind {
    PIM_MATRIX_AB = 0, PIM_VECTOR_GB = 1, PIM_VECTOR_PAIRED = 2, PIM_VECTOR_SB = 3,
} pim_kind_t;

typedef enum pim_dtype {
    PIM_BF16 = PIM_FMT_BF16, PIM_FP16 = PIM_FMT_FP16, PIM_FP32 = PIM_FMT_FP32,
    PIM_INT8 = PIM_FMT_INT8, PIM_INT16 = PIM_FMT_INT16, PIM_INT32 = PIM_FMT_INT32,
} pim_dtype_t;

typedef struct pim_tensor pim_tensor_t;   /* opaque */
```

Layouts, with U = channels × units per channel and B = banks per unit
(`util/pim/src/tensor.c:26-55`):

| Kind | Shape | Where element (m, j) or i lives | Slices | Used by |
|---|---|---|---|---|
| `PIM_MATRIX_AB` | rows × cols | Matrix row m in unit (m mod UB) / B, bank (m mod UB) mod B; column j in tile j / E (E = elements per bank row), at element j mod E; tile t of row group m / UB in slice (m / UB) × tiles + t | ceil(rows / UB) × ceil(cols / E) | `pim_gemv` (A) |
| `PIM_VECTOR_GB` | 1 × cols | Host memory inside the tensor; written into every unit's GB by the operation | none | `pim_gemv` (x) |
| `PIM_VECTOR_PAIRED` | 1 × cols | Runs of E elements spread over the UB / 2 bank pairs, then the next slice; a single tensor in the even bank 2k, the two tensors of a pair in banks 2k and 2k + 1 | ceil(cols / (UB / 2 × E)) | `pim_add`, `pim_mul` (AB) |
| `PIM_VECTOR_SB` | 1 × cols | Bank 0 of unit 0 (the second tensor of a pair: bank 1), E elements per slice | ceil(cols / E) | `pim_add`, `pim_mul` (SB); `pim_gemv` (y) |

A tensor owns whole row slices, so every bank of its slices belongs to
it, even banks it does not use (the operations may overwrite them: for
example `pim.ab.mov.regbk` writes every bank of c's slices).

#### Functions

| Function | Description | Errors |
|---|---|---|
| `pim_status_t pim_tensor_create(pim_tensor_t **t, pim_kind_t kind, size_t rows, size_t cols, pim_dtype_t dtype)` | A tensor of the kind; vectors need `rows = 1`. Allocates its slices (none for `PIM_VECTOR_GB`) | `PIM_E_STATE` before `pim_init()`; `PIM_E_ARG` zero shape, bad kind or type, vector with `rows ≠ 1`, `PIM_VECTOR_PAIRED` with fewer than 2 banks; `PIM_E_NOMEM` no slices or host memory |
| `pim_status_t pim_tensor_create_pair(pim_tensor_t **a, pim_tensor_t **b, pim_kind_t kind, size_t rows, size_t cols, pim_dtype_t dtype)` | Two tensors sharing slices: element i of `*a` in bank 2k, of `*b` in bank 2k + 1, same row and column. Kind `PIM_VECTOR_PAIRED` or `PIM_VECTOR_SB` | As above; `PIM_E_ARG` for another kind or a `NULL` pointer |
| `void pim_tensor_destroy(pim_tensor_t *t)` | Frees the tensor; the slices go back when both tensors of a pair are destroyed. `NULL` is a no-op | — |
| `pim_status_t pim_tensor_locate(const pim_tensor_t *t, size_t row, size_t col, pim_coord_t *c, uint32_t *byte)` | DRAM coordinates of an element (`c->row` absolute) and its byte within the column (`byte` may be `NULL`) | `PIM_E_ARG` out of range, or a `PIM_VECTOR_GB` tensor |
| `const pim_buf_t *pim_tensor_buf(const pim_tensor_t *t)` | The tensor's buffer, for the runtime layer; `NULL` for `PIM_VECTOR_GB` | — |
| `pim_status_t pim_copy_to(pim_tensor_t *t, const float *host)` | rows × cols fp32 values, row-major, converted to the tensor's type, stored at their encoder addresses, and flushed (flush + fence) | `PIM_E_STATE`; `PIM_E_ARG` if the type is neither `PIM_LAYOUT_NUM_FMT` nor fp32 |
| `pim_status_t pim_copy_from(const pim_tensor_t *t, float *host)` | The reverse; first drops stale cached copies (fence, flush, fence) if PIM commands wrote the tensor | As `pim_copy_to` |
| `pim_status_t pim_gemv(pim_tensor_t *y, const pim_tensor_t *A, const pim_tensor_t *x)` | y = A x. A: `PIM_MATRIX_AB` rows × cols, x: `PIM_VECTOR_GB` 1 × cols, both of `NUM_FMT`; y: `PIM_VECTOR_SB` 1 × rows of `ACC_FMT` | `PIM_E_STATE`; `PIM_E_ARG` for other kinds, shapes or types, or when `ACC_FMT` is neither fp32 nor `NUM_FMT` |
| `pim_status_t pim_add(pim_tensor_t *c, const pim_tensor_t *a, const pim_tensor_t *b)`, `pim_mul(...)` | c = a + b, c = a × b element-wise. a, b: the first and second tensor of one `pim_tensor_create_pair()`, of `NUM_FMT`; c: a tensor of the same kind, shape and type from `pim_tensor_create()` | `PIM_E_STATE`; `PIM_E_ARG` otherwise, or when `NUM_REGS ≠ 1` |

What the operations issue (`util/pim/src/ops.c:76-246`):

- **`pim_gemv`**, per row group of UB matrix rows: `pim.wr.reg` of
  zeros into every accumulator; per column tile, `pim.wr.gb` of the
  tile of x into every unit's GB and `pim.ab.mac` over the tile in every
  unit; then `pim.rd.reg` of each used accumulator, stored into y with
  normal stores. A host-written A is flushed first.
- **`pim_add`, `pim_mul` on `PIM_VECTOR_PAIRED`**, per slice and unit:
  `pim.ab.ewadd`/`pim.ab.ewmul` over the row, then `pim.ab.mov.regbk`
  into c's row.
- **On `PIM_VECTOR_SB`**, per slice: one `pim.sb.ewadd`/`pim.sb.ewmul`
  with `PIM_DESC(ncols)` on the used columns, then `pim.ab.mov.regbk`
  into c's row.

Results follow the PIM memory's arithmetic (ISA spec §1.4): with bf16
operands, GEMV results differ from a double-precision loop by bf16
rounding of the inputs (the test programs accept |error| ≤ Σ|a·x| / 128
+ 10⁻⁶, `tests/test-progs/pim-gemv/src/pim_gemv.c`).

### 5.2 `pim_runtime.h`: the runtime

#### 5.2.1 Status, configuration, start and stop

```c
typedef enum pim_status {
    PIM_OK = 0,
    PIM_E_ARG,      /* bad argument or shape */
    PIM_E_NOMEM,    /* no free row slices */
    PIM_E_DEVICE,   /* /dev/mem or the driver cannot be opened or mapped */
    PIM_E_MODE,     /* the run-time mode contradicts the system */
    PIM_E_NODE,     /* Way 2: no PIM NUMA node (not returned yet) */
    PIM_E_HUGEPAGE, /* Way 2: no free huge page (not returned yet) */
    PIM_E_STATE,    /* called before pim_init() or after pim_finalize() */
} pim_status_t;

typedef enum pim_mode { PIM_MODE_DEFAULT = 0, PIM_MODE_SE = 1, PIM_MODE_FS = 2 } pim_mode_t;

typedef struct pim_config {
    pim_mode_t mode;
    int way;            /* 1 (the only one implemented) */
    unsigned flags;     /* none yet */
    const char *device; /* NULL: /dev/mem */
} pim_config_t;
#define PIM_CONFIG_INIT {PIM_MODE_DEFAULT, 1, 0, NULL}
```

| Function | Description |
|---|---|
| `pim_status_t pim_init(const pim_config_t *cfg)` | Maps the PIM memory and starts the library; `NULL` means `PIM_CONFIG_INIT`. Mode: `cfg->mode`, else `PIM_MODE`, else SE. Refuses (`PIM_E_MODE`) a mode the system contradicts: SE requires the node name `sim.gem5.org` (gem5's SE mode), FS refuses it; FS also requires `/proc/iomem` to show System RAM addresses (root) none of which overlaps the PIM memory. Errors: `PIM_E_STATE` if already started, `PIM_E_ARG` for `way ≠ 1` or a bad mode, `PIM_E_DEVICE` if the open or `mmap` fails, `PIM_E_NOMEM` (`util/pim/src/init.c:59-235`) |
| `void pim_finalize(void)` | Unmaps the region and frees the allocator. Buffers and tensors must not be used afterwards |
| `const char *pim_strerror(pim_status_t s)` | Name of a status |
| `const char *pim_error_detail(void)` | Description of the last failure, or `""` |
| `pim_mode_t pim_mode(void)` | The chosen mode (`PIM_MODE_DEFAULT` before `pim_init()`) |
| `void *pim_region_base(void)`, `uint64_t pim_region_size(void)` | The mapping; `NULL`/0 before `pim_init()` |
| `uint64_t pim_region_phys(void)` | The PIM memory's physical base (`WINDOW_BASE_SE` or `_FS`) |

#### 5.2.2 Buffers

| Function | Description |
|---|---|
| `pim_status_t pim_buf_alloc(pim_buf_t **b, uint32_t n_slices)` | n row slices (`PIM_E_ARG` for 0, `PIM_E_NOMEM` if fewer are free, `PIM_E_STATE`) |
| `void pim_buf_free(pim_buf_t *b)` | Returns the slices |
| `uint32_t pim_buf_slices(const pim_buf_t *b)` | Slice count |
| `uint32_t pim_buf_row(const pim_buf_t *b, uint32_t slice)` | Absolute row of a slice |
| `void *pim_buf_addr(const pim_buf_t *b, const pim_coord_t *c)` | Address of the coordinates, `c->row` the slice index; `NULL` outside the buffer (with `PIM_DEBUG`: abort) |
| `uint32_t pim_free_slices(void)` | Free slices left |

#### 5.2.3 Register window and GB addresses

| Function | Description |
|---|---|
| `void *pim_reg_addr(const pim_coord_t *c, uint32_t j)` | Register j of the PU of `c`'s bank, at `c->col` (`c->row` ignored); `NULL` for j ≥ `NUM_REGS` (`util/pim/src/regs.c:75-88`) |
| `void *pim_acc_addr(const pim_coord_t *c)` | The accumulator row of `c`'s bank, at `c->col`; the accumulator is its first `ACC_BYTES` bytes at column 0 |
| `void *pim_gb_addr(uint32_t ch, uint32_t unit, uint32_t col)` | An address of the unit's GB at a column. The next piece of a GB is at `pim_gb_addr(ch, unit, next_col)`, not necessarily at the next address |

Note: the header comment at `include/gem5/pim/pim_runtime.h:249-250`
says "j = NUM_REGS gives the accumulator row"; the code returns `NULL`
for that j (`util/pim/src/regs.c:78`). Use `pim_acc_addr()`.

#### 5.2.4 Number formats

| Function | Description |
|---|---|
| `uint16_t pim_f32_to_bf16(float f)`, `float pim_bf16_to_f32(uint16_t b)` | bf16 conversions (RNE) |
| `void pim_f32_to_elem(void *dst, float f)`, `float pim_elem_to_f32(const void *src)` | One element of `NUM_FMT` |
| `float pim_acc_to_f32(const void *src)` | One accumulator of `ACC_FMT` |

#### 5.2.5 Copies (normal loads and stores)

| Function | Description |
|---|---|
| `void pim_copy_piece(void *pim, const void *host, size_t n)`, `void pim_copy_piece_out(void *host, const void *pim, size_t n)` | Byte copy of one contiguous piece (at most `CONTIG_BYTES`, inside one bank row); no flush |
| `pim_status_t pim_write_bank_row(const pim_buf_t *b, const pim_coord_t *c, size_t off, const void *src, size_t n)` | Bytes [off, off + n) of the bank row of `c` (`c->row` the slice, `c->col` ignored), piece by piece at the encoder's addresses; no flush. `PIM_E_ARG` if off + n exceeds the row, or the slice, channel, unit or bank is out of range |
| `pim_status_t pim_read_bank_row(...)` | The reverse |
| `pim_status_t pim_write_bank_row_f32(const pim_buf_t *b, const pim_coord_t *c, size_t first, const float *src, size_t n)`, `pim_read_bank_row_f32(...)` | n elements from element index `first`, converted between fp32 and `NUM_FMT` |

Because each column is written at its encoder address, the DRAM content
(what the PUs see) is the same under every policy; only the number of
pieces differs.

#### 5.2.6 Coherence

| Function | Flushes | Fence |
|---|---|---|
| `pim_flush_rows(b, first, n)` | Slices [first, first + n) of `b`, every bank and unit | no |
| `pim_flush_unit_row(b, ch, unit, slice)` | One slice in one unit (the rows of one AB command) | no |
| `pim_flush_bank_row(b, c)` | One bank row (one SB command; flush bank 2k + 1 too for an EW pair) | no |
| `pim_sync_to_device(b)` | Every row of `b` | after |
| `pim_sync_from_device(b)` | Every row of `b` | before and after |

Each cache line is flushed once, with `cbo.flush`
(`util/pim/src/sync.c:95-174`). All return `PIM_E_ARG` for slices
outside the buffer, `PIM_E_STATE` before `pim_init()`.

### 5.3 `pim_isa.h`: instruction wrappers

One `static inline` function per instruction, named after the assembly
name with `_` for `.`; each is one `asm volatile` statement and becomes
exactly that instruction (vector forms: a `vsetvli` and the
instruction; memory forms: a `vsetvli`, a vector load or store and the
instruction). They hold no policy and do no address checks unless
`PIM_DEBUG` is set.

| Form | Functions | Signature |
|---|---|---|
| Scalar SB (custom-1) | `pim_sb_ewadd`, `pim_sb_ewmul`, `pim_sb_ewadd_gbuf`, `pim_sb_ewmul_gbuf`, `pim_sb_mac` | `void f(void *addr, uint64_t desc)`; `addr` 8-byte aligned, `desc = PIM_DESC(ncols)` (0 = to the end of the row) |
| Vector, no host data (custom-2) | `pim_sb_mov_bkgbuf`, `pim_sb_mov_gbufbk`, `pim_ab_ewadd`, `pim_ab_ewmul`, `pim_ab_ewadd_gbuf`, `pim_ab_ewmul_gbuf`, `pim_ab_mac`, `pim_ab_mov_regbk`, `pim_ab_mov_bkreg` | `size_t f(void *addr, size_t nbytes)`; `vs3` is `v0`, ignored |
| GB and registers from or to host memory | `pim_wr_gb_mem`, `pim_wr_reg_mem` | `size_t f(void *addr, const void *src, size_t nbytes)` (through `v8`-`v15`) |
| | `pim_rd_gb_mem`, `pim_rd_reg_mem` | `size_t f(const void *addr, void *dst, size_t nbytes)` |
| GB and registers, vector registers (RISC-V with `__riscv_vector`) | `pim_wr_gb`, `pim_wr_reg` | `size_t f(void *addr, vuint8m8_t data, size_t nbytes)` |
| | `pim_rd_gb`, `pim_rd_reg` | `vuint8m8_t f(const void *addr, size_t nbytes, size_t *vl)` |
| Ordering | `pim_fence()` | `fence rw, rw` |
| | `pim_cbo_flush(const void *line)` | `cbo.flush`, encoded with `.insn` (no `zicbom` needed in `-march`) |
| | `pim_flush_range(const void *p, size_t n)` | `cbo.flush` of every 64-byte line of the range, no fence |

Every vector form sets `vl = min(nbytes, VLMAX)` at e8, LMUL 8, issues
the instruction on `vl` bytes and returns `vl`; loop until done (section
4.4). Clobbers: `vl`, `vtype`, `memory`, and `v8`-`v15` for the memory
forms (`include/gem5/pim/pim_isa.h:299-361`).

Constants: `PIM_E8`, `PIM_E16`, `PIM_E32`, `PIM_E64` (width codes 0, 5,
6, 7), `PIM_DESC_NCOLS_SHIFT`, `PIM_DESC_NCOLS_MASK`, `PIM_DESC(n)`,
`PIM_CACHE_LINE` (64), and `enum pim_op` (`PIM_OP_SB_EWADD` = 1 to
`PIM_OP_WR_REG` = 18).

**Assembler macros** (`.S` files, `include/gem5/pim/pim_isa.h:52-123`):

```asm
#include <gem5/pim/pim_isa.h>
    pim_sb_mac a2, 0, a1           # pim.sb.mac a2, 0(a1)
    vsetvli t0, a3, e64, m1, ta, ma
    pim_ab_mac v8, a1, PIM_E64     # pim.ab.mac v8, (a1), EEW 64
    pim_rd_reg v16, a1             # EEW 8 by default
```

### 5.4 `pim_addr.h`: addresses

```c
typedef struct pim_geom {
    uint32_t policy;          /* PIM_POLICY_* */
    uint32_t channels;
    uint64_t intlv_bytes;     /* channel interleave */
    uint32_t units;           /* ranks or pseudo channels per channel */
    uint32_t sids;            /* stack IDs per unit (1 if none) */
    uint32_t bank_groups;     /* bank groups per SID (1 if none) */
    uint32_t banks_per_group;
    uint32_t rows;            /* rows per bank */
    uint32_t cols;            /* columns per row */
    uint32_t col_bytes;
} pim_geom_t;
```

Policies: `PIM_POLICY_RoRaBaChCo` 0, `RoRaBaCoCh` 1, `RoCoRaBaCh` 2,
`RoBaRaCoCh` 3, `ChRaBaRoCo` 4, `MOP4CLXOR` 5
(`include/gem5/pim/pim_addr.h:37-43`).

| Function (geometry as argument) | With the included layout | Result |
|---|---|---|
| `uint64_t pim_geom_encode(const pim_geom_t *g, const pim_coord_t *c)` | `pim_encode(c)` | Offset of byte 0 of the coordinates' column |
| `uint32_t pim_geom_decode(const pim_geom_t *g, uint64_t off, pim_coord_t *c)` | `pim_decode(off, c)` | Coordinates of an offset; returns the byte within the column |
| `uint64_t pim_geom_contig_bytes(g)` | `PIM_LAYOUT_CONTIG_BYTES` | Size of a contiguous piece |
| `uint32_t pim_geom_pieces_per_row(g)` | — | Pieces per bank row |
| `uint64_t pim_geom_piece_offset(g, c, i)` | `pim_piece_offset(c, i)` | Offset of piece i of `c`'s bank row (`c->col` ignored) |
| `uint64_t pim_geom_row_bytes(g)`, `pim_geom_slice_bytes(g)`, `pim_geom_size(g)` | `PIM_LAYOUT_ROW_BYTES`, —, `PIM_LAYOUT_WINDOW_SIZE` | Sizes |
| `uint32_t pim_geom_banks_per_unit(g)`, `uint64_t pim_geom_bank_rows(g)` | `PIM_LAYOUT_BANKS_PER_UNIT`, — | Counts |
| `unsigned pim_geom_row_shift(g)` | `PIM_LAYOUT_ROW_SHIFT` | Bit position of row bit 0 |
| `int pim_geom_slice_contiguous(g)`, `uint64_t pim_geom_slice_offset(g, r)` | — | Whether slice r is the single range at `r << row shift` |
| — | `const pim_geom_t *pim_geom(void)` | The included layout's geometry |

The encoder is arithmetic only and `always_inline`, so with the layout's
constants it folds into shifts and masks (P-LIB L6: about 30
instructions per `pim.ab.mac` in the GEMV inner loop).

### 5.5 Layout constants

Every generated header defines these as `#define PIM_LAYOUT_<NAME>
<number>` (`include/gem5/pim/pim_layout_lpddr6.h:18-52`), checked by
`pim_layout.h:70-124`:

| Constant | Meaning |
|---|---|
| `STD`, `POLICY` | Standard (1 DDR5, 2 LPDDR6, 3 GDDR7, 4 HBM4) and policy number |
| `CHANNELS`, `INTLV_BYTES` | Channel count and interleave |
| `RANKS`, `PSEUDO_CHANNELS`, `SIDS`, `BANK_GROUPS`, `BANKS_PER_GROUP` | Organization (0 = level absent) |
| `ROWS`, `COLS`, `COL_BYTES`, `ROW_BYTES` | Rows per bank, columns per row, bytes |
| `UNITS`, `BANKS_PER_UNIT`, `PUS_PER_BANK` | Units per channel, banks per unit, 1 |
| `COL_SHIFT`, `CH_SHIFT`, `UNIT_SHIFT`, `SID_SHIFT`, `BG_SHIFT`, `BANK_SHIFT`, `ROW_SHIFT` | Bit positions of the fields |
| `CONTIG_BYTES` | Contiguous piece |
| `WINDOW_SIZE`, `WINDOW_ROW0`, `WINDOW_BASE_SE`, `WINDOW_BASE_FS` | PIM memory size, 0, physical bases |
| `NUM_REGS`, `REG_ROW0` | Registers per PU, first register row |
| `NUM_FMT`, `ACC_FMT`, `ELEM_BYTES`, `ELEMS_PER_COL`, `ACC_BYTES` | Formats (`PIM_FMT_BF16` 0 ... `PIM_FMT_INT32` 5) and sizes |

Plus `GEM5_PIM_LAYOUT_ID` (section 3.3).

## 6. Example: a dot product with the runtime layer

One bank row against the GB with `pim.sb.mac`, without the tensor API:

```c
#include <string.h>
#include <gem5/pim/pim.h>

/* r = a . x for n <= PIM_LAYOUT_ROW_BYTES / PIM_LAYOUT_ELEM_BYTES */
static pim_status_t dot(const float *a, const float *x, size_t n, float *r)
{
    enum { E = PIM_LAYOUT_ELEMS_PER_COL, ROW = PIM_LAYOUT_ROW_BYTES / PIM_LAYOUT_ELEM_BYTES };
    const size_t ncols = (n + E - 1) / E;           /* whole columns */
    float pa[ROW] = {0}, px[ROW] = {0};             /* zero padding */
    memcpy(pa, a, n * sizeof(float));
    memcpy(px, x, n * sizeof(float));

    pim_buf_t *b;
    pim_status_t s = pim_buf_alloc(&b, 1);
    if (s != PIM_OK) return s;
    pim_coord_t c = {0, 0, 0, 0, 0, 0, 0};          /* ch 0, unit 0, bank 0, slice 0 */

    /* a into the bank row with normal stores, then flush + fence. */
    pim_write_bank_row_f32(b, &c, 0, pa, ncols * E);
    pim_sync_to_device(b);

    /* x into unit 0's GB, piece by piece; reset the accumulator. */
    uint8_t gx[PIM_LAYOUT_ROW_BYTES];
    for (size_t i = 0; i < ncols * E; i++)
        pim_f32_to_elem(gx + i * PIM_LAYOUT_ELEM_BYTES, px[i]);
    const size_t bytes = ncols * PIM_LAYOUT_COL_BYTES;
    for (size_t off = 0; off < bytes;) {
        uint8_t *g = (uint8_t *)pim_gb_addr(0, 0, off / PIM_LAYOUT_COL_BYTES)
                     + off % PIM_LAYOUT_COL_BYTES;
        size_t left = PIM_LAYOUT_CONTIG_BYTES - off % PIM_LAYOUT_CONTIG_BYTES;
        off += pim_wr_gb_mem(g, gx + off, bytes - off < left ? bytes - off : left);
    }
    const uint8_t zero[PIM_LAYOUT_ACC_BYTES] = {0};
    pim_wr_reg_mem(pim_acc_addr(&c), zero, sizeof zero);

    /* One instruction: ncols MAC column commands in bank 0. */
    pim_sb_mac(pim_buf_addr(b, &c), PIM_DESC(ncols));

    uint8_t acc[PIM_LAYOUT_ACC_BYTES];
    pim_rd_reg_mem(pim_acc_addr(&c), acc, sizeof acc);
    *r = pim_acc_to_f32(acc);
    pim_buf_free(b);
    return PIM_OK;
}
```

Notes: the accumulator persists between MACs, hence the reset; the
three PIM steps need no fence between them, because PIM requests stay
in program order and the unit executes them in order (ISA spec §8);
the `pim.rd.reg` result is not cached, so no flush is needed before
reading it. Replacing `pim_sb_mac` by `pim_ab_mac(addr, bytes)` (looped
over pieces) gives one dot product per bank of the unit, one per
accumulator.

## 7. Running in gem5

`configs/example/gem5_library/riscv-pim.py` (needs
`build/RISCV_RAMULATOR2`) builds a `PimSystemMemory`: an OS memory and a
PIM memory of one standard with one policy, the PIM memory hidden from
the OS (Way 1).

| Option | Meaning |
|---|---|
| `--mode se\|fs`, `--cpu`, `--cores` | Mode, CPU model, cores |
| `--binary`, `--args` | SE program (default: `pim-gemv` of the standard's default layout) |
| `--memory lpddr6\|ddr5\|gddr7\|hbm4`, `--channels`, `--size`, `--policy`, `--channel-interleave` | PIM memory |
| `--os-standard`, `--os-channels`, `--os-size` | OS memory (default: like the PIM memory) |
| `--num-fmt`, `--acc-fmt`, `--no-values` | Number formats; timing only |
| `--ab-mode`, `--arb-policy`, `--row-policy`, `--posted`/`--no-posted` | Memory-side timing behaviour |
| `--dirty-read` | A PIM read meeting a dirty copy warns instead of stopping |
| `--layout` | The layout header the program was built with; required with any layout-changing option |
| `--vlen` | Vector length (default 8 × the column bytes) |
| `--kernel`, `--disk-image`, `--bootloader`, `--fs-exit` | FS boot |

In SE mode, an option that changes the layout (a non-default channel
count, policy, channel interleave, number formats, or any OS-memory
option, which moves the bases) needs `--layout` and a `--binary` built
against that header (`make ... PIM_LAYOUT=<header>`)
(`configs/example/gem5_library/riscv-pim.py:212-235`); the start-up
check refuses a mismatched pair.

## 8. Host reference backend

Built with `-DPIM_BACKEND_REF` and linked with
`util/pim/build/native/libpim_ref_<std>.a`, a program runs on the host:
the PIM memory is a host mapping, and a C model of the 18 commands
(`util/pim/src/ref/model.c`) keeps the GBs, registers and accumulators
with the same arithmetic as gem5's model. It gives golden outputs (the
committed references in `tests/gem5/pim/ref/`) and request-count
predictions (`PIM_REF_COUNTS`). Build a test program for it with
`make -C tests/test-progs/pim-gemv/src PIM_STD=<std> TARGET=host`.

## 9. Limitations and open items

- **Way 2 is not implemented:** `pim_init()` with `way ≠ 1` returns
  `PIM_E_ARG`; `PIM_E_NODE` and `PIM_E_HUGEPAGE` are reserved for it.
- **One thread, one session** (section 4.8).
- **EW operations need `NUM_REGS = 1`**, and `pim_gemv` needs `ACC_FMT`
  fp32 or equal to `NUM_FMT`.
- **Tensor copies convert only `NUM_FMT` and fp32;** a tensor of another
  type can be created but not copied.
- **`pim_gemv`'s result** is gathered by the host (`pim.rd.reg`, then
  normal stores into y's bank), so its cost grows with the number of
  rows.
- **No compute on a tensor slice offset:** operations start at element 0
  of each slice; finer placement uses the runtime layer.
- **The PIM compiler** that tiles tensors for the library is planned
  separately (`docs/pim/pim-compiler-plan.md`).
