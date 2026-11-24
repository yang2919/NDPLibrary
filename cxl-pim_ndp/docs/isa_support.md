# RVV Instruction Support

## Supported Instructions

| Category | Notes |
|---|---|
| Integer arithmetic, logical, shift, compare, min/max | |
| Integer widening, narrowing, sign and zero extension | |
| Integer multiply and multiply-add | |
| Integer reductions | |
| Fixed-point saturating and averaging arithmetic | |
| Floating-point arithmetic, fp32 and fp16 | Scalar `Zfh` on CVA6, no vector fp64 |
| fp8 (E5M2) arithmetic, floating-point instructions at SEW 8 | Selected by SEW 8 in `vtype` |
| fp8 to fp16 widening multiply-add, `vfwmacc.vv` / `.vf` at SEW 8 | |
| bf16 widening multiply-add, `vfwmaccbf16.vv` / `.vf` (Zvfbfwma) | The only bf16 instructions decoded |
| Floating-point reductions | Except `vfredmax` and `vfredmin` |
| Mask operations | |
| Permutation, slide and compress | |
| Unit-stride, strided and indexed load/store | |
| Indexed segment load/store (`vluxseg`, `vsuxseg`) | |
| `vfexp`, `vfrcp` on the custom-1 opcode | KETI extension |

## Unsupported Instructions

| Instruction | Behavior | Alternative |
|---|---|---|
| `vlxei`, `vsxei` | Not encodable, RVV 0.x opcodes | `vluxei` / `vsuxei` |
| `vfwcvtbf16`, `vfncvtbf16` (Zvfbfmin) | Illegal instruction, not decoded | Integer shift by 16 bits, truncating |
| Vector fp64 | Illegal instruction, not in the FPU format set | Scalar fp64 on CVA6 |

## Identified Issues

| Instruction | Behavior | Workaround | Observed on | RTL simulation |
|---|---|---|---|---|
| `vfredmax`, `vfredmin` | Returns the scalar seed | Chunked `vfmax` / `vfmin` with a scalar compare | VHK158 board | Pass |
| `vrgather` | Wrong result | `vluxei` where the access pattern allows | VPK180 board, 8 lanes | Pass |
| `vsseg`, `vssseg` | Hangs | Indexed `vsuxseg` | VPK180 board, 8 lanes | Pass |
| `viota` directly after `vid` | Wrong result | Another vector instruction between the two | VPK180 board, 8 lanes | Fail, upstream Ara mask unit |

## Vector Length (`vsetvl`)

`vsetvl(n)` returns `min(n, VLMAX)` with VLMAX = VLEN × LMUL / SEW, giving VLMAX 64 for `e32,m1` and 128 for `e16,m1` at VLEN 2048.
A single `vsetvl(n)` followed by an `n`-element operation processes only VLMAX elements when `n` exceeds VLMAX, and the kernel library in `sw/src/common` therefore uses the loop below.

```c
for (size_t i = 0; i < n; ) {
    size_t vl = __riscv_vsetvl_e32m1(n - i);
    /* process vl elements */
    i += vl;
}
```

## Verification

| Test | Platform |
|---|---|
| 16 kernels, `gather_bench` 31/31, `spm_diag`, `fp8_diag`, `bf16_diag`, `fp8_widen_diag`, `lsu_bench`, roofline 62 records in DRAM and in the scratchpad (bit-exact) | VHK158 board, release bitstream |
| 22 vector programs and 92 CVA6 scalar ISA tests (`rv64ui`, `um`, `uc`, `uf`, `ud`, `si`), 0 fail | Verilator, 2 lanes, VLEN 2048, release RTL in the simulation SoC (`ara_soc.sv`) |
| 67 integer ISA tests, 0 fail | Verilator, 2 lanes, VLEN 2048, RTL of 2026-08-14 |
| 8 reduction and 7 mask ISA tests with a directed seed-alignment test | Verilator, 8 lanes, earlier RTL |
| 182 `rv64uv` programs, 178 pass | VPK180 board, 8 lanes, VLEN 8192, earlier RTL |
