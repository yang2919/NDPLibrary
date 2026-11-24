# PIM test vectors

Inputs, command lists and expected results for the PIM command
semantics (`docs/pim/04-memctrl-plan.md` section 9.4). Two models read
them: the library's C reference model (`util/pim/src/ref`, run by
`util/pim/tests/test_model.c`) and, once it exists, the simulator side's
Python reference `tests/gem5/pim/pim_ref.py` (P4R step P4R.10). Their
outputs must be equal bit for bit (P-LIB section 8, LQ6 (c)):

```sh
build/native/test/bin/test_model_lpddr6 --dump tests/vectors/*.txt > c.txt
# pim_ref.py --dump tests/vectors/*.txt > py.txt; diff c.txt py.txt
```

Without `--dump`, `test_model` checks every `expect` line; with it, it
prints each `expect` line followed by ` => ` and the actual values, in
hexadecimal.

## Format

One statement per line; `#` starts a comment. Tokens are separated by
spaces. Coordinates are `key=value` tokens, each 0 when left out:
`ch`, `unit` (rank or pseudo channel), `sid`, `bg`, `bank`, `row`, `col`,
`byte` (the byte within the column). `row=reg<j>` is register row j
(`REG_ROW0 + j`, with `REG_ROW0 = rows - num_regs - 1`) and `row=acc` the
accumulator row. Values are `<fmt> v0 v1 ...`, `fmt` one of `bf16`,
`fp16`, `fp32`, `int8`, `int16`, `int32`; a value written `0x...` is the
element's bit pattern, any other value a decimal number (use only
decimal values the format represents exactly). Elements are little
endian and consecutive in the row, from `col` and `byte` on.

| Statement | Meaning |
|---|---|
| `geom policy= channels= intlv= units= sids= bgs= banks= rows= cols= col_bytes= [num_regs=1] [num_fmt=bf16] [acc_fmt=fp32] [vlenb=col_bytes]` | Start a new PIM memory, all zero: the `pim_geom_t` of `include/gem5/pim/pim_addr.h` (policy numbers as `PIM_LAYOUT_POLICY`), the register count, the formats and the vector register bytes (VLMAX at e8, m8 is 8 × `vlenb`) |
| `errors fatal` / `errors count` | A software error stops the run (the default, as gem5's `sw_error_fatal`), or is counted and the request does nothing (loads return zeros) |
| `row <coord> <values>` | Write elements into the bank row at the encoder's addresses |
| `gb ch= unit= [col=] <values>` | Set the global buffer (GB) of a unit |
| `reg <bank coord> j= [col=] <values>` | Set register j of the PU of a bank |
| `acc <bank coord> <values>` | Set the accumulator of the PU of a bank |
| `cmd <instruction> <coord> ncols=N` | A scalar form (`pim.sb.ewadd` ... `pim.sb.mac`) at the coordinates' address; the descriptor is N (0 = to the end of the row) |
| `cmd <instruction> <coord> bytes=N` | A vector form without data, or `pim.rd.gb` / `pim.rd.reg` reading N bytes |
| `cmd <instruction> <coord> <values>` | `pim.wr.gb` / `pim.wr.reg` with these data |
| `expect row\|gb\|reg\|acc <as above> <values>` | The state equals the values |
| `expect read <values>` | The last `pim.rd.*` returned these bytes |
| `expect errors N` | N software errors so far (this geometry) |
| `expect count <instruction> N` | N requests of that instruction so far |

## Semantics the files assume

- Element-wise (EW) results are computed in fp32 and rounded to
  `num_fmt` (round to nearest even); integers wrap. A NaN narrowed to
  bf16 keeps the top 16 bits of the fp32 NaN with the quiet bit set (the
  16-bit shift of `04-memctrl-plan.md` section 9.2); narrowed to fp16
  it is the canonical NaN 0x7e00 (as SoftFloat's RISC-V NaN).
- MAC: per column command, the products of the elements in range are
  summed in `acc_fmt` sequentially in ascending element order (the first
  product, then each next one added), and the column's sum is added to
  the accumulator once.
- Vector forms cover vl = min(bytes, 8 × `vlenb`) bytes. Bank operations
  need consecutive bytes of one bank row; `pim.rd.gb` and `pim.wr.gb`
  take GB bytes from the first byte's column and byte on;
  `pim.rd.reg` and `pim.wr.reg` decode every byte through the register
  window; accumulator-row bytes past the accumulator read as zero and
  ignore writes.
