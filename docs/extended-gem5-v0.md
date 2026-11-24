# V0: libpim build and FP16 profile integration

Update: the subsequent step built and statically inspected both RISC-V ELF files
using a dedicated Docker cross compiler. See [target smoke verification](extended-gem5-smoke-verification.md)
for the artifacts and user-run SE/FS procedure. The original V0 observations
below describe the native-only checkpoint; simulator profile agreement is still pending.

The default worker build now uses `gem5-ParallelPIM-main/libpim`. Outputs are
isolated under `build/extended-gem5`; the previous snapshot and worker under
`build/pim-kernel-worker-fp16` remain available as the old reference baseline.

## Generate and build

Run from the NDP repository root:

```sh
make -C tools/gem5/pim_kernel profile
make -C tools/gem5/pim_kernel manifest
python3 tools/gem5/pim_kernel/test_fp16.py \
    build/extended-gem5/pim-kernel-worker-fp16
```

`profile` explicitly generates `build/extended-gem5/lpddr6_fp16.h` using the
supplier generator. It selects LPDDR6, 16 channels, `RoRaBaChCo`, FP16 operands
and FP32 accumulation. The resulting profile has 16 banks per channel, 2 KiB
bank rows, 32 GiB PIM capacity and one register. Native and target builds consume
this same generated header, including its supplier layout-link identity.

`manifest` builds the native reference worker and writes a JSON file beside it.
It records profile constants, compiler identity, artifact SHA-256 hashes and
the relevant libpim/NDP source hashes. It is a worker build manifest, not a hash
of the entire simulator delivery. Simulator binary/source identity and actual
CPU, VLEN, cache, memory and kernel settings must be recorded at simulation time.

To use a different generated profile, pass `PIM_LAYOUT=/absolute/path/header.h`
to `fp16`, `manifest` or `riscv-fp16`. Those targets consume the file without
regenerating it. `profile` is an explicit regeneration command and overwrites
its selected output. Missing profiles fail with instructions instead of silently
falling back to the supplier BF16 default.

The old `lpddr6_fp16_native.h` is retained only for explicit legacy use. It
overrides a default header and is not suitable as a simulator-readable generated
profile. Do not use it for the new target build.

## RISC-V build handoff

On a machine with a Linux RISC-V C/C++ cross toolchain and SCons:

```sh
make -C tools/gem5/pim_kernel riscv-fp16 \
    CROSS_COMPILE=riscv64-linux-gnu-
```

This builds the supplier custom archive and a static worker at
`build/extended-gem5/riscv/pim-kernel-worker-fp16`, then writes its manifest.
The target uses `-march=rv64gcv_zicbom -mabi=lp64d`, C++20 and
`-ffp-contract=off`, with `NDP_PIM_TARGET_FP16` and without `PIM_BACKEND_REF`.
Its compile-time guard rejects a non-LPDDR6 or non-FP16/FP32 layout. The supplier
archive provides an additional link-time layout identity guard.

Use a toolchain with static Linux C++ libraries; a freestanding RISC-V compiler
alone does not build this Linux worker. Before simulation, inspect the resulting
ELF attributes/disassembly for actual ISA requirements, including any compiler
generated half-precision operations. These checks remain pending until the
target executable is available.

The eventual simulator invocation must use this exact header with
`--layout`, `--memory lpddr6 --channels 16 --policy RoRaBaChCo
--num-fmt fp16 --acc-fmt fp32`. The config validates header constants against
its memory configuration. This configuration check has not been executed in V0.

## Observed native results (2026-11-17)

- Profile generation and new supplier native archive/worker build succeeded.
- `test_fp16.py` passed: FP16-specific inputs, FP32 accumulation, resident result
  rounding/chaining, repeated input updates and rejection of a BF16 client.
- The existing `ndp_gem5_pim_tests` executable passed against the new worker.
- The native manifest was generated successfully.

The Mac has no `riscv64-linux-gnu-g++` on PATH. No cross compiler was installed,
and no gem5 build, SE simulation or FS boot was started. Therefore V0's native
portion is complete; target compilation, ELF inspection and simulator profile
agreement remain open. Passing native tests does not prove target instruction
execution, cache maintenance or timing behavior.

For the next stage, retain the generated header and manifest, produce the target
ELF on the chosen Linux build environment, then execute the small V1/V2 cases in
[the validation plan](extended-gem5-validation-plan.md). FS execution remains a
user/gem5-team verification step.
