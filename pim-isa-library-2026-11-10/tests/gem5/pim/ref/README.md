# Reference outputs of the PIM library's SE programs

Processing-in-Memory (PIM) library step L7
(`docs/pim/pim-library-plan.md`, LQ13 (b)): the expected standard
output of `pim-gemv` and `pim-eltwise` (`tests/test-progs/`) for each
memory standard's default layout header
(`include/gem5/pim/pim_layout_<std>.h`).

- `<program>/<std>.args`: the program's arguments for that standard.
- `<program>/<std>.out`: its standard output.

The outputs come from the host build only: the program linked with
`libpim_ref_<std>.a` (`PIM_BACKEND_REF`, the library's reference model).
Regenerate one with:

```sh
make -C tests/test-progs/pim-gemv/src TARGET=host PIM_STD=lpddr6
tests/test-progs/pim-gemv/bin/host/lpddr6/pim-gemv \
    $(cat tests/gem5/pim/ref/pim-gemv/lpddr6.args) \
    > tests/gem5/pim/ref/pim-gemv/lpddr6.out
```

Never regenerate a file to make a gem5 run pass. `scons -C util/pim
build/native/test` checks that the host build still reproduces every
file (`test_ref_<program>_<std>`), and `tests/gem5/pim/run_lib_se.py`
checks it again for each run's layout before comparing the gem5 run's
`program.out` with it. The output does not depend on the
address-mapping policy or the interleave, so one file per standard
serves every layout of that standard.

The shapes (rows and columns for `pim-gemv`; paired and single-bank
elements for `pim-eltwise`) are a few elements past a multiple of the
standard's units, row width or column tile, so every run has a partial
tile and crosses unit boundaries.
