#!/bin/sh
# Cross-check builds:
# libpim's test_model.c linked against gem5's PIM functional model
# (shim.cc) and against the C reference model, and geom_c, a sampler of
# pim_addr.h's pim_geom_*(). Run from the repository root:
#   tests/gem5/pim/xmodel/build.sh <out dir> [<libsoftfloat.a>]
# The SoftFloat archive defaults to the one of build/RISCV_RAMULATOR2.
set -e
O=$1
SF=${2:-build/RISCV_RAMULATOR2/ext/softfloat/libsoftfloat.a}
X=tests/gem5/pim/xmodel
F="-O2 -ffp-contract=off -DPIM_BACKEND_REF -DPIM_STD=lpddr6 -D_DEFAULT_SOURCE"
F="$F -Iinclude -Ilibpim/src -Ilibpim/tests"
mkdir -p "$O"
gcc -std=c11 $F -c libpim/tests/test_model.c -o "$O/test_model.o"
for f in pim_numfmt pim_func; do
    g++ -std=c++17 -O2 -ffp-contract=off -Isrc -Iext/softfloat \
        -c src/mem/ramulator2/pim/$f.cc -o "$O/$f.o"
done
g++ -std=c++17 $F -Isrc -Iext/softfloat -Wno-unused-function \
    -c $X/shim.cc -o "$O/shim.o"
gcc -std=c11 $F -c libpim/src/layout.c -o "$O/layout.o"
g++ -o "$O/test_model_gem5" "$O/test_model.o" "$O/shim.o" \
    "$O/pim_numfmt.o" "$O/pim_func.o" "$O/layout.o" "$SF" -lm
gcc -std=c11 $F -c libpim/src/ref/model.c -o "$O/model.o"
gcc -std=c11 $F -c libpim/src/ref/numfmt.c -o "$O/numfmt.o"
gcc -o "$O/test_model_c" "$O/test_model.o" "$O/model.o" "$O/numfmt.o" \
    "$O/layout.o" -lm
gcc -std=c11 -O2 -Iinclude -o "$O/geom_c" $X/geom_c.c
