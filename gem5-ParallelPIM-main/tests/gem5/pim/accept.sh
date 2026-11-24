#!/bin/bash
# The acceptance of the processing-in-memory (PIM) system.
#
#   tests/gem5/pim/accept.sh quick|full|release <out>
#
# Runs the items of the profile, one log per item under <out>, and prints
# one "PASS <item>" or "FAIL <item>" line per item (also in
# <out>/summary.txt); exits with 1 when an item fails.
#
#   quick    A1 (incremental builds), A3 (smoke tests, import, unit
#            tests), A4 (the PIM library's host tests and cross build), A5
#            (static check, negative runs), A6 (cpuless, config): about 30
#            minutes
#   full     quick, with every build variant, and A2 (no Ramulator 2.1 in
#            build/RISCV and build/ALL), A4 (Ramulator 2.1's suites), A5,
#            A6, A7 (the library in SE), A11 (TestLib quick): about 1.5
#            hours
#   release  full, and A9, A10, A12 (FS Linux runs): about 10 hours, the
#            full Ubuntu boot of A10 the longest part
#
# Which profile a change needs: under src/mem/ramulator2, ext/ramulator2 or
# the stdlib PIM component, full; under src/arch/riscv, src/mem/cache,
# src/sim, the board or the guest kernel, release; under libpim only, A4
# and A7 (full).
#
# The FS items need the rebuilt kernel and image of util/pim-kernel
# (PIM_KERNEL_WORK, default ~/pim-kernel). The drivers build the PIM
# library's shared objects of generated layouts in the source tree, so
# they run one after the other here; each runs its own cases in parallel
# (JOBS, default the number of cores). A restore of a boot checkpoint fills
# its guest memory in host memory (64 GiB, 256 GiB for fs-ddr5), so FS
# restores run two at a time (PIM_FS_PARALLEL).

set -u
PROFILE=${1:-}
OUT=${2:-}
case "$PROFILE" in
quick | full | release) ;;
*)
    echo "usage: $0 quick|full|release <out>" >&2
    exit 2
    ;;
esac
[ -n "$OUT" ] || {
    echo "usage: $0 quick|full|release <out>" >&2
    exit 2
}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
cd "$ROOT" || exit 2
JOBS=${JOBS:-$(nproc)}
export PIM_FS_PARALLEL=${PIM_FS_PARALLEL:-2}
SUMMARY=$OUT/summary.txt
: > "$SUMMARY"
FAILED=0
P4R="python3 tests/gem5/pim/run_p4r.py"
R2=build/RISCV_RAMULATOR2
HELLO=build_logs/smoke/hello-riscv

# Expected counts (they change only with new tests).
UNIT_ALL=${UNIT_ALL:-1000}
UNIT_R2=${UNIT_R2:-1039}
HOST_TESTS=${HOST_TESTS:-97}

verdict() {
    # verdict <item> <ok 0|1> <detail>
    local item=$1 ok=$2 detail=$3
    if [ "$ok" = 0 ]; then
        echo "PASS $item: $detail" | tee -a "$SUMMARY"
    else
        echo "FAIL $item: $detail" | tee -a "$SUMMARY"
        FAILED=1
    fi
}

p4r() {
    # p4r <item> <out name> <run_p4r options...>: one driver run, PASS when
    # it reports no FAIL and no SKIP.
    local item=$1 name=$2
    shift 2
    $P4R --out "$OUT/$name" -j "$JOBS" "$@" > "$OUT/$name.log" 2>&1
    local rc=$?
    local line
    line=$(grep -h '^run_p4r:' "$OUT/$name/summary.txt" 2>/dev/null | tail -1)
    if [ $rc = 0 ] && [ -n "$line" ] && ! echo "$line" | grep -q ' [1-9][0-9]* SKIP'; then
        verdict "$item" 0 "$name: $line"
    else
        verdict "$item" 1 "$name: rc $rc, ${line:-no summary} ($OUT/$name/summary.txt)"
    fi
}

# A1: builds; no warning in RISCV and ALL.
if [ "$PROFILE" = quick ]; then
    TARGETS="build/RISCV/gem5.opt build/ALL/gem5.opt $R2/gem5.opt"
else
    TARGETS="build/RISCV/gem5.opt build/RISCV/gem5.fast build/ALL/gem5.opt $R2/gem5.opt $R2/gem5.fast $R2/gem5.debug"
fi
bad=""
for t in $TARGETS; do
    log=$OUT/build_$(echo "$t" | tr / _).log
    scons "$t" -j"$JOBS" > "$log" 2>&1 || bad="$bad $t"
    case $t in
    build/RISCV/* | build/ALL/*)
        grep -q 'warning:' "$log" && bad="$bad $t(warnings)"
        ;;
    esac
done
verdict A1-builds "$([ -z "$bad" ] && echo 0 || echo 1)" "${bad:-$TARGETS}"

# A3: smoke tests, import, unit tests.
mkdir -p "$(dirname "$HELLO")"
[ -x "$HELLO" ] || riscv64-linux-gnu-gcc -static -O2 -o "$HELLO" \
    tests/test-progs/hello/src/hello.c
smoke=$(build/RISCV/gem5.opt -d "$OUT/smoke" \
    configs/deprecated/example/se.py --cmd="$HELLO" 2>&1)
# The smoke runs must print "Hello world!" and end normally. Their ticks
# depend on the compiler that built hello-riscv (and, in SE mode, on the
# binary's path, which lands on the guest stack, so the Ramulator 2.1
# runs use a fixed path); they are reported against the reference ticks
# of the development host, not judged.
ok=1
echo "$smoke" | grep -q 'Hello world!' &&
    echo "$smoke" | grep -q 'because exiting with last active thread context' &&
    ok=0
tick=$(echo "$smoke" | grep -o 'Exiting @ tick [0-9]*' | grep -o '[0-9]*$')
verdict A3-smoke $ok "tick ${tick:-none} (reference 67284500)"
r2=""
note=""
R2HELLO=/tmp/gem5-pim-smoke/hello-riscv
mkdir -p "$(dirname "$R2HELLO")" && cp "$HELLO" "$R2HELLO"
for spec in "DDR4 84488760" "LPDDR6 100224342"; do
    set -- $spec
    o=$($R2/gem5.opt -d "$OUT/smoke-r2-$1" \
        configs/example/gem5_library/riscv-ramulator2-hello.py \
        --binary "$R2HELLO" --std "$1" --channels 2 2>&1)
    echo "$o" | grep -q 'Hello world!' &&
        echo "$o" | grep -q 'because exiting with last active thread context' ||
        r2="$r2 $1"
    t=$(echo "$o" | grep -o 'Exiting @ tick [0-9]*' | grep -o '[0-9]*$')
    note="$note $1 ${t:-none} (reference $2);"
done
verdict A3-ramulator2-smoke "$([ -z "$r2" ] && echo 0 || echo 1)" \
    "${note# }${r2:+ failed:$r2}"
build/ALL/gem5.opt -c "import m5.objects" > "$OUT/import.log" 2>&1
verdict A3-import $? "import m5.objects on build/ALL"
unit() {
    # unit <tag> <expected>
    local log=$OUT/unittests-$1.log
    scons "build/$1/unittests.opt" -j"$JOBS" > "$log" 2>&1
    local rc=$?
    local n f
    n=$(grep -o '\[  PASSED  \] [0-9]* test' "$log" | awk '{s += $4} END {print s + 0}')
    f=$(grep -c '\[  FAILED  \]' "$log")
    verdict "A3-unittests-$1" "$([ $rc = 0 ] && [ "$f" = 0 ] && [ "$n" = "$2" ] && echo 0 || echo 1)" \
        "$n passed (expected $2), $f failed"
}
unit ALL "$UNIT_ALL"
unit RISCV_RAMULATOR2 "$UNIT_R2"

# A4: the PIM library's host tests and cross build.
rm -rf libpim/build/native/test/result
scons -C libpim build/native/test -j"$JOBS" > "$OUT/host.log" 2>&1
rc=$?
n=$(ls libpim/build/native/test/result 2>/dev/null | wc -l)
verdict A4-host "$([ $rc = 0 ] && [ "$n" = "$HOST_TESTS" ] && echo 0 || echo 1)" \
    "$n tests (expected $HOST_TESTS), scons rc $rc"
scons -C libpim riscv.CROSS_COMPILE=riscv64-linux-gnu- build/riscv \
    -j"$JOBS" > "$OUT/cross.log" 2>&1
verdict A4-cross $? "test_encoding, test_lto_link, wrong-layout link"

if [ "$PROFILE" = quick ]; then
    # A5 and A6, their quick part.
    p4r A5-decode-static quick-decode --groups decode \
        --only '^decode-(static|neg)'
    p4r A6-cpuless-config quick-a6 --groups cpuless,config
else
    p4r A2-r6 a2 --groups r6
    p4r A4-ramulator2-suites a4 --groups r2suites
    p4r A5-request-path a5 --groups decode,fsbm,path,cache,se-misc
    p4r A6-memory-side a6 --groups cpuless,refresh,c28,config,v16,xcheck
    # A7: the library in SE (after the run_p4r groups: shared objects).
    python3 tests/gem5/pim/run_lib_se.py -j "$JOBS" --work "$OUT/a7" \
        > "$OUT/a7.log" 2>&1
    rc=$?
    verdict A7-library-se "$rc" "$(grep -h 'passed' "$OUT/a7.log" | tail -1)"
    # A11: TestLib quick on build/ALL; the known failure is pyunit-tests
    # (its resource tests fail upstream as well).
    # pydot (the config_output_files tests) is in the local venv.
    VENV_SITE=$(ls -d "$ROOT"/.venv/lib/python3*/site-packages 2>/dev/null | head -1)
    (cd tests && PYTHONPATH=${VENV_SITE:-} ./main.py run --isa ALL \
        --length quick --variant opt --skip-build -j"$JOBS" -t"$JOBS") \
        > "$OUT/testlib.log" 2>&1
    res=$(grep -o 'Results: [0-9]* Passed, [0-9]* Failed' "$OUT/testlib.log" | tail -1)
    nfail=$(echo "$res" | awk '{print $4}')
    other=$(grep -E '^Test: .* Failed' "$OUT/testlib.log" | grep -v 'pyunit-tests' | head -3)
    verdict A11-testlib-quick \
        "$([ -n "$res" ] && [ "${nfail:-9}" -le 1 ] && [ -z "$other" ] && echo 0 || echo 1)" \
        "${res:-no result} (allowed: pyunit-tests)${other:+; also $other}"
fi

if [ "$PROFILE" = release ]; then
    # A10's full Ubuntu boot: Atomic, systemd, Way 2,
    # the rebuilt kernel, alongside the other FS items.
    W=${PIM_KERNEL_WORK:-$HOME/pim-kernel}
    cat > "$OUT/numa-boot.sh" << 'EOF'
#!/bin/bash
sudo_() { echo 12345 | sudo -S -p '' "$@"; }
echo "NUMA-BOOT begin $(uname -a)"
ls /sys/devices/system/node/
numactl -H
sudo_ sh -c 'echo 8 > /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages'
echo "node1 nr_hugepages $(cat /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages)"
echo "--- module errors"
sudo_ dmesg | grep -i -E 'disagrees|invalid module|unknown symbol|module verification'
echo "NUMA-BOOT end"
EOF
    ($R2/gem5.fast -re -d "$OUT/numa-boot" \
        configs/example/gem5_library/riscv-pim.py --mode fs --way 2 \
        --cpu atomic --kernel "$W/vmlinux" \
        --disk-image "$W/riscv-ubuntu-24.04-img-pim" \
        --fs-script "$OUT/numa-boot.sh" > "$OUT/numa-boot.log" 2>&1
     echo $? > "$OUT/numa-boot.rc") &
    BOOT=$!
    # A9, A10, A12; fs-ddr5 last, alone (256 GiB per restore).
    p4r A9-fs-way1 a9 --groups fs-way1,fs-way1-whole,fs-protocol
    p4r A10-fs-way2 a10 --groups fs-way2
    p4r A12-sv48 a12 --groups sv48
    p4r A9-boot a9boot --groups fs-boot
    p4r A9-ddr5 a9ddr5 --groups fs-ddr5
    wait $BOOT
    con=$OUT/numa-boot/board.platform.terminal
    ok=1
    if [ "$(cat "$OUT/numa-boot.rc")" = 0 ] &&
        grep -a -q 'NUMA-BOOT end' "$con" &&
        grep -a -q 'Initmem setup node 1' "$con" &&
        grep -a -q 'node1 nr_hugepages 8' "$con" &&
        ! sed -n '/--- module errors/,/NUMA-BOOT end/p' "$con" |
            grep -a -v -e '--- module errors' -e 'NUMA-BOOT end' | grep -q .; then
        ok=0
    fi
    verdict A10-numa-boot $ok "two nodes, 8 huge pages on node 1, no module error ($con)"
fi

echo "accept.sh $PROFILE: $(grep -c '^PASS' "$SUMMARY") PASS, $(grep -c '^FAIL' "$SUMMARY") FAIL" | tee -a "$SUMMARY"
exit $FAILED
