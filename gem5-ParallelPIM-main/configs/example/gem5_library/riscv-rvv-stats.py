# Copyright (c) 2026 Jongmin Kim
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Run a small RISC-V vector (RVV) kernel in SE mode and check the statistics
of its region of interest (ROI) against values computed from the kernel.

The workload is `tests/test-progs/rvv-vadd`: a strip-mined
`c[i] = a[i] + b[i]` on N int32 elements, written in assembly. The kernel
call is wrapped in `m5_work_begin()` / `m5_work_end()`. The program starts
on the fast AtomicSimpleCPU; at `m5_work_begin` this script switches to the
CPU model chosen with `--cpu` and resets the stats, and at `m5_work_end` it
dumps and checks them. The setup and the result check outside the ROI are
still executed, but are not part of the checked stats (the first dump in
`stats.txt` is the ROI; the second one, at exit, covers the rest).

Each strip executes one `vsetvli`, two `vle32.v`, one `vadd.vv`, one
`vse32.v` and 7 scalar instructions, so the ROI statistics are known in
advance:

* `vsetvli` sets `vl = min(remaining, VLMAX)` with
  `VLMAX = LMUL * VLEN / 32`, so the loop runs `ceil(N / VLMAX)` strips.
* gem5 splits a vector load, store or arithmetic instruction into one
  micro-op per vector register it touches, `ceil(vl / (VLEN / 32))`, at
  most LMUL. `committedInstType` and `numVecInsts` count micro-ops, so
  each strip adds 2 `SimdUnitStrideLoad`, 1 `SimdUnitStrideStore` and
  1 `SimdAdd` micro-op per register, and 1 `SimdConfig` op.
* `vsetvli` is flagged as a vector instruction, so `numVecInsts` is the
  sum of those four classes. Every other `Simd*` class must stay 0.
* `numInsts` counts each vector instruction once and `numOps` counts its
  micro-ops, so `numOps - numInsts` is the number of extra micro-ops.

The workload also checks `c` and its strip count in scalar code and exits
with a non-zero status on error. The ROI counts do not depend on the CPU
model; cycle counts do and are only printed, as is the number of ROI
instructions outside the kernel (the m5op calls around it). The script
exits with status 1 if a check fails.

`--cpu timing` is limited to VLEN <= 512 (64-byte cache lines), since
TimingSimpleCPU handles memory accesses of at most one cache line.

Usage
-----

# Build gem5 and the workload (needs a riscv64-linux-gnu cross compiler).
scons build/RISCV/gem5.opt
make -C tests/test-progs/rvv-vadd/src

# Run the kernel and check the statistics.
./build/RISCV/gem5.opt configs/example/gem5_library/riscv-rvv-stats.py \
    [--cpu {atomic,timing,minor,o3}] [--vlen VLEN] [--lmul LMUL] [-n N]
"""

import argparse
import math
import sys
from pathlib import Path

import m5

from _m5 import stats as _m5_stats

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory import SingleChannelDDR3_1600
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_switchable_processor import (
    SimpleSwitchableProcessor,
)
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.exit_event import ExitEvent
from gem5.simulate.simulator import Simulator

SEW = 32  # The kernel works on int32 elements (e32).
MAX_N = 65536  # Array size in rvv_vadd.c.

default_binary = (
    Path(__file__).resolve().parents[3]
    / "tests/test-progs/rvv-vadd/bin/riscv/linux/rvv-vadd"
)

cpu_types = {
    "atomic": CPUTypes.ATOMIC,
    "timing": CPUTypes.TIMING,
    "minor": CPUTypes.MINOR,
    "o3": CPUTypes.O3,
}

parser = argparse.ArgumentParser(
    description="Run the rvv-vadd kernel and verify its vector statistics."
)
parser.add_argument("--cpu", choices=cpu_types.keys(), default="o3")
parser.add_argument(
    "--vlen", type=int, default=256, help="VLEN in bits (default: 256)."
)
parser.add_argument(
    "--lmul", type=int, choices=[1, 2, 4, 8], default=1, help="LMUL."
)
parser.add_argument(
    "-n",
    type=int,
    default=1000,
    help=f"Number of int32 elements, at most {MAX_N} (default: 1000).",
)
parser.add_argument(
    "--binary",
    type=Path,
    default=default_binary,
    help="Path to the rvv-vadd binary.",
)
args = parser.parse_args()

if not 0 <= args.n <= MAX_N:
    parser.error(f"-n must be between 0 and {MAX_N}")
if not args.binary.is_file():
    parser.error(
        f"{args.binary} not found; build it with "
        "`make -C tests/test-progs/rvv-vadd/src`"
    )


def expected_roi_stats(n, lmul, vlen):
    """Op counts of one rvv-vadd kernel call, as gem5 commits them."""
    elems_per_reg = vlen // SEW
    vlmax = lmul * elems_per_reg
    strips = 0
    uops = 0  # Micro-ops of one strip-mined vector instruction, summed.
    remaining = n
    while remaining > 0:
        vl = min(remaining, vlmax)
        strips += 1
        uops += math.ceil(vl / elems_per_reg)
        remaining -= vl
    return {
        "committedInstType::SimdConfig": strips,
        "committedInstType::SimdUnitStrideLoad": 2 * uops,
        "committedInstType::SimdUnitStrideStore": uops,
        "committedInstType::SimdAdd": uops,
        "numVecInsts": strips + 4 * uops,
        "numOps - numInsts": 4 * (uops - strips),
    }


def kernel_insts(n, lmul, vlen):
    """Instructions executed by one call of the assembly kernel."""
    vlmax = lmul * vlen // SEW
    strips = math.ceil(n / vlmax)
    # li, beqz, mv and ret, plus 12 instructions per strip.
    return 4 + 12 * strips


def stat_values(group):
    """Return the scalar and vector stats of a stat group as a flat dict.

    Vector entries are named `<stat>::<subname>`, as in `stats.txt`. The
    group is read directly because `get_simstat()` drops `commitStats0`
    for the O3 CPU, whose `commit` stage group name matches it.
    """
    values = {}
    for info in group.getStats():
        if isinstance(info, _m5_stats.VectorInfo):
            for subname, value in zip(info.subnames, info.value):
                values[f"{info.name}::{subname}"] = value
        elif isinstance(info, _m5_stats.ScalarInfo):
            values[info.name] = info.value
    return values


# Run the code before the ROI on the fast AtomicSimpleCPU, then switch.
processor = SimpleSwitchableProcessor(
    starting_core_type=CPUTypes.ATOMIC,
    switch_core_type=cpu_types[args.cpu],
    num_cores=1,
    isa=ISA.RISCV,
)
for core in processor._all_cores():
    core.core.isa[0].vlen = args.vlen
    core.core.isa[0].elen = 64

board = SimpleBoard(
    clk_freq="1GHz",
    processor=processor,
    memory=SingleChannelDDR3_1600(),
    cache_hierarchy=PrivateL1PrivateL2CacheHierarchy(
        l1d_size="32KiB", l1i_size="32KiB", l2_size="512KiB"
    ),
)
# TimingSimpleCPU handles memory accesses of at most one cache line (split
# over two lines if unaligned), so it cannot load or store a wider vector
# register (`gem5.opt` asserts, `gem5.fast` loses data).
if args.cpu == "timing" and args.vlen // 8 > board.get_cache_line_size():
    parser.error(
        f"--cpu timing supports VLEN up to "
        f"{board.get_cache_line_size() * 8} (one cache line per register)"
    )

board.set_se_binary_workload(
    BinaryResource(str(args.binary)),
    arguments=[str(args.n), str(args.lmul)],
)

roi = {}


def roi_begin():
    """`m5_work_begin`: switch to the detailed CPU and reset the stats."""
    simulator.switch_processor()
    m5.stats.reset()
    return False


def roi_end():
    """`m5_work_end`: dump the ROI stats and keep a copy to check."""
    m5.stats.dump()
    core = processor.get_cores()[0].core
    roi["commit"] = stat_values(core.getStatGroups()["commitStats0"])
    roi["numCycles"] = stat_values(core)["numCycles"]
    return False


simulator = Simulator(
    board=board,
    on_exit_event={
        ExitEvent.WORKBEGIN: roi_begin,
        ExitEvent.WORKEND: roi_end,
    },
)
simulator.run()

print()
print(
    f"rvv-vadd: cpu={args.cpu} vlen={args.vlen} lmul={args.lmul} "
    f"n={args.n}"
)
failures = 0
exit_code = simulator.get_last_exit_event_code()
print(f"  workload exit code: {exit_code}")
if exit_code != 0:
    failures += 1
if not roi:
    print("rvv-vadd stats: FAIL (the ROI was never reached)")
    sys.exit(1)

commit = roi["commit"]
actual = {
    name: value
    for name, value in commit.items()
    if name == "numVecInsts" or name.startswith("committedInstType::Simd")
}
actual["numOps - numInsts"] = commit["numOps"] - commit["numInsts"]

expected = expected_roi_stats(args.n, args.lmul, args.vlen)
# Every other vector op class must stay at 0.
for name in actual:
    expected.setdefault(name, 0)

print(f"  {'ROI statistic':<42}{'expected':>10}{'actual':>10}")
for name, value in expected.items():
    ok = actual[name] == value
    failures += not ok
    # Skip the op classes the kernel does not use, unless they mismatch.
    if value != 0 or not ok or "::" not in name:
        mark = "" if ok else "  MISMATCH"
        print(f"  {name:<42}{value:>10}{actual[name]:>10.0f}{mark}")

# Results that depend on the CPU model or the compiler, printed only.
kernel = kernel_insts(args.n, args.lmul, args.vlen)
informational = {
    "numInsts": commit["numInsts"],
    "  in the kernel": kernel,
    "  outside the kernel (m5op calls)": commit["numInsts"] - kernel,
    "numCycles": roi["numCycles"],
    "IPC": commit["numInsts"] / roi["numCycles"],
}
for name, value in informational.items():
    print(f"  {name:<42}{'':>10}{value:>10.{2 if name == 'IPC' else 0}f}")

if failures:
    print(f"rvv-vadd stats: FAIL ({failures} check(s) failed)")
    sys.exit(1)
print("rvv-vadd stats: PASS")
