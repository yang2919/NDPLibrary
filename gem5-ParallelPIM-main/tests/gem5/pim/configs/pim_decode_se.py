"""SE decode test of the processing-in-memory (PIM) instructions.

Runs tests/test-progs/pim-decode (build it with
``make -C tests/test-progs/pim-decode/src``) on one RISC-V core in SE
mode. ``tests/gem5/pim/check_exec_trace.py`` checks the trace written
with ``--debug-flags=ExecEnable,ExecUser,ExecMacro,ExecMicro,ExecOpClass,
ExecEffAddr,ExecFlags,Decode,Cache,PIM``.

The memory is gem5's native ``SingleChannelDDR3_1600``: before a PIM
memory serves them, PIM requests reach it as plain uncacheable reads and
writes, so the test checks decoding and the requests' shape, never memory
contents. Atomic runs have no caches; the other CPU models have private
L1 and L2 caches.

Example::

    build/RISCV/gem5.opt --debug-flags=... --debug-file=trace.out \\
        tests/gem5/pim/configs/pim_decode_se.py --cpu o3 --vlen 512 \\
        --mode all
"""

import argparse
from pathlib import Path

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.no_cache import NoCache
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory import SingleChannelDDR3_1600
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator

default_binary = (
    Path(__file__).resolve().parents[4]
    / "tests/test-progs/pim-decode/bin/riscv/linux/pim-decode"
)
cpu_types = {
    "atomic": CPUTypes.ATOMIC,
    "timing": CPUTypes.TIMING,
    "minor": CPUTypes.MINOR,
    "o3": CPUTypes.O3,
}

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--cpu", choices=cpu_types.keys(), default="atomic")
parser.add_argument("--vlen", type=int, default=256, help="VLEN in bits")
parser.add_argument(
    "--mode",
    default="all",
    help="the program's mode: scalar, vstore, vload, all, 'bad <n>' or "
    "'guard <n>'",
)
parser.add_argument("--binary", type=Path, default=default_binary)
args = parser.parse_args()

if not args.binary.is_file():
    parser.error(
        f"{args.binary} not found; build it with "
        "`make -C tests/test-progs/pim-decode/src`"
    )

processor = SimpleProcessor(
    cpu_type=cpu_types[args.cpu], isa=ISA.RISCV, num_cores=1
)
for core in processor.get_cores():
    core.core.isa[0].vlen = args.vlen
    core.core.isa[0].elen = 64

if args.cpu == "atomic":
    cache_hierarchy = NoCache()
else:
    cache_hierarchy = PrivateL1PrivateL2CacheHierarchy(
        l1d_size="32KiB", l1i_size="32KiB", l2_size="512KiB"
    )
board = SimpleBoard(
    clk_freq="1GHz",
    processor=processor,
    memory=SingleChannelDDR3_1600(),
    cache_hierarchy=cache_hierarchy,
)
# TimingSimpleCPU handles accesses of at most one cache line
# (src/cpu/simple/timing.cc), so VLENB may not exceed it.
if args.cpu == "timing" and args.vlen // 8 > board.get_cache_line_size():
    parser.error(
        f"--cpu timing supports VLEN up to "
        f"{board.get_cache_line_size() * 8} (one cache line per register)"
    )

board.set_se_binary_workload(
    BinaryResource(str(args.binary)), arguments=args.mode.split()
)
simulator = Simulator(board=board)
simulator.run()
print(
    f"Exiting @ tick {simulator.get_current_tick()} because "
    f"{simulator.get_last_exit_event_cause()}"
)
