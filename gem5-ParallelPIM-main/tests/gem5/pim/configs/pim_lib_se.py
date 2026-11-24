"""SE runs of the PIM library's programs on a PimSystemMemory.

Runs a program built against the PIM library (tests/test-progs/pim-gemv,
tests/test-progs/pim-eltwise) in system-call emulation (SE) mode on the board
of ``pim_se.py``: a 3 GHz ``SimpleBoard``, one core, private 32 KiB L1 and 256
KiB L2 caches, and a ``PimSystemMemory`` (``gem5.components.memory.pim``): the
standard's factory with --policy and --interleave, its operating-system memory
of the same size. The program maps the PIM memory through ``/dev/mem``
(``PimMemDriver``).

--layout-file is required: it is the layout header the program and its
``libpim_<std>.a`` were built with, and ``PimRamulator2`` compares every
``PIM_LAYOUT_`` constant with the memory at start-up. The vector length
defaults to 8 x the header's column bytes. The program's standard output goes
to ``program.out`` in the output directory; gem5 exits with the program's exit
status.

Run it with ``build/RISCV_RAMULATOR2/gem5.opt``. Example::

    build/RISCV_RAMULATOR2/gem5.opt -d m5out \\
        tests/gem5/pim/configs/pim_lib_se.py \\
        --std LPDDR6 --layout-file include/gem5/pim/pim_layout_lpddr6.h \\
        --binary tests/test-progs/pim-gemv/bin/riscv/lpddr6/pim-gemv
"""

import argparse
import sys
from pathlib import Path

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory import pim
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("--binary", required=True)
p.add_argument("--args", nargs="*", default=[], help="program arguments")
p.add_argument("--std", default="LPDDR6", choices=pim.PIM_STANDARDS)
p.add_argument("--channels", type=int, default=None)
p.add_argument("--policy", default=pim.DEFAULT_POLICY, choices=pim.POLICIES)
p.add_argument("--interleave", type=int, default=None)
p.add_argument("--layout-file", required=True)
p.add_argument(
    "--cpu", default="TIMING", choices=("ATOMIC", "TIMING", "MINOR", "O3")
)
p.add_argument(
    "--prefetcher",
    choices=("on", "none", "excluded-off"),
    default="on",
    help="the caches' prefetchers: with the PIM memory excluded (default), "
    "removed, or without the exclusion",
)
p.add_argument(
    "--vlen",
    type=int,
    default=None,
    help="vector length in bits (default: 8 x the column bytes)",
)
a = p.parse_args()


def set_prefetchers(memory, setting):
    """Set the prefetchers: "on" keeps the stdlib caches' prefetchers
    with the PIM memory as an excluded range (the default), "none" removes
    every cache's prefetcher, "excluded-off" keeps them without the
    excluded range. Applied after the PIM memory's hook, which sets the
    range."""
    if setting == "on":
        return
    from m5.objects import (
        BaseCache,
        BasePrefetcher,
    )
    from m5.params import NULL

    hook = memory._pre_instantiate

    def pre_instantiate(root):
        hook(root)
        for obj in list(root.board.descendants()):
            if setting == "none" and isinstance(obj, BaseCache):
                obj.prefetcher = NULL
            elif setting == "excluded-off" and isinstance(obj, BasePrefetcher):
                obj.excluded_ranges = []

    memory._pre_instantiate = pre_instantiate


layout = dict(entry.split("=", 1) for entry in pim.read_layout(a.layout_file))
vlen = a.vlen or 8 * int(layout["COL_BYTES"], 0)

memory = pim.pim_memory_system(
    a.std,
    channels=a.channels,
    policy=a.policy,
    channel_interleave=a.interleave,
    layout_file=a.layout_file,
)
set_prefetchers(memory, a.prefetcher)
processor = SimpleProcessor(
    cpu_type=CPUTypes[a.cpu], isa=ISA.RISCV, num_cores=1
)
for core in processor.get_cores():
    core.core.isa[0].vlen = vlen
board = SimpleBoard(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=PrivateL1PrivateL2CacheHierarchy(
        l1d_size="32KiB", l1i_size="32KiB", l2_size="256KiB"
    ),
)
board.set_se_binary_workload(
    BinaryResource(local_path=a.binary),
    arguments=a.args,
    # SE opens it relative to the output directory.
    stdout_file=Path("program.out"),
)
simulator = Simulator(board=board)
simulator.run()
code = simulator.get_last_exit_event_code()
print(
    f"Exiting @ tick {simulator.get_current_tick()} because "
    f"{simulator.get_last_exit_event_cause()} (code {code}, VLEN {vlen})"
)
sys.exit(code)
