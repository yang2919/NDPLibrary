"""System-call emulation (SE) harness of the processing-in-memory (PIM)
system memory: a program without PIM instructions must give the same
results on a PimSystemMemory as on its operating-system memory alone
(the PIM memory sees no request); also checks V12, V14 and V15.

A 3 GHz SimpleBoard, one core, private 32 KiB L1 and 256 KiB L2 caches,
on
- --memory pim: a PimSystemMemory (gem5.components.memory.pim), the
  standard's factory with --policy and --interleave;
- --memory os: the same configuration's operating-system memory alone
  (pim_os_memory), the reference.

Run it with build/RISCV_RAMULATOR2/gem5.opt from the tree root, with fixed
binary paths (SE puts the binary's path on the guest stack).
"""

import argparse

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

p = argparse.ArgumentParser()
p.add_argument("--binary", required=True)
p.add_argument("--args", nargs="*", default=[], help="program arguments")
p.add_argument("--memory", choices=("pim", "os"), default="pim")
p.add_argument("--std", default="LPDDR6", choices=pim.PIM_STANDARDS)
p.add_argument("--channels", type=int, default=None)
p.add_argument("--policy", default="RoRaBaChCo", choices=pim.POLICIES)
p.add_argument("--interleave", type=int, default=None)
p.add_argument("--os-size", default=None)
p.add_argument("--layout-file", default=None)
p.add_argument("--dirty-read-warn", action="store_true")
p.add_argument("--cpu", default="TIMING")
p.add_argument("--vlen", type=int, default=None, help="vector length, bits")
a = p.parse_args()

kw = dict(
    channels=a.channels,
    policy=a.policy,
    channel_interleave=a.interleave,
    layout_file=a.layout_file,
    dirty_read_fatal=not a.dirty_read_warn,
)
if a.memory == "pim":
    memory = pim.pim_memory_system(a.std, os_size=a.os_size, **kw)
else:
    memory = pim.pim_os_memory(
        pim.PimMemory(standard=a.std, **kw), size=a.os_size
    )

processor = SimpleProcessor(
    cpu_type=CPUTypes[a.cpu], isa=ISA.RISCV, num_cores=1
)
if a.vlen:
    for core in processor.get_cores():
        core.core.isa[0].vlen = a.vlen
board = SimpleBoard(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=PrivateL1PrivateL2CacheHierarchy(
        l1d_size="32KiB", l1i_size="32KiB", l2_size="256KiB"
    ),
)
board.set_se_binary_workload(
    BinaryResource(local_path=a.binary), arguments=a.args
)
simulator = Simulator(board=board)
simulator.run()
print(
    f"Exiting @ tick {simulator.get_current_tick()} because "
    f"{simulator.get_last_exit_event_cause()}"
)
