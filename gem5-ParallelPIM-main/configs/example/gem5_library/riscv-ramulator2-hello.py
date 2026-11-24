"""Run a local RISC-V binary in SE mode on a Ramulator 2.1 memory.

This is the Ramulator 2.1 smoke test. It needs a gem5 built with Ramulator
2.1 (the Kconfig option USE_RAMULATOR2), for example:

```
scons build/RISCV_RAMULATOR2/gem5.opt -j$(nproc)
build/RISCV_RAMULATOR2/gem5.opt \
    configs/example/gem5_library/riscv-ramulator2-hello.py \
    --binary build_logs/smoke/hello-riscv --std DDR4 --channels 2
```

The board is a 3 GHz SimpleBoard with one core and private 32 KiB L1 and
256 KiB L2 caches. Ramulator 2.1's statistics are written to
board.memory.ramulator2.ramulator_stats.yaml in the output directory. The
binary is a local file, so nothing is downloaded.
"""

import argparse

from m5.defines import buildEnv
from m5.util import (
    fatal,
    warn,
)

if not buildEnv.get("USE_RAMULATOR2"):
    fatal(
        "This gem5 binary is built without Ramulator 2.1. Build "
        "build/RISCV_RAMULATOR2/gem5.opt (build_opts/RISCV_RAMULATOR2) "
        "instead."
    )

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory.ramulator2 import (
    STANDARDS,
    ramulator2_memory,
)
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator

parser = argparse.ArgumentParser(
    description="Run a local RISC-V binary on a Ramulator 2.1 memory."
)
parser.add_argument(
    "--binary", required=True, help="Path of a static RISC-V binary."
)
parser.add_argument(
    "--std", default="DDR4", choices=STANDARDS, help="DRAM standard."
)
parser.add_argument(
    "--channels", type=int, default=2, help="Number of channels."
)
parser.add_argument("--size", default="2GiB", help="Memory size.")
parser.add_argument(
    "--cpu",
    default="TIMING",
    choices=[t.name for t in CPUTypes],
    help="CPU type.",
)
args = parser.parse_args()

cpu_type = CPUTypes[args.cpu]
if cpu_type in (CPUTypes.ATOMIC, CPUTypes.KVM):
    warn(
        "An Atomic CPU does not use Ramulator 2.1's timing: atomic accesses "
        "return a fixed latency and Ramulator 2.1 stays idle."
    )

board = SimpleBoard(
    clk_freq="3GHz",
    processor=SimpleProcessor(cpu_type=cpu_type, isa=ISA.RISCV, num_cores=1),
    memory=ramulator2_memory(args.std, args.channels, args.size),
    cache_hierarchy=PrivateL1PrivateL2CacheHierarchy(
        l1d_size="32KiB", l1i_size="32KiB", l2_size="256KiB"
    ),
)
board.set_se_binary_workload(BinaryResource(local_path=args.binary))

simulator = Simulator(board=board)
simulator.run()
print(
    f"Exiting @ tick {simulator.get_current_tick()} because "
    f"{simulator.get_last_exit_event_cause()}"
)
