"""Run an LLM decode-block example (workload/llm) in SE mode on a RISC-V
PIM system memory simulated by Ramulator 2.1.

One PIM core and --rvv-cores RVV cores: the program's main thread runs on
core 0 and issues every PIM instruction; its worker threads land on cores
1 .. N in creation order. The program initializes its weights and KV cache
on --init-cpu (Atomic by default, fast) and calls hypercall 4 before the
decode step: this configuration then switches every core to --cpu and
resets the statistics. The program's regions dump one statistics block
each (workload/common/bench.h).

    build/RISCV_RAMULATOR2/gem5.opt workload/llm/configs/llm_se.py \\
        --binary workload/llm/gqa/bin/lpddr6/gqa --rvv-cores 1 \\
        --args 4 short 1

Needs build/RISCV_RAMULATOR2 (Kconfig USE_RAMULATOR2).
"""

import argparse
import sys
from pathlib import Path

import m5
from m5.defines import buildEnv
from m5.util import fatal

if not buildEnv.get("USE_RAMULATOR2"):
    fatal("Build build/RISCV_RAMULATOR2/gem5.opt (Ramulator 2.1) instead.")

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory import pim
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.processors.simple_switchable_processor import (
    SimpleSwitchableProcessor,
)
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.exit_handler import ExitHandler
from gem5.simulate.simulator import Simulator
from gem5.utils.override import overrides

TREE = Path(__file__).resolve().parents[3]
MEMORIES = {
    "lpddr6": "LPDDR6",
    "ddr5": "DDR5",
    "gddr7": "GDDR7",
    "hbm4": "HBM4",
}
CPUS = {
    "atomic": CPUTypes.ATOMIC,
    "timing": CPUTypes.TIMING,
    "minor": CPUTypes.MINOR,
    "o3": CPUTypes.O3,
}

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--binary", required=True)
parser.add_argument("--args", nargs="*", default=[])
parser.add_argument("--memory", choices=tuple(MEMORIES), default="lpddr6")
parser.add_argument("--cpu", choices=tuple(CPUS), default="o3")
parser.add_argument(
    "--init-cpu",
    choices=tuple(CPUS) + ("none",),
    default="atomic",
    help="CPU before hypercall 4; none: --cpu throughout",
)
parser.add_argument("--rvv-cores", type=int, default=1)
parser.add_argument(
    "--ab-mode",
    choices=("ideal", "aim", "baseline", "aespa"),
    default="baseline",
)
args = parser.parse_args()

std = MEMORIES[args.memory]
memory = pim.pim_memory_system(
    std,
    ab_mode=args.ab_mode,
    layout_file=str(
        TREE / "include" / "gem5" / "pim" / f"pim_layout_{args.memory}.h"
    ),
)
cores = 1 + args.rvv_cores
if args.init_cpu != "none":
    processor = SimpleSwitchableProcessor(
        starting_core_type=CPUS[args.init_cpu],
        switch_core_type=CPUS[args.cpu],
        isa=ISA.RISCV,
        num_cores=cores,
    )
    all_cores = list(processor._all_cores())
else:
    processor = SimpleProcessor(
        cpu_type=CPUS[args.cpu], isa=ISA.RISCV, num_cores=cores
    )
    all_cores = processor.get_cores()
# VLEN as riscv-pim.py: one DRAM column (256 bits on LPDDR6).
vlen = 8 * memory.pim_memory.get_device().col_bytes
for core in all_cores:
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
    BinaryResource(local_path=str(Path(args.binary).resolve())),
    arguments=args.args,
    stdout_file=Path("program.out"),
)


class SwitchToDetailed(ExitHandler, hypercall_num=4):
    @overrides(ExitHandler)
    def _process(self, simulator: "Simulator") -> None:
        if args.init_cpu != "none":
            simulator.switch_processor()
        m5.stats.reset()
        print(
            f"LLM: decode step starts on {args.cpu} at tick {m5.curTick()}",
            flush=True,
        )

    @overrides(ExitHandler)
    def _exit_simulation(self) -> bool:
        return False


simulator = Simulator(board=board)
simulator.run()
code = simulator.get_last_exit_event_code()
print(
    f"Exiting @ tick {simulator.get_current_tick()} because "
    f"{simulator.get_last_exit_event_cause()} (code {code}, VLEN {vlen})"
)
sys.exit(code)
