"""SE request-path test of the processing-in-memory (PIM) instructions.

Runs tests/test-progs/pim-path (build it with ``make -C
tests/test-progs/pim-path/src``) in SE mode on gem5's native memory. The board
has an operating-system memory and a separate window memory, the stand-in for
the PIM memory: it is hidden from the SE frame pool (``conf_table_reported =
False``, as the PIM memory in Way 1), starts at a multiple of its size, and
every SE process gets a ``PimMemDriver`` over it, which the program maps
through ``/dev/mem``. Before a PIM memory serves them, PIM requests reach
native memory as plain uncacheable reads and writes.

The hook also sets, as the PIM memory's does, the dirty-read knob of every
cache and, unless --no-exclude, the window as every prefetcher's excluded
range. ``tests/gem5/pim/check_request_path.py`` checks a run traced with
``--debug-flags=PIM,Cache,MemCtrl,ExecEnable,ExecMicro,ExecEffAddr``.

With ``--memory lpddr6|ddr5|gddr7|hbm4`` the board has a ``PimSystemMemory`` of
that standard's default layout instead (``gem5.components.memory.pim``; needs
``build/RISCV_RAMULATOR2``), whose own hook sets the knob, the prefetchers'
excluded range and the driver. The program then maps the whole PIM memory, and
the kernels' register forms go to the first register row (``REG_ROW0 <<
ROW_SHIFT`` from ``include/gem5/pim/pim_layout_<std>.h``).

Example::

    build/RISCV/gem5.opt tests/gem5/pim/configs/pim_path_se.py \\
        --cpu o3 --vlen 512 --mode all
"""

import argparse
from pathlib import Path
from typing import (
    List,
    Sequence,
    Tuple,
)

from m5.objects import (
    AddrRange,
    BaseCache,
    BasePrefetcher,
    PimMemDriver,
    Port,
    Process,
)
from m5.util.convert import toMemorySize

from gem5.components.boards.abstract_board import AbstractBoard
from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.no_cache import NoCache
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory import SingleChannelDDR3_1600
from gem5.components.memory.abstract_memory_system import (
    AbstractMemorySystem,
)
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator
from gem5.utils.override import overrides

default_binary = (
    Path(__file__).resolve().parents[4]
    / "tests/test-progs/pim-path/bin/riscv/linux/pim-path"
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
parser.add_argument("--cores", type=int, default=1)
parser.add_argument(
    "--caches",
    action="store_true",
    help="private L1/L2 caches with an atomic CPU too (they are always on "
    "for the other CPU models)",
)
parser.add_argument("--mode", default="all", help="the program's mode")
parser.add_argument("--os-size", default="512MiB")
parser.add_argument("--window-size", default="64MiB")
parser.add_argument(
    "--dirty-read-warn",
    action="store_true",
    help="pim_dirty_read_fatal = False on every cache",
)
parser.add_argument(
    "--no-exclude",
    action="store_true",
    help="leave the prefetchers' excluded_ranges empty",
)
parser.add_argument("--binary", type=Path, default=default_binary)
parser.add_argument(
    "--memory",
    choices=("native", "lpddr6", "ddr5", "gddr7", "hbm4"),
    default="native",
    help="native: a hidden native window memory; otherwise a "
    "PimSystemMemory of that standard",
)
args = parser.parse_args()

if not args.binary.is_file():
    parser.error(
        f"{args.binary} not found; build it with "
        "`make -C tests/test-progs/pim-path/src`"
    )


class WindowedMemory(AbstractMemorySystem):
    """An OS memory and a hidden window memory above it (Way 1)."""

    def __init__(self, os_size: str, window_size: str) -> None:
        super().__init__()
        self.os_memory = SingleChannelDDR3_1600(size=os_size)
        self.window_memory = SingleChannelDDR3_1600(size=window_size)
        for mem in self.window_memory.get_mem_interfaces():
            mem.conf_table_reported = False
        self._window = None

    def window(self) -> AddrRange:
        return self._window

    @overrides(AbstractMemorySystem)
    def incorporate_memory(self, board: AbstractBoard) -> None:
        self.os_memory.incorporate_memory(board)
        self.window_memory.incorporate_memory(board)

    @overrides(AbstractMemorySystem)
    def get_mem_ports(self) -> Sequence[Tuple[AddrRange, Port]]:
        return list(self.os_memory.get_mem_ports()) + list(
            self.window_memory.get_mem_ports()
        )

    @overrides(AbstractMemorySystem)
    def get_memory_controllers(self) -> List:
        return (
            self.os_memory.get_memory_controllers()
            + self.window_memory.get_memory_controllers()
        )

    @overrides(AbstractMemorySystem)
    def get_mem_interfaces(self) -> List:
        return (
            self.os_memory.get_mem_interfaces()
            + self.window_memory.get_mem_interfaces()
        )

    @overrides(AbstractMemorySystem)
    def get_size(self) -> int:
        return self.os_memory.get_size()

    @overrides(AbstractMemorySystem)
    def set_memory_range(self, ranges: List[AddrRange]) -> None:
        self.os_memory.set_memory_range(ranges)
        size = self.window_memory.get_size()
        end = max(int(r.end) for r in ranges)
        base = -(-end // size) * size
        self._window = AddrRange(base, size=size)
        self.window_memory.set_memory_range([self._window])

    @overrides(AbstractMemorySystem)
    def get_uninterleaved_range(self) -> List[AddrRange]:
        return self.os_memory.get_uninterleaved_range()

    def _pre_instantiate(self, root) -> None:
        window = self._window
        for obj in root.board.descendants():
            if isinstance(obj, BaseCache):
                obj.pim_dirty_read_fatal = not args.dirty_read_warn
            elif isinstance(obj, BasePrefetcher) and not args.no_exclude:
                obj.excluded_ranges = [window]
            elif isinstance(obj, Process):
                obj.drivers = [
                    PimMemDriver(window=window, alignment=window.size())
                ]


processor = SimpleProcessor(
    cpu_type=cpu_types[args.cpu], isa=ISA.RISCV, num_cores=args.cores
)
for core in processor.get_cores():
    core.core.isa[0].vlen = args.vlen
    core.core.isa[0].elen = 64

if args.cpu == "atomic" and not args.caches:
    cache_hierarchy = NoCache()
else:
    cache_hierarchy = PrivateL1PrivateL2CacheHierarchy(
        l1d_size="32KiB", l1i_size="32KiB", l2_size="512KiB"
    )


def layout_values(std: str) -> dict:
    header = (
        Path(__file__).resolve().parents[4]
        / f"include/gem5/pim/pim_layout_{std}.h"
    )
    values = {}
    for line in header.read_text().splitlines():
        f = line.split()
        if len(f) == 3 and f[0] == "#define" and f[1].startswith("PIM_"):
            try:
                values[f[1][len("PIM_LAYOUT_") :]] = int(f[2], 0)
            except ValueError:
                pass
    return values, str(header)


if args.memory == "native":
    memory = WindowedMemory(args.os_size, args.window_size)
else:
    from gem5.components.memory import pim

    class PathPimMemory(pim.PimSystemMemory):
        """The PIM system memory; --no-exclude empties the prefetchers'
        excluded ranges after its hook has set them."""

        def _pre_instantiate(self, root) -> None:
            super()._pre_instantiate(root)
            if args.no_exclude:
                for obj in root.board.descendants():
                    if isinstance(obj, BasePrefetcher):
                        obj.excluded_ranges = []

    layout, layout_file = layout_values(args.memory)
    memory = PathPimMemory(
        pim.PimMemory(
            standard=args.memory.upper(),
            layout_file=layout_file,
            dirty_read_fatal=not args.dirty_read_warn,
        ),
        os_size=args.os_size,
    )
board = SimpleBoard(
    clk_freq="1GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)
if args.cpu == "timing" and args.vlen // 8 > board.get_cache_line_size():
    parser.error(
        f"--cpu timing supports VLEN up to "
        f"{board.get_cache_line_size() * 8} (one cache line per register)"
    )
# SimpleBoard's memory starts at 0 and gets its ranges later, at
# instantiation; the window lands as WindowedMemory.set_memory_range puts
# it.
os_size = toMemorySize(args.os_size)
program_args = []
if args.memory == "native":
    window_size = toMemorySize(args.window_size)
else:
    window_size = memory.pim_memory.get_size()
    program_args = [hex(layout["REG_ROW0"] << layout["ROW_SHIFT"])]
window = AddrRange(-(-os_size // window_size) * window_size, size=window_size)
board.set_se_binary_workload(
    BinaryResource(str(args.binary)),
    arguments=[hex(int(window.start)), hex(window.size()), args.mode]
    + program_args,
)
print(f"PIM window {int(window.start):#x} size {window.size():#x}")
simulator = Simulator(board=board)
simulator.run()
print(
    f"Exiting @ tick {simulator.get_current_tick()} because "
    f"{simulator.get_last_exit_event_cause()}"
)
