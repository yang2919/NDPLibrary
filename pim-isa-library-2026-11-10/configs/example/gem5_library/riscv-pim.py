"""Run a RISC-V Processing-in-Memory (PIM) program, or boot Linux, on a
PIM system memory simulated by Ramulator 2.1.

The memory is ``gem5.components.memory.pim``'s ``PimSystemMemory``: an
operating-system (OS) memory and a PIM memory of one DRAM standard (LPDDR6,
DDR5, GDDR7 or HBM4) with one address-mapping policy. The PIM memory is
hidden from the OS ("Way 1"): a program reaches it by mapping ``/dev/mem``.

System-call emulation (SE, the default) runs a program built against the
PIM library. The default is ``pim-gemv``, a matrix-vector product checked
against plain double loops, built for the standard's default layout:

```
scons build/RISCV_RAMULATOR2/gem5.opt -j$(nproc)
scons -C util/pim riscv.CROSS_COMPILE=riscv64-linux-gnu- build/riscv
make -C tests/test-progs/pim-gemv/src PIM_STD=lpddr6
build/RISCV_RAMULATOR2/gem5.opt configs/example/gem5_library/riscv-pim.py
```

The program's standard output goes to ``program.out`` in the output
directory, and gem5 exits with the program's exit status (``pim-gemv``: 0
when every result is within bf16 tolerance). Another policy, channel
count, channel interleave or number format changes the layout: generate
its header with ``util/pim/gen_pim_layout.py``, build the program with
``PIM_LAYOUT=<header>`` and pass both with ``--layout`` and ``--binary``.

Full system (``--mode fs``) boots the Ubuntu 24.04 image of
``riscv-ubuntu-run.py`` on a ``RiscvBoard`` whose device tree lists only
the OS memory. ``--fs-exit kernel-booted`` stops once the kernel has
booted.

This needs a gem5 built with Ramulator 2.1 (``build/RISCV_RAMULATOR2``,
Kconfig ``USE_RAMULATOR2``). An Atomic CPU sees fixed latencies: a PIM
request takes its unloaded latency, a normal read the standard's
closed-page latency.
"""

import argparse
import sys
from pathlib import Path

from m5.defines import buildEnv
from m5.util import fatal

if not buildEnv.get("USE_RAMULATOR2"):
    fatal(
        "This gem5 binary is built without Ramulator 2.1. Build "
        "build/RISCV_RAMULATOR2/gem5.opt (build_opts/RISCV_RAMULATOR2) "
        "instead."
    )

from gem5.components.boards.riscv_board import RiscvBoard
from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.cachehierarchies.classic.private_l1_private_l2_walk_cache_hierarchy import (
    PrivateL1PrivateL2WalkCacheHierarchy,
)
from gem5.components.memory import pim
from gem5.components.memory.ramulator2 import STANDARDS as OS_STANDARDS
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import (
    BinaryResource,
    BootloaderResource,
    DiskImageResource,
    KernelResource,
    obtain_resource,
)
from gem5.simulate.exit_handler import (
    ExitHandler,
    KernelBootedExitHandler,
)
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
FORMATS = ("bf16", "fp16", "fp32", "int8", "int16", "int32")
# The Ubuntu 24.04 boot of riscv-ubuntu-run.py.
FS_WORKLOAD = ("riscv-ubuntu-24.04-boot", "2.0.0")

parser = argparse.ArgumentParser(
    description="Run a RISC-V PIM program, or boot Linux, on a PIM system "
    "memory (Ramulator 2.1)."
)
parser.add_argument("--mode", choices=("se", "fs"), default="se")
parser.add_argument("--cpu", choices=tuple(CPUS), default="timing")
parser.add_argument("--cores", type=int, default=1, help="Number of cores.")
parser.add_argument(
    "--binary",
    default=None,
    help="SE program (default: pim-gemv for the standard's default layout).",
)
parser.add_argument(
    "--args", nargs="*", default=[], help="Arguments of the SE program."
)
parser.add_argument(
    "--kernel", default=None, help="FS kernel: a path or a resource ID."
)
parser.add_argument(
    "--disk-image",
    default=None,
    help="FS disk image: a path (root partition 1) or a resource ID.",
)
parser.add_argument(
    "--bootloader",
    default=None,
    help="FS bootloader: a path or a resource ID.",
)
parser.add_argument(
    "--fs-exit",
    choices=("kernel-booted", "after-boot", "end"),
    default="end",
    help="Where an FS run stops: the kernel has booted, after_boot.sh has "
    "started, or the run script has ended.",
)
parser.add_argument("--memory", choices=tuple(MEMORIES), default="lpddr6")
parser.add_argument(
    "--channels",
    type=int,
    default=None,
    help="PIM memory channels (default: the standard's).",
)
parser.add_argument(
    "--size",
    default=None,
    help="PIM memory size; must be its capacity (default).",
)
parser.add_argument(
    "--policy", choices=pim.POLICIES, default=pim.DEFAULT_POLICY
)
parser.add_argument(
    "--channel-interleave",
    type=int,
    default=None,
    help="Channel interleave in bytes, where the policy leaves it open.",
)
parser.add_argument(
    "--os-standard",
    choices=OS_STANDARDS,
    default=None,
    help="OS memory standard (default: the PIM memory's).",
)
parser.add_argument("--os-channels", type=int, default=None)
parser.add_argument(
    "--os-size", default=None, help="OS memory size (default: its capacity)."
)
parser.add_argument(
    "--row-policy", choices=("Open", "ClosedCAP"), default="Open"
)
parser.add_argument(
    "--ab-mode",
    choices=("ideal", "aim", "baseline", "aespa"),
    default="baseline",
    help="How all-bank PIM operations open and close rows.",
)
parser.add_argument(
    "--arb-policy",
    choices=("frfcfs", "chopim", "asyncdimm", "cosm"),
    default="frfcfs",
    help="PIM and host arbitration (cosm stops the run: pending).",
)
parser.add_argument(
    "--posted",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Acknowledge store-like PIM requests at acceptance.",
)
parser.add_argument("--num-fmt", choices=FORMATS, default="bf16")
parser.add_argument("--acc-fmt", choices=FORMATS, default="fp32")
parser.add_argument(
    "--no-values",
    action="store_true",
    help="Do not compute PIM results (timing only; programs that check "
    "their results then fail).",
)
parser.add_argument(
    "--dirty-read",
    choices=("fatal", "warn"),
    default="fatal",
    help="A PIM read that finds a dirty copy in a peer cache.",
)
parser.add_argument(
    "--layout",
    default=None,
    help="The PIM library's layout header the program was built with "
    "(default: the standard's, include/gem5/pim/pim_layout_<std>.h).",
)
parser.add_argument(
    "--vlen",
    type=int,
    default=None,
    help="Vector length in bits (default: 8 x the standard's column).",
)
args = parser.parse_args()

std = MEMORIES[args.memory]
custom_layout = (
    (args.channels not in (None, pim.PIM_PRESETS[std]["channels"]))
    or args.policy != pim.DEFAULT_POLICY
    or args.channel_interleave is not None
    or (args.num_fmt, args.acc_fmt) != ("bf16", "fp32")
    or (args.os_standard, args.os_channels, args.os_size) != (None,) * 3
)
layout = args.layout
if layout is None and not custom_layout:
    layout = str(
        TREE / "include" / "gem5" / "pim" / f"pim_layout_{args.memory}.h"
    )
if args.mode == "se":
    if layout is None:
        parser.error(
            "these options change the PIM layout: generate its header with "
            "util/pim/gen_pim_layout.py, build the program with "
            "PIM_LAYOUT=<header>, and pass --layout and --binary"
        )
    if args.layout is not None and args.binary is None:
        parser.error(
            "--layout needs --binary: a program built with "
            "PIM_LAYOUT=<that header>"
        )
    binary = Path(
        args.binary
        or TREE
        / "tests"
        / "test-progs"
        / "pim-gemv"
        / "bin"
        / "riscv"
        / args.memory
        / "pim-gemv"
    )
    if not binary.is_file():
        parser.error(
            f"no program {binary}; build it with: make -C "
            f"tests/test-progs/pim-gemv/src PIM_STD={args.memory}"
        )

memory = pim.pim_memory_system(
    std,
    os_standard=args.os_standard,
    os_channels=args.os_channels,
    os_size=args.os_size,
    channels=args.channels,
    size=args.size,
    policy=args.policy,
    channel_interleave=args.channel_interleave,
    row_policy=args.row_policy,
    ab_mode=args.ab_mode,
    arb_policy=args.arb_policy,
    posted=args.posted,
    values=not args.no_values,
    num_fmt=args.num_fmt,
    acc_fmt=args.acc_fmt,
    dirty_read_fatal=args.dirty_read == "fatal",
    layout_file=layout,
)

# P4R Q15 (b): VLENB is one column (256 bits on LPDDR6, 512 on DDR5).
vlen = args.vlen or 8 * memory.pim_memory.get_device().col_bytes
processor = SimpleProcessor(
    cpu_type=CPUS[args.cpu], isa=ISA.RISCV, num_cores=args.cores
)
for core in processor.get_cores():
    core.core.isa[0].vlen = vlen

if args.mode == "se":
    board = SimpleBoard(
        clk_freq="3GHz",
        processor=processor,
        memory=memory,
        cache_hierarchy=PrivateL1PrivateL2CacheHierarchy(
            l1d_size="32KiB", l1i_size="32KiB", l2_size="256KiB"
        ),
    )
    board.set_se_binary_workload(
        BinaryResource(local_path=str(binary)),
        arguments=args.args,
        # SE opens it relative to the output directory.
        stdout_file=Path("program.out"),
    )
else:
    board = RiscvBoard(
        clk_freq="3GHz",
        processor=processor,
        memory=memory,
        cache_hierarchy=PrivateL1PrivateL2WalkCacheHierarchy(
            l1d_size="16KiB", l1i_size="16KiB", l2_size="256KiB"
        ),
    )
    workload = obtain_resource(FS_WORKLOAD[0], resource_version=FS_WORKLOAD[1])
    if (args.kernel, args.disk_image, args.bootloader) == (None,) * 3:
        board.set_workload(workload)
    else:
        default = workload.get_parameters()

        def resource(value, name, local):
            if value is None:
                return default[name]
            if Path(value).exists():
                return local(value)
            return obtain_resource(value)

        board.set_kernel_disk_workload(
            kernel=resource(
                args.kernel, "kernel", lambda p: KernelResource(local_path=p)
            ),
            disk_image=resource(
                args.disk_image,
                "disk_image",
                lambda p: DiskImageResource(local_path=p, root_partition="1"),
            ),
            bootloader=resource(
                args.bootloader,
                "bootloader",
                lambda p: BootloaderResource(local_path=p),
            ),
        )

    class PimKernelBooted(KernelBootedExitHandler):
        @overrides(KernelBootedExitHandler)
        def _process(self, simulator: "Simulator") -> None:
            print("First exit: kernel booted", flush=True)

        @overrides(KernelBootedExitHandler)
        def _exit_simulation(self) -> bool:
            return args.fs_exit == "kernel-booted"

    class PimAfterBoot(ExitHandler, hypercall_num=2):
        @overrides(ExitHandler)
        def _process(self, simulator: "Simulator") -> None:
            print("Second exit: started after_boot.sh", flush=True)

        @overrides(ExitHandler)
        def _exit_simulation(self) -> bool:
            return args.fs_exit == "after-boot"

    class PimScriptDone(ExitHandler, hypercall_num=3):
        @overrides(ExitHandler)
        def _process(self, simulator: "Simulator") -> None:
            print("Third exit: the run script has ended", flush=True)

        @overrides(ExitHandler)
        def _exit_simulation(self) -> bool:
            return True


simulator = Simulator(board=board)
simulator.run()
code = simulator.get_last_exit_event_code()
print(
    f"Exiting @ tick {simulator.get_current_tick()} because "
    f"{simulator.get_last_exit_event_cause()} (code {code}, VLEN {vlen})"
)
if args.mode == "se":
    sys.exit(code)
