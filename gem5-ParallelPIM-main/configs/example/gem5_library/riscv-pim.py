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
scons -C libpim riscv.CROSS_COMPILE=riscv64-linux-gnu- build/riscv
make -C tests/test-progs/pim-gemv/src PIM_STD=lpddr6
build/RISCV_RAMULATOR2/gem5.opt configs/example/gem5_library/riscv-pim.py
```

The program's standard output goes to ``program.out`` in the output
directory, and gem5 exits with the program's exit status (``pim-gemv``: 0
when every result is within bf16 tolerance). Another policy, channel
count, DDR5 rank count, channel interleave or number format changes the
layout: generate its header with ``libpim/gen_pim_layout.py``, build the
program with ``PIM_LAYOUT=<header>`` and pass both with ``--layout`` and
``--binary``.

Full system (``--mode fs``) boots the Ubuntu 24.04 image of
``riscv-ubuntu-run.py`` on a ``RiscvBoard`` whose device tree lists only
the OS memory. ``--fs-exit kernel-booted`` stops once the kernel has
booted. A PIM program runs in FS with ``--fs-program <static binary>``
(its arguments in ``--args``; ``tests/gem5/pim/fs_script.py`` embeds it in
the guest's run script and runs it as root) or any guest script with
``--fs-script``. ``--boot no-systemd`` skips systemd, and ``--boot-cpu
atomic`` boots on the Atomic CPU and switches to ``--cpu`` when
``after_boot.sh`` starts. The harts accept Sv48 paging, so Linux boots in
Sv48; ``--paging-mode sv39`` limits them to Sv39 (gem5's default). The
program's statistics are reset when it starts and dumped when it ends
(hypercalls 4 and 5); ``--debug-from switch|work`` starts the debug trace
(``--debug-flags``) at the CPU switch, or keeps it to the program (from
hypercall 4 to 5):

```
build/RISCV_RAMULATOR2/gem5.fast configs/example/gem5_library/riscv-pim.py \
    --mode fs --boot no-systemd --boot-cpu atomic --cpu timing \
    --fs-program tests/test-progs/pim-path/bin/riscv/linux/pim-path \
    --args 0x1000000000 0x800000000 all 0x7fff00000
```

A program that maps the whole PIM memory (the PIM library's, ``pim_init``)
spends most of its host time while Linux fills page-table entries.
Built with ``FS_MARKERS=1`` it marks the end of that setup and of its
output (hypercalls 10 and 11): ``--switch-at
marker`` switches to ``--cpu`` and resets the statistics at the first
marker, and dumps them and ends the run at the second. ``--fs-env
PIM_MODE=fs`` sets the library's mode. ``--checkpoint-at after-boot``
writes a boot checkpoint to ``<outdir>/ckpt`` when ``after_boot.sh``
starts; a run with ``--restore <dir>`` continues from it and reads its
own ``--fs-program`` (``--switch-at work`` or ``marker``: the restored
run does not see the after-boot hypercall again). ``--kernel`` and
``--disk-image`` take the rebuilt kernel of ``util/pim-kernel``.

This needs a gem5 built with Ramulator 2.1 (``build/RISCV_RAMULATOR2``,
Kconfig ``USE_RAMULATOR2``). An Atomic CPU sees fixed latencies: a PIM
request takes its unloaded latency, a normal read the standard's
closed-page latency.
"""

import argparse
import shlex
import sys
from pathlib import Path

import m5
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
from gem5.components.processors.simple_switchable_processor import (
    SimpleSwitchableProcessor,
)
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
    "--args",
    nargs="*",
    default=[],
    help="Arguments of the SE program or of --fs-program.",
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
parser.add_argument(
    "--boot",
    choices=("full", "no-systemd"),
    default="full",
    help="FS boot: with systemd, or the image's init straight to the run "
    "script (kernel argument no_systemd).",
)
parser.add_argument(
    "--boot-cpu",
    choices=tuple(CPUS),
    default=None,
    help="FS: boot on this CPU and switch to --cpu when after_boot.sh "
    "starts (default: --cpu throughout).",
)
parser.add_argument(
    "--fs-script",
    default=None,
    help="FS: a guest script, run by after_boot.sh through readfile.",
)
parser.add_argument(
    "--fs-program",
    default=None,
    help="FS: a static guest program, run as root with --args "
    "(tests/gem5/pim/fs_script.py embeds it in the run script).",
)
parser.add_argument(
    "--fs-env",
    action="append",
    default=[],
    metavar="NAME=VALUE",
    help="FS: an environment variable of --fs-program (repeatable), for "
    "example PIM_MODE=fs for the PIM library's programs.",
)
parser.add_argument(
    "--fs-setup",
    action="append",
    default=[],
    metavar="COMMAND",
    help="FS: a shell command run as root before --fs-program "
    "(repeatable), for example reserving huge pages on the PIM node.",
)
parser.add_argument(
    "--fs-wrap",
    default="",
    metavar="COMMAND",
    help="FS: a command that starts --fs-program, for example "
    "'numactl --membind=1'.",
)
parser.add_argument(
    "--switch-at",
    choices=("after-boot", "work", "marker"),
    default="after-boot",
    help="FS, with --boot-cpu: switch to --cpu when after_boot.sh starts, "
    "when the run script starts the program (hypercall 4; for a run "
    "restored from a boot checkpoint), or at the program's first marker (hypercall 10: a program built with "
    "FS_MARKERS=1, after pim_init() has mapped the PIM memory), which "
    "also resets the statistics. At the second marker "
    "(hypercall 11) the statistics are dumped and the run ends.",
)
parser.add_argument(
    "--checkpoint-at",
    choices=("after-boot", "marker"),
    default=None,
    help="FS: write a checkpoint to <outdir>/ckpt and end the run when "
    "after_boot.sh starts (a boot checkpoint, before any switch), or at the "
    "program's checkpoint marker (hypercall 12).",
)
parser.add_argument(
    "--restore",
    default=None,
    help="FS: restore this checkpoint directory, taken with the same "
    "memory, kernel, disk image, --boot and --boot-cpu; after_boot.sh then "
    "reads this run's --fs-program or --fs-script.",
)
parser.add_argument(
    "--debug-from",
    choices=("switch", "work"),
    default=None,
    help="FS: the debug trace (--debug-flags) starts at the CPU switch "
    "(needs --boot-cpu), or runs only while the program runs (hypercalls 4 "
    "to 5; with --switch-at marker, 10 to 11).",
)
parser.add_argument(
    "--way",
    type=int,
    choices=(1, 2),
    default=1,
    help="FS: 1, the PIM memory hidden from the OS (default); 2, the PIM "
    "memory a NUMA node without its register rows (needs a kernel with "
    "CONFIG_NUMA, util/pim-kernel).",
)
parser.add_argument(
    "--paging-mode",
    choices=("sv39", "sv48"),
    default=None,
    help="FS: the largest virtual-memory scheme the harts accept "
    "(RiscvISA.max_paging_mode; default sv48). Linux takes the largest one.",
)
DEFAULT_CHANNELS = ", ".join(
    f"{std} {pim.PIM_PRESETS[std]['channels']}" for std in MEMORIES.values()
)
parser.add_argument(
    "--memory",
    choices=tuple(MEMORIES),
    default="lpddr6",
    help="PIM memory standard; each default PIM memory is 32 GiB.",
)
parser.add_argument(
    "--channels",
    type=int,
    default=None,
    help=f"PIM memory channels (default: the standard's: {DEFAULT_CHANNELS}).",
)
parser.add_argument(
    "--size",
    default=None,
    help="PIM memory size; must be its capacity (default).",
)
parser.add_argument(
    "--ranks",
    type=int,
    default=None,
    help="DDR5 ranks per channel, 1 or 2 (default: the preset's, 1); the "
    "OS memory follows it.",
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
    "--row-policy",
    choices=("Open", "ClosedCAP"),
    default="Open",
    help="Ramulator 2.1's row policy; ClosedCAP also closes PIM rows "
    "(auto-precharge SB columns, closing PREs after all-bank jobs).",
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
    "--prefetcher",
    choices=("on", "none", "excluded-off"),
    default="on",
    help="The caches' prefetchers: with the PIM memory excluded (default), "
    "removed, or without the exclusion (an ablation).",
)
parser.add_argument(
    "--vlen",
    type=int,
    default=None,
    help="Vector length in bits (default: 8 x the standard's column).",
)
args = parser.parse_args()


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


if args.mode == "se" and (
    args.boot != "full"
    or args.boot_cpu
    or args.fs_script
    or args.fs_program
    or args.fs_env
    or args.fs_setup
    or args.fs_wrap
    or args.debug_from
    or args.switch_at != "after-boot"
    or args.checkpoint_at
    or args.restore
    or args.way != 1
    or args.paging_mode
):
    parser.error(
        "--boot, --boot-cpu, --fs-script, --fs-program, --fs-env, "
        "--fs-setup, --fs-wrap, --debug-from, --switch-at, "
        "--checkpoint-at, --restore, --way and --paging-mode are FS "
        "options (--mode fs)"
    )
if (args.fs_env or args.fs_setup or args.fs_wrap) and not args.fs_program:
    parser.error("--fs-env, --fs-setup and --fs-wrap need --fs-program")
if args.fs_script and args.fs_program:
    parser.error("give --fs-script or --fs-program, not both")
if args.debug_from == "switch" and not args.boot_cpu:
    parser.error("--debug-from switch needs --boot-cpu")

std = MEMORIES[args.memory]
custom_layout = (
    (args.channels not in (None, pim.PIM_PRESETS[std]["channels"]))
    or (args.ranks is not None and args.ranks != pim.default_ranks(std))
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
            "libpim/gen_pim_layout.py, build the program with "
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
    ranks=args.ranks,
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
    way=args.way,
)

set_prefetchers(memory, args.prefetcher)

# By default VLENB is one column (256 bits on LPDDR6, 512 on DDR5).
vlen = args.vlen or 8 * memory.pim_memory.get_device().col_bytes
if args.boot_cpu:
    # Boot on --boot-cpu, switch to --cpu at after_boot.sh.
    processor = SimpleSwitchableProcessor(
        starting_core_type=CPUS[args.boot_cpu],
        switch_core_type=CPUS[args.cpu],
        isa=ISA.RISCV,
        num_cores=args.cores,
    )
    cores = list(processor._all_cores())
else:
    processor = SimpleProcessor(
        cpu_type=CPUS[args.cpu], isa=ISA.RISCV, num_cores=args.cores
    )
    cores = processor.get_cores()
for core in cores:
    core.core.isa[0].vlen = vlen
    if args.mode == "fs":
        # PIM FS runs boot Linux in Sv48 unless
        # told otherwise; SE never walks guest page tables.
        core.core.isa[0].max_paging_mode = args.paging_mode or "sv48"

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
    script = None
    if args.fs_script:
        script = Path(args.fs_script).read_text()
    elif args.fs_program:
        sys.path.insert(0, str(TREE / "tests" / "gem5" / "pim"))
        from fs_script import make_script

        script = make_script(
            args.fs_program,
            args.args,
            env=args.fs_env,
            setup=args.fs_setup,
            wrap=shlex.split(args.fs_wrap),
            # The second marker ends the run: the output must have left
            # the console by then.
            console=args.switch_at == "marker",
        )
    plain = script is None and args.boot == "full" and not args.restore
    if plain and (args.kernel, args.disk_image, args.bootloader) == (
        (None,) * 3
    ):
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
            kernel_args=(
                list(
                    default.get("kernel_args")
                    or board.get_default_kernel_args()
                )
                + (["no_systemd"] if args.boot == "no-systemd" else [])
            ),
            readfile_contents=script,
            checkpoint=Path(args.restore) if args.restore else None,
        )
    if args.debug_from:
        m5.trace.disable()

    class PimKernelBooted(KernelBootedExitHandler):
        @overrides(KernelBootedExitHandler)
        def _process(self, simulator: "Simulator") -> None:
            print("First exit: kernel booted", flush=True)

        @overrides(KernelBootedExitHandler)
        def _exit_simulation(self) -> bool:
            return args.fs_exit == "kernel-booted"

    def switch(simulator):
        simulator.switch_processor()
        print(
            f"PIM-FS switched from {args.boot_cpu} to {args.cpu} "
            f"at tick {m5.curTick()}",
            flush=True,
        )

    def checkpoint():
        path = Path(m5.options.outdir) / "ckpt"
        m5.checkpoint(str(path))
        print(f"PIM-FS checkpoint {path} at tick {m5.curTick()}", flush=True)

    marker = args.switch_at == "marker"
    at_work = args.switch_at == "work"

    class PimAfterBoot(ExitHandler, hypercall_num=2):
        @overrides(ExitHandler)
        def _process(self, simulator: "Simulator") -> None:
            print("Second exit: started after_boot.sh", flush=True)
            if args.checkpoint_at == "after-boot":
                checkpoint()
            elif args.boot_cpu and args.switch_at == "after-boot":
                switch(simulator)
                if args.debug_from == "switch":
                    m5.trace.enable()

        @overrides(ExitHandler)
        def _exit_simulation(self) -> bool:
            return (
                args.fs_exit == "after-boot"
                or args.checkpoint_at == "after-boot"
            )

    class PimScriptDone(ExitHandler, hypercall_num=3):
        @overrides(ExitHandler)
        def _process(self, simulator: "Simulator") -> None:
            print("Third exit: the run script has ended", flush=True)

        @overrides(ExitHandler)
        def _exit_simulation(self) -> bool:
            return True

    class PimWorkBegin(ExitHandler, hypercall_num=4):
        @overrides(ExitHandler)
        def _process(self, simulator: "Simulator") -> None:
            if at_work and args.boot_cpu:
                switch(simulator)
            m5.stats.reset()
            if not marker and (
                args.debug_from == "work"
                or (at_work and args.debug_from == "switch")
            ):
                m5.trace.enable()
            print(f"PIM-FS work begin at tick {m5.curTick()}", flush=True)

        @overrides(ExitHandler)
        def _exit_simulation(self) -> bool:
            return False

    class PimWorkEnd(ExitHandler, hypercall_num=5):
        @overrides(ExitHandler)
        def _process(self, simulator: "Simulator") -> None:
            m5.stats.dump()
            if not marker and args.debug_from == "work":
                m5.trace.disable()
            print(f"PIM-FS work end at tick {m5.curTick()}", flush=True)

        @overrides(ExitHandler)
        def _exit_simulation(self) -> bool:
            return False

    class PimMarkerBegin(ExitHandler, hypercall_num=10):
        """The program has mapped the PIM memory."""

        @overrides(ExitHandler)
        def _process(self, simulator: "Simulator") -> None:
            if marker and args.boot_cpu:
                switch(simulator)
            m5.stats.reset()
            if marker and args.debug_from:
                m5.trace.enable()
            print(f"PIM-FS marker begin at tick {m5.curTick()}", flush=True)

        @overrides(ExitHandler)
        def _exit_simulation(self) -> bool:
            return False

    class PimMarkerEnd(ExitHandler, hypercall_num=11):
        """The program has printed its output; end before it
        unmaps the PIM memory."""

        @overrides(ExitHandler)
        def _process(self, simulator: "Simulator") -> None:
            m5.stats.dump()
            if marker and args.debug_from:
                m5.trace.disable()
            print(f"PIM-FS marker end at tick {m5.curTick()}", flush=True)

        @overrides(ExitHandler)
        def _exit_simulation(self) -> bool:
            return marker

    class PimCheckpoint(ExitHandler, hypercall_num=12):
        """The program's checkpoint marker."""

        @overrides(ExitHandler)
        def _process(self, simulator: "Simulator") -> None:
            if args.checkpoint_at == "marker":
                checkpoint()

        @overrides(ExitHandler)
        def _exit_simulation(self) -> bool:
            return args.checkpoint_at == "marker"


simulator = Simulator(board=board)
simulator.run()
code = simulator.get_last_exit_event_code()
print(
    f"Exiting @ tick {simulator.get_current_tick()} because "
    f"{simulator.get_last_exit_event_cause()} (code {code}, VLEN {vlen})"
)
if args.mode == "se":
    sys.exit(code)
