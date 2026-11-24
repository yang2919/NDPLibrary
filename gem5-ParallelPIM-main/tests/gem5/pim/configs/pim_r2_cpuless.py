"""A CPU-less PIM run: one PimTester on a Ramulator 2.1 memory.

A ``System`` with a ``PimTester`` (src/cpu/testers/pim_tester) that plays a
script through a non-coherent crossbar to a Ramulator 2.1 memory with one port
per channel: a plain ``Ramulator2Memory`` (``--memory ramulator2``) or a PIM
memory (``--memory pim``), the stdlib's ``PimMemory``
(``gem5.components.memory.pim``: a ``PimRamulator2`` on ``PimDRAM`` with a
``PimSequencer`` on every controller and the standard's PIM-derived DRAM).
Needs ``build/RISCV_RAMULATOR2``.

The run exits with status 0 when the tester finished every line without a
mismatch, 1 otherwise. ``tests/gem5/pim/check_pim_ctrl.py`` then checks
``pim_tester.csv`` and ``stats.txt``; ``--print-geometry`` prints the PIM
memory's geometry for ``tests/gem5/pim/pim_ref.py`` and exits.

Example:
  build/RISCV_RAMULATOR2/gem5.opt -d m5out tests/gem5/pim/configs/\\
      pim_r2_cpuless.py --script rdwr.txt --std LPDDR6 --channels 2
"""

import argparse
import json
import sys

import m5
from m5.objects import (
    AddrRange,
    NoncoherentXBar,
    PimTester,
    Root,
    SrcClockDomain,
    System,
    VoltageDomain,
)

from gem5.components.memory.ramulator2 import (
    STANDARDS,
    ramulator2_memory,
)

POLICIES = (
    "RoRaBaChCo",
    "RoRaBaCoCh",
    "RoCoRaBaCh",
    "RoBaRaCoCh",
    "ChRaBaRoCo",
    "MOP4CLXOR",
)

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--script", help="PimTester script")
parser.add_argument(
    "--memory", default="ramulator2", choices=("ramulator2", "pim")
)
parser.add_argument("--std", default="LPDDR6", choices=STANDARDS)
parser.add_argument("--channels", type=int, default=2)
parser.add_argument(
    "--size", default="512MiB", help="memory size (ramulator2 only)"
)
parser.add_argument(
    "--mode",
    default="timing",
    choices=("timing", "atomic", "atomic_noncaching"),
)
parser.add_argument(
    "--use-backdoor",
    action="store_true",
    help="atomic requests through sendAtomicBackdoor",
)
parser.add_argument(
    "--interleave",
    type=int,
    default=None,
    help="channel interleave in bytes (default: 64; for --memory pim "
    "fixed at one row under RoRaBaChCo and one channel under ChRaBaRoCo)",
)
parser.add_argument("--csv", default="pim_tester.csv")
parser.add_argument("--max-ticks", type=int, default=10**12)
pim = parser.add_argument_group("PIM memory (--memory pim)")
pim.add_argument("--policy", default="RoRaBaChCo", choices=POLICIES)
pim.add_argument("--num-fmt", default="bf16")
pim.add_argument("--acc-fmt", default="fp32")
pim.add_argument("--num-regs", type=int, default=1)
pim.add_argument("--no-values", action="store_true")
pim.add_argument(
    "--sw-errors-counted", action="store_true", help="sw_error_fatal off"
)
pim.add_argument("--ab-mode", default="baseline")
pim.add_argument("--arb-policy", default="frfcfs")
pim.add_argument("--close-rows", action="store_true")
pim.add_argument("--no-posted", action="store_true", help="posted off")
pim.add_argument("--pim-queue-size", type=int, default=32)
pim.add_argument(
    "--row-policy", default="Open", help="Ramulator 2.1 row policy"
)
pim.add_argument(
    "--refresh", default="AllBank", help="Ramulator 2.1 refresh manager"
)
pim.add_argument(
    "--override",
    action="append",
    default=[],
    metavar="NAME=VALUE",
    help="a DRAM organization or timing value of the PIM memory (an "
    "integer, for example nREFI=1000); repeatable",
)
pim.add_argument(
    "--ranks",
    type=int,
    default=None,
    help="DDR5 ranks per channel (PimMemory's ranks)",
)
pim.add_argument("--read-buffer", type=int, default=64)
pim.add_argument("--write-buffer", type=int, default=128)
pim.add_argument(
    "--trace",
    action="store_true",
    help="record every DRAM command (CmdTraceRecorder) to cmd_trace.csv.ch<i>",
)
parser.add_argument(
    "--checkpoint-at",
    type=int,
    help="at this tick, drain and save a checkpoint to --checkpoint-dir, "
    "then continue",
)
parser.add_argument(
    "--checkpoint-at-end",
    action="store_true",
    help="when the testers finish, drain and save a checkpoint",
)
parser.add_argument("--checkpoint-dir", default=None)
parser.add_argument("--restore", default=None, help="checkpoint to restore")
parser.add_argument(
    "--host-script",
    help="a second PimTester (host traffic) playing this script; its "
    "results go to host.csv",
)
pim.add_argument("--null", action="store_true", help="no backing store")
pim.add_argument(
    "--layout-file",
    help="the PIM library's layout header to check at start-up",
)
pim.add_argument(
    "--print-geometry",
    action="store_true",
    help="print the geometry as JSON and exit",
)
args = parser.parse_args()


class _LineSize:
    """What Ramulator2Memory.incorporate_memory() asks of a board."""

    def get_cache_line_size(self):
        return 64


def pim_memory():
    """The stdlib's PimMemory at [0, its capacity), and
    the geometry of pim_ref.Geometry."""
    from gem5.components.memory.pim import (
        PIM_PRESETS,
        PimMemory,
    )

    if args.std not in PIM_PRESETS:
        sys.exit("--memory pim needs one of %s" % ", ".join(PIM_PRESETS))
    memory = PimMemory(
        standard=args.std,
        channels=args.channels,
        ranks=args.ranks,
        policy=args.policy,
        channel_interleave=args.interleave,
        row_policy=args.row_policy,
        refresh_manager=args.refresh,
        read_buffer_size=args.read_buffer,
        write_buffer_size=args.write_buffer,
        ab_mode=args.ab_mode,
        arb_policy=args.arb_policy,
        # Unset: the row policy decides (ClosedCAP closes rows).
        close_rows=args.close_rows or None,
        posted=not args.no_posted,
        values=not args.no_values,
        num_fmt=args.num_fmt,
        acc_fmt=args.acc_fmt,
        num_regs=args.num_regs,
        pim_queue_size=args.pim_queue_size,
        layout_file=args.layout_file,
        overrides={
            k: int(v, 0) for k, v in (o.split("=", 1) for o in args.override)
        },
        cmd_trace=(
            (m5.options.outdir + "/cmd_trace.csv") if args.trace else None
        ),
    )
    memory.incorporate_memory(_LineSize())
    rng = AddrRange(memory.get_size())
    memory.set_memory_range([rng])
    mem = memory.ramulator2
    mem.sw_error_fatal = not args.sw_errors_counted
    mem.null = args.null
    dev = memory.get_device()
    probe = dev.make()
    o, _ = probe.resolve()
    levels = [lv.lower() for lv in type(probe).levels]
    count = {lv: o.get(lv, 1) for lv in levels}
    geometry = dict(
        policy=POLICIES.index(args.policy),
        channels=args.channels,
        intlv=memory.get_interleave(),
        units=count.get("rank") or count.get("pseudochannel") or 1,
        sids=count.get("sid", 1),
        bgs=count.get("bankgroup", 1),
        banks=count["bank"],
        rows=count["row"],
        cols=dev.row_bytes // dev.col_bytes,
        col_bytes=dev.col_bytes,
    )
    return memory, rng, geometry


system = System()
system.clk_domain = SrcClockDomain(
    clock="1GHz", voltage_domain=VoltageDomain()
)
system.mem_mode = args.mode
system.membus = NoncoherentXBar(
    width=64,
    frontend_latency=0,
    forward_latency=0,
    response_latency=0,
)
if args.memory == "pim":
    memory, rng, geometry = pim_memory()
    if args.print_geometry:
        print(json.dumps(geometry))
        sys.exit(0)
    system.mem_ranges = [rng]
    system.memory = memory
    for _, port in memory.get_mem_ports():
        system.membus.mem_side_ports = port
else:
    memory = ramulator2_memory(args.std, args.channels, args.size)
    memory._set_interleaving(args.interleave or 64)
    memory.incorporate_memory(_LineSize())
    system.mem_ranges = [AddrRange(memory.get_size())]
    memory.set_memory_range(system.mem_ranges)
    system.memory = memory
    for _, port in memory.get_mem_ports():
        system.membus.mem_side_ports = port
if not args.script:
    sys.exit("--script is required")
system.tester = PimTester(
    script=args.script, csv_file=args.csv, use_backdoor=args.use_backdoor
)
system.tester.port = system.membus.cpu_side_ports
if args.host_script:
    system.host = PimTester(script=args.host_script, csv_file="host.csv")
    system.host.port = system.membus.cpu_side_ports
system.system_port = system.membus.cpu_side_ports

root = Root(full_system=False, system=system)
m5.instantiate(args.restore)
if args.restore:
    print("Restored from %s at tick %d" % (args.restore, m5.curTick()))


def checkpoint():
    start = m5.curTick()
    m5.checkpoint(args.checkpoint_dir)
    print(
        "Checkpoint at tick %d: the drain took %d ticks"
        % (m5.curTick(), m5.curTick() - start)
    )


if args.checkpoint_at is not None:
    event = m5.simulate(args.checkpoint_at - m5.curTick())
    if event.getCause() != "simulate() limit reached":
        sys.exit("checkpoint tick not reached: %s" % event.getCause())
    checkpoint()
# Each tester exits the simulation when it finishes; run until both have.
testers = 2 if args.host_script else 1
failed = False
while testers:
    event = m5.simulate(args.max_ticks - m5.curTick())
    cause, code = event.getCause(), event.getCode()
    print(
        "Exiting @ tick %d because %s (code %d)" % (m5.curTick(), cause, code)
    )
    if cause != "PimTester finished":
        failed = True
        break
    failed = failed or code != 0
    testers -= 1
if args.checkpoint_at_end and not failed:
    checkpoint()
sys.exit(1 if failed else 0)
