"""CPU-less cache cases of the processing-in-memory (PIM) request path.

Cases P2-P6 and the diagnostics D-1 to D-3 (``run_p4r.py``'s ``cache``
group). Two ``PimTester``s, A (PIM requests)
and B (the peer), each behind a private L1 cache on a coherent crossbar.
The memory is a ``SimpleMemory``, which treats a PIM packet as a plain
read or write, or (``--memory pim``) the stdlib ``PimMemory`` (LPDDR6 x 2)
at address 0. ``--noncoherent`` puts A behind a ``NoncoherentCache``.

Needs ``build/RISCV_RAMULATOR2`` (``PimTester``). Exit status 0 when every
tester finished with no mismatch, 1 otherwise.
"""

import argparse
import sys

import m5
from m5.objects import (
    AddrRange,
    Cache,
    NoncoherentCache,
    PimTester,
    Root,
    SimpleMemory,
    SrcClockDomain,
    System,
    SystemXBar,
    VoltageDomain,
)

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--a", required=True, help="script of tester A")
ap.add_argument("--b", help="script of tester B (the peer)")
ap.add_argument("--mode", default="timing", choices=("timing", "atomic"))
ap.add_argument("--dirty-read", default="fatal", choices=("fatal", "warn"))
ap.add_argument("--l1-size", default="32KiB")
ap.add_argument("--l1-assoc", type=int, default=8)
ap.add_argument("--noncoherent", action="store_true")
ap.add_argument(
    "--memory",
    default="simple",
    choices=("simple", "pim"),
    help="SimpleMemory, or the stdlib PimMemory (LPDDR6 x 2) at 0",
)
args = ap.parse_args()


def l1(cls=Cache):
    return cls(
        size=args.l1_size,
        assoc=args.l1_assoc,
        tag_latency=1,
        data_latency=1,
        response_latency=1,
        mshrs=4,
        tgts_per_mshr=8,
        pim_dirty_read_fatal=args.dirty_read == "fatal",
    )


system = System()
system.clk_domain = SrcClockDomain(
    clock="1GHz", voltage_domain=VoltageDomain()
)
system.mem_mode = args.mode
system.cache_line_size = 64
system.membus = SystemXBar()
if args.memory == "pim":
    from gem5.components.memory.pim import PimMemory

    class Line:
        def get_cache_line_size(self):
            return 64

    system.memory = PimMemory(standard="LPDDR6", channels=2)
    system.memory.incorporate_memory(Line())
    system.mem_ranges = [AddrRange(system.memory.get_size())]
    system.memory.set_memory_range(system.mem_ranges)
    for _, port in system.memory.get_mem_ports():
        system.membus.mem_side_ports = port
else:
    system.mem_ranges = [AddrRange("512MiB")]
    system.mem = SimpleMemory(range=system.mem_ranges[0])
    system.mem.port = system.membus.mem_side_ports
system.system_port = system.membus.cpu_side_ports
testers = [("a", args.a)] + ([("b", args.b)] if args.b else [])
for n, script in testers:
    t = PimTester(script=script, csv_file=n + ".csv")
    c = l1(NoncoherentCache if (args.noncoherent and n == "a") else Cache)
    setattr(system, "tester_" + n, t)
    setattr(system, "l1" + n, c)
    t.port = c.cpu_side
    c.mem_side = system.membus.cpu_side_ports
root = Root(full_system=False, system=system)
m5.instantiate()
left, failed = len(testers), False
while left:
    ev = m5.simulate()
    print(
        "Exiting @ tick %d because %s (code %d)"
        % (m5.curTick(), ev.getCause(), ev.getCode())
    )
    if ev.getCause() != "PimTester finished":
        failed = True
        break
    failed = failed or ev.getCode() != 0
    left -= 1
sys.exit(1 if failed else 0)
