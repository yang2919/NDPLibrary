"""A RiscvBoard's device tree and config.ini without booting.

Checks V13 (the device tree lists only the operating-system (OS) memory), V14
(every cache has the dirty-read knob and every prefetcher excludes the
processing-in-memory (PIM) memory) and V15 (the layout header check at
start-up) in full-system mode. ``--memory``:

- ``r2``: a plain ``Ramulator2Memory`` (LPDDR6 x 16, 32 GiB, the OS memory
  of the default ``PimSystemMemory``);
- ``pim``: the default ``PimSystemMemory`` (LPDDR6);
- ``pim-<std>``: a factory's default (``lpddr6``, ``ddr5``, ``gddr7``,
  ``hbm4``).

``--way 2`` makes the PIM memory NUMA node 1 of a ``pim`` memory; ``--policy``
sets its address-mapping policy.

``--instantiate`` also writes config.ini; ``--simulate`` simulates 1 us,
which runs the start-up check. The kernel, disk image and bootloader are
the ones of ``riscv-ubuntu-run.py``; they are never read before the boot.
Run it with ``build/RISCV_RAMULATOR2/gem5.opt``.
"""

import argparse
import os

import m5

from gem5.components.boards.riscv_board import RiscvBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_walk_cache_hierarchy import (
    PrivateL1PrivateL2WalkCacheHierarchy,
)
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import (
    BootloaderResource,
    DiskImageResource,
    KernelResource,
    obtain_resource,
)

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--memory", default="pim")
ap.add_argument("--instantiate", action="store_true")
ap.add_argument("--dirty-read-warn", action="store_true")
ap.add_argument("--layout-file", default=None)
ap.add_argument("--simulate", action="store_true")
ap.add_argument("--way", type=int, default=None)
ap.add_argument("--policy", default=None)
args = ap.parse_args()

if args.memory == "r2":
    from gem5.components.memory.ramulator2 import ramulator2_memory

    memory = ramulator2_memory("LPDDR6", 16, "32GiB")
else:
    from gem5.components.memory import pim

    std = args.memory.split("-")[1] if "-" in args.memory else "lpddr6"
    kw = {"dirty_read_fatal": not args.dirty_read_warn}
    if args.layout_file:
        kw["layout_file"] = args.layout_file
    if args.way is not None:
        kw["way"] = args.way
    if args.policy is not None:
        kw["policy"] = args.policy
    memory = getattr(pim, f"pim_{std}_memory")(**kw)

board = RiscvBoard(
    clk_freq="3GHz",
    processor=SimpleProcessor(
        cpu_type=CPUTypes.TIMING, isa=ISA.RISCV, num_cores=1
    ),
    memory=memory,
    cache_hierarchy=PrivateL1PrivateL2WalkCacheHierarchy(
        l1d_size="32KiB", l1i_size="32KiB", l2_size="512KiB"
    ),
)


def resource(cls, rid, version, **kw):
    """The cached file when present (no checksum pass), else download."""
    path = os.path.expanduser(f"~/.cache/gem5/{rid}-{version}")
    if os.path.exists(path):
        return cls(local_path=path, **kw)
    return obtain_resource(rid, resource_version=version)


board.set_kernel_disk_workload(
    kernel=resource(KernelResource, "riscv-linux-6.8.12-kernel", "1.0.0"),
    disk_image=resource(
        DiskImageResource,
        "riscv-ubuntu-24.04-img",
        "2.0.0",
        root_partition="1",
    ),
    bootloader=resource(
        BootloaderResource, "riscv-bootloader-opensbi-1.3.1", "1.0.0"
    ),
)
board._pre_instantiate()
if args.instantiate or args.simulate:
    m5.instantiate()
if args.simulate:
    ev = m5.simulate(10**6)
    print(f"simulated to tick {m5.curTick()}: {ev.getCause()}")
print("device tree written to", m5.options.outdir)
