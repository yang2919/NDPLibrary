"""fs.py with a PMA-uncacheable range over the PIM FS test's window.

Case F7 of the PIM FS bare-metal test: runs
configs/deprecated/example/fs.py unchanged, with all its arguments, and
before instantiation adds [0x10000000, +4 MiB), the window of
tests/test-progs/pim-fs-bm, to every CPU's PMAChecker.uncacheable (fs.py's
--param cannot set a vector of address ranges). PIM requests keep their
flags; normal accesses to the window become uncacheable.
"""

import os
import runpy
import sys

import m5
from m5.objects import (
    AddrRange,
    Root,
)

WINDOW = AddrRange(0x10000000, size="4MiB")
_instantiate = m5.instantiate


def _instantiate_with_pma(*args, **kwargs):
    cpus = Root.getInstance().system.cpu
    for cpu in cpus if isinstance(cpus, list) else [cpus]:
        cpu.mmu.pma_checker.uncacheable = [WINDOW]
    return _instantiate(*args, **kwargs)


m5.instantiate = _instantiate_with_pma
fs_py = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../../configs/deprecated/example/fs.py",
)
fs_py = os.path.realpath(fs_py)
sys.argv[0] = fs_py
# fs.py's addToPath() resolves against sys.path[0], its own directory.
sys.path[0] = os.path.dirname(fs_py)
runpy.run_path(fs_py, run_name="__main__")
