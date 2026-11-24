"""CPU-less PIM controller cases on Ramulator 2.1: the ``cpuless``,
``config`` and ``cache`` groups of ``run_p4r.py`` (the ``gem5.debug``
case ``r3-swap`` is in ``test_pim_system.py``). See ``pim_testlib.py``."""

from pim_testlib import p4r_suite
from testlib import *

p4r_suite(
    "pim-cpuless", ["cpuless"], constants.quick_tag, only=r"^(?!r3-swap$)"
)
p4r_suite("pim-config", ["config"], constants.quick_tag)
p4r_suite("pim-cache", ["cache"], constants.long_tag)
