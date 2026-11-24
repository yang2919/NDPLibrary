"""PIM system cases on Ramulator 2.1: the request path, refresh, C28,
V16, the SE driver regressions, Ramulator 2.1's own suites with the model
cross-checks, and the cases that need ``gem5.debug``. See
``pim_testlib.py``."""

from pim_testlib import (
    ALL,
    R2,
    p4r_suite,
)
from testlib import *

p4r_suite("pim-path", ["path"], constants.long_tag)
p4r_suite(
    "pim-refresh", ["refresh"], constants.long_tag, only=r"^(?!.*-debug)"
)
p4r_suite("pim-c28", ["c28"], constants.long_tag)
# v16-refuse-noram runs the example on a build without Ramulator 2.1.
p4r_suite(
    "pim-v16",
    ["v16"],
    constants.long_tag,
    extra_builds=[(ALL, constants.opt_tag)],
)
p4r_suite(
    "pim-se-misc", ["se-misc"], constants.long_tag, only=r"^(order|dpn)-"
)
p4r_suite("pim-models", ["r2suites", "xcheck"], constants.long_tag)
p4r_suite(
    "pim-debug",
    ["cpuless", "refresh"],
    constants.very_long_tag,
    variant=constants.debug_tag,
    only=r"^r3-swap$|-debug",
    extra_builds=[(R2, constants.opt_tag)],
)
