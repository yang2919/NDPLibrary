"""Full-system (FS) Linux runs on Ramulator 2.1: V16's boot, the PIM
request-path program under Linux (Way 1), the full flush protocol, Way 2,
Linux in Sv48 and the 128 GiB DDR5 system. These need the rebuilt kernel
of ``util/pim-kernel``, and several take hours. See ``pim_testlib.py``."""

from pim_testlib import (
    R2,
    p4r_suite,
)
from testlib import *

for name, group, only in (("pim-fs-boot", "fs-boot", None),):
    p4r_suite(
        name,
        [group],
        constants.very_long_tag,
        variant=constants.fast_tag,
        only=only,
        extra_builds=[(R2, constants.opt_tag)],
    )

# The other FS groups, on gem5.opt (fs-ddr5's Sv39 boot also uses
# gem5.fast).
for name, group, extra in (
    ("pim-fs-way1", "fs-way1", []),
    ("pim-fs-way1-whole", "fs-way1-whole", []),
    ("pim-fs-protocol", "fs-protocol", []),
    ("pim-fs-way2", "fs-way2", []),
    ("pim-sv48", "sv48", []),
    ("pim-fs-ddr5", "fs-ddr5", [(R2, constants.fast_tag)]),
):
    p4r_suite(name, [group], constants.very_long_tag, extra_builds=extra)
