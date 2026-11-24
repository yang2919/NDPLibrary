"""The PIM request path and P1-P7 on gem5's native memory, and the Ruby
guard, on the native build. See ``pim_testlib.py``."""

from pim_testlib import (
    ALL,
    p4r_suite,
)
from testlib import *

p4r_suite(
    "pim-path-native",
    ["se-misc"],
    constants.long_tag,
    tag=ALL,
    only=r"^(native|ruby)-",
)
