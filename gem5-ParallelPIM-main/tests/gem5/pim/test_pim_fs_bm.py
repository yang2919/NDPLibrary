"""PIM translation cases in FS bare metal on the native build (F1-F13;
``pim-fs-bm-nopim``; Sv48 F1-F18). See ``pim_testlib.py``."""

from pim_testlib import (
    ALL,
    p4r_suite,
)
from testlib import *

p4r_suite("pim-fs-bm", ["fsbm"], constants.long_tag, tag=ALL)
