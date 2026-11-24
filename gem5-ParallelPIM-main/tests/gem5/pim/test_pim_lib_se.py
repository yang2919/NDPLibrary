"""The PIM library's SE runs (``run_lib_se.py``). See ``pim_testlib.py``."""

from pim_testlib import lib_se_suite
from testlib import *

lib_se_suite("pim-lib-se", constants.long_tag)
