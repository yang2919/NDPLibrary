"""PIM decode cases on the native build: the static check, the negative
runs and the Atomic and Timing matrix at VLEN 256 (quick), and the rest
of the matrix (long). See ``pim_testlib.py``."""

from pim_testlib import (
    ALL,
    p4r_suite,
)
from testlib import *

p4r_suite(
    "pim-decode-quick",
    ["decode"],
    constants.quick_tag,
    tag=ALL,
    only=r"^decode-(static|neg-\w+|\w+-(atomic|timing)-256)$",
)
p4r_suite(
    "pim-decode-long",
    ["decode"],
    constants.long_tag,
    tag=ALL,
    only=r"^decode-\w+-(timing-512|minor-\d+|o3-\d+)$",
)
