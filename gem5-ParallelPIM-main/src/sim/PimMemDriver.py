# An emulated /dev/mem for Processing-in-Memory (PIM) programs in
# system-call emulation (SE) mode.

from m5.objects.Process import EmulatedDriver
from m5.params import *


class PimMemDriver(EmulatedDriver):
    """``/dev/mem`` for SE mode, limited to one physical range.

    ``mmap(addr, length, prot, flags, fd, offset)`` on the file descriptor
    maps the physical bytes ``[offset, offset + length)``, which must lie
    inside ``window``, linearly and cacheable at a virtual base congruent
    to ``offset`` modulo ``alignment``, so every address bit below
    ``alignment`` is the same in the virtual and the physical address. The
    window is not in the SE frame pool (its memory has
    ``conf_table_reported = False``), so no ordinary page ever uses it.
    """

    type = "PimMemDriver"
    cxx_header = "sim/pim_mem_driver.hh"
    cxx_class = "gem5::PimMemDriver"

    filename = "mem"
    window = Param.AddrRange("Physical range that mmap may map")
    alignment = Param.UInt64(
        0,
        "Virtual alignment of a mapping, a power of two (0: the window "
        "size rounded up to a power of two)",
    )
