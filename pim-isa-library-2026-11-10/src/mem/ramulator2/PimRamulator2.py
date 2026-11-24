# A Processing-in-Memory (PIM) memory on Ramulator 2.1
# (docs/pim/04r-ramulator-pim-plan.md, sections 2, 6 and 7).

from m5.objects.Ramulator2VectorPorts import Ramulator2VectorPorts
from m5.params import *


class PimRamulator2(Ramulator2VectorPorts):
    """A Ramulator 2.1 memory whose whole range is PIM memory.

    Its Ramulator 2.1 memory system must be ``PimDRAM`` with a
    ``PimSequencer`` on every controller, and its size the devices'
    capacity. ``policy`` and ``channel_interleave`` must describe the
    controllers' address mapper and the channel mapper's interleave: the
    memory checks at start-up that the memory system decodes sampled
    addresses as include/gem5/pim/pim_addr.h does under them.
    """

    type = "PimRamulator2"
    cxx_class = "gem5::memory::PimRamulator2"
    cxx_header = "mem/ramulator2/pim_ramulator2.hh"

    policy = Param.String(
        "RoRaBaChCo",
        "Address-mapping policy of the controllers: RoRaBaChCo, "
        "RoRaBaCoCh, RoCoRaBaCh, RoBaRaCoCh, ChRaBaRoCo or MOP4CLXOR",
    )
    channel_interleave = Param.UInt64(
        2048, "Channel interleave in bytes (as the channel mapper's)"
    )
    num_regs = Param.Unsigned(1, "Registers per processing unit")
    num_fmt = Param.String(
        "bf16", "Operand format: bf16, fp16, fp32, int8, int16 or int32"
    )
    acc_fmt = Param.String("fp32", "Accumulator format, as num_fmt")
    values = Param.Bool(True, "Compute PIM values (off: timing only)")
    posted = Param.Bool(
        True, "Acknowledge store-like PIM requests at acceptance"
    )
    pim_queue_size = Param.Unsigned(32, "PIM queue entries per channel")
    sw_error_fatal = Param.Bool(
        True, "A PIM software error is fatal (off: counted, a no-op)"
    )
    expected_layout = VectorParam.String(
        [],
        "NAME=VALUE pairs of the PIM library's layout header "
        "(PIM_LAYOUT_<NAME>); start-up stops unless they equal the "
        "memory's own values (empty: no check)",
    )
