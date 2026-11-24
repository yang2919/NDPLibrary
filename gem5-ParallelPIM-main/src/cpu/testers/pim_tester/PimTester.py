# A CPU-less tester for Processing-in-Memory (PIM) memories.

from m5.objects.ClockedObject import ClockedObject
from m5.params import *
from m5.proxy import *


class PimTester(ClockedObject):
    """Plays a script of PIM, normal and functional requests on one port.

    Each script line is ``<delay> <kind> <addr> <size> [data] [expect]``:

    - ``delay``: ticks after the previous line was issued (a ``drain`` line
      is issued when every earlier response has arrived);
    - ``kind``: ``pim:<Op>`` (``pim:SbEwadd`` ... ``pim:WrReg``), ``rd``,
      ``wr``, ``frd``, ``fwr`` (functional), ``swap`` (an uncacheable
      atomic swap: writes ``data`` and returns the old bytes, checked
      against ``expect``) or ``drain``;
    - ``addr``, ``size``: the physical address (decimal or ``0x`` hex) and
      the size in bytes;
    - ``data``: the payload of a store-like request, as hexadecimal bytes
      in address order (``-`` or left out: zeros); a scalar PIM form's
      payload is its descriptor;
    - ``expect``: the response data of a load-like request, in the same
      form (``-`` or left out: not checked).

    Requests are timing or atomic as ``system.mem_mode`` says. When every
    line has completed, the tester writes ``csv_file`` (line, kind,
    address, size, send tick, response tick, data, result) to the output
    directory and exits the simulation with the number of mismatches as
    the exit code.
    """

    type = "PimTester"
    cxx_header = "cpu/testers/pim_tester/pim_tester.hh"
    cxx_class = "gem5::PimTester"

    script = Param.String("Script file")
    csv_file = Param.String("pim_tester.csv", "Results in the output dir")
    use_backdoor = Param.Bool(
        False,
        "In atomic mode, send with sendAtomicBackdoor and count the "
        "backdoors returned",
    )
    port = RequestPort("Port to the memory system")
    system = Param.System(Parent.any, "System this tester is part of")
