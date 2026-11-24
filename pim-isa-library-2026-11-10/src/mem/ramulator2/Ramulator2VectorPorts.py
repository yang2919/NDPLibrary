# Derived from Ramulator 2.1, resources/gem5_wrappers/Ramulator2VectorPorts.py
# at upstream commit c6c7a5d (ext/ramulator2), with the static latencies of
# docs/pim/s1-ramulator-integration-plan.md section 4.1.
#
# MIT License
#
# Copyright (c) 2023 SAFARI Research Group at ETH Zurich and Carnegie Mellon
# University
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from m5.objects.AbstractMemory import *
from m5.params import *
from m5.SimObject import *


class Ramulator2VectorPorts(AbstractMemory):
    """A Ramulator 2.1 memory system with one response port per channel.

    Built by gem5.components.memory.ramulator2 (or by Ramulator 2.1's own
    ramulator.gem5.VectorPortMemory), which also sets ramulator_config and
    port_ranges. Statistics: <name>.ramulator_stats.yaml and
    <name>.ramulator_stats.<tick>.yaml in the output directory.
    """

    type = "Ramulator2VectorPorts"
    cxx_class = "gem5::memory::Ramulator2VectorPorts"
    cxx_header = "mem/ramulator2/ramulator2_vector_ports.hh"

    ports = VectorResponsePort(
        "Vector ports for receiving memory requests and sending responses"
    )
    port_ranges = VectorParam.AddrRange(
        [], "Address range served by each vector port"
    )
    ramulator_config = Param.String(
        "Ramulator 2 configuration (set automatically by "
        "gem5.components.memory.ramulator2 or "
        "ramulator.gem5.VectorPortMemory)"
    )

    # Verification using a gem5 native model: these defaults equal MemCtrl's
    # static latencies (src/mem/MemCtrl.py), and were checked against native
    # DDR4_2400_8x8 on the same device (docs/pim/ramulator2-dram-backend.md
    # section 4.3): with them, Ramulator 2.1 on DDR4-2400 ran within 1.3% of
    # native; without them, 19.5% to 24.5% fast.
    static_frontend_latency = Param.Latency(
        "10ns", "Charged on every response, as MemCtrl"
    )
    static_backend_latency = Param.Latency(
        "10ns", "Charged on reads only, as MemCtrl"
    )
    # Atomic accesses do not reach Ramulator 2.1. As MemCtrl::recvAtomic
    # (src/mem/mem_ctrl.cc), a read returns a closed-page latency and a
    # write 0; gem5.components.memory.ramulator2 sets nRP + nRCD + nCL of
    # the standard (P4R Q9 (d)).
    atomic_read_latency = Param.Latency(
        "50ns", "Atomic latency of a read (writes: 0)"
    )
