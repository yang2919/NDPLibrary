/*
 * Derived from Ramulator 2.1,
 * resources/gem5_wrappers/ramulator2_vector_ports.cc at upstream commit
 * c6c7a5d (ext/ramulator2), with the hardening of
 * docs/pim/s1-ramulator-integration-plan.md section 4.
 *
 * MIT License
 *
 * Copyright (c) 2023 SAFARI Research Group at ETH Zurich and Carnegie Mellon
 * University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the
 * following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
 * NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mem/ramulator2/ramulator2_vector_ports.hh"

#include <string>

#include "base/trace.hh"
#include "debug/Ramulator2.hh"

namespace gem5
{

namespace memory
{

Ramulator2VectorPorts::Ramulator2VectorPorts(const Params &p) :
    Ramulator2Base(p, p.ramulator_config, p.port_ports_connection_count,
                   p.static_frontend_latency, p.static_backend_latency,
                   p.atomic_read_latency),
    portRanges(p.port_ranges)
{
    for (PortID i = 0;
         i < static_cast<PortID>(p.port_ports_connection_count); ++i) {
        ports.push_back(std::make_unique<MemorySystemPort>(
            name() + ".ports" + std::to_string(i), *this, i));
    }

    DPRINTF(Ramulator2, "Instantiated vector-port Ramulator2 with %zu "
            "ports\n", ports.size());
}

void
Ramulator2VectorPorts::init()
{
    AbstractMemory::init();

    if (ports.empty()) {
        fatal("Ramulator2VectorPorts %s has no connected vector ports!\n",
              name());
    }

    if (portRanges.size() != ports.size()) {
        fatal("Ramulator2VectorPorts %s requires port_ranges.size() (%zu) "
              "to match connected vector ports (%zu)\n",
              name(), portRanges.size(), ports.size());
    }

    for (PortID i = 0; i < static_cast<PortID>(ports.size()); ++i) {
        if (!ports[i]->isConnected()) {
            fatal("Ramulator2VectorPorts %s port %d is unconnected!\n",
                  name(), i);
        }
        ports[i]->sendRangeChange();
    }

    initRamulator();
}

Ramulator2VectorPorts::MemorySystemPort&
Ramulator2VectorPorts::getMemoryPort(PortID port_id)
{
    return *ports.at(port_id);
}

AddrRange
Ramulator2VectorPorts::getPortRange(PortID port_id) const
{
    return portRanges.at(port_id);
}

int
Ramulator2VectorPorts::getIngressId(PortID port_id) const
{
    return static_cast<int>(port_id);
}

Port&
Ramulator2VectorPorts::getPort(const std::string& if_name, PortID idx)
{
    if (if_name == "ports" &&
        idx >= 0 && idx < static_cast<PortID>(ports.size())) {
        return *ports[idx];
    }
    return ClockedObject::getPort(if_name, idx);
}

} // namespace memory
} // namespace gem5
