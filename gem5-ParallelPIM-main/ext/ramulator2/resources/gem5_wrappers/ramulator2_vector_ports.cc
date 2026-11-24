#include "mem/ramulator2/ramulator2_vector_ports.hh"

#include <string>

#include "base/trace.hh"
#include "debug/Ramulator2.hh"

namespace gem5
{

namespace memory
{

Ramulator2VectorPorts::Ramulator2VectorPorts(const Params &p) :
    Ramulator2Base(p, p.ramulator_config, p.port_ports_connection_count),
    portRanges(p.port_ranges)
{
    for (PortID i = 0;
         i < static_cast<PortID>(p.port_ports_connection_count); ++i) {
        ports.push_back(std::make_unique<MemorySystemPort>(
            name() + ".ports" + std::to_string(i), *this, i));
    }

    DPRINTF(Ramulator2, "Instantiated vector-port Ramulator2 with %zu ports\n",
            ports.size());
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
