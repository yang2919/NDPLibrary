/*
 * Derived from Ramulator 2.1,
 * resources/gem5_wrappers/ramulator2_vector_ports.hh at upstream commit
 * c6c7a5d (ext/ramulator2), hardened for this gem5 tree.
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

#ifndef __MEM_RAMULATOR2_RAMULATOR2_VECTOR_PORTS_HH__
#define __MEM_RAMULATOR2_RAMULATOR2_VECTOR_PORTS_HH__

#include <memory>
#include <vector>

#include "mem/ramulator2/ramulator2_base.hh"
#include "params/Ramulator2VectorPorts.hh"

namespace gem5
{

namespace memory
{

/**
 * A Ramulator 2.1 memory system with one response port per channel. gem5
 * selects the channel through each port's interleaved address range;
 * Ramulator 2.1's Gem5PortInterleave channel mapper sends a request to the
 * controller of the port it came from. One channel uses one port.
 */
class Ramulator2VectorPorts : public Ramulator2Base
{
  private:
    std::vector<std::unique_ptr<MemorySystemPort>> ports;
    std::vector<AddrRange> portRanges;

  protected:
    MemorySystemPort& getMemoryPort(PortID port_id) override;
    AddrRange getPortRange(PortID port_id) const override;
    int getIngressId(PortID port_id) const override;

  public:
    typedef Ramulator2VectorPortsParams Params;
    Ramulator2VectorPorts(const Params &p);

    Port& getPort(const std::string& if_name,
                  PortID idx = InvalidPortID) override;

    void init() override;
};

} // namespace memory
} // namespace gem5

#endif // __MEM_RAMULATOR2_RAMULATOR2_VECTOR_PORTS_HH__
