#ifndef __MEM_RAMULATOR2_VECTOR_PORTS_HH__
#define __MEM_RAMULATOR2_VECTOR_PORTS_HH__

#include <memory>
#include <vector>

#include "mem/ramulator2/ramulator2_base.hh"
#include "params/Ramulator2VectorPorts.hh"

namespace gem5
{

namespace memory
{

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

#endif // __MEM_RAMULATOR2_VECTOR_PORTS_HH__
