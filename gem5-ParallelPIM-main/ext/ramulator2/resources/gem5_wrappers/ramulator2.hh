#ifndef __MEM_RAMULATOR2_HH__
#define __MEM_RAMULATOR2_HH__

#include "mem/ramulator2/ramulator2_base.hh"
#include "params/Ramulator2.hh"

namespace gem5
{

namespace memory
{

class Ramulator2 : public Ramulator2Base
{
  private:
    MemorySystemPort port;

  protected:
    MemorySystemPort& getMemoryPort(PortID port_id) override;
    AddrRange getPortRange(PortID port_id) const override;
    int getIngressId(PortID port_id) const override;

  public:
    typedef Ramulator2Params Params;
    Ramulator2(const Params &p);

    Port& getPort(const std::string& if_name,
                  PortID idx = InvalidPortID) override;

    void init() override;
};

} // namespace memory
} // namespace gem5

#endif // __MEM_RAMULATOR2_HH__
