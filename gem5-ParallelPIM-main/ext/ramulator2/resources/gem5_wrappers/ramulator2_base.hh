#ifndef __MEM_RAMULATOR2_BASE_HH__
#define __MEM_RAMULATOR2_BASE_HH__

#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mem/abstract_mem.hh"
#include "params/AbstractMemory.hh"

namespace Ramulator
{
  class IFrontEnd;
  class IMemorySystem;
}

namespace gem5
{

namespace memory
{

class Ramulator2Base : public AbstractMemory
{
  protected:
    class MemorySystemPort : public ResponsePort
    {
      private:
        Ramulator2Base& ramulator2;
        PortID portId;

      public:
        MemorySystemPort(const std::string& _name,
                         Ramulator2Base& _ramulator2,
                         PortID _port_id);

      protected:
        Tick recvAtomic(PacketPtr pkt) override
        {
            return ramulator2.recvAtomic(pkt);
        };
        void recvFunctional(PacketPtr pkt) override
        {
            ramulator2.recvFunctional(pkt);
        };
        bool recvTimingReq(PacketPtr pkt) override
        {
            return ramulator2.recvTimingReq(pkt, portId);
        };
        void recvRespRetry() override
        {
            ramulator2.recvRespRetry(portId);
        };

        AddrRangeList getAddrRanges() const override;
    };

    struct PortState
    {
        bool retryReq = false;
        bool retryResp = false;
        std::deque<PacketPtr> responseQueue;
        EventFunctionWrapper sendResponseEvent;
        std::unique_ptr<Packet> pendingDelete;

        PortState(Ramulator2Base& ramulator2, PortID port_id);
    };

    std::vector<std::unique_ptr<PortState>> portStates;

    std::string ramulator_config;
    Ramulator::IFrontEnd* ramulator2_frontend;
    Ramulator::IMemorySystem* ramulator2_memorysystem;
    bool ramulator2_finalized;

    Tick startTick;
    std::unordered_map<Addr, std::deque<PacketPtr>> outstandingReads;

    unsigned int nbrOutstandingReads;
    unsigned int nbrOutstandingWrites;

    Ramulator2Base(const AbstractMemoryParams& p,
                   const std::string& ramulator_config,
                   size_t num_ports);
    ~Ramulator2Base();

    void initRamulator();
    unsigned int nbrOutstanding() const;

    virtual MemorySystemPort& getMemoryPort(PortID port_id) = 0;
    virtual AddrRange getPortRange(PortID port_id) const = 0;
    virtual int getIngressId(PortID port_id) const = 0;

    void accessAndRespond(PacketPtr pkt, PortID port_id);
    void sendResponse(PortID port_id);

    enum class StatsWriteMode
    {
        Snapshot,
        Final
    };
    void writeRamulatorStats(const std::string& path, StatsWriteMode mode);

    void tick();
    EventFunctionWrapper tickEvent;

  public:
    DrainState drain() override;

    void startup() override;
    void resetStats() override;
    void preDumpStats() override;

  protected:
    Tick recvAtomic(PacketPtr pkt);
    void recvFunctional(PacketPtr pkt);
    bool recvTimingReq(PacketPtr pkt, PortID port_id);
    void recvRespRetry(PortID port_id);
};

} // namespace memory
} // namespace gem5

#endif // __MEM_RAMULATOR2_BASE_HH__
