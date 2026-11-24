/*
 * Derived from Ramulator 2.1, resources/gem5_wrappers/ramulator2_base.hh at
 * upstream commit c6c7a5d (ext/ramulator2), with the hardening of
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

#ifndef __MEM_RAMULATOR2_RAMULATOR2_BASE_HH__
#define __MEM_RAMULATOR2_RAMULATOR2_BASE_HH__

#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/statistics.hh"
#include "mem/abstract_mem.hh"
#include "params/AbstractMemory.hh"

namespace Ramulator
{
class ConfigNode;
class IFrontEnd;
class IMemorySystem;
} // namespace Ramulator

namespace gem5
{

namespace memory
{

/**
 * Common part of the gem5 wrappers around a Ramulator 2.1 memory system.
 *
 * A packet larger than one DRAM transaction is split: it is accepted once
 * its first transaction is in Ramulator 2.1, and the rest are fed from
 * tick(), while the port refuses new packets. Responses are queued per port
 * in order of their due time. Static frontend and backend latencies are
 * charged as MemCtrl does. Addresses are decoded from the start of the
 * memory's range. Atomic accesses do not reach Ramulator 2.1: as MemCtrl,
 * a read returns a closed-page latency and a write 0. Ramulator 2.1's
 * state is not checkpointed.
 *
 * Ramulator 2.1's statistics are written as YAML into the output directory,
 * per instance: <name>.ramulator_stats.yaml at exit, and
 * <name>.ramulator_stats.<tick>.yaml at every statistics dump. They are
 * also bridged into gem5's statistics: one group for the memory system
 * (memory_system), one per controller (channel0, channel1, ...) and, if it
 * has statistics, one for the frontend, with one value per numeric leaf.
 * At every dump, each value equals the tick-named file's.
 */
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
        Tick
        recvAtomic(PacketPtr pkt) override
        {
            return ramulator2.recvAtomic(pkt);
        };
        void
        recvFunctional(PacketPtr pkt) override
        {
            ramulator2.recvFunctional(pkt);
        };
        bool
        recvTimingReq(PacketPtr pkt) override
        {
            return ramulator2.recvTimingReq(pkt, portId);
        };
        void
        recvRespRetry() override
        {
            ramulator2.recvRespRetry(portId);
        };

        AddrRangeList getAddrRanges() const override;
    };

    /** A packet that spans several DRAM transactions. */
    struct SplitPacket
    {
        PacketPtr pkt;
        /** Transactions not yet completed (reads only). */
        unsigned left;
        /** For a subclass: its sequence number of the packet. */
        uint64_t seq = 0;
    };

    struct PortState
    {
        bool retryReq = false;
        bool retryResp = false;
        /** Responses with their due tick, sorted by it. */
        std::deque<std::pair<Tick, PacketPtr>> responseQueue;
        /**
         * The packet whose remaining transactions still wait to be sent,
         * from pendingNext up to pendingEnd. While it is set, the port
         * accepts no new packet.
         */
        std::shared_ptr<SplitPacket> pendingSplit;
        Addr pendingNext = 0;
        Addr pendingEnd = 0;
        bool pendingWrite = false;
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

    unsigned int nbrOutstandingReads;
    unsigned int nbrOutstandingWrites;

    /** Charged on every response, as MemCtrl's static frontend latency. */
    const Tick frontendLatency;
    /** Charged on reads only, as MemCtrl's static backend latency. */
    const Tick backendLatency;
    /** Returned by atomic reads (writes return 0), as MemCtrl. */
    const Tick atomicReadLatency;

    /** The DRAM clock period, an integer number of picoseconds. */
    Tick dramClockPeriod;

    Ramulator2Base(const AbstractMemoryParams& p,
                   const std::string& ramulator_config,
                   size_t num_ports,
                   Tick frontend_latency,
                   Tick backend_latency,
                   Tick atomic_read_latency);
    ~Ramulator2Base();

    void initRamulator();

    /**
     * Requests not yet finished: reads in Ramulator 2.1, writes in
     * Ramulator 2.1, queued responses, and the parts of split packets
     * that are not sent yet.
     */
    unsigned int nbrOutstanding() const;

    /** Whether nothing is outstanding: draining may end. A subclass with
     * more state extends it. */
    virtual bool drained() const { return nbrOutstanding() == 0; }

    virtual MemorySystemPort& getMemoryPort(PortID port_id) = 0;
    virtual AddrRange getPortRange(PortID port_id) const = 0;
    virtual int getIngressId(PortID port_id) const = 0;

    /**
     * Send the transaction of [addr, end) that starts at addr to Ramulator
     * 2.1. Returns false if Ramulator 2.1 refused it.
     */
    bool sendChunk(Addr addr, Addr end, bool is_write, PortID port_id,
                   std::shared_ptr<SplitPacket> split);
    /** Send as many of the port's waiting transactions as are accepted. */
    void feedPendingSplit(PortID port_id);
    virtual void accessAndRespond(PacketPtr pkt, PortID port_id,
                                  Tick static_latency);
    /** Queue a response packet, due after static_latency. */
    void queueResponse(PacketPtr pkt, PortID port_id, Tick static_latency);
    void sendResponse(PortID port_id);

    /**
     * Hooks for a subclass, no-ops here. chunkSent: Ramulator 2.1 accepted
     * the transaction token at ctrl_addr (an offset in the memory) of the
     * packet of split; chunkDone: its callback fired (writes when the
     * write issues, reads when their data returns). cycle: every DRAM
     * cycle in timing mode, after Ramulator 2.1's tick.
     */
    virtual void chunkSent(uint64_t token, Addr ctrl_addr, bool is_write,
                           PortID port_id, SplitPacket& split) {}
    virtual void chunkDone(uint64_t token) {}
    virtual void cycle() {}
    uint64_t nextChunkToken = 1;

    enum class StatsWriteMode
    {
        Snapshot,
        Final
    };
    /** The per-instance YAML statistics file for the given suffix. */
    std::string statsFileName(const std::string& suffix) const;
    void writeRamulatorStats(const std::string& path, StatsWriteMode mode);

    /** Ramulator 2.1's statistics as YAML text, as in the files. */
    std::string ramulatorStatsText() const;

    /**
     * The statistics bridge (H10): Ramulator 2.1's numeric statistics as
     * gem5 statistics::Value objects. Built once in regStats() from the
     * statistics tree; each value reads the cache that preDumpStats()
     * refreshes from the same text it writes to the tick-named file.
     */
    struct StatsBridge
    {
        std::vector<std::unique_ptr<statistics::Group>> groups;
        std::vector<std::unique_ptr<statistics::Value>> values;
        /** Statistics-tree path of each value, to its cache index. */
        std::unordered_map<std::string, size_t> index;
        std::vector<double> cache;
    };
    StatsBridge statsBridge;

    /**
     * Walk one map of the statistics tree. With create, add a group per
     * map that holds numeric leaves and a value per numeric leaf; without
     * it, update the cache.
     */
    void bridgeStats(const Ramulator::ConfigNode& node,
                     statistics::Group* group, const std::string& path,
                     bool create);
    void updateStatsBridge(const std::string& stats_text);

    void tick();
    EventFunctionWrapper tickEvent;

  public:
    DrainState drain() override;

    void startup() override;
    void regStats() override;
    void resetStats() override;
    void preDumpStats() override;

  protected:
    // Virtual, so that a subclass (for example a PIM memory) can extend
    // the handling of the requests it receives.
    virtual Tick recvAtomic(PacketPtr pkt);
    virtual void recvFunctional(PacketPtr pkt);
    virtual bool recvTimingReq(PacketPtr pkt, PortID port_id);
    virtual void recvRespRetry(PortID port_id);
};

} // namespace memory
} // namespace gem5

#endif // __MEM_RAMULATOR2_RAMULATOR2_BASE_HH__
