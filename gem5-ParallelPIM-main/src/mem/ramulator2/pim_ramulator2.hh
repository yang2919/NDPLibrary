/*
 * A Processing-in-Memory (PIM) memory on Ramulator 2.1.
 *
 * PimRamulator2 extends the vector-port wrapper with the PIM request path:
 * decode and validation on PimLayout, the functional model
 * (PimFunctionalModel) at acceptance, functional accesses to the register
 * window, and the checks of normal requests against the window. Its
 * Ramulator 2.1 memory system must be PimDRAM, with a PimSequencer on
 * every controller.
 *
 * Timing mode: one PIM queue per channel, in order per unit; a request
 * waits in Hazard for older conflicting normal requests, then goes to its
 * channel's PimSequencer, which issues its commands; younger conflicting
 * normal requests are held until it is Done. The functional effect happens
 * at acceptance; store-like requests are acknowledged at acceptance when
 * posted.
 *
 * Draining waits until every PIM queue, hold list and release queue is
 * empty and every sequencer idle, as well as the base's requests.
 * Checkpoints hold the functional state: the GBs,
 * registers and accumulators; Ramulator 2.1's own state is not saved.
 */

#ifndef __MEM_RAMULATOR2_PIM_RAMULATOR2_HH__
#define __MEM_RAMULATOR2_PIM_RAMULATOR2_HH__

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/statistics.hh"
#include "mem/ramulator2/pim/pim_func.hh"
#include "mem/ramulator2/pim/pim_layout.hh"
#include "mem/ramulator2/pim/pim_scope.hh"
#include "mem/ramulator2/ramulator2_vector_ports.hh"
#include "params/PimRamulator2.hh"

namespace Ramulator
{
class IPimMemorySystem;
struct PimJob;
} // namespace Ramulator

namespace gem5
{

namespace pim
{
class PimUnloaded;
} // namespace pim

namespace memory
{

class PimRamulator2 : public Ramulator2VectorPorts
{
  public:
    using Params = PimRamulator2Params;
    PimRamulator2(const Params &p);
    ~PimRamulator2();

    void init() override;
    void startup() override;
    DrainState drain() override;
    void serialize(CheckpointOut &cp) const override;
    void unserialize(CheckpointIn &cp) override;

  protected:
    Tick recvAtomic(PacketPtr pkt) override;
    void recvFunctional(PacketPtr pkt) override;
    bool recvTimingReq(PacketPtr pkt, PortID port_id) override;
    void accessAndRespond(PacketPtr pkt, PortID port_id,
                          Tick static_latency) override;
    void chunkSent(uint64_t token, Addr ctrl_addr, bool is_write,
                   PortID port_id, SplitPacket& split) override;
    void chunkDone(uint64_t token) override;
    void cycle() override;
    bool drained() const override;

  private:
    /** Bank columns in the backing store, at their encoded addresses. */
    class BankStore : public pim::PimBankStore
    {
      public:
        explicit BankStore(PimRamulator2 &m) : mem(m) {}
        uint8_t *column(unsigned unit, unsigned bank, uint32_t row,
                        unsigned col) override;

      private:
        PimRamulator2 &mem;
    };

    /** Offset of an address in the memory. */
    uint64_t offsetOf(Addr addr) const;

    /**
     * Decode and validate a PIM packet. A
     * simulator bug panics; a software error is fatal, or with
     * sw_error_fatal off is counted, and the function returns false: the
     * request is then a no-op.
     */
    bool decodeAndValidate(PacketPtr pkt, pim::Decoded &d);

    /** The functional effect of a valid request. */
    void execute(const pim::Decoded &d, PacketPtr pkt);

    /**
     * A normal packet that touches a register row is a software error.
     * Returns true if the packet does (and the error
     * was not fatal).
     */
    bool normalInWindow(PacketPtr pkt);

    /** Report a software error; returns only if it is not fatal. */
    void swError(PacketPtr pkt, const pim::Decoded *d,
                 const std::string &rule);

    /**
     * The unloaded latency U: acceptance to Done of the request on an
     * idle channel with every bank closed, in ticks.
     */
    Tick unloadedTicks(const pim::Decoded &d, int64_t accept_clk = -1) const;

    /** The sequencer job of a decoded request. */
    Ramulator::PimJob jobOf(const pim::Decoded &d, uint64_t id) const;

    /** Self-check of the layout against the memory system's decode. */
    void checkLayout() const;

    /**
     * gem5's value of every constant of the layout header
     * (include/gem5/pim/pim_layout_<std>.h), by name without the
     * PIM_LAYOUT_ prefix.
     */
    std::map<std::string, uint64_t> layoutValues() const;

    /** Compare expected_layout with layoutValues(); fatal on a mismatch. */
    void checkExpectedLayout() const;

    // Timing path ---------------------------------------------------------

    enum class State
    {
        Hazard,      // waiting for older conflicting normal requests
        Dispatched,  // with the sequencer, commands to issue
        LastCommand, // its last command issued; data still moving
        Done,
    };

    /** A queued PIM request. */
    struct Entry
    {
        uint64_t seq = 0;
        PacketPtr pkt = nullptr; // until its response, if at Done
        PortID port = 0;
        pim::Decoded d;
        pim::Scope scope;
        Tick acceptTick = 0;
        int64_t acceptClk = 0;
        State state = State::Hazard;
        bool waitsWrite = false;
        Tick blockedSince = MaxTick; // first found waiting for an older one
        int64_t doneClk = -1;
    };

    /** A normal transaction in Ramulator 2.1 whose column has not issued. */
    struct Record
    {
        uint64_t seq = 0;
        unsigned channel = 0;
        Addr ctrlAddr = 0;
        bool write = false;
        pim::Scope scope;
        bool retired = false; // its column issued
    };

    /** A held or released normal packet. */
    struct Held
    {
        uint64_t seq = 0;
        PacketPtr pkt = nullptr; // reads only
        PortID port = 0;
        bool write = false;
        Addr addr = 0;
        unsigned size = 0;
        std::vector<pim::Scope> scopes;
        Tick since = 0;
        Addr next = 0;
        std::shared_ptr<SplitPacket> split;
    };

    struct Channel
    {
        std::deque<Entry> queue;
        std::deque<Held> held;
        std::deque<Held> release;
        unsigned waitingReads = 0;  // held or released, not yet sent
        unsigned waitingWrites = 0;
        bool drainOn = false;
    };

    /** An accepted normal read not yet answered. */
    struct OutRead
    {
        uint64_t seq = 0;
        unsigned channel = 0;
        std::vector<pim::Scope> scopes;
        bool hasCopy = false;
        std::vector<uint8_t> copy;
    };

    /** The scopes of the transactions of [addr, addr + size). */
    std::vector<pim::Scope> scopesOf(Addr addr, unsigned size) const;
    /** True if a PIM entry of ch older than seq and not Done conflicts. */
    bool heldByPim(unsigned ch, const std::vector<pim::Scope>& scopes,
                   uint64_t seq) const;
    /** No older conflicting normal request is outstanding. */
    bool hazardClear(const Entry& e, bool& waitsWrite) const;
    void tryDispatch(unsigned ch);
    void processDone(unsigned ch);
    void releaseHolds(unsigned ch);
    void feedRelease(unsigned ch);
    /** Answer a request that a counted software error made a no-op. */
    void respondNoop(PacketPtr pkt, PortID port_id);
    /** Take copies for older reads of rows a PIM request writes. */
    void copyOlderReads(const pim::Decoded& d, const pim::Scope& scope);
    Entry* findEntry(unsigned ch, uint64_t seq);
    void hostColumnIssued(int channel, Addr addr, bool is_read);

    std::vector<Channel> chans;
    std::unordered_map<uint64_t, Record> records;
    /** Not retired records by (channel, address, read): the host-column
     * listener's lookup. */
    std::map<std::tuple<unsigned, Addr, bool>, std::deque<uint64_t>>
        recordIndex;
    std::unordered_set<uint64_t> doneEarly;
    std::unordered_map<PacketPtr, OutRead> outReads;
    uint64_t nextSeq = 1;
    uint64_t sendingSeq = 0;
    const bool posted;
    const unsigned queueSize;

    Ramulator::IPimMemorySystem *pimSystem = nullptr;

    const std::string policyParam;
    const uint64_t interleaveParam;
    const std::vector<std::string> expectedLayout;
    /** The DRAM standard's name, as Ramulator 2.1 reports it. */
    std::string standardName;
    const bool swErrorFatal;
    pim::FuncConfig funcCfg;
    pim::PimLayout layout;
    BankStore store;
    std::unique_ptr<pim::PimFunctionalModel> model;
    std::unique_ptr<pim::PimUnloaded> unloaded;

    struct PimStats : public statistics::Group
    {
        PimStats(PimRamulator2 &mem);
        void regStats() override;

        statistics::Vector reqs;
        statistics::Vector bytes;
        statistics::Vector atomicReqs;
        statistics::Vector atomicLat;
        statistics::Vector totLat;
        statistics::Formula avgLat;
        statistics::Vector hazardWaitTicks;
        statistics::Scalar heldPkts;
        statistics::Scalar heldTicks;
        statistics::Scalar holdFull;
        statistics::Scalar numRetry;
        statistics::Scalar writeDrainTicks;
        statistics::Scalar partialCols;
        statistics::Scalar swErrors;
        statistics::Scalar regWindowFunctional;
    } pimStats;
};

} // namespace memory
} // namespace gem5

#endif // __MEM_RAMULATOR2_PIM_RAMULATOR2_HH__
