/*
 * A CPU-less tester for Processing-in-Memory (PIM) memories: it plays a
 * script of PIM, normal and functional requests on one request port and
 * checks the data of load-like responses. See PimTester.py for the script
 * format.
 *
 * PIM requests are built as the CPU builds them: Request::PIM,
 * UNCACHEABLE, STRICT_ORDER and the operation in Request::pimOpFlags(),
 * then Packet::createRead() for the load-like operations (RdGb, RdReg) and
 * Packet::createWrite() for the others.
 */

#ifndef __CPU_TESTERS_PIM_TESTER_PIM_TESTER_HH__
#define __CPU_TESTERS_PIM_TESTER_PIM_TESTER_HH__

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/statistics.hh"
#include "mem/packet.hh"
#include "mem/pim_op.hh"
#include "mem/port.hh"
#include "params/PimTester.hh"
#include "sim/clocked_object.hh"
#include "sim/eventq.hh"

namespace gem5
{

class System;

class PimTester : public ClockedObject
{
  public:
    using Params = PimTesterParams;
    PimTester(const Params &p);

    Port &getPort(const std::string &if_name,
                  PortID idx = InvalidPortID) override;
    void startup() override;

  private:
    enum class Kind
    {
        Pim,
        Read,
        Write,
        FuncRead,
        FuncWrite,
        Drain,
        Swap,
    };

    struct Line
    {
        unsigned lineNo = 0; // in the script file
        Tick delay = 0;
        Kind kind = Kind::Read;
        PimOp op = PimOp::None;
        Addr addr = 0;
        unsigned size = 0;
        std::vector<uint8_t> data;
        bool hasExpect = false;
        std::vector<uint8_t> expect;
    };

    struct Result
    {
        Tick send = MaxTick;
        Tick resp = MaxTick;
        bool done = false;
        bool hasData = false;
        bool ok = true;
        std::vector<uint8_t> data;
    };

    class TesterPort : public RequestPort
    {
      public:
        TesterPort(const std::string &name, PimTester &tester)
            : RequestPort(name), tester(tester)
        {}

      protected:
        bool recvTimingResp(PacketPtr pkt) override;
        void recvReqRetry() override;

      private:
        PimTester &tester;
    };

    /** Read and check the script; fatal on a malformed line. */
    void parseScript(const std::string &path);

    /** True for the lines whose response carries data. */
    bool loadLike(const Line &l) const;

    PacketPtr makePacket(const Line &l) const;

    /** Issue lines from next on, as far as time and the port allow. */
    void issue();

    /** Record a completed line and check its data. */
    void complete(size_t idx, PacketPtr pkt, Tick resp);

    bool recvTimingResp(PacketPtr pkt);
    void recvReqRetry();

    /** Schedule the next line, or finish. */
    void advance();
    void finish();

    TesterPort port;
    System *const sys;
    const std::string csvFile;
    const RequestorID requestorId;
    const bool useBackdoor;
    bool atomic = false;

    std::vector<Line> lines;
    std::vector<Result> results;
    size_t next = 0;
    unsigned outstanding = 0;
    PacketPtr retryPkt = nullptr;
    bool finished = false;
    std::unordered_map<PacketPtr, size_t> inflight;

    EventFunctionWrapper issueEvent;

    struct TesterStats : public statistics::Group
    {
        TesterStats(statistics::Group *parent);
        statistics::Scalar pimReqs;
        statistics::Scalar normalReqs;
        statistics::Scalar functionalReqs;
        statistics::Scalar mismatches;
        statistics::Scalar totLatency;
        statistics::Scalar backdoors;
    } stats;
};

} // namespace gem5

#endif // __CPU_TESTERS_PIM_TESTER_PIM_TESTER_HH__
