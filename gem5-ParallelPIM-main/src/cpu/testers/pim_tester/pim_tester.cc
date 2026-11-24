/*
 * A CPU-less tester for PIM memories; see pim_tester.hh and PimTester.py.
 */

#include "cpu/testers/pim_tester/pim_tester.hh"

#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>

#include "base/logging.hh"
#include "base/output.hh"
#include "base/trace.hh"
#include "debug/PimTester.hh"
#include "mem/request.hh"
#include "sim/sim_exit.hh"
#include "sim/system.hh"

namespace gem5
{

namespace
{

bool
parseHex(const std::string &s, unsigned size, std::vector<uint8_t> &out)
{
    if (s.size() != 2 * size)
        return false;
    out.resize(size);
    for (unsigned i = 0; i < size; i++) {
        unsigned v = 0;
        for (int k = 0; k < 2; k++) {
            const char c = s[2 * i + k];
            unsigned d;
            if (c >= '0' && c <= '9')
                d = c - '0';
            else if (c >= 'a' && c <= 'f')
                d = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F')
                d = c - 'A' + 10;
            else
                return false;
            v = v * 16 + d;
        }
        out[i] = uint8_t(v);
    }
    return true;
}

std::string
toHex(const std::vector<uint8_t> &v)
{
    static const char digits[] = "0123456789abcdef";
    std::string s;
    s.reserve(2 * v.size());
    for (uint8_t b : v) {
        s += digits[b >> 4];
        s += digits[b & 0xf];
    }
    return s;
}

} // anonymous namespace

PimTester::PimTester(const Params &p)
    : ClockedObject(p), port(name() + ".port", *this), sys(p.system),
      csvFile(p.csv_file),
      requestorId(p.system->getRequestorId(this)),
      useBackdoor(p.use_backdoor),
      issueEvent([this] { issue(); }, name() + ".issue"), stats(this)
{
    parseScript(p.script);
    results.resize(lines.size());
}

Port &
PimTester::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "port")
        return port;
    return ClockedObject::getPort(if_name, idx);
}

void
PimTester::parseScript(const std::string &path)
{
    std::ifstream in(path);
    fatal_if(!in, "PimTester: cannot open script '%s'", path);
    std::string text;
    unsigned lineNo = 0;
    while (std::getline(in, text)) {
        lineNo++;
        const auto hash = text.find('#');
        if (hash != std::string::npos)
            text.erase(hash);
        std::istringstream ts(text);
        std::vector<std::string> tok;
        for (std::string t; ts >> t;)
            tok.push_back(t);
        if (tok.empty())
            continue;
        Line l;
        l.lineNo = lineNo;
        const char *where = path.c_str();
        fatal_if(tok.size() < 2, "%s:%u: '<delay> <kind> ...' expected",
                 where, lineNo);
        l.delay = std::stoull(tok[0], nullptr, 0);
        const std::string &kind = tok[1];
        if (kind == "drain") {
            fatal_if(tok.size() != 2, "%s:%u: drain takes no operands",
                     where, lineNo);
            l.kind = Kind::Drain;
            lines.push_back(l);
            continue;
        }
        if (kind == "rd") {
            l.kind = Kind::Read;
        } else if (kind == "wr") {
            l.kind = Kind::Write;
        } else if (kind == "frd") {
            l.kind = Kind::FuncRead;
        } else if (kind == "fwr") {
            l.kind = Kind::FuncWrite;
        } else if (kind == "swap") {
            l.kind = Kind::Swap;
        } else if (kind.rfind("pim:", 0) == 0) {
            l.kind = Kind::Pim;
            const std::string op = kind.substr(4);
            for (unsigned i = 1; i <= NumPimOps; i++) {
                if (op == pimOpName(PimOp(i)))
                    l.op = PimOp(i);
            }
            fatal_if(l.op == PimOp::None, "%s:%u: unknown PIM operation '%s'",
                     where, lineNo, op);
        } else {
            fatal("%s:%u: unknown kind '%s'", where, lineNo, kind);
        }
        fatal_if(tok.size() < 4 || tok.size() > 6,
                 "%s:%u: '<delay> <kind> <addr> <size> [data] [expect]' "
                 "expected",
                 where, lineNo);
        l.addr = std::stoull(tok[2], nullptr, 0);
        l.size = unsigned(std::stoul(tok[3], nullptr, 0));
        fatal_if(l.size == 0, "%s:%u: size 0", where, lineNo);
        const bool load = loadLike(l);
        l.data.assign(l.size, 0);
        if (tok.size() > 4 && tok[4] != "-") {
            fatal_if(load && l.kind != Kind::Swap,
                     "%s:%u: a load-like request has no data", where,
                     lineNo);
            fatal_if(!parseHex(tok[4], l.size, l.data),
                     "%s:%u: data must be %u hexadecimal bytes", where,
                     lineNo, l.size);
        }
        if (tok.size() > 5 && tok[5] != "-") {
            fatal_if(!load, "%s:%u: only a load-like request has an expect",
                     where, lineNo);
            fatal_if(!parseHex(tok[5], l.size, l.expect),
                     "%s:%u: expect must be %u hexadecimal bytes", where,
                     lineNo, l.size);
            l.hasExpect = true;
        }
        lines.push_back(l);
    }
}

bool
PimTester::loadLike(const Line &l) const
{
    return l.kind == Kind::Read || l.kind == Kind::FuncRead ||
           l.kind == Kind::Swap ||
           (l.kind == Kind::Pim && pimOpIsLoad(l.op));
}

PacketPtr
PimTester::makePacket(const Line &l) const
{
    Request::Flags flags = 0;
    if (l.kind == Kind::Pim) {
        flags = Request::PIM | Request::UNCACHEABLE | Request::STRICT_ORDER |
                Request::pimOpFlags(l.op);
    }
    if (l.kind == Kind::Swap) {
        // An uncacheable atomic swap (an AMO): both a read and a write.
        flags = Request::MEM_SWAP | Request::UNCACHEABLE;
    }
    auto req = std::make_shared<Request>(l.addr, l.size, flags, requestorId);
    PacketPtr pkt = l.kind == Kind::Swap ? new Packet(req, MemCmd::SwapReq)
                    : loadLike(l)        ? Packet::createRead(req)
                                         : Packet::createWrite(req);
    uint8_t *buf = new uint8_t[l.size];
    std::memcpy(buf, l.data.data(), l.size);
    pkt->dataDynamic(buf);
    return pkt;
}

void
PimTester::startup()
{
    atomic = sys->isAtomicMode();
    fatal_if(!atomic && !sys->isTimingMode(),
             "PimTester needs the timing or the atomic memory mode");
    if (lines.empty()) {
        finish();
        return;
    }
    schedule(issueEvent, curTick() + lines[0].delay);
}

void
PimTester::issue()
{
    assert(next < lines.size() && !retryPkt);
    const Line &l = lines[next];
    Result &r = results[next];
    if (l.kind == Kind::Drain) {
        // Issued once every earlier response has arrived.
        if (outstanding > 0)
            return;
        r.send = r.resp = curTick();
        r.done = true;
        advance();
        return;
    }
    PacketPtr pkt = makePacket(l);
    DPRINTF(PimTester, "line %u: %s %#x size %u\n", l.lineNo,
            pkt->cmdString(), l.addr, l.size);
    if (l.kind == Kind::FuncRead || l.kind == Kind::FuncWrite) {
        stats.functionalReqs++;
        r.send = curTick();
        port.sendFunctional(pkt);
        complete(next, pkt, curTick());
        advance();
        return;
    }
    if (l.kind == Kind::Pim)
        stats.pimReqs++;
    else
        stats.normalReqs++;
    if (atomic) {
        r.send = curTick();
        MemBackdoorPtr backdoor = nullptr;
        const Tick lat = useBackdoor ? port.sendAtomicBackdoor(pkt, backdoor)
                                     : port.sendAtomic(pkt);
        if (backdoor)
            stats.backdoors++;
        complete(next, pkt, curTick() + lat);
        advance();
        return;
    }
    if (!port.sendTimingReq(pkt)) {
        retryPkt = pkt;
        return;
    }
    r.send = curTick();
    inflight[pkt] = next;
    outstanding++;
    advance();
}

void
PimTester::recvReqRetry()
{
    assert(retryPkt);
    PacketPtr pkt = retryPkt;
    if (!port.sendTimingReq(pkt))
        return;
    retryPkt = nullptr;
    results[next].send = curTick();
    inflight[pkt] = next;
    outstanding++;
    advance();
}

bool
PimTester::recvTimingResp(PacketPtr pkt)
{
    auto it = inflight.find(pkt);
    panic_if(it == inflight.end(), "PimTester: unexpected response %s",
             pkt->print());
    const size_t idx = it->second;
    inflight.erase(it);
    assert(outstanding > 0);
    outstanding--;
    complete(idx, pkt, curTick());
    // A drain line waits for the last outstanding response.
    if (outstanding == 0 && next < lines.size() &&
        lines[next].kind == Kind::Drain && !issueEvent.scheduled() &&
        !retryPkt) {
        issue();
    } else if (outstanding == 0 && next == lines.size()) {
        finish();
    }
    return true;
}

void
PimTester::complete(size_t idx, PacketPtr pkt, Tick resp)
{
    const Line &l = lines[idx];
    Result &r = results[idx];
    r.resp = resp;
    r.done = true;
    if (l.kind != Kind::FuncRead && l.kind != Kind::FuncWrite)
        stats.totLatency += resp - r.send;
    if (pkt->isError()) {
        r.ok = false;
        stats.mismatches++;
        warn("PimTester: line %u: error response %s", l.lineNo,
             pkt->print());
    } else if (loadLike(l)) {
        r.hasData = true;
        const uint8_t *p = pkt->getConstPtr<uint8_t>();
        r.data.assign(p, p + l.size);
        if (l.hasExpect && r.data != l.expect) {
            r.ok = false;
            stats.mismatches++;
            warn("PimTester: line %u: read %s, expected %s", l.lineNo,
                 toHex(r.data), toHex(l.expect));
        }
    }
    DPRINTF(PimTester, "line %u: done at %llu%s\n", l.lineNo, resp,
            r.ok ? "" : " (mismatch)");
    delete pkt;
}

void
PimTester::advance()
{
    next++;
    if (next < lines.size()) {
        schedule(issueEvent, curTick() + lines[next].delay);
    } else if (outstanding == 0) {
        finish();
    }
}

void
PimTester::finish()
{
    if (finished)
        return;
    finished = true;
    OutputStream *os = simout.create(csvFile);
    std::ostream &out = *os->stream();
    out << "line,kind,addr,size,send_tick,resp_tick,data,result\n";
    for (size_t i = 0; i < lines.size(); i++) {
        const Line &l = lines[i];
        const Result &r = results[i];
        std::string kind;
        switch (l.kind) {
          case Kind::Pim:
            kind = std::string("pim:") + pimOpName(l.op);
            break;
          case Kind::Read:
            kind = "rd";
            break;
          case Kind::Write:
            kind = "wr";
            break;
          case Kind::FuncRead:
            kind = "frd";
            break;
          case Kind::FuncWrite:
            kind = "fwr";
            break;
          case Kind::Drain:
            kind = "drain";
            break;
          case Kind::Swap:
            kind = "swap";
            break;
        }
        out << l.lineNo << ',' << kind << ",0x" << std::hex << l.addr
            << std::dec << ',' << l.size << ',' << r.send << ',' << r.resp
            << ',' << (r.hasData ? toHex(r.data) : "-") << ','
            << (!r.ok ? "mismatch" : l.hasExpect ? "ok" : "-") << '\n';
    }
    simout.close(os);
    const unsigned bad = unsigned(stats.mismatches.value());
    inform("PimTester: %u lines, %u mismatches", lines.size(), bad);
    exitSimLoop("PimTester finished", bad ? 1 : 0);
}

bool
PimTester::TesterPort::recvTimingResp(PacketPtr pkt)
{
    return tester.recvTimingResp(pkt);
}

void
PimTester::TesterPort::recvReqRetry()
{
    tester.recvReqRetry();
}

PimTester::TesterStats::TesterStats(statistics::Group *parent)
    : statistics::Group(parent),
      ADD_STAT(pimReqs, statistics::units::Count::get(),
               "PIM requests sent (timing or atomic)"),
      ADD_STAT(normalReqs, statistics::units::Count::get(),
               "Normal requests sent (timing or atomic)"),
      ADD_STAT(functionalReqs, statistics::units::Count::get(),
               "Functional requests"),
      ADD_STAT(mismatches, statistics::units::Count::get(),
               "Responses with an error or unexpected data"),
      ADD_STAT(totLatency, statistics::units::Tick::get(),
               "Sum of the response latencies"),
      ADD_STAT(backdoors, statistics::units::Count::get(),
               "Backdoors returned to atomic requests (use_backdoor)")
{}

} // namespace gem5
