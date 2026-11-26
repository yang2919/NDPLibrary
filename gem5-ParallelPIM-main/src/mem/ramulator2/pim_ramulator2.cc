/*
 * A PIM memory on Ramulator 2.1; see pim_ramulator2.hh.
 */

#include "mem/ramulator2/pim_ramulator2.hh"

#include <algorithm>
#include <cstring>
#include <map>
#include <random>
#include <set>

#include "base/logging.hh"
#include "base/str.hh"
#include "base/trace.hh"
#include "debug/Drain.hh"
#include "debug/PIM.hh"
#include "mem/packet_access.hh"
#include "mem/ramulator2/pim/pim_scope.hh"
#include "mem/ramulator2/pim/pim_unloaded.hh"
#include "sim/full_system.hh"
#include "sim/serialize.hh"
#include "sim/system.hh"

// gem5 defines warn as a macro; protect Ramulator 2.1's headers from it.
#pragma push_macro("warn")
#undef warn

#include "ramulator/memory_system/i_memory_system.h"
#include "ramulator/memory_system/i_pim_memory_system.h"

#pragma pop_macro("warn")

namespace gem5
{

namespace memory
{

namespace
{

Ramulator::PimClass
classOf(PimOp op)
{
    using C = Ramulator::PimClass;
    switch (op) {
      case PimOp::SbEwadd:
      case PimOp::SbEwmul:
        return C::SbRd2;
      case PimOp::SbEwaddGbuf:
      case PimOp::SbEwmulGbuf:
      case PimOp::SbMac:
      case PimOp::SbMovBkgbuf:
        return C::SbRd;
      case PimOp::SbMovGbufbk:
        return C::SbWr;
      case PimOp::AbEwadd:
      case PimOp::AbEwmul:
      case PimOp::AbEwaddGbuf:
      case PimOp::AbEwmulGbuf:
        return C::AbEw;
      case PimOp::AbMac:
        return C::AbMac;
      case PimOp::AbMovRegbk:
        return C::AbRegbk;
      case PimOp::AbMovBkreg:
        return C::AbBkreg;
      case PimOp::RdGb:
        return C::GbRd;
      case PimOp::WrGb:
        return C::GbWr;
      case PimOp::RdReg:
        return C::RegRd;
      default:
        return C::RegWr;
    }
}

} // anonymous namespace

PimRamulator2::PimRamulator2(const Params &p)
    : Ramulator2VectorPorts(p), posted(p.posted),
      queueSize(p.pim_queue_size), policyParam(p.policy),
      interleaveParam(p.channel_interleave),
      expectedLayout(p.expected_layout),
      swErrorFatal(p.sw_error_fatal), store(*this), pimStats(*this)
{
    funcCfg.numRegs = p.num_regs;
    funcCfg.values = p.values;
    fatal_if(!pim::fmtFromName(p.num_fmt.c_str(), funcCfg.numFmt),
             "%s: unknown num_fmt '%s'", name(), p.num_fmt);
    fatal_if(!pim::fmtFromName(p.acc_fmt.c_str(), funcCfg.accFmt),
             "%s: unknown acc_fmt '%s'", name(), p.acc_fmt);
}

PimRamulator2::~PimRamulator2() = default;

void
PimRamulator2::init()
{
    Ramulator2VectorPorts::init();

    pimSystem =
        dynamic_cast<Ramulator::IPimMemorySystem *>(ramulator2_memorysystem);
    fatal_if(!pimSystem,
             "%s: a PIM memory needs the Ramulator 2.1 memory system PimDRAM",
             name());
    const Ramulator::PimGeometry geo = pimSystem->geometry();
    for (int ch = 0; ch < geo.channels; ch++) {
        fatal_if(!pimSystem->sequencer(ch),
                 "%s: channel %d's controller has no PimSequencer plugin",
                 name(), ch);
    }
    fatal_if(size_t(geo.channels) != portStates.size(),
             "%s: %d channels but %zu ports", name(), geo.channels,
             portStates.size());

    const int policy = pim::policyFromName(policyParam);
    fatal_if(policy < 0, "%s: unknown address-mapping policy '%s'", name(),
             policyParam);
    pim::LevelInfo levels;
    levels.names = geo.level_names;
    levels.counts = geo.level_counts;
    levels.channels = geo.channels;
    levels.txBytes = geo.tx_bytes;
    levels.unitLevel = geo.unit_level;
    const std::string why =
        layout.build(levels, policy, interleaveParam, funcCfg.numRegs);
    fatal_if(!why.empty(), "%s: %s", name(), why);
    fatal_if(layout.size() != getAddrRange().size(),
             "%s: the memory is %llu bytes, but its devices hold %llu; a PIM "
             "memory must be exactly its capacity",
             name(), getAddrRange().size(), layout.size());
    checkLayout();

    funcCfg.units = layout.units();
    funcCfg.banksPerUnit = layout.banksPerUnit();
    funcCfg.rowsPerBank = layout.rowsPerBank();
    funcCfg.colsPerRow = layout.colsPerRow();
    funcCfg.colBytes = layout.colBytes();
    if (funcCfg.values && !pmemAddr) {
        warn("%s: no backing store (null), so PIM values are off", name());
        funcCfg.values = false;
    }
    const char *bad = pim::PimFunctionalModel::checkConfig(funcCfg);
    fatal_if(bad, "%s: %s (num_fmt %s, acc_fmt %s, %u-byte columns)",
             name(), bad, pim::fmtName(funcCfg.numFmt),
             pim::fmtName(funcCfg.accFmt), funcCfg.colBytes);
    model = std::make_unique<pim::PimFunctionalModel>(
        funcCfg, funcCfg.values ? &store : nullptr);
    fatal_if(queueSize == 0, "%s: pim_queue_size must be at least 1", name());
    const Ramulator::PimTimingInfo info = pimSystem->timing_info(0);
    standardName = info.standard;
    unloaded = std::make_unique<pim::PimUnloaded>(info);
    chans.resize(layout.channels());
    pimSystem->set_host_column_listener(
        [this](int channel, Ramulator::Addr_t addr, bool is_read) {
            hostColumnIssued(channel, Addr(addr), is_read);
        });

    inform("%s: PIM memory, policy %s, %u units of %u banks, %u-byte rows, "
           "register rows from %u, %s operands, %s accumulator",
           name(), pim::policyName(policy), layout.units(),
           layout.banksPerUnit(), unsigned(layout.rowBytes()),
           layout.regRow0(), pim::fmtName(funcCfg.numFmt),
           pim::fmtName(funcCfg.accFmt));
}

void
PimRamulator2::checkLayout() const
{
    // Decode through the memory system equals the
    // inverse of the layout's encode() on sampled addresses, and the
    // round trip is the identity.
    std::mt19937_64 rng(1);
    const uint64_t size = layout.size();
    const uint64_t tx = layout.colBytes();
    for (int i = 0; i < 1024; i++) {
        uint64_t off = rng() % size;
        // Every column of the first and the last row too.
        if (i < int(layout.colsPerRow()))
            off = uint64_t(i) * tx;
        pim_coord_t c;
        const uint32_t byte = layout.decode(off, c);
        fatal_if(layout.encode(c) + byte != off,
                 "%s: the layout's encode() is not the inverse of its "
                 "decode() at offset %#x", name(), off);
        const std::vector<int> want = layout.toAddrVec(c);
        const Ramulator::AddrVec_t got = pimSystem->decode(int(c.ch), off);
        fatal_if(got.size() != want.size() ||
                     !std::equal(want.begin(), want.end(), got.begin()),
                 "%s: the memory system decodes offset %#x unlike the "
                 "layout (policy %s, interleave %llu); check the "
                 "controllers' addr_mapper and the channel interleave",
                 name(), off, policyParam, interleaveParam);
    }
}

void
PimRamulator2::startup()
{
    Ramulator2VectorPorts::startup();
    checkExpectedLayout();
}

std::map<std::string, uint64_t>
PimRamulator2::layoutValues() const
{
    // The constants libpim/gen_pim_layout.py writes, with gem5's values:
    // the constructed objects are the truth.
    const pim_geom_t &g = layout.geom();
    const Ramulator::PimGeometry geo = pimSystem->geometry();
    const std::string unitName =
        geo.unit_level > 0 ? geo.level_names[geo.unit_level] : "";
    auto levelCount = [&](const char *n) -> uint64_t {
        for (size_t i = 1; i < geo.level_names.size(); i++) {
            if (geo.level_names[i] == n)
                return uint64_t(geo.level_counts[i]);
        }
        return 0;
    };
    uint64_t std_id = 0;
    for (const auto &[prefix, id] :
         {std::pair<const char *, uint64_t>{"DDR5", 1}, {"LPDDR6", 2},
          {"GDDR7", 3}, {"HBM4", 4}}) {
        if (standardName.rfind(prefix, 0) == 0)
            std_id = id;
    }
    const pim::Shifts sh = layout.shifts(interleaveParam);
    const unsigned elem = pim::fmtBytes(funcCfg.numFmt);
    const AddrRange &r = getAddrRange();
    return {
        {"STD", std_id},
        {"POLICY", g.policy},
        {"CHANNELS", g.channels},
        {"INTLV_BYTES", interleaveParam},
        {"RANKS", unitName == "Rank" ? g.units : 0},
        {"PSEUDO_CHANNELS", unitName == "PseudoChannel" ? g.units : 0},
        {"SIDS", levelCount("Sid")},
        {"BANK_GROUPS", levelCount("BankGroup")},
        {"BANKS_PER_GROUP", g.banks_per_group},
        {"ROWS", g.rows},
        {"COLS", g.cols},
        {"COL_BYTES", g.col_bytes},
        {"ROW_BYTES", layout.rowBytes()},
        {"UNITS", g.units},
        {"BANKS_PER_UNIT", layout.banksPerUnit()},
        {"PUS_PER_BANK", 1},
        {"COL_SHIFT", sh.col},
        {"CH_SHIFT", sh.ch},
        {"UNIT_SHIFT", sh.unit},
        {"SID_SHIFT", sh.sid},
        {"BG_SHIFT", sh.bg},
        {"BANK_SHIFT", sh.bank},
        {"ROW_SHIFT", sh.row},
        {"CONTIG_BYTES", layout.contigBytes()},
        {"WINDOW_SIZE", r.size()},
        {"WINDOW_ROW0", 0},
        // The physical base in the mode this run is in; the other mode's
        // base is not compared.
        {"WINDOW_BASE_SE", r.start()},
        {"WINDOW_BASE_FS", r.start()},
        {"NUM_REGS", layout.numRegs()},
        {"REG_ROW0", layout.regRow0()},
        {"NUM_FMT", unsigned(funcCfg.numFmt)},
        {"ACC_FMT", unsigned(funcCfg.accFmt)},
        {"ELEM_BYTES", elem},
        {"ELEMS_PER_COL", elem ? g.col_bytes / elem : 0},
        {"ACC_BYTES", pim::fmtBytes(funcCfg.accFmt)},
    };
}

void
PimRamulator2::checkExpectedLayout() const
{
    if (expectedLayout.empty())
        return; // the configuration warned (no layout_file)
    const std::map<std::string, uint64_t> mine = layoutValues();
    const char *skipped = FullSystem ? "WINDOW_BASE_SE" : "WINDOW_BASE_FS";
    std::set<std::string> seen;
    for (const std::string &entry : expectedLayout) {
        const size_t eq = entry.find('=');
        fatal_if(eq == std::string::npos || eq == 0,
                 "%s: malformed expected_layout entry '%s'", name(), entry);
        const std::string key = entry.substr(0, eq);
        uint64_t value = 0;
        fatal_if(!to_number(entry.substr(eq + 1), value),
                 "%s: malformed expected_layout entry '%s'", name(), entry);
        const auto it = mine.find(key);
        fatal_if(it == mine.end(),
                 "PIM layout mismatch: %s: unknown constant in the library "
                 "header", key);
        fatal_if(!seen.insert(key).second,
                 "PIM layout mismatch: %s: given twice", key);
        if (key == skipped)
            continue;
        fatal_if(it->second != value,
                 "PIM layout mismatch: %s: library %llu, configuration %llu",
                 key, value, it->second);
    }
    for (const auto &[key, value] : mine) {
        fatal_if(!seen.count(key),
                 "PIM layout mismatch: %s: missing from the library header "
                 "(configuration %llu)", key, value);
    }
    inform("%s: the PIM layout header matches (%d constants)", name(),
           seen.size());
}

uint64_t
PimRamulator2::offsetOf(Addr addr) const
{
    return getAddrRange().getOffset(addr);
}

uint8_t *
PimRamulator2::BankStore::column(unsigned unit, unsigned bank, uint32_t row,
                                 unsigned col)
{
    const pim_coord_t c = mem.layout.coordOf(unit, bank, row, col);
    return mem.toHostAddr(mem.getAddrRange().start() + mem.layout.encode(c));
}

void
PimRamulator2::swError(PacketPtr pkt, const pim::Decoded *d,
                       const std::string &rule)
{
    const RequestorID id = pkt->req->requestorId();
    const std::string who = system()->getRequestorName(id);
    std::string where;
    if (d) {
        const pim_coord_t &c = d->coord;
        where = csprintf(" (channel %u, unit %u, bank %u, row %u, column %u)",
                         c.ch, c.unit, layout.bankOf(c), c.row, c.col);
    }
    const char *op = pkt->isPim() ? pimOpName(pkt->req->getPimOp())
                                  : pkt->cmdString().c_str();
    if (swErrorFatal) {
        fatal("%s: PIM software error: %s of %u bytes at %#x%s from %s: %s",
              name(), op, pkt->getSize(), pkt->getAddr(), where, who, rule);
    }
    pimStats.swErrors++;
    warn_once("%s: PIM software errors are counted, not fatal; the first: "
              "%s at %#x%s from %s: %s",
              name(), op, pkt->getAddr(), where, who, rule);
    DPRINTF(PIM, "software error: %s at %#x%s from %s: %s\n", op,
            pkt->getAddr(), where, who, rule);
}

bool
PimRamulator2::decodeAndValidate(PacketPtr pkt, pim::Decoded &d)
{
    const PimOp op = pkt->req->getPimOp();
    panic_if(!getAddrRange().contains(pkt->getAddr()) ||
                 !getAddrRange().contains(pkt->getAddr() + pkt->getSize() - 1),
             "%s: PIM request %s of %u bytes at %#x outside %s", name(),
             pimOpName(op), pkt->getSize(), pkt->getAddr(),
             getAddrRange().to_string());
    const bool scalar = op >= PimOp::SbEwadd && op <= PimOp::SbMac;
    const uint64_t desc =
        scalar && pkt->getSize() == 8 ? pkt->getLE<uint64_t>() : 0;
    pim::Check chk = layout.decodeRequest(op, offsetOf(pkt->getAddr()),
                                          pkt->getSize(), desc, d);
    panic_if(chk.fault == pim::Fault::Panic, "%s: PIM request %s at %#x: %s",
             name(), pimOpName(op), pkt->getAddr(), chk.rule);
    if (!chk && pim::opKind(op) != pim::OpKind::NonBank) {
        if (const char *why =
                model->checkBank(op, d.bank, d.row, d.lo, d.hi)) {
            chk = {pim::Fault::SwError, why};
        }
    }
    if (chk) {
        swError(pkt, &d, chk.rule);
        return false;
    }
    return true;
}

void
PimRamulator2::execute(const pim::Decoded &d, PacketPtr pkt)
{
    const PimOp op = d.op;
    const unsigned size = pkt->getSize();
    switch (op) {
      case PimOp::RdGb:
        model->readGb(d.unit, d.lo, pkt->getPtr<uint8_t>(), size);
        break;
      case PimOp::WrGb:
        model->writeGb(d.unit, d.lo, pkt->getConstPtr<uint8_t>(), size);
        break;
      case PimOp::RdReg:
        for (const pim::Piece &p : d.pieces) {
            model->readWindow(p.unit, p.bank, p.row, p.lo,
                              pkt->getPtr<uint8_t>() + p.reqOff, p.hi - p.lo);
        }
        break;
      case PimOp::WrReg: {
        uint64_t ignored = 0;
        for (const pim::Piece &p : d.pieces) {
            ignored += model->writeWindow(
                p.unit, p.bank, p.row, p.lo,
                pkt->getConstPtr<uint8_t>() + p.reqOff, p.hi - p.lo);
        }
        if (ignored && funcCfg.values) {
            DPRINTF(PIM, "WrReg at %#x: %llu accumulator-row bytes past "
                    "the accumulator ignored\n", pkt->getAddr(), ignored);
        }
        break;
      }
      default:
        model->bankOp(op, d.unit, d.bank, d.row, d.lo, d.hi);
        break;
    }
    DPRINTF(PIM, "%s at %#x size %u: unit %u bank %u row %u bytes [%llu, "
            "%llu), %u columns\n", pimOpName(op), pkt->getAddr(), size,
            d.unit, d.bank, d.row, d.lo, d.hi, d.ncols);
}

Ramulator::PimJob
PimRamulator2::jobOf(const pim::Decoded &d, uint64_t id) const
{
    Ramulator::PimJob job;
    job.id = id;
    job.cls = classOf(d.op);
    pim_coord_t at = d.coord;
    at.col = d.firstCol;
    const std::vector<int> v = layout.toAddrVec(at);
    job.addr.assign(v.begin(), v.end());
    job.ncols = int(d.ncols);
    return job;
}

Tick
PimRamulator2::unloadedTicks(const pim::Decoded &d, int64_t accept_clk) const
{
    // Two controller clocks pass from acceptance to the sequencer's first
    // chance to push: the job is handed over in the cycle after
    // acceptance and pushed in the next. U is never 0. Without an
    // acceptance clock (atomic mode) the first push is taken to be on a
    // rising edge, which only HBM4 distinguishes (odd clocks), so that U
    // is fixed per operation and size.
    const Ramulator::PimJob job = jobOf(d, 0);
    const std::vector<int> addr(job.addr.begin(), job.addr.end());
    const int64_t first = accept_clk >= 0 ? accept_clk + 2 : 1;
    const int64_t clocks =
        2 + unloaded->clocks(job.cls, addr, job.ncols, first);
    return Tick(clocks) * dramClockPeriod;
}

bool
PimRamulator2::normalInWindow(PacketPtr pkt)
{
    const uint64_t tx = layout.colBytes();
    const uint64_t start = offsetOf(pkt->getAddr());
    const uint64_t end = start + pkt->getSize();
    for (uint64_t off = start; off < end; off = (off | (tx - 1)) + 1) {
        pim_coord_t c;
        layout.decode(off, c);
        if (layout.isWindowRow(c.row)) {
            pim::Decoded d;
            d.coord = c;
            swError(pkt, &d, "a normal request reaches a register row");
            return true;
        }
    }
    return false;
}

Tick
PimRamulator2::recvAtomic(PacketPtr pkt)
{
    if (!pkt->isPim()) {
        if (normalInWindow(pkt)) {
            // Counted, not fatal: a no-op; loads return zeros.
            if (pkt->isRead())
                std::memset(pkt->getPtr<uint8_t>(), 0, pkt->getSize());
            if (pkt->needsResponse())
                pkt->makeResponse();
            return 0;
        }
        return Ramulator2VectorPorts::recvAtomic(pkt);
    }
    panic_if(pkt->cacheResponding(), "Should not see packets where cache "
             "is responding");
    const PimOp op = pkt->req->getPimOp();
    const unsigned i = unsigned(op) - 1;
    pim::Decoded d;
    // A counted software error issues no command: one DRAM clock, as the
    // latency must not be 0.
    Tick lat = dramClockPeriod;
    if (decodeAndValidate(pkt, d)) {
        execute(d, pkt);
        lat = unloadedTicks(d);
        pimStats.reqs[i]++;
        pimStats.bytes[i] += pkt->getSize();
        pimStats.atomicReqs[i]++;
        pimStats.atomicLat[i] += lat;
        DPRINTF(PIM, "atomic %s at %#x size %u: U %llu ticks\n", pimOpName(op),
                pkt->getAddr(), pkt->getSize(), lat);
    } else if (pkt->isRead()) {
        std::memset(pkt->getPtr<uint8_t>(), 0, pkt->getSize());
    }
    // Never the backing store: the response carries the load data, or
    // none.
    if (pkt->needsResponse())
        pkt->makeResponse();
    return lat;
}

void
PimRamulator2::recvFunctional(PacketPtr pkt)
{
    panic_if(pkt->isPim(), "%s: a functional PIM request (%s)", name(),
             pkt->print());
    const uint64_t tx = layout.colBytes();
    const uint64_t start = offsetOf(pkt->getAddr());
    const uint64_t end = start + pkt->getSize();
    bool window = false;
    for (uint64_t off = start; off < end && !window;
         off = (off | (tx - 1)) + 1) {
        pim_coord_t c;
        layout.decode(off, c);
        window = layout.isWindowRow(c.row);
    }
    if (!window) {
        Ramulator2VectorPorts::recvFunctional(pkt);
        return;
    }
    // Window rows read and write the register state;
    // other columns of the packet the backing store.
    pimStats.regWindowFunctional++;
    const bool read = pkt->isRead();
    panic_if(!read && !pkt->isWrite(), "%s: functional %s", name(),
             pkt->cmdString());
    uint8_t *data = pkt->getPtr<uint8_t>();
    for (uint64_t off = start; off < end;) {
        pim_coord_t c;
        const uint32_t byte = layout.decode(off, c);
        const uint64_t n = std::min<uint64_t>(end - off, tx - byte);
        uint8_t *p = data + (off - start);
        if (layout.isWindowRow(c.row)) {
            const uint64_t lo = uint64_t(c.col) * tx + byte;
            if (read) {
                model->readWindow(layout.unitOf(c), layout.bankOf(c), c.row,
                                  lo, p, n);
            } else {
                model->writeWindow(layout.unitOf(c), layout.bankOf(c), c.row,
                                   lo, p, n);
            }
        } else if (pmemAddr) {
            uint8_t *host = toHostAddr(getAddrRange().start() + off);
            if (read)
                std::memcpy(p, host, n);
            else
                std::memcpy(host, p, n);
        }
        off += n;
    }
    pkt->makeResponse();
}

bool
PimRamulator2::recvTimingReq(PacketPtr pkt, PortID port_id)
{
    panic_if(pkt->cacheResponding(), "Should not see packets where cache "
             "is responding");
    auto &st = *portStates.at(port_id);
    // As the base: a port waiting for a retry, or feeding a split packet,
    // refuses new packets.
    if (st.retryReq || st.pendingSplit) {
        st.retryReq = true;
        return false;
    }
    const unsigned ch = unsigned(port_id);
    if (!pkt->isPim()) {
        panic_if(!(pkt->isRead() || pkt->isWrite()),
                 "Should only see read and writes at memory controller, "
                 "saw %s to %#llx\n", pkt->cmdString(), pkt->getAddr());
        if (normalInWindow(pkt)) {
            respondNoop(pkt, port_id);
            return true;
        }
        Channel &c = chans.at(ch);
        std::vector<pim::Scope> scopes;
        if (!c.queue.empty()) {
            scopes = scopesOf(pkt->getAddr(), pkt->getSize());
            if (heldByPim(ch, scopes, nextSeq)) {
                // Held until the conflicting PIM requests are Done; held
                // packets share the buffers' capacity.
                const bool write = pkt->isWrite();
                const size_t used =
                    write ? c.waitingWrites +
                                pimSystem->write_buffer_size(int(ch))
                          : c.waitingReads +
                                pimSystem->read_buffer_size(int(ch));
                const size_t cap =
                    write ? pimSystem->write_buffer_capacity(int(ch))
                          : pimSystem->read_buffer_capacity(int(ch));
                if (used >= cap) {
                    pimStats.holdFull++;
                    st.retryReq = true;
                    return false;
                }
                Held h;
                h.seq = nextSeq++;
                h.port = port_id;
                h.write = write;
                h.addr = pkt->getAddr();
                h.size = pkt->getSize();
                h.scopes = scopes;
                h.since = curTick();
                pimStats.heldPkts++;
                DPRINTF(PIM, "hold %s %#x size %u (seq %llu)\n",
                        pkt->cmdString(), h.addr, h.size, h.seq);
                if (write) {
                    // Its data goes to the backing store now, in
                    // acceptance order, and it is acknowledged; only its
                    // Ramulator 2.1 requests wait.
                    c.waitingWrites++;
                    c.held.push_back(std::move(h));
                    accessAndRespond(pkt, port_id, frontendLatency);
                } else {
                    h.pkt = pkt;
                    c.waitingReads++;
                    ++nbrOutstandingReads;
                    outReads[pkt] = OutRead{h.seq, ch, scopes, false, {}};
                    c.held.push_back(std::move(h));
                }
                return true;
            }
        }
        const uint64_t seq = nextSeq;
        // Only pure reads: an AMO (SwapReq) is also a write, which the base
        // answers at once, before a read entry could ever be erased.
        const bool read = pkt->isRead() && !pkt->isWrite();
        sendingSeq = seq;
        const bool ok = Ramulator2VectorPorts::recvTimingReq(pkt, port_id);
        sendingSeq = 0;
        if (!ok)
            return false;
        nextSeq++;
        if (read) {
            if (scopes.empty())
                scopes = scopesOf(pkt->getAddr(), pkt->getSize());
            outReads[pkt] = OutRead{seq, ch, scopes, false, {}};
        }
        return true;
    }

    const PimOp op = pkt->req->getPimOp();
    pim::Decoded d;
    if (!decodeAndValidate(pkt, d)) {
        respondNoop(pkt, port_id);
        return true;
    }
    panic_if(d.coord.ch != ch, "%s: %s at %#x decodes to channel %u but "
             "arrived on port %d", name(), pimOpName(op), pkt->getAddr(),
             d.coord.ch, port_id);
    Channel &c = chans.at(ch);
    if (c.queue.size() >= queueSize) {
        pimStats.numRetry++;
        st.retryReq = true;
        return false;
    }
    Entry e;
    e.seq = nextSeq++;
    e.port = port_id;
    e.d = d;
    e.scope = pim::Scope::pim(op, d.unit, d.bank, d.row,
                              layout.banksPerUnit());
    e.acceptTick = curTick();
    e.acceptClk = pimSystem->clk(int(ch));
    if (op == PimOp::SbMovGbufbk || op == PimOp::AbMovRegbk)
        copyOlderReads(d, e.scope);
    execute(d, pkt);
    const unsigned i = unsigned(op) - 1;
    pimStats.reqs[i]++;
    pimStats.bytes[i] += pkt->getSize();
    pimStats.partialCols += d.partialCols;
    if (posted && !pimOpIsLoad(op)) {
        // Posted: acknowledged now, without the backing store (3.4).
        pkt->makeResponse();
        queueResponse(pkt, port_id, frontendLatency);
    } else {
        e.pkt = pkt;
    }
    DPRINTF(PIM, "accept %s at %#x size %u (seq %llu), unit %u bank %u "
            "row %u, %u columns\n", pimOpName(op), pkt->getAddr(),
            d.ncols ? unsigned(d.hi - d.lo) : 0, e.seq, d.unit, d.bank,
            d.row, d.ncols);
    c.queue.push_back(std::move(e));
    return true;
}

void
PimRamulator2::respondNoop(PacketPtr pkt, PortID port_id)
{
    const bool read = pkt->isRead();
    if (read)
        std::memset(pkt->getPtr<uint8_t>(), 0, pkt->getSize());
    if (pkt->needsResponse()) {
        pkt->makeResponse();
        queueResponse(pkt, port_id,
                      frontendLatency + (read ? backendLatency : 0));
    } else {
        portStates.at(port_id)->pendingDelete.reset(pkt);
    }
}

std::vector<pim::Scope>
PimRamulator2::scopesOf(Addr addr, unsigned size) const
{
    std::vector<pim::Scope> out;
    const uint64_t tx = layout.colBytes();
    const uint64_t start = offsetOf(addr);
    for (uint64_t off = start; off < start + size;
         off = (off | (tx - 1)) + 1) {
        pim_coord_t c;
        layout.decode(off, c);
        out.push_back(pim::Scope::normal(layout.unitOf(c), layout.bankOf(c),
                                         c.row));
    }
    return out;
}

bool
PimRamulator2::heldByPim(unsigned ch, const std::vector<pim::Scope> &scopes,
                         uint64_t seq) const
{
    for (const Entry &e : chans[ch].queue) {
        if (e.seq >= seq || e.state == State::Done)
            continue;
        for (const pim::Scope &s : scopes) {
            if (pim::scopesConflict(e.scope, s))
                return true;
        }
    }
    return false;
}

void
PimRamulator2::copyOlderReads(const pim::Decoded &d, const pim::Scope &scope)
{
    // The functional effect happens at acceptance, so an older read of a
    // row this request writes is answered from the row as it was before
    // the request.
    (void)d;
    for (auto &[pkt, r] : outReads) {
        if (r.hasCopy || r.channel != scope.unit / layout.unitsPerChannel())
            continue;
        bool hit = false;
        for (const pim::Scope &s : r.scopes)
            hit = hit || pim::scopesConflict(scope, s);
        if (!hit || !pmemAddr)
            continue;
        r.hasCopy = true;
        r.copy.resize(pkt->getSize());
        std::memcpy(r.copy.data(), toHostAddr(pkt->getAddr()),
                    pkt->getSize());
    }
}

void
PimRamulator2::accessAndRespond(PacketPtr pkt, PortID port_id,
                                Tick static_latency)
{
    if (pkt->isRead()) {
        auto it = outReads.find(pkt);
        if (it != outReads.end()) {
            const bool copy = it->second.hasCopy;
            if (copy) {
                std::memcpy(pkt->getPtr<uint8_t>(), it->second.copy.data(),
                            pkt->getSize());
            }
            outReads.erase(it);
            if (copy) {
                DPRINTF(PIM, "read %#x answered from its copy\n",
                        pkt->getAddr());
                // Counted as access() counts a read.
                if (collectStats) {
                    stats.numReads[pkt->req->requestorId()]++;
                    stats.bytesRead[pkt->req->requestorId()] +=
                        pkt->getSize();
                    if (pkt->req->isInstFetch()) {
                        stats.bytesInstRead[pkt->req->requestorId()] +=
                            pkt->getSize();
                    }
                }
                pkt->makeResponse();
                queueResponse(pkt, port_id, static_latency);
                return;
            }
        }
    }
    Ramulator2VectorPorts::accessAndRespond(pkt, port_id, static_latency);
}

void
PimRamulator2::chunkSent(uint64_t token, Addr ctrl_addr, bool is_write,
                         PortID port_id, SplitPacket &split)
{
    if (split.seq == 0)
        split.seq = sendingSeq;
    if (doneEarly.erase(token))
        return;
    pim_coord_t c;
    layout.decode(ctrl_addr, c);
    Record r;
    r.seq = split.seq;
    r.channel = unsigned(port_id);
    r.ctrlAddr = ctrl_addr;
    r.write = is_write;
    r.scope = pim::Scope::normal(layout.unitOf(c), layout.bankOf(c), c.row);
    records[token] = r;
    recordIndex[{r.channel, ctrl_addr, !is_write}].push_back(token);
}

void
PimRamulator2::chunkDone(uint64_t token)
{
    auto it = records.find(token);
    if (it == records.end()) {
        doneEarly.insert(token);
        return;
    }
    const Record &r = it->second;
    if (!r.retired) {
        auto ix = recordIndex.find({r.channel, r.ctrlAddr, !r.write});
        if (ix != recordIndex.end()) {
            auto &q = ix->second;
            q.erase(std::remove(q.begin(), q.end(), token), q.end());
            if (q.empty())
                recordIndex.erase(ix);
        }
    }
    records.erase(it);
}

void
PimRamulator2::hostColumnIssued(int channel, Addr addr, bool is_read)
{
    // The oldest matching record's column issued.
    auto ix = recordIndex.find({unsigned(channel), addr, is_read});
    if (ix == recordIndex.end())
        return;
    const uint64_t token = ix->second.front();
    ix->second.pop_front();
    if (ix->second.empty())
        recordIndex.erase(ix);
    auto it = records.find(token);
    if (it != records.end())
        it->second.retired = true;
}

PimRamulator2::Entry *
PimRamulator2::findEntry(unsigned ch, uint64_t seq)
{
    for (Entry &e : chans.at(ch).queue) {
        if (e.seq == seq)
            return &e;
    }
    return nullptr;
}

bool
PimRamulator2::hazardClear(const Entry &e, bool &waitsWrite) const
{
    const unsigned ch = e.d.coord.ch;
    bool clear = true;
    waitsWrite = false;
    for (const auto &[token, r] : records) {
        if (r.retired || r.channel != ch || r.seq >= e.seq ||
            !pim::scopesConflict(r.scope, e.scope)) {
            continue;
        }
        clear = false;
        waitsWrite = waitsWrite || r.write;
    }
    for (const std::deque<Held> *q : {&chans[ch].held, &chans[ch].release}) {
        for (const Held &h : *q) {
            if (h.seq >= e.seq)
                continue;
            for (const pim::Scope &s : h.scopes) {
                if (pim::scopesConflict(s, e.scope)) {
                    clear = false;
                    waitsWrite = waitsWrite || h.write;
                }
            }
        }
    }
    return clear;
}


void
PimRamulator2::tryDispatch(unsigned ch)
{
    Channel &c = chans[ch];
    // In order per unit: a unit's next request goes to the sequencer once
    // the previous one issued its last command and it passes Hazard.
    std::unordered_set<unsigned> busy, seen;
    for (const Entry &e : c.queue) {
        if (e.state == State::Dispatched)
            busy.insert(e.d.unit);
    }
    for (Entry &e : c.queue) {
        if (e.state != State::Hazard || !seen.insert(e.d.unit).second)
            continue;
        if (busy.count(e.d.unit))
            continue;
        bool waitsWrite = false;
        if (!hazardClear(e, waitsWrite)) {
            e.waitsWrite = waitsWrite;
            if (e.blockedSince == MaxTick)
                e.blockedSince = curTick();
            continue;
        }
        e.waitsWrite = false;
        const PimOp op = e.d.op;
        if (e.blockedSince != MaxTick) {
            pimStats.hazardWaitTicks[unsigned(op) - 1] +=
                curTick() - e.blockedSince;
        }
        Ramulator::PimJob job = jobOf(e.d, e.seq);
        job.accept_clk = e.acceptClk;
        e.state = State::Dispatched;
        busy.insert(e.d.unit);
        DPRINTF(PIM, "dispatch %s (seq %llu) to channel %u, %u columns\n",
                pimOpName(op), e.seq, ch, e.d.ncols);
        pimSystem->sequencer(int(ch))->submit(
            job,
            [this, ch](uint64_t id) {
                if (Entry *x = findEntry(ch, id)) {
                    DPRINTF(PIM, "last command %s (seq %llu) channel %u\n",
                            pimOpName(x->d.op), id, ch);
                    x->state = State::LastCommand;
                }
            },
            [this, ch](uint64_t id, Ramulator::Clk_t clk) {
                if (Entry *x = findEntry(ch, id)) {
                    x->state = State::Done;
                    x->doneClk = clk;
                }
            });
    }
}

void
PimRamulator2::processDone(unsigned ch)
{
    Channel &c = chans[ch];
    bool any = false;
    for (auto it = c.queue.begin(); it != c.queue.end();) {
        if (it->state != State::Done) {
            ++it;
            continue;
        }
        const unsigned i = unsigned(it->d.op) - 1;
        pimStats.totLat[i] += curTick() - it->acceptTick;
        if (it->pkt) {
            it->pkt->makeResponse();
            queueResponse(it->pkt, it->port,
                          frontendLatency + backendLatency);
        }
        DPRINTF(PIM, "done %s (seq %llu) after %llu ticks, clocks %lld to "
                "%lld, U %llu ticks\n", pimOpName(it->d.op), it->seq,
                curTick() - it->acceptTick, (long long)it->acceptClk,
                (long long)it->doneClk, unloadedTicks(it->d, it->acceptClk));
        it = c.queue.erase(it);
        any = true;
    }
    if (any)
        releaseHolds(ch);
}

void
PimRamulator2::releaseHolds(unsigned ch)
{
    Channel &c = chans[ch];
    for (auto it = c.held.begin(); it != c.held.end();) {
        if (heldByPim(ch, it->scopes, it->seq)) {
            ++it;
            continue;
        }
        pimStats.heldTicks += curTick() - it->since;
        DPRINTF(PIM, "release %s %#x (seq %llu)\n",
                it->write ? "write" : "read", it->addr, it->seq);
        c.release.push_back(std::move(*it));
        it = c.held.erase(it);
    }
}

void
PimRamulator2::feedRelease(unsigned ch)
{
    Channel &c = chans[ch];
    const Addr tx = ramulator2_memorysystem->get_tx_bytes();
    while (!c.release.empty()) {
        Held &h = c.release.front();
        const Addr end = h.addr + h.size;
        if (!h.split) {
            unsigned chunks = 0;
            for (Addr a = h.addr; a < end; a = (a | (tx - 1)) + 1)
                ++chunks;
            h.split = std::make_shared<SplitPacket>(
                SplitPacket{h.write ? nullptr : h.pkt, chunks, h.seq});
            h.next = h.addr;
        }
        while (h.next < end) {
            if (!sendChunk(h.next, end, h.write, h.port, h.split))
                return; // Ramulator 2.1 is full: again next cycle
            h.next = (h.next | (tx - 1)) + 1;
        }
        if (h.write)
            c.waitingWrites--;
        else
            c.waitingReads--;
        c.release.pop_front();
    }
}

void
PimRamulator2::cycle()
{
    for (unsigned ch = 0; ch < chans.size(); ch++) {
        Channel &c = chans[ch];
        processDone(ch);
        feedRelease(ch);
        tryDispatch(ch);
        // Chopim's backlog: packets released or split but
        // not yet sent. Held packets do not count: they wait for PIM.
        size_t backlog = c.release.size();
        const auto &ps = *portStates.at(ch);
        if (ps.pendingSplit) {
            const Addr tx = ramulator2_memorysystem->get_tx_bytes();
            backlog += (ps.pendingEnd - ps.pendingNext + tx - 1) / tx;
        }
        pimSystem->sequencer(int(ch))->set_external_backlog(int(backlog));
        // Keep the controller writing while a head request
        // waits for an older write.
        bool drain = false;
        for (const Entry &e : c.queue)
            drain = drain || (e.state == State::Hazard && e.waitsWrite);
        if (drain != c.drainOn) {
            c.drainOn = drain;
            pimSystem->request_write_drain(int(ch), drain);
        }
        if (drain)
            pimStats.writeDrainTicks += dramClockPeriod;
    }
    if (drainState() == DrainState::Draining && drained()) {
        DPRINTF(Drain, "PIM memory drained\n");
        signalDrainDone();
    }
}

bool
PimRamulator2::drained() const
{
    if (!Ramulator2VectorPorts::drained())
        return false;
    for (size_t ch = 0; ch < chans.size(); ch++) {
        const Channel &c = chans[ch];
        if (!c.queue.empty() || !c.held.empty() || !c.release.empty())
            return false;
        if (!pimSystem->sequencer(int(ch))->idle())
            return false;
    }
    // Every read has been answered, so none may still wait for a copy.
    panic_if(!outReads.empty(), "%s: drained with %d read entries left",
             name(), outReads.size());
    return true;
}

DrainState
PimRamulator2::drain()
{
    size_t entries = 0, held = 0, released = 0, busy = 0;
    for (size_t ch = 0; ch < chans.size(); ch++) {
        entries += chans[ch].queue.size();
        held += chans[ch].held.size();
        released += chans[ch].release.size();
        busy += pimSystem->sequencer(int(ch))->idle() ? 0 : 1;
    }
    const DrainState s = Ramulator2VectorPorts::drain();
    DPRINTF(Drain, "PIM drain: %d PIM requests, %d held, %d released, "
            "%d busy sequencers, %d outstanding in the base: %s\n",
            entries, held, released, busy, nbrOutstanding(),
            s == DrainState::Drained ? "drained" : "draining");
    return s;
}

void
PimRamulator2::serialize(CheckpointOut &cp) const
{
    Ramulator2VectorPorts::serialize(cp);
    // The queues are empty after the drain; the functional state remains.
    paramOut(cp, "pim_units", layout.units());
    paramOut(cp, "pim_banks_per_unit", layout.banksPerUnit());
    paramOut(cp, "pim_row_bytes", layout.rowBytes());
    paramOut(cp, "pim_num_regs", funcCfg.numRegs);
    paramOut(cp, "pim_acc_bytes", pim::fmtBytes(funcCfg.accFmt));
    paramOut(cp, "pim_num_fmt", std::string(pim::fmtName(funcCfg.numFmt)));
    paramOut(cp, "pim_acc_fmt", std::string(pim::fmtName(funcCfg.accFmt)));
    paramOut(cp, "pim_values", funcCfg.values);
    arrayParamOut(cp, "pim_gb", model->gbState());
    arrayParamOut(cp, "pim_regs", model->regState());
    arrayParamOut(cp, "pim_accs", model->accState());
}

void
PimRamulator2::unserialize(CheckpointIn &cp)
{
    Ramulator2VectorPorts::unserialize(cp);
    if (!cp.entryExists(Serializable::currentSection(), "pim_gb")) {
        warn("%s: the checkpoint holds no PIM state; global buffers, "
             "registers and accumulators start at zero", name());
        return;
    }
    unsigned units, banks, regs, acc;
    uint64_t row;
    bool values;
    paramIn(cp, "pim_units", units);
    paramIn(cp, "pim_banks_per_unit", banks);
    paramIn(cp, "pim_row_bytes", row);
    paramIn(cp, "pim_num_regs", regs);
    paramIn(cp, "pim_acc_bytes", acc);
    paramIn(cp, "pim_values", values);
    fatal_if(units != layout.units() || banks != layout.banksPerUnit() ||
                 row != layout.rowBytes() || regs != funcCfg.numRegs ||
                 acc != pim::fmtBytes(funcCfg.accFmt),
             "%s: the checkpoint's PIM state (%u units of %u banks, %llu-byte "
             "rows, %u registers, %u-byte accumulators) does not fit this "
             "memory", name(), units, banks, row, regs, acc);
    // The bytes mean values only in their formats: a checkpoint
    // of bf16 values must not restore as fp16 or int16.
    std::string num_fmt, acc_fmt;
    if (cp.entryExists(Serializable::currentSection(), "pim_num_fmt")) {
        paramIn(cp, "pim_num_fmt", num_fmt);
        paramIn(cp, "pim_acc_fmt", acc_fmt);
        fatal_if(num_fmt != pim::fmtName(funcCfg.numFmt) ||
                     acc_fmt != pim::fmtName(funcCfg.accFmt),
                 "%s: the checkpoint's PIM state holds %s operands and %s "
                 "accumulators, this memory uses %s and %s", name(),
                 num_fmt, acc_fmt, pim::fmtName(funcCfg.numFmt),
                 pim::fmtName(funcCfg.accFmt));
    } else {
        warn("%s: the checkpoint names no PIM number formats; assuming "
             "%s operands and %s accumulators", name(),
             pim::fmtName(funcCfg.numFmt), pim::fmtName(funcCfg.accFmt));
    }
    std::vector<uint8_t> gb, rg, ac;
    arrayParamIn(cp, "pim_gb", gb);
    arrayParamIn(cp, "pim_regs", rg);
    arrayParamIn(cp, "pim_accs", ac);
    if (!funcCfg.values) {
        if (values)
            warn("%s: values are off; the checkpoint's PIM state is "
                 "dropped", name());
        return;
    }
    if (!values) {
        warn("%s: the checkpoint was taken with values off; PIM state "
             "starts at zero", name());
        return;
    }
    fatal_if(gb.size() != model->gbState().size() ||
                 rg.size() != model->regState().size() ||
                 ac.size() != model->accState().size(),
             "%s: the checkpoint's PIM state has the wrong size", name());
    model->gbState() = gb;
    model->regState() = rg;
    model->accState() = ac;
}

PimRamulator2::PimStats::PimStats(PimRamulator2 &mem)
    : statistics::Group(&mem, "pim"),
      ADD_STAT(reqs, statistics::units::Count::get(),
               "Accepted PIM requests per operation"),
      ADD_STAT(bytes, statistics::units::Byte::get(),
               "Bytes of accepted PIM requests per operation"),
      ADD_STAT(atomicReqs, statistics::units::Count::get(),
               "Atomic PIM requests per operation"),
      ADD_STAT(atomicLat, statistics::units::Tick::get(),
               "Sum of the atomic latencies per operation"),
      ADD_STAT(totLat, statistics::units::Tick::get(),
               "Sum of the latencies from acceptance to Done (timing)"),
      ADD_STAT(avgLat,
               statistics::units::Rate<statistics::units::Tick,
                                       statistics::units::Count>::get(),
               "Average latency from acceptance to Done of the timing "
               "requests"),
      ADD_STAT(hazardWaitTicks, statistics::units::Tick::get(),
               "Time requests waited for older conflicting normal requests"),
      ADD_STAT(heldPkts, statistics::units::Count::get(),
               "Normal packets held behind PIM requests"),
      ADD_STAT(heldTicks, statistics::units::Tick::get(),
               "Sum of the times normal packets were held"),
      ADD_STAT(holdFull, statistics::units::Count::get(),
               "Refused attempts to send a normal packet: held and queued "
               "requests filled a buffer (once per retry)"),
      ADD_STAT(numRetry, statistics::units::Count::get(),
               "Refused attempts to send a PIM request: the channel's PIM "
               "queue was full (once per retry)"),
      ADD_STAT(writeDrainTicks, statistics::units::Tick::get(),
               "Time a write drain was requested for a waiting request"),
      ADD_STAT(partialCols, statistics::units::Count::get(),
               "Column commands of partly covered transactions"),
      ADD_STAT(swErrors, statistics::units::Count::get(),
               "PIM software errors (sw_error_fatal off)"),
      ADD_STAT(regWindowFunctional, statistics::units::Count::get(),
               "Functional accesses to the register window")
{}

void
PimRamulator2::PimStats::regStats()
{
    statistics::Group::regStats();
    for (statistics::Vector *v :
         {&reqs, &bytes, &atomicReqs, &atomicLat, &totLat, &hazardWaitTicks}) {
        v->init(NumPimOps).flags(statistics::nozero | statistics::total);
        for (unsigned i = 0; i < NumPimOps; i++)
            v->subname(i, pimOpName(PimOp(i + 1)));
    }
    avgLat.flags(statistics::nozero | statistics::nonan | statistics::total);
    // Timing requests only: totLat sums no atomic latency.
    avgLat = totLat / (reqs - atomicReqs);
    for (unsigned i = 0; i < NumPimOps; i++)
        avgLat.subname(i, pimOpName(PimOp(i + 1)));
    for (statistics::Scalar *v :
         {&heldPkts, &heldTicks, &holdFull, &numRetry, &writeDrainTicks,
          &partialCols, &swErrors, &regWindowFunctional}) {
        v->flags(statistics::nozero);
    }
}

} // namespace memory
} // namespace gem5
