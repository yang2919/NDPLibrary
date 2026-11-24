/*
 * Geometry and address decode of a PIM memory; see pim_layout.hh.
 */

#include "mem/ramulator2/pim/pim_layout.hh"

#include <algorithm>

#include "base/intmath.hh"
#include "mem/ramulator2/pim/pim_scope.hh"

namespace gem5
{

namespace pim
{

namespace
{

constexpr const char *policyNames[] = {
    "RoRaBaChCo", "RoRaBaCoCh", "RoCoRaBaCh",
    "RoBaRaCoCh", "ChRaBaRoCo", "MOP4CLXOR",
};

bool
isPow2(uint64_t v)
{
    return v && !(v & (v - 1));
}

} // anonymous namespace

int
policyFromName(const std::string &name)
{
    for (int i = 0; i < 6; i++) {
        if (name == policyNames[i])
            return i;
    }
    return -1;
}

const char *
policyName(int policy)
{
    return policy >= 0 && policy < 6 ? policyNames[policy] : "invalid";
}

std::string
PimLayout::build(const LevelInfo &l, int policy, uint64_t interleave,
                 unsigned numRegs)
{
    if (l.names.size() != l.counts.size() || l.names.size() < 3)
        return "the organization needs at least channel, row and column";
    if (l.names.front() != "Channel" || l.names.back() != "Column")
        return "the organization must start with Channel and end with "
               "Column";
    if (policy < 0 || policy > 5)
        return "unknown address-mapping policy";
    nLevels = l.names.size();
    unitLevel = l.unitLevel > 0 ? l.unitLevel : -1;
    sidLevel = bgLevel = bankLevel = rowLevel = -1;
    colLevel = int(nLevels) - 1;
    // Below the unit: SID, bank group and bank, in this order, then row.
    const int first = unitLevel > 0 ? unitLevel + 1 : 1;
    int expect = 0; // 0: Sid, 1: BankGroup, 2: Bank, 3: Row
    for (int i = first; i < colLevel; i++) {
        const std::string &n = l.names[i];
        const int k = n == "Sid" ? 0
                      : n == "BankGroup" ? 1
                      : n == "Bank"      ? 2
                      : n == "Row"       ? 3
                                         : -1;
        if (k < expect)
            return "unexpected level " + n + " in the organization";
        expect = k + 1;
        (k == 0 ? sidLevel : k == 1 ? bgLevel : k == 2 ? bankLevel
                                                       : rowLevel) = i;
    }
    if (bankLevel < 0 || rowLevel < 0 || rowLevel != colLevel - 1)
        return "the organization needs bank and row levels above the column";
    if (unitLevel > 0 && unitLevel != 1)
        return "the unit must be the level below the channel";
    for (size_t i = 1; i < nLevels; i++) {
        if (!isPow2(l.counts[i]))
            return "level " + l.names[i] + " count is not a power of two";
    }
    if (!isPow2(l.channels) || !isPow2(l.txBytes))
        return "channels and transaction bytes must be powers of two";
    g.policy = uint32_t(policy);
    g.channels = uint32_t(l.channels);
    g.intlv_bytes = l.channels > 1 ? interleave : 1;
    g.units = unitLevel > 0 ? uint32_t(l.counts[unitLevel]) : 1;
    g.sids = sidLevel > 0 ? uint32_t(l.counts[sidLevel]) : 1;
    g.bank_groups = bgLevel > 0 ? uint32_t(l.counts[bgLevel]) : 1;
    g.banks_per_group = uint32_t(l.counts[bankLevel]);
    g.rows = uint32_t(l.counts[rowLevel]);
    g.cols = uint32_t(l.counts[colLevel]);
    g.col_bytes = uint32_t(l.txBytes);
    regs = numRegs;
    if (l.channels > 1 &&
        (!isPow2(interleave) || interleave < uint64_t(l.txBytes)))
        return "the channel interleave must be a power of two of at least "
               "one transaction";
    if (policy == PIM_POLICY_MOP4CLXOR && g.cols < 4)
        return "MOP4CLXOR needs at least four columns";
    if (g.banks_per_group % 2)
        return "an element-wise pair needs an even number of banks per "
               "bank group";
    if (numRegs < 1 || g.rows < numRegs + 2)
        return "the register window leaves no bank row";
    return "";
}

Shifts
PimLayout::shifts(uint64_t interleave) const
{
    // The fields in the order pim_geom_encode() places them, least
    // significant first (include/gem5/pim/pim_addr.h), each with its
    // width and whether it holds bit 0 of its field.
    const pim__bits_t b = pim__geom_bits(&g);
    enum F { Col, Unit, Sid, Bg, Bank, Row };
    struct Part
    {
        F f;
        unsigned w;
        bool low;
    };
    Part parts[7];
    size_t n = 0;
    auto put = [&](F f, unsigned w, bool low = true) {
        parts[n++] = Part{f, w, low};
    };
    switch (g.policy) {
      case PIM_POLICY_RoRaBaChCo:
      case PIM_POLICY_RoRaBaCoCh:
        put(Col, b.col);
        put(Bg, b.bg);
        put(Bank, b.bank);
        put(Sid, b.sid);
        put(Unit, b.unit);
        put(Row, b.row);
        break;
      case PIM_POLICY_RoCoRaBaCh:
        put(Col, b.stripe);
        put(Bg, b.bg);
        put(Bank, b.bank);
        put(Sid, b.sid);
        put(Unit, b.unit);
        put(Col, b.col - b.stripe, false);
        put(Row, b.row);
        break;
      case PIM_POLICY_RoBaRaCoCh:
        put(Col, b.col);
        put(Unit, b.unit);
        put(Sid, b.sid);
        put(Bg, b.bg);
        put(Bank, b.bank);
        put(Row, b.row);
        break;
      case PIM_POLICY_ChRaBaRoCo:
        put(Col, b.col);
        put(Row, b.row);
        put(Bank, b.bank);
        put(Bg, b.bg);
        put(Sid, b.sid);
        put(Unit, b.unit);
        break;
      default: // MOP4CLXOR
        put(Col, 2);
        put(Unit, b.unit);
        put(Sid, b.sid);
        put(Bg, b.bg);
        put(Bank, b.bank);
        put(Col, b.col - 2, false);
        put(Row, b.row);
        break;
    }
    const unsigned chShift = interleave ? unsigned(floorLog2(interleave)) : 0;
    const unsigned chBits = b.ch;
    unsigned pos[6] = {};
    unsigned at = b.tx;
    for (size_t i = 0; i < n; i++) {
        const Part &p = parts[i];
        if (p.low)
            pos[p.f] = at >= chShift ? at + chBits : at;
        at += p.w;
    }
    return Shifts{pos[Col], chShift,   pos[Unit], pos[Sid],
                  pos[Bg],  pos[Bank], pos[Row]};
}

uint64_t
PimLayout::encode(const pim_coord_t &c) const
{
    return pim_geom_encode(&g, &c);
}

uint32_t
PimLayout::decode(uint64_t off, pim_coord_t &c) const
{
    return pim_geom_decode(&g, off, &c);
}

unsigned
PimLayout::unitOf(const pim_coord_t &c) const
{
    return c.ch * g.units + c.unit;
}

unsigned
PimLayout::bankOf(const pim_coord_t &c) const
{
    return (c.sid * g.bank_groups + c.bg) * g.banks_per_group + c.bank;
}

pim_coord_t
PimLayout::coordOf(unsigned unit, unsigned bank, uint32_t row,
                   unsigned col) const
{
    pim_coord_t c;
    c.ch = unit / g.units;
    c.unit = unit % g.units;
    c.bank = bank % g.banks_per_group;
    c.bg = (bank / g.banks_per_group) % g.bank_groups;
    c.sid = bank / (g.banks_per_group * g.bank_groups);
    c.row = row;
    c.col = col;
    return c;
}

std::vector<int>
PimLayout::toAddrVec(const pim_coord_t &c) const
{
    std::vector<int> v(nLevels, 0);
    v[0] = int(c.ch);
    if (unitLevel > 0)
        v[unitLevel] = int(c.unit);
    if (sidLevel > 0)
        v[sidLevel] = int(c.sid);
    if (bgLevel > 0)
        v[bgLevel] = int(c.bg);
    v[bankLevel] = int(c.bank);
    v[rowLevel] = int(c.row);
    v[colLevel] = int(c.col);
    return v;
}

pim_coord_t
PimLayout::fromAddrVec(const std::vector<int> &v) const
{
    pim_coord_t c = {};
    c.ch = uint32_t(v.at(0));
    c.unit = unitLevel > 0 ? uint32_t(v.at(unitLevel)) : 0;
    c.sid = sidLevel > 0 ? uint32_t(v.at(sidLevel)) : 0;
    c.bg = bgLevel > 0 ? uint32_t(v.at(bgLevel)) : 0;
    c.bank = uint32_t(v.at(bankLevel));
    c.row = uint32_t(v.at(rowLevel));
    c.col = uint32_t(v.at(colLevel));
    return c;
}

Check
PimLayout::decodeRequest(PimOp op, uint64_t off, unsigned nbytes,
                         uint64_t desc, Decoded &d) const
{
    d = Decoded();
    d.op = op;
    const uint64_t cb = g.col_bytes;
    const OpKind kind = opKind(op);
    if (kind == OpKind::None)
        return {Fault::Panic, "not a PIM operation"};
    if (nbytes == 0 || off + nbytes > size())
        return {Fault::Panic, "request outside the PIM memory"};
    const uint32_t byte = decode(off, d.coord);
    d.unit = unitOf(d.coord);
    d.bank = bankOf(d.coord);
    d.row = d.coord.row;

    if (kind == OpKind::SingleBank && op <= PimOp::SbMac) {
        // The scalar forms: an 8-byte descriptor at an 8-byte address.
        if (nbytes != 8 || off % 8)
            return {Fault::Panic, "a scalar PIM request must be 8 bytes, "
                                  "8-byte aligned"};
        if (desc >> 16)
            return {Fault::SwError, "reserved descriptor bits [63:16] are "
                                    "not zero"};
        const unsigned n = unsigned(desc & PimDesc::NcolsMask)
                               ? unsigned(desc & PimDesc::NcolsMask)
                               : g.cols - d.coord.col;
        if (d.coord.col + n > g.cols)
            return {Fault::SwError, "the columns run past the end of the "
                                    "row"};
        d.firstCol = d.coord.col;
        d.ncols = n;
        d.lo = uint64_t(d.coord.col) * cb;
        d.hi = uint64_t(d.coord.col + n) * cb;
        return {};
    }

    if (op == PimOp::RdReg || op == PimOp::WrReg) {
        // Byte by byte through the register window (P4 section 8.3), one
        // column at a time.
        d.firstCol = d.coord.col;
        for (uint64_t i = 0; i < nbytes;) {
            pim_coord_t c;
            const uint32_t b = decode(off + i, c);
            const uint64_t n = std::min<uint64_t>(nbytes - i, cb - b);
            if (!isWindowRow(c.row))
                return {Fault::SwError, "the request leaves the register "
                                        "window"};
            Piece p;
            p.unit = unitOf(c);
            p.bank = bankOf(c);
            p.row = c.row;
            p.lo = uint64_t(c.col) * cb + b;
            p.hi = p.lo + n;
            p.reqOff = i;
            d.pieces.push_back(p);
            d.ncols++;
            if (b != 0 || n != cb)
                d.partialCols++;
            i += n;
        }
        return {};
    }

    // Vector bank operations and the GB operations: consecutive bytes of
    // one row from the first byte's column and byte on.
    d.lo = uint64_t(d.coord.col) * cb + byte;
    d.hi = d.lo + nbytes;
    if (d.hi > rowBytes())
        return {Fault::SwError, "the request crosses the end of the row"};
    if (kind != OpKind::NonBank) {
        for (uint64_t i = cb - byte; i < nbytes; i += cb) {
            pim_coord_t c;
            decode(off + i, c);
            if (c.ch != d.coord.ch || c.unit != d.coord.unit ||
                c.sid != d.coord.sid || c.bg != d.coord.bg ||
                c.bank != d.coord.bank || c.row != d.coord.row ||
                uint64_t(c.col) * cb != d.lo + i) {
                return {Fault::SwError, "the request crosses a row, bank or "
                                        "unit"};
            }
        }
    }
    d.firstCol = unsigned(d.lo / cb);
    d.ncols = unsigned((d.hi + cb - 1) / cb - d.lo / cb);
    if (d.lo % cb)
        d.partialCols++;
    if (d.hi % cb && (d.ncols > 1 || d.lo % cb == 0))
        d.partialCols++;
    return {};
}

} // namespace pim
} // namespace gem5
