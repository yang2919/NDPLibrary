/*
 * Tests of the PIM memory layout (docs/pim/04r-ramulator-pim-plan.md
 * sections 6.1 and 9.2): decode and encode against the bit positions of
 * the default layout headers (include/gem5/pim/pim_layout_<std>.h), the
 * address vectors, and the request checks of P4 section 8.5.
 */

#include <gtest/gtest.h>

#include <random>

#include "mem/ramulator2/pim/pim_layout.hh"

using namespace gem5;
using namespace gem5::pim;

namespace
{

struct Std
{
    const char *name;
    LevelInfo levels;
    uint64_t interleave;
    // Bit positions of the default header (policy RoRaBaChCo).
    unsigned colShift, chShift, rowShift;
    uint32_t regRow0;
    uint64_t size;
};

LevelInfo
info(std::vector<std::string> names, std::vector<int> counts, int channels,
     int tx, int unit)
{
    LevelInfo l;
    l.names = names;
    l.counts = counts;
    l.channels = channels;
    l.txBytes = tx;
    l.unitLevel = unit;
    return l;
}

const std::vector<Std> &
standards()
{
    static const std::vector<Std> s = {
        {"LPDDR6",
         info({"Channel", "Rank", "BankGroup", "Bank", "Row", "Column"},
              {1, 1, 4, 4, 65536, 64}, 8, 32, 1),
         2048, 5, 11, 18, 65534, 16ULL << 30},
        {"DDR5",
         info({"Channel", "Rank", "BankGroup", "Bank", "Row", "Column"},
              {1, 2, 8, 4, 131072, 128}, 2, 64, 1),
         8192, 6, 13, 20, 131070, 128ULL << 30},
        {"GDDR7",
         info({"Channel", "Bank", "Row", "Column"}, {1, 16, 65536, 64}, 8,
              32, 0),
         2048, 5, 11, 18, 65534, 16ULL << 30},
        {"HBM4",
         info({"Channel", "PseudoChannel", "Sid", "BankGroup", "Bank", "Row",
               "Column"},
              {1, 2, 4, 2, 8, 16384, 32}, 8, 32, 1),
         1024, 5, 10, 20, 16382, 16ULL << 30},
    };
    return s;
}

PimLayout
make(const Std &s, int policy = PIM_POLICY_RoRaBaChCo, uint64_t intlv = 0)
{
    PimLayout l;
    const std::string why =
        l.build(s.levels, policy, intlv ? intlv : s.interleave, 1);
    EXPECT_EQ(why, "") << s.name;
    return l;
}

pim_coord_t
coord(unsigned ch, unsigned unit, unsigned sid, unsigned bg, unsigned bank,
      uint32_t row, unsigned col)
{
    pim_coord_t c;
    c.ch = ch;
    c.unit = unit;
    c.sid = sid;
    c.bg = bg;
    c.bank = bank;
    c.row = row;
    c.col = col;
    return c;
}

} // anonymous namespace

TEST(PimLayout, DefaultHeaderBitPositions)
{
    for (const Std &s : standards()) {
        const PimLayout l = make(s);
        EXPECT_EQ(l.size(), s.size) << s.name;
        EXPECT_EQ(l.regRow0(), s.regRow0) << s.name;
        EXPECT_EQ(l.encode(coord(0, 0, 0, 0, 0, 0, 1)), 1ULL << s.colShift)
            << s.name;
        EXPECT_EQ(l.encode(coord(1, 0, 0, 0, 0, 0, 0)), 1ULL << s.chShift)
            << s.name;
        EXPECT_EQ(l.encode(coord(0, 0, 0, 0, 0, 1, 0)), 1ULL << s.rowShift)
            << s.name;
    }
    // LPDDR6 in full: bank group at 14, bank at 16 (pim_layout_lpddr6.h).
    const PimLayout l = make(standards()[0]);
    EXPECT_EQ(l.encode(coord(0, 0, 0, 1, 0, 0, 0)), 1ULL << 14);
    EXPECT_EQ(l.encode(coord(0, 0, 0, 0, 1, 0, 0)), 1ULL << 16);
    EXPECT_EQ(l.units(), 8);
    EXPECT_EQ(l.banksPerUnit(), 16);
    EXPECT_EQ(l.rowBytes(), 2048);
    // HBM4: 64 banks per pseudo channel, the pseudo channel the unit.
    const PimLayout h = make(standards()[3]);
    EXPECT_EQ(h.units(), 16);
    EXPECT_EQ(h.banksPerUnit(), 64);
    // GDDR7: the channel is the unit.
    const PimLayout g = make(standards()[2]);
    EXPECT_EQ(g.unitsPerChannel(), 1);
    EXPECT_EQ(g.units(), 8);
}

TEST(PimLayout, RoundTripsEveryPolicy)
{
    std::mt19937_64 rng(5);
    for (const Std &s : standards()) {
        for (int policy = 0; policy < 6; policy++) {
            const uint64_t intlv =
                policy == PIM_POLICY_RoRaBaChCo ? s.interleave
                : policy == PIM_POLICY_ChRaBaRoCo
                    ? s.size / s.levels.channels
                    : 256;
            const PimLayout l = make(s, policy, intlv);
            for (int i = 0; i < 2000; i++) {
                const uint64_t off = rng() % l.size();
                pim_coord_t c;
                const uint32_t byte = l.decode(off, c);
                ASSERT_LT(byte, l.colBytes());
                ASSERT_EQ(l.encode(c) + byte, off) << s.name << policy;
                const pim_coord_t back = l.fromAddrVec(l.toAddrVec(c));
                ASSERT_EQ(l.encode(back), l.encode(c));
                ASSERT_EQ(l.toAddrVec(c).size(), s.levels.names.size());
                const pim_coord_t again = l.coordOf(
                    l.unitOf(c), l.bankOf(c), c.row, c.col);
                ASSERT_EQ(l.encode(again), l.encode(c));
            }
        }
    }
}

TEST(PimLayout, AddressVectorsAndBanks)
{
    const PimLayout h = make(standards()[3]);
    const pim_coord_t c = coord(3, 1, 2, 1, 5, 77, 9);
    const std::vector<int> v = h.toAddrVec(c);
    EXPECT_EQ(v, (std::vector<int>{3, 1, 2, 1, 5, 77, 9}));
    // Banks: (SID x bank groups + bank group) x banks + bank; a pair's
    // banks are consecutive.
    EXPECT_EQ(h.bankOf(c), (2 * 2 + 1) * 8 + 5);
    EXPECT_EQ(h.unitOf(c), 3 * 2 + 1);
    const pim_coord_t even = coord(0, 0, 3, 1, 6, 0, 0);
    const pim_coord_t odd = coord(0, 0, 3, 1, 7, 0, 0);
    EXPECT_EQ(h.bankOf(odd), h.bankOf(even) + 1);
    const PimLayout g = make(standards()[2]);
    EXPECT_EQ(g.toAddrVec(coord(5, 0, 0, 0, 3, 8, 2)),
              (std::vector<int>{5, 3, 8, 2}));
}

TEST(PimLayout, BuildErrors)
{
    PimLayout l;
    Std s = standards()[0];
    EXPECT_EQ(l.build(s.levels, 9, 2048, 1), "unknown address-mapping "
                                             "policy");
    LevelInfo odd = s.levels;
    odd.counts[3] = 3;
    EXPECT_NE(l.build(odd, 0, 2048, 1), "");
    LevelInfo order = s.levels;
    std::swap(order.names[2], order.names[3]);
    EXPECT_NE(l.build(order, 0, 2048, 1), "");
    LevelInfo extra = s.levels;
    extra.names[2] = "Subarray";
    EXPECT_NE(l.build(extra, 0, 2048, 1), "");
    EXPECT_NE(l.build(s.levels, 0, 48, 1), "");    // interleave not pow2
    EXPECT_NE(l.build(s.levels, 0, 2048, 70000), ""); // no bank row left
    LevelInfo one = s.levels;
    one.counts[3] = 1;
    EXPECT_NE(l.build(one, 0, 2048, 1), ""); // no bank pair
}

TEST(PimLayout, ScalarRequests)
{
    const PimLayout l = make(standards()[0]);
    Decoded d;
    const uint64_t at = l.encode(coord(2, 0, 0, 1, 2, 9, 10));
    // ncols 0: to the end of the row.
    EXPECT_FALSE(l.decodeRequest(PimOp::SbMac, at, 8, 0, d));
    EXPECT_EQ(d.firstCol, 10);
    EXPECT_EQ(d.ncols, 54);
    EXPECT_EQ(d.lo, 10 * 32);
    EXPECT_EQ(d.hi, 64 * 32);
    EXPECT_EQ(d.unit, 2);
    EXPECT_EQ(d.bank, 1 * 4 + 2);
    EXPECT_EQ(d.row, 9);
    EXPECT_FALSE(l.decodeRequest(PimOp::SbEwadd, at, 8, 54, d));
    EXPECT_EQ(d.ncols, 54);
    // Past the end, reserved descriptor bits, size or alignment.
    EXPECT_EQ(l.decodeRequest(PimOp::SbMac, at, 8, 55, d).fault,
              Fault::SwError);
    EXPECT_EQ(l.decodeRequest(PimOp::SbMac, at, 8, 1ULL << 16, d).fault,
              Fault::SwError);
    EXPECT_EQ(l.decodeRequest(PimOp::SbMac, at, 16, 1, d).fault,
              Fault::Panic);
    EXPECT_EQ(l.decodeRequest(PimOp::SbMac, at + 4, 8, 1, d).fault,
              Fault::Panic);
    EXPECT_EQ(l.decodeRequest(PimOp::SbMac, l.size(), 8, 1, d).fault,
              Fault::Panic);
}

TEST(PimLayout, VectorRequests)
{
    const PimLayout l = make(standards()[0]);
    Decoded d;
    const uint64_t at = l.encode(coord(1, 0, 0, 2, 3, 100, 4)) + 8;
    // 100 bytes from byte 8 of column 4: columns 4 .. 7, two partial.
    EXPECT_FALSE(l.decodeRequest(PimOp::AbMac, at, 100, 0, d));
    EXPECT_EQ(d.lo, 4 * 32 + 8);
    EXPECT_EQ(d.hi, 4 * 32 + 108);
    EXPECT_EQ(d.firstCol, 4);
    EXPECT_EQ(d.ncols, 4);
    EXPECT_EQ(d.partialCols, 2);
    // Inside one column: one partial column.
    EXPECT_FALSE(l.decodeRequest(PimOp::SbMovBkgbuf, at, 8, 0, d));
    EXPECT_EQ(d.ncols, 1);
    EXPECT_EQ(d.partialCols, 1);
    // Whole columns: none partial.
    EXPECT_FALSE(l.decodeRequest(PimOp::AbEwadd, at - 8, 64, 0, d));
    EXPECT_EQ(d.partialCols, 0);
    // The end of the row.
    const uint64_t last = l.encode(coord(1, 0, 0, 2, 3, 100, 63));
    EXPECT_FALSE(l.decodeRequest(PimOp::AbMac, last, 32, 0, d));
    EXPECT_EQ(l.decodeRequest(PimOp::AbMac, last, 64, 0, d).fault,
              Fault::SwError);
    EXPECT_EQ(l.decodeRequest(PimOp::WrGb, last + 16, 32, 0, d).fault,
              Fault::SwError);
    // GB operations: bytes from the first byte's column on.
    EXPECT_FALSE(l.decodeRequest(PimOp::RdGb, at, 64, 0, d));
    EXPECT_EQ(d.lo, 4 * 32 + 8);
    EXPECT_EQ(d.unit, 1);
    // Under RoRaBaCoCh with a 64-byte interleave a 128-byte request
    // crosses into another channel.
    const PimLayout c = make(standards()[0], PIM_POLICY_RoRaBaCoCh, 64);
    const uint64_t a = c.encode(coord(0, 0, 0, 0, 0, 5, 0));
    EXPECT_FALSE(c.decodeRequest(PimOp::AbMac, a, 64, 0, d));
    EXPECT_EQ(c.decodeRequest(PimOp::AbMac, a, 128, 0, d).fault,
              Fault::SwError);
    EXPECT_EQ(c.decodeRequest(PimOp::SbMovGbufbk, a + 32, 64, 0, d).fault,
              Fault::SwError);
}

TEST(PimLayout, RegisterRequests)
{
    // Under RoRaBaCoCh on one channel, consecutive bytes of a register row
    // walk one bank's row, then the next bank's.
    Std one = standards()[0];
    one.levels.channels = 1;
    const PimLayout l = make(one, PIM_POLICY_RoRaBaCoCh, 64);
    Decoded d;
    const uint64_t at = l.encode(coord(0, 0, 0, 0, 0, l.regRow0(), 62));
    EXPECT_FALSE(l.decodeRequest(PimOp::RdReg, at + 16, 112, 0, d));
    ASSERT_EQ(d.pieces.size(), 4u);
    EXPECT_EQ(d.pieces[0].bank, 0);
    EXPECT_EQ(d.pieces[0].lo, 62 * 32 + 16);
    EXPECT_EQ(d.pieces[0].hi, 63 * 32);
    EXPECT_EQ(d.pieces[1].lo, 63 * 32);
    EXPECT_EQ(d.pieces[2].bank, 4); // the bank field's next value
    EXPECT_EQ(d.pieces[2].lo, 0);
    EXPECT_EQ(d.pieces[2].reqOff, 48);
    EXPECT_EQ(d.pieces[3].hi - d.pieces[3].lo, 112 - 48 - 32);
    EXPECT_EQ(d.ncols, 4);
    EXPECT_EQ(d.partialCols, 1);
    // The accumulator row is part of the window; row REG_ROW0 - 1 is not.
    const uint64_t acc = l.encode(coord(0, 0, 0, 1, 1, l.regRow0() + 1, 0));
    EXPECT_FALSE(l.decodeRequest(PimOp::WrReg, acc, 32, 0, d));
    const uint64_t below = l.encode(coord(0, 0, 0, 1, 1, l.regRow0() - 1, 0));
    EXPECT_EQ(l.decodeRequest(PimOp::RdReg, below, 32, 0, d).fault,
              Fault::SwError);
    EXPECT_EQ(l.decodeRequest(PimOp::None, below, 32, 0, d).fault,
              Fault::Panic);
}

TEST(PimLayout, ShiftsOfTheDefaultHeaders)
{
    // PIM_LAYOUT_{COL,CH,UNIT,SID,BG,BANK,ROW}_SHIFT and CONTIG_BYTES of
    // include/gem5/pim/pim_layout_<std>.h, in the order of standards();
    // an absent level has the position it would have.
    const unsigned want[4][7] = {
        {5, 11, 18, 18, 14, 16, 18},
        {6, 13, 19, 19, 14, 17, 20},
        {5, 11, 18, 18, 14, 14, 18},
        {5, 10, 19, 17, 13, 14, 20},
    };
    const uint64_t contig[4] = {2048, 8192, 2048, 1024};
    for (size_t i = 0; i < standards().size(); i++) {
        const Std &s = standards()[i];
        const PimLayout l = make(s);
        const Shifts sh = l.shifts(s.interleave);
        const unsigned got[7] = {sh.col, sh.ch,   sh.unit, sh.sid,
                                 sh.bg,  sh.bank, sh.row};
        for (int k = 0; k < 7; k++)
            EXPECT_EQ(got[k], want[i][k]) << s.name << " field " << k;
        EXPECT_EQ(l.contigBytes(), contig[i]) << s.name;
    }
}
