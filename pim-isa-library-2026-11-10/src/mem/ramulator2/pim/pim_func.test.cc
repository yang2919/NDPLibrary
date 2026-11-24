/*
 * Tests of the PIM functional model (docs/pim/04-memctrl-plan.md sections
 * 9.4 and 12.1) on a vector-backed bank store.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "mem/ramulator2/pim/pim_func.hh"

using namespace gem5;
using namespace gem5::pim;

namespace
{

/** Two units of four banks, eight rows of four 32-byte columns. */
FuncConfig
smallConfig(NumFmt num = NumFmt::Bf16, NumFmt acc = NumFmt::Fp32)
{
    FuncConfig c;
    c.units = 2;
    c.banksPerUnit = 4;
    c.rowsPerBank = 8; // rows 6 (register 0) and 7 (accumulator)
    c.colsPerRow = 4;
    c.colBytes = 32;
    c.numFmt = num;
    c.accFmt = acc;
    return c;
}

class VectorStore : public PimBankStore
{
  public:
    explicit VectorStore(const FuncConfig &c)
        : cfg(c), data(size_t(c.units) * c.banksPerUnit * c.rowsPerBank *
                       c.rowBytes())
    {}

    uint8_t *
    column(unsigned unit, unsigned bank, uint32_t row, unsigned col) override
    {
        EXPECT_LT(unit, cfg.units);
        EXPECT_LT(bank, cfg.banksPerUnit);
        EXPECT_LT(row, cfg.regRow0()); // never a window row
        EXPECT_LT(col, cfg.colsPerRow);
        return rowPtr(unit, bank, row) + size_t(col) * cfg.colBytes;
    }

    uint8_t *
    rowPtr(unsigned unit, unsigned bank, uint32_t row)
    {
        return data.data() +
               ((size_t(unit) * cfg.banksPerUnit + bank) * cfg.rowsPerBank +
                row) *
                   cfg.rowBytes();
    }

    FuncConfig cfg;
    std::vector<uint8_t> data;
};

/** Element i of a row as bf16 bits, or as an integer. */
uint32_t
bf16At(const uint8_t *row, unsigned i)
{
    return loadBits(NumFmt::Bf16, row + 2 * i);
}

void
setBf16(uint8_t *row, unsigned i, uint32_t bits)
{
    storeBits(NumFmt::Bf16, row + 2 * i, bits);
}

/** bf16 bits of a small integer value v (exact). */
uint32_t
bf16Of(int v)
{
    float f = float(v);
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u >> 16;
}

float
accValue(PimFunctionalModel &m, unsigned unit, unsigned bank)
{
    uint32_t u = loadBits(NumFmt::Fp32, m.acc(unit, bank));
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

struct Fixture : public ::testing::Test
{
    Fixture() : cfg(smallConfig()), store(cfg), model(cfg, &store)
    {
        // Row 3 of every bank of both units holds small integers v(u, k,
        // i), so that sums and the products used below are exact in
        // bf16.
        for (unsigned u = 0; u < cfg.units; u++)
            for (unsigned k = 0; k < cfg.banksPerUnit; k++)
                for (unsigned i = 0; i < 64; i++)
                    setBf16(store.rowPtr(u, k, 3), i, bf16Of(v(u, k, i)));
    }

    int v(unsigned u, unsigned k, unsigned i) const
    {
        return int((10 * k + i) % 64 + 100 * u) % 128;
    }

    FuncConfig cfg;
    VectorStore store;
    PimFunctionalModel model;
};

} // anonymous namespace

TEST(PimFunc, Config)
{
    FuncConfig c = smallConfig();
    EXPECT_EQ(PimFunctionalModel::checkConfig(c), nullptr);
    EXPECT_EQ(c.rowBytes(), 128);
    EXPECT_EQ(c.regRow0(), 6);
    c.numRegs = 0;
    EXPECT_NE(PimFunctionalModel::checkConfig(c), nullptr);
    c = smallConfig(NumFmt::Fp32, NumFmt::Bf16);
    EXPECT_NE(PimFunctionalModel::checkConfig(c), nullptr);
    c = smallConfig();
    c.rowsPerBank = 2;
    EXPECT_NE(PimFunctionalModel::checkConfig(c), nullptr);
    c = smallConfig();
    c.colBytes = 33;
    EXPECT_NE(PimFunctionalModel::checkConfig(c), nullptr);
}

TEST_F(Fixture, Checks)
{
    EXPECT_EQ(model.checkBank(PimOp::SbEwadd, 2, 3, 0, 64), nullptr);
    // Window rows.
    EXPECT_TRUE(model.isWindowRow(6));
    EXPECT_TRUE(model.isWindowRow(7));
    EXPECT_FALSE(model.isWindowRow(5));
    EXPECT_NE(model.checkBank(PimOp::SbMovBkgbuf, 0, 6, 0, 32), nullptr);
    EXPECT_NE(model.checkBank(PimOp::AbMac, 0, 7, 0, 32), nullptr);
    // Odd bank of an element-wise pair (the address's bank for AB).
    EXPECT_NE(model.checkBank(PimOp::SbEwmul, 1, 3, 0, 32), nullptr);
    EXPECT_NE(model.checkBank(PimOp::AbEwadd, 3, 3, 0, 32), nullptr);
    EXPECT_EQ(model.checkBank(PimOp::SbEwaddGbuf, 1, 3, 0, 32), nullptr);
    EXPECT_EQ(model.checkBank(PimOp::AbMac, 3, 3, 0, 32), nullptr);
    // Compute ranges must be element-aligned; copies need not be.
    EXPECT_NE(model.checkBank(PimOp::SbMac, 0, 3, 1, 32), nullptr);
    EXPECT_NE(model.checkBank(PimOp::AbEwaddGbuf, 0, 3, 0, 31), nullptr);
    EXPECT_EQ(model.checkBank(PimOp::SbMovGbufbk, 0, 3, 1, 31), nullptr);
    EXPECT_EQ(model.checkBank(PimOp::AbMovBkreg, 0, 3, 3, 5), nullptr);
    // More than one register: EW and the register copies are refused.
    FuncConfig c = smallConfig();
    c.numRegs = 2;
    VectorStore s(c);
    PimFunctionalModel m(c, &s);
    EXPECT_EQ(c.regRow0(), 5);
    EXPECT_NE(m.checkBank(PimOp::SbEwadd, 0, 3, 0, 32), nullptr);
    EXPECT_NE(m.checkBank(PimOp::AbEwmulGbuf, 0, 3, 0, 32), nullptr);
    EXPECT_NE(m.checkBank(PimOp::AbMovRegbk, 0, 3, 0, 32), nullptr);
    EXPECT_NE(m.checkBank(PimOp::AbMovBkreg, 0, 3, 0, 32), nullptr);
    EXPECT_EQ(m.checkBank(PimOp::SbMac, 0, 3, 0, 32), nullptr);
    EXPECT_EQ(m.checkBank(PimOp::SbMovBkgbuf, 0, 3, 0, 32), nullptr);
}

TEST_F(Fixture, SingleBankElementWise)
{
    // Elements 1 .. 19 of bank pair (2, 3) of unit 1, across the column
    // boundary at element 16; the other elements stay zero.
    model.bankOp(PimOp::SbEwadd, 1, 2, 3, 2, 40);
    const uint8_t *r = model.reg(1, 2, 0);
    for (unsigned i = 0; i < 64; i++) {
        const uint32_t want =
            i >= 1 && i < 20 ? bf16Of(v(1, 2, i) + v(1, 3, i)) : 0;
        EXPECT_EQ(bf16At(r, i), want) << i;
    }
    // Only the even bank's PU is written; unit 0 is untouched.
    for (unsigned i = 0; i < 128; i++) {
        EXPECT_EQ(model.reg(1, 3, 0)[i], 0);
        EXPECT_EQ(model.reg(0, 2, 0)[i], 0);
    }
    model.bankOp(PimOp::SbEwmul, 0, 0, 3, 0, 8);
    for (unsigned i = 0; i < 4; i++)
        EXPECT_EQ(bf16At(model.reg(0, 0, 0), i),
                  bf16Of(v(0, 0, i) * v(0, 1, i)));
}

TEST_F(Fixture, GlobalBufferForms)
{
    uint8_t g[128];
    for (unsigned i = 0; i < 64; i++)
        setBf16(g, i, bf16Of(i % 4));
    model.writeGb(0, 0, g, 128);
    model.bankOp(PimOp::SbEwaddGbuf, 0, 1, 3, 0, 128);
    model.bankOp(PimOp::SbEwmulGbuf, 0, 2, 3, 64, 96);
    for (unsigned i = 0; i < 64; i++) {
        EXPECT_EQ(bf16At(model.reg(0, 1, 0), i), bf16Of(int(i % 4) +
                                                         v(0, 1, i)));
        const uint32_t mul =
            i >= 32 && i < 48 ? bf16Of(int(i % 4) * v(0, 2, i)) : 0;
        EXPECT_EQ(bf16At(model.reg(0, 2, 0), i), mul) << i;
    }
    // The GB of unit 1 is separate.
    uint8_t g1[128];
    model.readGb(1, 0, g1, 128);
    for (unsigned i = 0; i < 128; i++)
        EXPECT_EQ(g1[i], 0);
    // RdGb returns the bytes from an offset on.
    uint8_t part[5];
    model.readGb(0, 3, part, 5);
    EXPECT_EQ(std::memcmp(part, g + 3, 5), 0);
}

TEST_F(Fixture, Moves)
{
    // SbMovBkgbuf of bytes 5 .. 69 (byte-exact), then SbMovGbufbk into
    // another bank's row 4.
    model.bankOp(PimOp::SbMovBkgbuf, 1, 3, 3, 5, 70);
    uint8_t g[128];
    model.readGb(1, 0, g, 128);
    for (unsigned i = 0; i < 128; i++)
        EXPECT_EQ(g[i], i >= 5 && i < 70 ? store.rowPtr(1, 3, 3)[i] : 0);
    model.bankOp(PimOp::SbMovGbufbk, 1, 0, 4, 0, 128);
    EXPECT_EQ(std::memcmp(store.rowPtr(1, 0, 4), g, 128), 0);
    // AbMovBkreg then AbMovRegbk round trip: every bank's row 3 into its
    // register, out into row 5.
    model.bankOp(PimOp::AbMovBkreg, 0, 2, 3, 0, 128);
    model.bankOp(PimOp::AbMovRegbk, 0, 0, 5, 0, 128);
    for (unsigned k = 0; k < 4; k++) {
        EXPECT_EQ(std::memcmp(model.reg(0, k, 0), store.rowPtr(0, k, 3), 128),
                  0);
        EXPECT_EQ(std::memcmp(store.rowPtr(0, k, 5), store.rowPtr(0, k, 3),
                              128),
                  0);
        // Unit 1 is untouched.
        for (unsigned i = 0; i < 128; i++)
            EXPECT_EQ(store.rowPtr(1, k, 5)[i], 0);
    }
}

TEST_F(Fixture, AllBankElementWise)
{
    model.bankOp(PimOp::AbEwadd, 0, 0, 3, 0, 128);
    for (unsigned k = 0; k < 4; k += 2) {
        for (unsigned i = 0; i < 64; i++)
            EXPECT_EQ(bf16At(model.reg(0, k, 0), i),
                      bf16Of(v(0, k, i) + v(0, k + 1, i)));
        for (unsigned i = 0; i < 128; i++)
            EXPECT_EQ(model.reg(0, k + 1, 0)[i], 0);
    }
    model.bankOp(PimOp::AbEwmul, 0, 0, 3, 0, 2);
    EXPECT_EQ(bf16At(model.reg(0, 2, 0), 0), bf16Of(v(0, 2, 0) *
                                                    v(0, 3, 0)));
    EXPECT_EQ(bf16At(model.reg(0, 2, 0), 1),
              bf16Of(v(0, 2, 1) + v(0, 3, 1)));
    uint8_t g[128] = {};
    setBf16(g, 0, bf16Of(2));
    model.writeGb(0, 0, g, 128);
    model.bankOp(PimOp::AbEwmulGbuf, 0, 1, 3, 0, 32);
    model.bankOp(PimOp::AbEwaddGbuf, 0, 1, 3, 32, 64);
    for (unsigned k = 0; k < 4; k++) {
        EXPECT_EQ(bf16At(model.reg(0, k, 0), 0), bf16Of(2 * v(0, k, 0)));
        EXPECT_EQ(bf16At(model.reg(0, k, 0), 1), 0);
        EXPECT_EQ(bf16At(model.reg(0, k, 0), 16), bf16Of(v(0, k, 16)));
    }
}

TEST_F(Fixture, MacAndSplitRequests)
{
    uint8_t g[128];
    for (unsigned i = 0; i < 64; i++)
        setBf16(g, i, bf16Of(i % 3 == 0 ? 1 : -1));
    model.writeGb(0, 0, g, 128);
    // SbMac over the whole row of bank 1.
    model.bankOp(PimOp::SbMac, 0, 1, 3, 0, 128);
    float want = 0;
    for (unsigned i = 0; i < 64; i++)
        want += float(i % 3 == 0 ? 1 : -1) * float(v(0, 1, i));
    EXPECT_EQ(accValue(model, 0, 1), want);
    EXPECT_EQ(accValue(model, 0, 0), 0.0f);
    // The same row in pieces that split columns: each product is added
    // once (exact values, so the order does not matter).
    model.bankOp(PimOp::SbMac, 0, 2, 3, 0, 10);
    model.bankOp(PimOp::SbMac, 0, 2, 3, 10, 50);
    model.bankOp(PimOp::SbMac, 0, 2, 3, 50, 128);
    want = 0;
    for (unsigned i = 0; i < 64; i++)
        want += float(i % 3 == 0 ? 1 : -1) * float(v(0, 2, i));
    EXPECT_EQ(accValue(model, 0, 2), want);
    // AbMac adds to every bank's accumulator.
    float before[4];
    for (unsigned k = 0; k < 4; k++)
        before[k] = accValue(model, 0, k);
    model.bankOp(PimOp::AbMac, 0, 0, 3, 0, 32);
    for (unsigned k = 0; k < 4; k++) {
        float add = 0;
        for (unsigned i = 0; i < 16; i++)
            add += float(i % 3 == 0 ? 1 : -1) * float(v(0, k, i));
        EXPECT_EQ(accValue(model, 0, k), before[k] + add) << k;
        EXPECT_EQ(accValue(model, 1, k), 0.0f);
    }
}

TEST(PimFunc, MacRoundsPerColumn)
{
    // fp32 operands: one column of 8 elements per update. Products
    // 2^24, 1, 1, ... are summed in order inside a column (RNE drops each
    // 1), but a 1 in the next column is added to the accumulator.
    FuncConfig c = smallConfig(NumFmt::Fp32, NumFmt::Fp32);
    VectorStore s(c);
    PimFunctionalModel m(c, &s);
    const float x[16] = {0x1p24f, 1, 1, 1, 0, 0, 0, 0,
                         1,       1, 0, 0, 0, 0, 0, 0};
    std::memcpy(s.rowPtr(0, 0, 0), x, sizeof(x));
    uint8_t g[128] = {};
    const float one = 1.0f;
    for (unsigned i = 0; i < 16; i++)
        std::memcpy(g + 4 * i, &one, 4);
    m.writeGb(0, 0, g, 128);
    m.bankOp(PimOp::SbMac, 0, 0, 0, 0, 64);
    // Column 0: 2^24 (+1 +1 +1 dropped); column 1: 2; 2^24 + 2 exact.
    EXPECT_EQ(accValue(m, 0, 0), 0x1p24f + 2);
}

TEST_F(Fixture, RegisterWindow)
{
    // WrReg in two chunks of register 0 of bank 2 of unit 1, then RdReg.
    uint8_t w[128];
    for (unsigned i = 0; i < 128; i++)
        w[i] = uint8_t(3 * i + 1);
    EXPECT_EQ(model.writeWindow(1, 2, 6, 0, w, 40), 0);
    EXPECT_EQ(model.writeWindow(1, 2, 6, 40, w + 40, 88), 0);
    EXPECT_EQ(std::memcmp(model.reg(1, 2, 0), w, 128), 0);
    uint8_t r[128];
    model.readWindow(1, 2, 6, 7, r, 50);
    EXPECT_EQ(std::memcmp(r, w + 7, 50), 0);
    // The accumulator row: 4 bytes of fp32, the rest reads 0 and ignores
    // writes.
    EXPECT_EQ(model.writeWindow(1, 2, 7, 2, w, 10), 8);
    EXPECT_EQ(model.acc(1, 2)[2], w[0]);
    EXPECT_EQ(model.acc(1, 2)[3], w[1]);
    EXPECT_EQ(model.writeWindow(1, 2, 7, 4, w, 10), 10);
    model.readWindow(1, 2, 7, 0, r, 32);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], w[0]);
    EXPECT_EQ(r[3], w[1]);
    for (unsigned i = 4; i < 32; i++)
        EXPECT_EQ(r[i], 0);
    // A register an EW command wrote is read back through the window.
    model.bankOp(PimOp::SbEwadd, 0, 0, 3, 0, 4);
    model.readWindow(0, 0, 6, 0, r, 4);
    EXPECT_EQ(bf16At(r, 0), bf16Of(v(0, 0, 0) + v(0, 1, 0)));
    EXPECT_EQ(bf16At(r, 1), bf16Of(v(0, 0, 1) + v(0, 1, 1)));
}

TEST(PimFunc, ValuesOff)
{
    FuncConfig c = smallConfig();
    c.values = false;
    PimFunctionalModel m(c, nullptr);
    uint8_t w[64];
    std::memset(w, 0x5a, sizeof(w));
    // Nothing is touched (the store is null) and loads return zeros.
    m.bankOp(PimOp::AbMac, 0, 0, 3, 0, 128);
    m.bankOp(PimOp::SbMovGbufbk, 1, 2, 3, 0, 128);
    m.writeGb(0, 0, w, 64);
    EXPECT_EQ(m.writeWindow(0, 0, 6, 0, w, 64), 64);
    uint8_t r[64];
    std::memset(r, 0xff, sizeof(r));
    m.readGb(0, 0, r, 64);
    for (uint8_t b : r)
        EXPECT_EQ(b, 0);
    std::memset(r, 0xff, sizeof(r));
    m.readWindow(0, 0, 6, 0, r, 64);
    for (uint8_t b : r)
        EXPECT_EQ(b, 0);
    EXPECT_TRUE(m.gbState().empty());
    EXPECT_TRUE(m.regState().empty());
}

TEST(PimFunc, IntegerFormats)
{
    FuncConfig c = smallConfig(NumFmt::Int8, NumFmt::Int32);
    VectorStore s(c);
    PimFunctionalModel m(c, &s);
    uint8_t *a = s.rowPtr(0, 0, 1), *b = s.rowPtr(0, 1, 1);
    for (unsigned i = 0; i < 128; i++) {
        a[i] = uint8_t(i);
        b[i] = uint8_t(200 - i);
    }
    m.bankOp(PimOp::SbEwadd, 0, 0, 1, 0, 128);
    for (unsigned i = 0; i < 128; i++)
        EXPECT_EQ(m.reg(0, 0, 0)[i], uint8_t(200));
    // MAC with the GB all ones: the sum of bank 1's signed elements.
    uint8_t g[128];
    std::memset(g, 1, sizeof(g));
    m.writeGb(0, 0, g, 128);
    m.bankOp(PimOp::SbMac, 0, 1, 1, 0, 128);
    int64_t want = 0;
    for (unsigned i = 0; i < 128; i++)
        want += int8_t(uint8_t(200 - i));
    EXPECT_EQ(loadInt(NumFmt::Int32, m.acc(0, 1)), want);
}
