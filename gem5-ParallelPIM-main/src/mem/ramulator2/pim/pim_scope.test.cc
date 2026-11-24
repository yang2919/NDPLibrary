/*
 * Tests of the PIM request scopes and their conflict predicate.
 */

#include <gtest/gtest.h>

#include "mem/ramulator2/pim/pim_scope.hh"

using namespace gem5;
using namespace gem5::pim;

namespace
{

constexpr unsigned Banks = 16;

Scope
pimAt(PimOp op, unsigned unit, unsigned bank, uint32_t row)
{
    return Scope::pim(op, unit, bank, row, Banks);
}

} // anonymous namespace

TEST(PimScope, Kinds)
{
    EXPECT_EQ(opKind(PimOp::None), OpKind::None);
    for (unsigned i = 1; i <= NumPimOps; i++) {
        const PimOp op = PimOp(i);
        const OpKind want = i <= 7 ? OpKind::SingleBank
                            : i <= 14 ? OpKind::AllBank
                                      : OpKind::NonBank;
        EXPECT_EQ(opKind(op), want) << pimOpName(op);
        EXPECT_EQ(opIsPairEw(op), op == PimOp::SbEwadd ||
                                      op == PimOp::SbEwmul ||
                                      op == PimOp::AbEwadd ||
                                      op == PimOp::AbEwmul)
            << pimOpName(op);
        EXPECT_EQ(opComputes(op), (i >= 1 && i <= 5) || (i >= 8 && i <= 12))
            << pimOpName(op);
    }
    EXPECT_TRUE(opUsesReg0(PimOp::AbMovRegbk));
    EXPECT_TRUE(opUsesReg0(PimOp::SbEwaddGbuf));
    EXPECT_FALSE(opUsesReg0(PimOp::SbMac));
    EXPECT_FALSE(opUsesReg0(PimOp::SbMovBkgbuf));
    EXPECT_FALSE(opUsesReg0(PimOp::WrReg));
}

TEST(PimScope, SingleBankAndPairs)
{
    const Scope mov = pimAt(PimOp::SbMovBkgbuf, 0, 4, 9);
    EXPECT_TRUE(scopesConflict(mov, Scope::normal(0, 4, 9)));
    EXPECT_FALSE(scopesConflict(mov, Scope::normal(0, 5, 9)));
    EXPECT_FALSE(scopesConflict(mov, Scope::normal(0, 4, 8)));
    // SbEwadd and SbEwmul also cover bank + 1.
    for (PimOp op : {PimOp::SbEwadd, PimOp::SbEwmul}) {
        const Scope ew = pimAt(op, 0, 4, 9);
        EXPECT_TRUE(scopesConflict(ew, Scope::normal(0, 4, 9)));
        EXPECT_TRUE(scopesConflict(ew, Scope::normal(0, 5, 9)));
        EXPECT_FALSE(scopesConflict(ew, Scope::normal(0, 3, 9)));
        EXPECT_FALSE(scopesConflict(ew, Scope::normal(0, 6, 9)));
        EXPECT_FALSE(scopesConflict(ew, Scope::normal(0, 5, 10)));
    }
    // The GB forms read one bank only.
    const Scope gbuf = pimAt(PimOp::SbEwaddGbuf, 0, 4, 9);
    EXPECT_FALSE(scopesConflict(gbuf, Scope::normal(0, 5, 9)));
}

TEST(PimScope, AllBankRow)
{
    // Row r in any bank of the unit, other rows free; the
    // address's bank is ignored.
    for (PimOp op : {PimOp::AbMac, PimOp::AbEwadd, PimOp::AbMovRegbk}) {
        const Scope ab = pimAt(op, 1, 7, 20);
        for (unsigned k = 0; k < Banks; k++) {
            EXPECT_TRUE(scopesConflict(ab, Scope::normal(1, k, 20))) << k;
            EXPECT_TRUE(scopesConflict(Scope::normal(1, k, 20), ab)) << k;
            EXPECT_FALSE(scopesConflict(ab, Scope::normal(1, k, 21))) << k;
            EXPECT_FALSE(scopesConflict(ab, Scope::normal(0, k, 20))) << k;
        }
    }
}

TEST(PimScope, NonBankUnitWide)
{
    for (PimOp op : {PimOp::RdGb, PimOp::WrGb, PimOp::RdReg, PimOp::WrReg}) {
        const Scope nb = pimAt(op, 2, 0, 0);
        EXPECT_TRUE(nb.unitWide);
        EXPECT_TRUE(scopesConflict(nb, Scope::normal(2, 15, 1234)));
        EXPECT_TRUE(scopesConflict(Scope::normal(2, 0, 0), nb));
        EXPECT_FALSE(scopesConflict(nb, Scope::normal(3, 0, 0)));
        EXPECT_TRUE(scopesConflict(nb, pimAt(PimOp::SbMac, 2, 3, 3)));
    }
}

TEST(PimScope, DifferentUnitsAndPimPairs)
{
    EXPECT_FALSE(scopesConflict(Scope::normal(0, 1, 2),
                                Scope::normal(1, 1, 2)));
    EXPECT_TRUE(scopesConflict(Scope::normal(0, 1, 2),
                               Scope::normal(0, 1, 2)));
    // Between PIM scopes the same rule holds.
    EXPECT_TRUE(scopesConflict(pimAt(PimOp::SbEwadd, 0, 2, 5),
                               pimAt(PimOp::SbMac, 0, 3, 5)));
    EXPECT_FALSE(scopesConflict(pimAt(PimOp::SbEwadd, 0, 2, 5),
                                pimAt(PimOp::SbMac, 0, 4, 5)));
    EXPECT_TRUE(scopesConflict(pimAt(PimOp::AbMac, 0, 0, 5),
                               pimAt(PimOp::SbMac, 0, 9, 5)));
    EXPECT_FALSE(scopesConflict(pimAt(PimOp::AbMac, 0, 0, 5),
                                pimAt(PimOp::AbMac, 0, 0, 6)));
    EXPECT_FALSE(scopesConflict(pimAt(PimOp::AbMac, 0, 0, 5),
                                pimAt(PimOp::WrGb, 1, 0, 0)));
    // The predicate is usable at compile time.
    static_assert(scopesConflict(Scope::normal(0, 1, 2),
                                 Scope::pim(PimOp::SbEwmul, 0, 0, 2, 4)),
                  "pair bank");
}
