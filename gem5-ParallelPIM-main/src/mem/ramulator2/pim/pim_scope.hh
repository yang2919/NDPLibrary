/*
 * Scopes of Processing-in-Memory (PIM) and normal requests, and the
 * conflict predicate that orders them.
 *
 * | Requester      | Scope                                              |
 * |----------------|----------------------------------------------------|
 * | Normal burst   | (unit, bank, row)                                  |
 * | Single bank    | (unit, {bank}, row); SbEwadd and SbEwmul add bank+1 |
 * | All bank       | (unit, every bank, row)                            |
 * | Non-bank       | (unit): every request of the unit                  |
 *
 * Two scopes conflict if they are in the same unit and either is
 * unit-wide, or their bank sets intersect and their rows are equal.
 * Units are the parallelism units of the whole memory (every channel);
 * banks are numbered within a unit, the two banks of an element-wise pair
 * being 2k and 2k + 1 of one bank group.
 */

#ifndef __MEM_RAMULATOR2_PIM_PIM_SCOPE_HH__
#define __MEM_RAMULATOR2_PIM_PIM_SCOPE_HH__

#include <cstdint>

#include "mem/pim_op.hh"

namespace gem5
{

namespace pim
{

/** What a PIM operation addresses. */
enum class OpKind : uint8_t
{
    None,       // not a PIM operation
    SingleBank, // SbEwadd ... SbMovGbufbk
    AllBank,    // AbEwadd ... AbMovBkreg
    NonBank,    // RdGb, WrGb, RdReg, WrReg
};

constexpr OpKind
opKind(PimOp op)
{
    return op >= PimOp::SbEwadd && op <= PimOp::SbMovGbufbk ? OpKind::SingleBank
           : op >= PimOp::AbEwadd && op <= PimOp::AbMovBkreg ? OpKind::AllBank
           : op >= PimOp::RdGb && op <= PimOp::WrReg         ? OpKind::NonBank
                                                             : OpKind::None;
}

/** Element-wise operations on a bank pair (2k, 2k + 1). */
constexpr bool
opIsPairEw(PimOp op)
{
    return op == PimOp::SbEwadd || op == PimOp::SbEwmul ||
           op == PimOp::AbEwadd || op == PimOp::AbEwmul;
}

/** Operations that compute on elements (EW and MAC). */
constexpr bool
opComputes(PimOp op)
{
    return (op >= PimOp::SbEwadd && op <= PimOp::SbMac) ||
           (op >= PimOp::AbEwadd && op <= PimOp::AbMac);
}

/** Operations that write or read register 0 of the PUs. */
constexpr bool
opUsesReg0(PimOp op)
{
    return (op >= PimOp::SbEwadd && op <= PimOp::SbEwmulGbuf) ||
           (op >= PimOp::AbEwadd && op <= PimOp::AbEwmulGbuf) ||
           op == PimOp::AbMovRegbk || op == PimOp::AbMovBkreg;
}

/** The banks and row a request covers in one unit. */
struct Scope
{
    unsigned unit = 0;
    bool unitWide = false;
    unsigned bankLo = 0; // first bank, inclusive
    unsigned bankHi = 0; // last bank, inclusive
    uint32_t row = 0;

    /** A normal burst to (unit, bank, row). */
    static constexpr Scope
    normal(unsigned unit, unsigned bank, uint32_t row)
    {
        return Scope{unit, false, bank, bank, row};
    }

    /**
     * A PIM request of operation op at (unit, bank, row), in a unit of
     * banksPerUnit banks. The bank of an all-bank or non-bank request is
     * ignored.
     */
    static constexpr Scope
    pim(PimOp op, unsigned unit, unsigned bank, uint32_t row,
        unsigned banksPerUnit)
    {
        switch (opKind(op)) {
          case OpKind::SingleBank:
            return Scope{unit, false, bank, opIsPairEw(op) ? bank + 1 : bank,
                         row};
          case OpKind::AllBank:
            return Scope{unit, false, 0, banksPerUnit - 1, row};
          default:
            return Scope{unit, true, 0, banksPerUnit - 1, 0};
        }
    }
};

/** True if requests of scopes a and b must stay in acceptance order. */
constexpr bool
scopesConflict(const Scope &a, const Scope &b)
{
    if (a.unit != b.unit)
        return false;
    if (a.unitWide || b.unitWide)
        return true;
    return a.bankLo <= b.bankHi && b.bankLo <= a.bankHi && a.row == b.row;
}

} // namespace pim
} // namespace gem5

#endif // __MEM_RAMULATOR2_PIM_PIM_SCOPE_HH__
