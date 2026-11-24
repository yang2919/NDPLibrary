/*
 * Functional model of the PIM commands; see pim_func.hh.
 */

#include "mem/ramulator2/pim/pim_func.hh"

#include <algorithm>
#include <cassert>
#include <cstring>

#include "mem/ramulator2/pim/pim_scope.hh"

namespace gem5
{

namespace pim
{

const char *
PimFunctionalModel::checkConfig(const FuncConfig &cfg)
{
    if (const char *why = checkFormats(cfg.numFmt, cfg.accFmt, cfg.colBytes))
        return why;
    if (cfg.units == 0 || cfg.banksPerUnit == 0 || cfg.colsPerRow == 0)
        return "empty geometry";
    if (cfg.numRegs < 1)
        return "at least one register per PU is needed";
    if (cfg.rowsPerBank < cfg.numRegs + 2)
        return "the register window leaves no bank row";
    return nullptr;
}

PimFunctionalModel::PimFunctionalModel(const FuncConfig &c,
                                       PimBankStore *s)
    : cfg(c), store(s), rowBytes(c.rowBytes()), elem(fmtBytes(c.numFmt)),
      accBytes(fmtBytes(c.accFmt)),
      gbs(c.values ? size_t(c.units) * rowBytes : 0),
      regs(c.values ? size_t(c.units) * c.banksPerUnit * c.numRegs *
                          rowBytes
                    : 0),
      accs(c.values ? size_t(c.units) * c.banksPerUnit * accBytes : 0)
{
    assert(checkConfig(c) == nullptr);
    assert(store || !c.values);
}

uint8_t *
PimFunctionalModel::gb(unsigned unit)
{
    return gbs.data() + size_t(unit) * rowBytes;
}

uint8_t *
PimFunctionalModel::reg(unsigned unit, unsigned bank, unsigned j)
{
    return regs.data() + (size_t(pu(unit, bank)) * cfg.numRegs + j) *
                             rowBytes;
}

uint8_t *
PimFunctionalModel::acc(unsigned unit, unsigned bank)
{
    return accs.data() + size_t(pu(unit, bank)) * accBytes;
}

const char *
PimFunctionalModel::checkBank(PimOp op, unsigned bank, uint32_t row,
                              uint64_t lo, uint64_t hi) const
{
    if (isWindowRow(row))
        return "bank operation on a register-window row";
    if (opIsPairEw(op) && (bank & 1))
        return "odd bank of an element-wise pair";
    if (opUsesReg0(op) && cfg.numRegs > 1)
        return "the register operations need one register per PU";
    if (opComputes(op) && (lo % elem || hi % elem))
        return "compute range is not element-aligned";
    return nullptr;
}

void
PimFunctionalModel::bankOp(PimOp op, unsigned unit, unsigned bank,
                           uint32_t row, uint64_t lo, uint64_t hi)
{
    assert(lo <= hi && hi <= rowBytes);
    assert(unit < cfg.units && bank < cfg.banksPerUnit);
    if (!cfg.values)
        return;
    if (opKind(op) == OpKind::AllBank) {
        for (unsigned k = 0; k < cfg.banksPerUnit; k++)
            bankPart(op, unit, k, row, lo, hi);
    } else {
        assert(opKind(op) == OpKind::SingleBank);
        bankPart(op, unit, bank, row, lo, hi);
    }
}

void
PimFunctionalModel::bankPart(PimOp op, unsigned unit, unsigned k,
                             uint32_t row, uint64_t lo, uint64_t hi)
{
    // The even bank of each pair computes the pair's result.
    if (opIsPairEw(op) && (k & 1))
        return;
    uint8_t *reg0 = reg(unit, k, 0);
    uint8_t *gbuf = gb(unit);
    const unsigned cb = cfg.colBytes;
    for (uint64_t i = lo; i < hi;) {
        const unsigned col = unsigned(i / cb);
        const uint64_t end = std::min<uint64_t>(uint64_t(col + 1) * cb, hi);
        const unsigned at = unsigned(i % cb);
        const uint64_t n = end - i;
        uint8_t *bk = store->column(unit, k, row, col) + at;
        switch (op) {
          case PimOp::SbEwadd:
          case PimOp::SbEwmul:
          case PimOp::AbEwadd:
          case PimOp::AbEwmul: {
            const uint8_t *pair = store->column(unit, k + 1, row, col) + at;
            const bool mul = op == PimOp::SbEwmul || op == PimOp::AbEwmul;
            for (uint64_t e = 0; e < n; e += elem)
                ewElement(cfg.numFmt, mul, bk + e, pair + e, reg0 + i + e);
            break;
          }
          case PimOp::SbEwaddGbuf:
          case PimOp::SbEwmulGbuf:
          case PimOp::AbEwaddGbuf:
          case PimOp::AbEwmulGbuf: {
            const bool mul =
                op == PimOp::SbEwmulGbuf || op == PimOp::AbEwmulGbuf;
            for (uint64_t e = 0; e < n; e += elem)
                ewElement(cfg.numFmt, mul, gbuf + i + e, bk + e, reg0 + i + e);
            break;
          }
          case PimOp::SbMac:
          case PimOp::AbMac:
            // One accumulator update per column command.
            macColumn(cfg.numFmt, cfg.accFmt, gbuf + i, bk,
                      unsigned(n / elem), acc(unit, k));
            break;
          case PimOp::SbMovBkgbuf:
            std::memcpy(gbuf + i, bk, n);
            break;
          case PimOp::SbMovGbufbk:
            std::memcpy(bk, gbuf + i, n);
            break;
          case PimOp::AbMovRegbk:
            std::memcpy(bk, reg0 + i, n);
            break;
          case PimOp::AbMovBkreg:
            std::memcpy(reg0 + i, bk, n);
            break;
          default:
            assert(false);
            return;
        }
        i = end;
    }
}

void
PimFunctionalModel::readGb(unsigned unit, uint64_t off, uint8_t *dst,
                           uint64_t n) const
{
    assert(unit < cfg.units && off + n <= rowBytes);
    if (!cfg.values) {
        std::memset(dst, 0, n);
        return;
    }
    std::memcpy(dst, gbs.data() + size_t(unit) * rowBytes + off, n);
}

void
PimFunctionalModel::writeGb(unsigned unit, uint64_t off, const uint8_t *src,
                            uint64_t n)
{
    assert(unit < cfg.units && off + n <= rowBytes);
    if (cfg.values)
        std::memcpy(gb(unit) + off, src, n);
}

void
PimFunctionalModel::readWindow(unsigned unit, unsigned bank, uint32_t row,
                               uint64_t off, uint8_t *dst, uint64_t n) const
{
    assert(isWindowRow(row) && row < cfg.rowsPerBank && off + n <= rowBytes);
    std::memset(dst, 0, n);
    if (!cfg.values)
        return;
    const unsigned j = row - cfg.regRow0();
    const size_t p = pu(unit, bank);
    if (j < cfg.numRegs) {
        std::memcpy(dst,
                    regs.data() + (p * cfg.numRegs + j) * rowBytes + off, n);
    } else if (off < accBytes) {
        std::memcpy(dst, accs.data() + p * accBytes + off,
                    std::min<uint64_t>(n, accBytes - off));
    }
}

uint64_t
PimFunctionalModel::writeWindow(unsigned unit, unsigned bank, uint32_t row,
                                uint64_t off, const uint8_t *src, uint64_t n)
{
    assert(isWindowRow(row) && row < cfg.rowsPerBank && off + n <= rowBytes);
    if (!cfg.values)
        return n;
    const unsigned j = row - cfg.regRow0();
    if (j < cfg.numRegs) {
        std::memcpy(reg(unit, bank, j) + off, src, n);
        return 0;
    }
    if (off >= accBytes)
        return n;
    const uint64_t m = std::min<uint64_t>(n, accBytes - off);
    std::memcpy(acc(unit, bank) + off, src, m);
    return n - m;
}

} // namespace pim
} // namespace gem5
