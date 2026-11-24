/*
 * Functional model of the Processing-in-Memory (PIM) commands.
 *
 * State per parallelism unit (rank, pseudo channel or channel, over every
 * channel of the memory): the global buffer (GB), one row; per processing
 * unit (PU, one per bank): numRegs registers of one row each and one
 * accumulator in acc_fmt. Bank data is not kept here: the model reaches
 * it column by column through a PimBankStore (the memory's backing
 * store).
 *
 * The register window is the top numRegs + 1 rows of every bank: row
 * regRow0() + j is register j of the bank's PU, and the next row is the
 * accumulator row, whose first fmtBytes(acc_fmt) bytes are the
 * accumulator; its other bytes read as zero and ignore writes.
 *
 * Requests reach the model decoded and checked by the caller (address
 * range, descriptor, row crossing); checkBank() gives the checks the
 * model can decide itself. Every effect applies only to the bytes, or
 * elements, inside a request's row-byte range [lo, hi). With values off
 * the model keeps no state: loads return zeros and stores change nothing.
 */

#ifndef __MEM_RAMULATOR2_PIM_PIM_FUNC_HH__
#define __MEM_RAMULATOR2_PIM_PIM_FUNC_HH__

#include <cstdint>
#include <vector>

#include "mem/pim_op.hh"
#include "mem/ramulator2/pim/pim_numfmt.hh"

namespace gem5
{

namespace pim
{

/** Bank data of the PIM memory, one column at a time. */
class PimBankStore
{
  public:
    virtual ~PimBankStore() = default;

    /** The colBytes bytes of column col of row row of a bank. */
    virtual uint8_t *column(unsigned unit, unsigned bank, uint32_t row,
                            unsigned col) = 0;
};

/** Geometry and formats of the model. */
struct FuncConfig
{
    unsigned units = 1;        // parallelism units of the whole memory
    unsigned banksPerUnit = 1; // one PU per bank
    uint32_t rowsPerBank = 2;
    unsigned colsPerRow = 1;
    unsigned colBytes = 32; // B: one column (transaction)
    unsigned numRegs = 1;
    NumFmt numFmt = NumFmt::Bf16;
    NumFmt accFmt = NumFmt::Fp32;
    bool values = true;

    uint64_t rowBytes() const { return uint64_t(colsPerRow) * colBytes; }
    uint32_t regRow0() const { return rowsPerBank - numRegs - 1; }
};

class PimFunctionalModel
{
  public:
    /** Check a configuration; nullptr if valid, else the rule. */
    static const char *checkConfig(const FuncConfig &cfg);

    /** cfg must pass checkConfig(); store may be null with values off. */
    PimFunctionalModel(const FuncConfig &cfg, PimBankStore *store);

    const FuncConfig &config() const { return cfg; }

    /** True for the register and accumulator rows of a bank. */
    bool isWindowRow(uint32_t row) const { return row >= cfg.regRow0(); }

    /**
     * The request checks of a bank operation on row bytes [lo, hi) of
     * row row of bank bank (the address's bank for all-bank operations):
     * a window row, an odd bank of an element-wise pair, more than one
     * register for the register operations, a compute range that is not
     * element-aligned. nullptr if none fails, else the rule.
     */
    const char *checkBank(PimOp op, unsigned bank, uint32_t row, uint64_t lo,
                          uint64_t hi) const;

    /**
     * Execute a bank operation (single bank, or every bank of the unit)
     * on row bytes [lo, hi) of row row, which lie in one row.
     */
    void bankOp(PimOp op, unsigned unit, unsigned bank, uint32_t row,
                uint64_t lo, uint64_t hi);

    /** RdGb and WrGb: GB bytes [off, off + n) of a unit. */
    void readGb(unsigned unit, uint64_t off, uint8_t *dst, uint64_t n) const;
    void writeGb(unsigned unit, uint64_t off, const uint8_t *src, uint64_t n);

    /**
     * RdReg, WrReg and functional accesses: bytes [off, off + n) of the
     * window row row of a bank, within the row. writeWindow() returns the
     * number of bytes it ignored (accumulator-row bytes past the
     * accumulator, or every byte with values off).
     */
    void readWindow(unsigned unit, unsigned bank, uint32_t row, uint64_t off,
                    uint8_t *dst, uint64_t n) const;
    uint64_t writeWindow(unsigned unit, unsigned bank, uint32_t row,
                         uint64_t off, const uint8_t *src, uint64_t n);

    /** State, for tests and checkpoints. */
    uint8_t *gb(unsigned unit);
    uint8_t *reg(unsigned unit, unsigned bank, unsigned j);
    uint8_t *acc(unsigned unit, unsigned bank);
    std::vector<uint8_t> &gbState() { return gbs; }
    std::vector<uint8_t> &regState() { return regs; }
    std::vector<uint8_t> &accState() { return accs; }
    const std::vector<uint8_t> &gbState() const { return gbs; }
    const std::vector<uint8_t> &regState() const { return regs; }
    const std::vector<uint8_t> &accState() const { return accs; }

  private:
    unsigned pu(unsigned unit, unsigned bank) const
    {
        return unit * cfg.banksPerUnit + bank;
    }

    /** One bank's part of a bank operation. */
    void bankPart(PimOp op, unsigned unit, unsigned k, uint32_t row,
                  uint64_t lo, uint64_t hi);

    const FuncConfig cfg;
    PimBankStore *const store;
    const uint64_t rowBytes;
    const unsigned elem;
    const unsigned accBytes;
    std::vector<uint8_t> gbs;  // units x rowBytes
    std::vector<uint8_t> regs; // PUs x numRegs x rowBytes
    std::vector<uint8_t> accs; // PUs x accBytes
};

} // namespace pim
} // namespace gem5

#endif // __MEM_RAMULATOR2_PIM_PIM_FUNC_HH__
