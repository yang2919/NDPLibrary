/*
 * Geometry and address decode of a Processing-in-Memory (PIM) memory on
 * Ramulator 2.1 (docs/pim/04r-ramulator-pim-plan.md section 6.1; P4
 * sections 8.3-8.5).
 *
 * The layout is built from the Ramulator 2.1 organization (level names and
 * counts, the column counted in transactions), the address-mapping policy
 * and the channel interleave. It encodes and decodes offsets in the memory
 * with include/gem5/pim/pim_addr.h's pim_geom_*() functions, the inverse
 * of the controllers' mappers; PimRamulator2 checks at start-up that the
 * memory system decodes sampled offsets alike.
 *
 * Coordinates: the unit is the parallelism unit (rank, pseudo channel, or
 * the channel itself on GDDR7), numbered over the whole memory; banks are
 * numbered within a unit as (SID x bank groups + bank group) x banks +
 * bank, so that the two banks of an element-wise pair, 2k and 2k + 1 of
 * one bank group, are consecutive.
 */

#ifndef __MEM_RAMULATOR2_PIM_PIM_LAYOUT_HH__
#define __MEM_RAMULATOR2_PIM_PIM_LAYOUT_HH__

#include <gem5/pim/pim_addr.h>

#include <cstdint>
#include <string>
#include <vector>

#include "mem/pim_op.hh"

namespace gem5
{

namespace pim
{

/** The organization as the memory system reports it. */
struct LevelInfo
{
    std::vector<std::string> names; // Channel first, Column last
    std::vector<int> counts;        // the column in transactions
    int channels = 0;
    int txBytes = 0;
    int unitLevel = 0; // Rank, PseudoChannel, or Channel
};

/** PIM_POLICY_* of a policy name, or -1. */
int policyFromName(const std::string &name);
const char *policyName(int policy);

/** Row bytes [lo, hi) of row row of bank bank of unit unit, request bytes
 * from reqOff on. */
struct Piece
{
    unsigned unit = 0;
    unsigned bank = 0;
    uint32_t row = 0;
    uint64_t lo = 0;
    uint64_t hi = 0;
    uint64_t reqOff = 0;
};

/** A PIM request, decoded (P4 sections 8.4 and 8.5). */
struct Decoded
{
    PimOp op = PimOp::None;
    pim_coord_t coord = {}; // of the first byte
    unsigned unit = 0;
    unsigned bank = 0;
    uint32_t row = 0;
    uint64_t lo = 0; // row bytes [lo, hi) of bank and register operations
    uint64_t hi = 0;
    unsigned firstCol = 0;
    unsigned ncols = 0;       // column commands (bursts spanned)
    unsigned partialCols = 0; // of them, the partly covered ones
    std::vector<Piece> pieces; // RdReg and WrReg: per column
};

/** The outcome of a check. */
enum class Fault
{
    None,
    Panic,   // a simulator bug (decoder or routing), P4 section 8.5
    SwError, // a software error, P4 section 8.5 and Q6
};

struct Check
{
    Fault fault = Fault::None;
    std::string rule;

    explicit operator bool() const { return fault != Fault::None; }
};

/**
 * Physical offset bit of bit 0 of each field, as the layout header's
 * PIM_LAYOUT_<FIELD>_SHIFT (util/pim/gen_pim_layout.py): also defined for
 * a level the standard lacks, at the position it would have.
 */
struct Shifts
{
    unsigned col, ch, unit, sid, bg, bank, row;
};

class PimLayout
{
  public:
    /** An empty layout; build() fills it. */
    PimLayout() = default;

    /**
     * Build from the organization; returns an empty string on success,
     * else why the organization is not a PIM memory's.
     */
    std::string build(const LevelInfo &levels, int policy,
                      uint64_t interleave, unsigned numRegs);

    const pim_geom_t &geom() const { return g; }
    unsigned channels() const { return g.channels; }
    unsigned unitsPerChannel() const { return g.units; }
    unsigned units() const { return g.channels * g.units; }
    unsigned banksPerUnit() const
    {
        return g.sids * g.bank_groups * g.banks_per_group;
    }
    uint32_t rowsPerBank() const { return g.rows; }
    unsigned colsPerRow() const { return g.cols; }
    unsigned colBytes() const { return g.col_bytes; }
    uint64_t rowBytes() const { return uint64_t(g.cols) * g.col_bytes; }
    uint64_t size() const { return pim_geom_size(&g); }
    unsigned numRegs() const { return regs; }
    uint32_t regRow0() const { return g.rows - regs - 1; }
    bool isWindowRow(uint32_t row) const { return row >= regRow0(); }

    uint64_t encode(const pim_coord_t &c) const;
    /** Coordinates of an offset; returns the byte within the column. */
    uint32_t decode(uint64_t off, pim_coord_t &c) const;

    unsigned unitOf(const pim_coord_t &c) const;
    unsigned bankOf(const pim_coord_t &c) const;
    /** The coordinates of a unit's bank, row and column. */
    pim_coord_t coordOf(unsigned unit, unsigned bank, uint32_t row,
                        unsigned col) const;

    /** Ramulator 2.1's address vector of coordinates, and back. */
    std::vector<int> toAddrVec(const pim_coord_t &c) const;
    pim_coord_t fromAddrVec(const std::vector<int> &v) const;
    size_t levels() const { return nLevels; }

    /** The field positions; interleave as the header's INTLV_BYTES. */
    Shifts shifts(uint64_t interleave) const;
    /** PIM_LAYOUT_CONTIG_BYTES. */
    uint64_t contigBytes() const { return pim_geom_contig_bytes(&g); }

    /**
     * Decode a PIM request of op at offset off with nbytes bytes and, for
     * the scalar forms, descriptor desc, and apply the checks of P4
     * section 8.5 that the address decides (the model's own, P4 section
     * 8.5's register and alignment rules, follow in
     * PimFunctionalModel::checkBank).
     */
    Check decodeRequest(PimOp op, uint64_t off, unsigned nbytes,
                        uint64_t desc, Decoded &d) const;

  private:
    pim_geom_t g = {};
    unsigned regs = 1;
    size_t nLevels = 0;
    int unitLevel = -1; // -1: the channel is the unit
    int sidLevel = -1;
    int bgLevel = -1;
    int bankLevel = -1;
    int rowLevel = -1;
    int colLevel = -1;
};

} // namespace pim
} // namespace gem5

#endif // __MEM_RAMULATOR2_PIM_PIM_LAYOUT_HH__
