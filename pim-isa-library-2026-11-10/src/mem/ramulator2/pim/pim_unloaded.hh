/*
 * The unloaded latency U of a Processing-in-Memory (PIM) request, as a
 * pure function (docs/pim/04r-ramulator-pim-plan.md section 7.3, Q8 (b);
 * P5a section 6.1).
 *
 * U is the time from acceptance to Done of one request on an idle unit
 * with every bank closed, without refresh or host traffic. On Ramulator
 * 2.1 the sequencer issues every command at its earliest legal clock, so
 * U follows from the command sequence of P5a section 6.1's table (per
 * class and all-bank mode: activations, columns, and AESPA's walk over
 * the banks) and the derived standard's timing constraints, read as the
 * device applies them (ext/ramulator2/src/ramulator/dram/node.cpp: ready
 * clocks per node, windowed and shared-window histories, siblings), plus
 * the controllers' rules a request meets: one PIM command per cycle, a
 * write-clock synchronization (CAS) before LPDDR6's data-carrying column
 * commands, and HBM4's rising edges. The result is in controller clocks;
 * C26 checks it against timing mode for every class, mode and size.
 */

#ifndef __MEM_RAMULATOR2_PIM_PIM_UNLOADED_HH__
#define __MEM_RAMULATOR2_PIM_PIM_UNLOADED_HH__

#include <cstdint>
#include <string>
#include <vector>

// gem5 defines warn as a macro; protect Ramulator 2.1's headers from it.
#pragma push_macro("warn")
#undef warn
#include "ramulator/memory_system/i_pim_memory_system.h"
#pragma pop_macro("warn")

namespace gem5
{

namespace pim
{

class PimUnloaded
{
  public:
    explicit PimUnloaded(const Ramulator::PimTimingInfo &info);

    /**
     * Controller clocks from the first clock at which the sequencer can
     * push the request's first command to its Done clock. addr is the
     * job's address vector (unit levels; bank; row; first column). first
     * is that first clock; only its parity matters (HBM4's rising edges
     * are the odd clocks).
     */
    int64_t clocks(Ramulator::PimClass cls, const std::vector<int> &addr,
                   int ncols, int64_t first = 1) const;

    /** A named timing value, or -1. */
    int timing(const std::string &name) const;

  private:
    Ramulator::PimTimingInfo t;
    int levels = 0;
    int unitLevel = 0;
    int bankLevel = 0;
    int rowLevel = 0;
    int colLevel = 0;
    bool lpddr = false; // write-clock sync before data-carrying columns
    bool hbm34 = false; // commands on rising edges, PRE on either
    int cmdCas = -1;
    std::vector<bool> isAccess; // LPDDR6: needs the write clock
    std::vector<bool> isPre;
    // Per level and command: the history window, and per shared group.
    std::vector<std::vector<int>> window;
    std::vector<std::vector<int>> groupWindow;
    // Per level and command: the entries.
    std::vector<std::vector<std::vector<Ramulator::PimTimingEntry>>> cons;
};

} // namespace pim
} // namespace gem5

#endif // __MEM_RAMULATOR2_PIM_PIM_UNLOADED_HH__
