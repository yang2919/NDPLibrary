#ifndef RAMULATOR_DRAM_COMMANDS_PIM_SB_RD_H
#define RAMULATOR_DRAM_COMMANDS_PIM_SB_RD_H

#include "ramulator/dram/node.h"

namespace Ramulator::Cmd {

// gem5 processing-in-memory (PIM) command: single-bank column of the RD-class SB operations (SB EW
// with the GB, MAC, MOV_BKGBUF). Issued directly by gem5's PIM sequencer: no preq, and not is_accessing, so the row
// policy never counts it.
template <class T>
struct PIM_SB_RD {
  static constexpr DRAMCommandMeta meta = {};
  static constexpr BankTarget bank_target = BankTarget::Single;
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_PIM_SB_RD_H
