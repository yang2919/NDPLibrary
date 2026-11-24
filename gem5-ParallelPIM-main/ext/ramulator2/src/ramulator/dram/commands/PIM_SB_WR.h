#ifndef RAMULATOR_DRAM_COMMANDS_PIM_SB_WR_H
#define RAMULATOR_DRAM_COMMANDS_PIM_SB_WR_H

#include "ramulator/dram/node.h"

namespace Ramulator::Cmd {

// gem5 processing-in-memory (PIM) command: single-bank column of the WR-class SB operation
// (MOV_GBUFBK). Issued directly by gem5's PIM sequencer: no preq, and not is_accessing, so the row policy never counts
// it.
template <class T>
struct PIM_SB_WR {
  static constexpr DRAMCommandMeta meta = {};
  static constexpr BankTarget bank_target = BankTarget::Single;
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_PIM_SB_WR_H
