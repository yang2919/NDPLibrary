#ifndef RAMULATOR_DRAM_COMMANDS_PIM_AB_MAC_H
#define RAMULATOR_DRAM_COMMANDS_PIM_AB_MAC_H

#include "ramulator/dram/node.h"

namespace Ramulator::Cmd {

// gem5 processing-in-memory (PIM) command: all-bank MAC column.
// Issued directly by gem5's PIM sequencer: no preq, and not is_accessing, so the row
// policy never counts it.
template <class T>
struct PIM_AB_MAC {
  static constexpr DRAMCommandMeta meta = {};
  static constexpr BankTarget bank_target = BankTarget::All;
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_PIM_AB_MAC_H
