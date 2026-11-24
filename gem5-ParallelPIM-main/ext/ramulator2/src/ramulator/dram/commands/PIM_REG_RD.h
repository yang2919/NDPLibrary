#ifndef RAMULATOR_DRAM_COMMANDS_PIM_REG_RD_H
#define RAMULATOR_DRAM_COMMANDS_PIM_REG_RD_H

#include "ramulator/dram/node.h"

namespace Ramulator::Cmd {

// gem5 processing-in-memory (PIM) command: non-bank read of the unit's PU registers (RD_REG); carries
// host data. Issued directly by gem5's PIM sequencer: no preq, and not is_accessing, so the row policy never counts it
//.
template <class T>
struct PIM_REG_RD {
  static constexpr DRAMCommandMeta meta = {};
  static constexpr BankTarget bank_target = BankTarget::All;
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_PIM_REG_RD_H
