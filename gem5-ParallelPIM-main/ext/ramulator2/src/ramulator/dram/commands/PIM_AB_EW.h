#ifndef RAMULATOR_DRAM_COMMANDS_PIM_AB_EW_H
#define RAMULATOR_DRAM_COMMANDS_PIM_AB_EW_H

#include "ramulator/dram/node.h"

namespace Ramulator::Cmd {

// gem5 processing-in-memory (PIM) command: all-bank column of the AB EW operations (EWADD, EWMUL and
// their GB forms, one timing class). Issued directly by gem5's PIM sequencer: no preq, and not is_accessing, so
// the row policy never counts it.
template <class T>
struct PIM_AB_EW {
  static constexpr DRAMCommandMeta meta = {};
  static constexpr BankTarget bank_target = BankTarget::All;
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_PIM_AB_EW_H
