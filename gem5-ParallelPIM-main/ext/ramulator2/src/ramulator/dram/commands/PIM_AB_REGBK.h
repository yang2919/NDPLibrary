#ifndef RAMULATOR_DRAM_COMMANDS_PIM_AB_REGBK_H
#define RAMULATOR_DRAM_COMMANDS_PIM_AB_REGBK_H

#include "ramulator/dram/node.h"

namespace Ramulator::Cmd {

// gem5 processing-in-memory (PIM) command: all-bank column writing the PU registers into a row
// (AB_MOV_REGBK). Issued directly by gem5's PIM sequencer: no preq, and not is_accessing, so the row policy never
// counts it.
template <class T>
struct PIM_AB_REGBK {
  static constexpr DRAMCommandMeta meta = {};
  static constexpr BankTarget bank_target = BankTarget::All;
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_PIM_AB_REGBK_H
