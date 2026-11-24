#ifndef RAMULATOR_DRAM_COMMANDS_PIM_SB_RD2_H
#define RAMULATOR_DRAM_COMMANDS_PIM_SB_RD2_H

#include "ramulator/dram/node.h"

namespace Ramulator::Cmd {

// gem5 processing-in-memory (PIM) command: single-bank column on a bank pair 2k, 2k + 1 (SB EWADD and
// EWMUL). Issued directly by gem5's PIM sequencer: no preq, and not is_accessing, so the row policy never
// counts it.
template <class T>
struct PIM_SB_RD2 {
  static constexpr DRAMCommandMeta meta = {};
  static constexpr BankTarget bank_target = BankTarget::Pair;
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_PIM_SB_RD2_H
