#ifndef RAMULATOR_DRAM_COMMANDS_ACTSB1_H
#define RAMULATOR_DRAM_COMMANDS_ACTSB1_H

#include "ramulator/dram/node.h"

namespace Ramulator::Cmd {

// gem5 processing-in-memory (PIM) command: first phase of a per-bank ACT issued directly on LPDDR6
// (the controller special-cases ACT1). Issued directly by gem5's PIM sequencer: no preq, and not is_opening, so it
// never enters the active buffer, whose bookkeeping assumes one bank.
template <class T>
struct ACTsb1 {
  static constexpr DRAMCommandMeta meta = {};
  static constexpr BankTarget bank_target = BankTarget::Single;

  static void action(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    bank->m_state = T::State::Activating;
  }
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_ACTSB1_H
