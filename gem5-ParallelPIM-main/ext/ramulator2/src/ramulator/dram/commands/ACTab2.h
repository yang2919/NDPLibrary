#ifndef RAMULATOR_DRAM_COMMANDS_ACTAB2_H
#define RAMULATOR_DRAM_COMMANDS_ACTAB2_H

#include "ramulator/dram/node.h"

namespace Ramulator::Cmd {

// gem5 processing-in-memory (PIM) command: second phase of the LPDDR6 all-bank ACT: opens the row in
// every bank of the unit. Issued directly by gem5's PIM sequencer: no preq, and not is_opening, so it never enters the
// active buffer, whose bookkeeping assumes one bank.
template <class T>
struct ACTab2 {
  static constexpr DRAMCommandMeta meta = {};
  static constexpr BankTarget bank_target = BankTarget::All;

  static void action(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    bank->m_state = T::State::Opened;
    bank->m_row_state[addr_vec[T::Level::Row]] = T::State::Opened;
  }
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_ACTAB2_H
