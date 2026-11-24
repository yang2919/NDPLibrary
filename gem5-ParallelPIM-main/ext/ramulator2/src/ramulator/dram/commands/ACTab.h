#ifndef RAMULATOR_DRAM_COMMANDS_ACTAB_H
#define RAMULATOR_DRAM_COMMANDS_ACTAB_H

#include "ramulator/dram/node.h"

namespace Ramulator::Cmd {

// gem5 processing-in-memory (PIM) command: all-bank ACT: opens the row in every bank of the unit
// (Ideal, AiM). Issued directly by gem5's PIM sequencer: no preq, and not is_opening, so it never enters the active
// buffer, whose bookkeeping assumes one bank.
template <class T>
struct ACTab {
  static constexpr DRAMCommandMeta meta = {};
  static constexpr BankTarget bank_target = BankTarget::All;

  static void action(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    bank->m_state = T::State::Opened;
    bank->m_row_state[addr_vec[T::Level::Row]] = T::State::Opened;
  }
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_ACTAB_H
