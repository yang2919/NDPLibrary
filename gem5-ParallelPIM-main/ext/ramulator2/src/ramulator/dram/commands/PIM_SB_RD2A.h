#ifndef RAMULATOR_DRAM_COMMANDS_PIM_SB_RD2A_H
#define RAMULATOR_DRAM_COMMANDS_PIM_SB_RD2A_H

#include "ramulator/dram/commands/PREpb.h"
#include "ramulator/dram/node.h"

namespace Ramulator::Cmd {

// gem5 processing-in-memory (PIM) command: PIM_SB_RD2 with auto-precharge of both banks.
// Issued directly by gem5's PIM sequencer: no preq, and not is_accessing, so the row
// policy never counts it.
template <class T>
struct PIM_SB_RD2A {
  static constexpr DRAMCommandMeta meta = {.is_closing = true};
  static constexpr BankTarget bank_target = BankTarget::Pair;

  static void action(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    PREpb<T>::action(bank, cmd, addr_vec, clk);
  }
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_PIM_SB_RD2A_H
