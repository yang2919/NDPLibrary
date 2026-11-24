#ifndef RAMULATOR_MEMORY_SYSTEM_I_PIM_MEMORY_SYSTEM_H
#define RAMULATOR_MEMORY_SYSTEM_I_PIM_MEMORY_SYSTEM_H

// gem5's processing-in-memory (PIM) interfaces: between gem5's wrapper
// PimRamulator2, the memory system PimDRAMSystem and one PimSequencer per channel controller.
//
// C++17 only: gem5's wrapper, built as C++17, includes this header.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "ramulator/base/type.h"

namespace Ramulator {

// The PIM timing classes, with the single-bank read class split into the
// single and the paired form. One command per class.
enum class PimClass : int {
  SbRd,     // PIM_SB_RD: SB EW with the GB, MAC, MOV_BKGBUF
  SbRd2,    // PIM_SB_RD2: SB EWADD, EWMUL on banks 2k and 2k + 1
  SbWr,     // PIM_SB_WR: MOV_GBUFBK
  AbEw,     // PIM_AB_EW
  AbMac,    // PIM_AB_MAC
  AbBkreg,  // PIM_AB_BKREG
  AbRegbk,  // PIM_AB_REGBK
  GbRd,     // PIM_GB_RD
  GbWr,     // PIM_GB_WR
  RegRd,    // PIM_REG_RD
  RegWr,    // PIM_REG_WR
  COUNT
};

inline const char* pim_class_name(PimClass c) {
  static const char* const names[] = {"SbRd",    "SbRd2", "SbWr", "AbEw",  "AbMac", "AbBkreg",
                                      "AbRegbk", "GbRd",  "GbWr", "RegRd", "RegWr"};
  int i = static_cast<int>(c);
  return (i >= 0 && i < static_cast<int>(PimClass::COUNT)) ? names[i] : "Invalid";
}

inline bool pim_class_is_sb(PimClass c) {
  return c == PimClass::SbRd || c == PimClass::SbRd2 || c == PimClass::SbWr;
}

inline bool pim_class_is_ab(PimClass c) {
  return c == PimClass::AbEw || c == PimClass::AbMac || c == PimClass::AbBkreg || c == PimClass::AbRegbk;
}

inline bool pim_class_is_non_bank(PimClass c) {
  return c == PimClass::GbRd || c == PimClass::GbWr || c == PimClass::RegRd || c == PimClass::RegWr;
}

// RD-class or WR-class (a PIM command that writes a row is WR-class).
inline bool pim_class_writes_row(PimClass c) {
  return c == PimClass::SbWr || c == PimClass::AbRegbk;
}

// One PIM request, as the wrapper hands it to a channel's sequencer.
struct PimJob {
  uint64_t id = 0;  // wrapper sequence number (acceptance order)
  PimClass cls = PimClass::SbRd;
  AddrVec_t addr;        // unit levels; bank (SB) or -1 below the unit; row; first column
  int ncols = 1;         // column commands
  Clk_t accept_clk = 0;  // controller clock at acceptance, for FR-FCFS age
};

// The organization the wrapper needs from the memory system.
struct PimGeometry {
  std::vector<std::string> level_names;  // Channel first, Column last
  std::vector<int> level_counts;         // per level; the column in transactions
  int channels = 0;
  int tx_bytes = 0;    // bytes per transaction (one PIM column)
  int unit_level = 0;  // the parallelism unit (Rank, PseudoChannel, or Channel)
};

// What gem5's closed-form atomic latency U reads:
// one channel's DRAM specification as the device applies it, and the
// sequencer's commands per class. Values are in controller clocks.
struct PimTimingEntry {
  int level = 0;  // where the constraint applies
  int cmd = 0;    // the command issued
  int next = 0;   // the command it constrains
  int val = 0;
  int window = 1;
  bool sibling = false;
  int history_group = -1;
};

struct PimTimingInfo {
  std::string standard;
  std::vector<std::string> level_names;  // Channel ... Column
  std::vector<int> level_sizes;
  std::vector<std::string> command_names;
  std::vector<int> command_cycles;  // command-bus ticks per command
  std::vector<bool> is_closing;
  std::vector<bool> is_pair;                        // BankTarget::Pair: banks 2k and 2k + 1
  std::vector<std::vector<bool>> has_sibling_cons;  // [level][command]
  std::vector<PimTimingEntry> entries;
  std::vector<std::string> timing_names;
  std::vector<int> timing_vals;
  // The sequencer (PimSequencer::describe).
  std::vector<int> col_cmd, col_ap_cmd, done_lat;  // per PimClass
  int cmd_act = -1, cmd_act1 = -1, cmd_act2 = -1;
  int cmd_actab = -1, cmd_actab1 = -1, cmd_actab2 = -1;
  int nAAD = -1;
  std::string ab_mode;
  bool close_rows = false;
};

// gem5's PIM command sequencer of one channel controller (the plugin).
class IPimSequencer {
 public:
  virtual ~IPimSequencer() = default;
  // on_last_command(id) fires when the job's last command issues;
  // on_done(id, clk) at the controller clock its data ends.
  virtual void submit(const PimJob& job, std::function<void(uint64_t)> on_last_command,
                      std::function<void(uint64_t, Clk_t)> on_done) = 0;
  // Packets the wrapper holds or has released but not sent yet (for Chopim).
  virtual void set_external_backlog(int packets) = 0;
  // Counts the cycles of a requested write drain (pim_drain_cycles).
  virtual void note_write_drain(bool on) = 0;
  virtual bool idle() const = 0;
  // Fill the sequencer's part of PimTimingInfo.
  virtual void describe(PimTimingInfo& info) const = 0;
};

// gem5's PIM memory system (PimDRAMSystem, registered as "PimDRAM").
class IPimMemorySystem {
 public:
  virtual ~IPimMemorySystem() = default;
  // The address vector of an offset that arrives on port (= channel), through
  // the channel mapper and the controller's address mapper.
  virtual AddrVec_t decode(int port, Addr_t offset) = 0;
  virtual PimGeometry geometry() const = 0;
  virtual IPimSequencer* sequencer(int channel) = 0;
  virtual void register_sequencer(int channel, IPimSequencer* s) = 0;
  virtual Clk_t clk(int channel) const = 0;
  // While on and writes wait, the channel's controller stays in write mode.
  virtual void request_write_drain(int channel, bool on) = 0;
  // The buffer views of a channel's controller (README.gem5, change 6), for
  // the shared capacity of held packets.
  virtual size_t read_buffer_size(int channel) const = 0;
  virtual size_t read_buffer_capacity(int channel) const = 0;
  virtual size_t write_buffer_size(int channel) const = 0;
  virtual size_t write_buffer_capacity(int channel) const = 0;
  // The DRAM specification and sequencer of a channel, for U.
  virtual PimTimingInfo timing_info(int channel) const = 0;
  // The sequencer reports each host column command (type 0 or 1,
  // command == final command) it sees in on_issue; the wrapper listens.
  virtual void set_host_column_listener(std::function<void(int channel, Addr_t addr, bool is_read)> f) = 0;
  virtual void host_column_issued(int channel, Addr_t addr, bool is_read) = 0;
};

// The type_id of a PIM command in a controller's priority buffer:
// not -1, so the stock maintenance counters do not count it.
constexpr int kPimTypeId = 2;

}  // namespace Ramulator

#endif  // RAMULATOR_MEMORY_SYSTEM_I_PIM_MEMORY_SYSTEM_H
