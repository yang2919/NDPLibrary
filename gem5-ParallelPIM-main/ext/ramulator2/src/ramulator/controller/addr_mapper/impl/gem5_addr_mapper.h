#ifndef RAMULATOR_CONTROLLER_ADDR_MAPPER_IMPL_GEM5_ADDR_MAPPER_H
#define RAMULATOR_CONTROLLER_ADDR_MAPPER_IMPL_GEM5_ADDR_MAPPER_H

#include <stdexcept>
#include <string>
#include <vector>

#include "ramulator/controller/addr_mapper/addr_mapper_base.h"

namespace Ramulator {

// Common part of the ports of gem5's address-mapping policies
// (DRAMInterface::decodePacket, gem5 src/mem/dram_interface.cc:856-895).
//
// gem5 decodes a channel-local address (the channel bits are already removed,
// here by the channel mapper) into a column, a flat bank, a rank and a row.
// The flat bank is split with the bank group lowest
// (gem5 src/mem/dram_interface.cc:1140-1151), then the bank and, on HBM, the
// stack ID; gem5's rank is the rank or the pseudo channel. So, least
// significant first, a flat bank and rank are: BankGroup, Bank, Sid, then
// Rank or PseudoChannel. A level a standard lacks takes no bits.
class Gem5AddrMapperBase : public AddrMapperBase {
 protected:
  // Indices into m_addr_bits (level id - 1), least significant first.
  std::vector<int> m_bank_rank_idx;

 public:
  Gem5AddrMapperBase(const ConfigNode& config, Implementation* parent) : AddrMapperBase(config, parent) {
  }

  void init() override {
    AddrMapperBase::init();
    const auto& spec = *m_ctrl->m_device.m_spec;
    if (m_row_idx != m_col_idx - 1) {
      throw std::runtime_error(get_name() + ": expected the Row level directly above the Column level");
    }
    int unit = -1, sid = -1, bg = -1, bank = -1;
    for (int i = 0; i < m_row_idx; i++) {
      const std::string& name = spec.level_names[i + 1];
      int* slot = nullptr;
      if (name == "Rank" || name == "PseudoChannel") {
        slot = &unit;
      } else if (name == "Sid") {
        slot = &sid;
      } else if (name == "BankGroup") {
        slot = &bg;
      } else if (name == "Bank") {
        slot = &bank;
      }
      if (slot == nullptr || *slot != -1) {
        throw std::runtime_error(get_name() + ": level " + name + " has no place in gem5's decode");
      }
      *slot = i;
    }
    if (bank == -1) {
      throw std::runtime_error(get_name() + ": the standard has no Bank level");
    }
    m_bank_rank_idx.clear();
    for (int i : {bg, bank, sid, unit}) {
      if (i != -1) {
        m_bank_rank_idx.push_back(i);
      }
    }
  }

  // Slices the flat bank and the rank from addr into req.addr_vec.
  void slice_bank_rank(Addr_t& addr, Request& req) const {
    for (int i : m_bank_rank_idx) {
      req.addr_vec[i + 1] = slice_lower_bits(addr, m_addr_bits[i]);
    }
  }

  // The decode of RoRaBaChCo and RoRaBaCoCh inside a channel: the column, the
  // flat bank and the rank, the row (gem5 src/mem/dram_interface.cc:856-872).
  void apply_column_bank_rank_row(Request& req) const {
    req.addr_vec.resize(m_num_mapped_levels + 1, -1);
    Addr_t addr = req.intra_channel_addr >> m_tx_offset;
    req.addr_vec[m_col_idx + 1] = slice_lower_bits(addr, m_addr_bits[m_col_idx]);
    slice_bank_rank(addr, req);
    req.addr_vec[m_row_idx + 1] = slice_lower_bits(addr, m_addr_bits[m_row_idx]);
  }
};

}  // namespace Ramulator

#endif  // RAMULATOR_CONTROLLER_ADDR_MAPPER_IMPL_GEM5_ADDR_MAPPER_H
