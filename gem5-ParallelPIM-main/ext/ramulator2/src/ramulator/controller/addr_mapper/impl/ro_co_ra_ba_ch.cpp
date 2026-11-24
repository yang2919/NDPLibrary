#include <fmt/format.h>

#include "ramulator/controller/addr_mapper/i_addr_mapper.h"
#include "ramulator/controller/addr_mapper/impl/gem5_addr_mapper.h"

namespace Ramulator {

// gem5's RoCoRaBaCh (gem5 src/mem/dram_interface.cc:873-895). Least
// significant first: the column bits of one channel stripe, the flat bank, the
// rank (gem5_addr_mapper.h), the other column bits and the row. The stripe is
// gem5's burstsPerStripe: the channel interleave when the memory's address
// range is interleaved, otherwise one transaction
// (gem5 src/mem/mem_interface.cc:64-65), at most one row.
class RoCoRaBaCh : public IAddrMapper, public Gem5AddrMapperBase {
  RAMULATOR_REGISTER_IMPLEMENTATION_DERIVED(IAddrMapper, RoCoRaBaCh, Gem5AddrMapperBase, "RoCoRaBaCh")

  // Bytes per channel stripe: the channel interleave with more than one
  // channel; 0 (the default) for one transaction, as gem5 without channel
  // interleaving.
  int m_stripe_size = 0;
  int m_stripe_bits = 0;

  void init() override;
  void apply(Request& req) override;
};

void RoCoRaBaCh::init() {
  Gem5AddrMapperBase::init();
  RAMULATOR_PARSE_PARAM(m_stripe_size, int, "stripe_size").default_val(0);
  const int tx_bytes = 1 << m_tx_offset;
  if (m_stripe_size == 0) {
    m_stripe_bits = 0;
    return;
  }
  if (m_stripe_size < tx_bytes || (m_stripe_size & (m_stripe_size - 1)) != 0) {
    throw std::runtime_error(
        fmt::format("RoCoRaBaCh: stripe_size ({}) must be 0 or a power of two of at least one transaction ({})",
                    m_stripe_size, tx_bytes));
  }
  m_stripe_bits = std::min(calc_log2(m_stripe_size / tx_bytes), m_addr_bits[m_col_idx]);
}

void RoCoRaBaCh::apply(Request& req) {
  req.addr_vec.resize(m_num_mapped_levels + 1, -1);
  Addr_t addr = req.intra_channel_addr >> m_tx_offset;
  Addr_t col = slice_lower_bits(addr, m_stripe_bits);
  slice_bank_rank(addr, req);
  col |= slice_lower_bits(addr, m_addr_bits[m_col_idx] - m_stripe_bits) << m_stripe_bits;
  req.addr_vec[m_col_idx + 1] = col;
  req.addr_vec[m_row_idx + 1] = slice_lower_bits(addr, m_addr_bits[m_row_idx]);
}

}  // namespace Ramulator
