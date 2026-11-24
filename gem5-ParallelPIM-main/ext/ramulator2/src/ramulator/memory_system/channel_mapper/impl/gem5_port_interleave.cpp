#include <fmt/format.h>

#include "ramulator/base/base.h"
#include "ramulator/memory_system/channel_mapper/i_channel_mapper.h"

namespace Ramulator {

class Gem5PortInterleave final : public IChannelMapper, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IChannelMapper, Gem5PortInterleave, "Gem5PortInterleave");

  Addr_t m_interleave_size = 0;  // 64-bit: one channel can be 2 GiB or more (gem5)
  int m_num_channels = 1;
  int m_ch_shift = 0;
  int m_ch_width = 0;

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_interleave_size, Addr_t, "interleave_size").required();
  }

  void setup(int num_channels, int tx_offset) override {
    if (num_channels <= 0) {
      throw std::runtime_error(fmt::format("Gem5PortInterleave requires a positive channel count, got {}", num_channels));
    }
    if ((num_channels & (num_channels - 1)) != 0) {
      throw std::runtime_error(
          fmt::format("Gem5PortInterleave requires a power-of-two channel count, got {}", num_channels));
    }
    if (m_interleave_size <= 0 || (m_interleave_size & (m_interleave_size - 1)) != 0) {
      throw std::runtime_error(
          fmt::format("Gem5PortInterleave requires power-of-two interleave_size, got {}", m_interleave_size));
    }

    const int tx_bytes = 1 << tx_offset;
    if (m_interleave_size < tx_bytes) {
      throw std::runtime_error(fmt::format(
          "Gem5PortInterleave interleave_size ({}) must be at least one Ramulator transaction ({})",
          m_interleave_size, tx_bytes));
    }

    m_num_channels = num_channels;
    m_ch_shift = calc_log2(m_interleave_size);
    m_ch_width = calc_log2(num_channels);
  }

  void apply(Request& req) const override {
    if (req.addr_vec.empty()) {
      req.addr_vec.resize(1, -1);
    }
    req.addr_vec[0] = req.ingress_id;

    if (m_num_channels == 1) {
      req.intra_channel_addr = req.addr;
      return;
    }

    const Addr_t low_mask = (1LL << m_ch_shift) - 1;
    const Addr_t low = req.addr & low_mask;
    const Addr_t high = req.addr >> (m_ch_shift + m_ch_width);
    req.intra_channel_addr = (high << m_ch_shift) | low;
  }
};

}  // namespace Ramulator
