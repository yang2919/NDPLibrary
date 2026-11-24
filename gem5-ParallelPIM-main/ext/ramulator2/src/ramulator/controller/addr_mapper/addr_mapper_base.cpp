#include "ramulator/controller/addr_mapper/addr_mapper_base.h"

namespace Ramulator {

void AddrMapperBase::init() {
  const auto& dram_spec = *m_ctrl->m_device.m_spec;
  const auto& level_sizes = dram_spec.organization.level_sizes;
  m_num_mapped_levels = dram_spec.level_count - 1;  // skip channel
  m_addr_bits.resize(m_num_mapped_levels);
  for (int i = 0; i < m_num_mapped_levels; i++) {
    int count = level_sizes[i + 1];
    if (count <= 0 || (count & (count - 1)) != 0) {
      throw std::runtime_error(get_name() + ": " + dram_spec.level_names[i + 1] +
                               " count " + std::to_string(count) +
                               " is currently unsupported; expected a positive power of two");
    }
    m_addr_bits[i] = calc_log2(count);
  }

  // Column adjusted for prefetch
  int prefetch = dram_spec.internal_prefetch_size;
  if (prefetch <= 0 || (prefetch & (prefetch - 1)) != 0) {
    throw std::runtime_error(get_name() + ": internal prefetch size " + std::to_string(prefetch) +
                             " is currently unsupported; expected a positive power of two");
  }
  int column_count = level_sizes[dram_spec.get_level_id("Column")];
  if (column_count < prefetch || column_count % prefetch != 0) {
    throw std::runtime_error(get_name() + ": Column count " + std::to_string(column_count) +
                             " is currently unsupported; expected a positive multiple of internal prefetch size " +
                             std::to_string(prefetch));
  }
  m_addr_bits[m_num_mapped_levels - 1] -= calc_log2(prefetch);

  // Transaction offset
  m_tx_offset = calc_log2(dram_spec.get_tx_bytes());

  // Level indices (offset by 1 since channel is removed)
  m_row_idx = dram_spec.get_level_id("Row") - 1;
  m_col_idx = dram_spec.level_count - 2;  // Column is always last mapped level
}

}  // namespace Ramulator
