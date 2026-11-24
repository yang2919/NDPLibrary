#ifndef RAMULATOR_PIM_PIM_UNIT_H
#define RAMULATOR_PIM_PIM_UNIT_H

#include "ramulator/dram/dram_spec.h"

namespace Ramulator {

// The level of gem5's PIM parallelism unit (P-DRAM section 4.1): the rank on
// DDR5 and LPDDR6, the pseudo channel on HBM4, otherwise the channel (GDDR7).
// It is also each standard's all-bank refresh scope.
inline int pim_unit_level(const DRAMSpec& spec) {
  if (spec.has_level("Rank")) {
    return spec.get_level_id("Rank");
  }
  if (spec.has_level("PseudoChannel")) {
    return spec.get_level_id("PseudoChannel");
  }
  return spec.get_level_id("Channel");
}

}  // namespace Ramulator

#endif  // RAMULATOR_PIM_PIM_UNIT_H
