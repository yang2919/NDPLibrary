#include "ramulator/controller/addr_mapper/i_addr_mapper.h"
#include "ramulator/controller/addr_mapper/impl/gem5_addr_mapper.h"

namespace Ramulator {

// gem5's RoRaBaChCo and RoRaBaCoCh (gem5 src/mem/dram_interface.cc:856-872).
// Inside a channel both decode, least significant first, the column, the flat
// bank, the rank and the row (gem5_addr_mapper.h). They differ only in where
// the channel bits sit, which gem5's port ranges and the channel mapper handle:
// one rank-wide row above the column for RoRaBaChCo, the channel interleave
// for RoRaBaCoCh.
class RoRaBaChCo : public IAddrMapper, public Gem5AddrMapperBase {
  RAMULATOR_REGISTER_IMPLEMENTATION_DERIVED(IAddrMapper, RoRaBaChCo, Gem5AddrMapperBase, "RoRaBaChCo")
  void init() override;
  void apply(Request& req) override;
};

class RoRaBaCoCh : public IAddrMapper, public Gem5AddrMapperBase {
  RAMULATOR_REGISTER_IMPLEMENTATION_DERIVED(IAddrMapper, RoRaBaCoCh, Gem5AddrMapperBase, "RoRaBaCoCh")
  void init() override;
  void apply(Request& req) override;
};

void RoRaBaChCo::init() {
  Gem5AddrMapperBase::init();
}

void RoRaBaChCo::apply(Request& req) {
  apply_column_bank_rank_row(req);
}

void RoRaBaCoCh::init() {
  Gem5AddrMapperBase::init();
}

void RoRaBaCoCh::apply(Request& req) {
  apply_column_bank_rank_row(req);
}

}  // namespace Ramulator
