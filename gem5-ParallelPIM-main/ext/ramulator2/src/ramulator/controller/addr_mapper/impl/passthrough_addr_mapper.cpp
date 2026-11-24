#include "ramulator/controller/addr_mapper/i_addr_mapper.h"

namespace Ramulator {

// Pass-through: assumes addr_vec is already populated by the frontend.
// Used with addr_vec-native frontends (LatencyThroughputTrace, ReadWriteTrace).
class PassThroughAddrMapper : public IAddrMapper, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IAddrMapper, PassThroughAddrMapper, "PassThroughAddrMapper")
  void init() override {
  }
  void apply(Request& req) override;
};

void PassThroughAddrMapper::apply(Request&) {
  // addr_vec already populated by frontend — nothing to do
}

}  // namespace Ramulator
