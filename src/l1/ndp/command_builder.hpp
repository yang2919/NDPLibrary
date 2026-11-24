// Lower existing host metadata to commands using allocation ranges, not device-side tensors.
#pragma once
#include "command.hpp"
#include "ndp/l1/ndp/backend.hpp"

namespace ndp::detail::command {
// Each allocation reserves consecutive rows across every channel/bank.
struct Region {
  size_t first_row, bytes;
};

PnmCompute build_pnm(const NdpSpec &, const KernelPlan &, std::span<const Region>, Region);
size_t stripe_bytes(const NdpSpec &);
size_t tile_elements(const NdpSpec &);
uint64_t shared_address(const NdpSpec &, Region, size_t logical_offset);
Result<LayoutDesc> weight_layout(const NdpSpec &, const TensorDesc &);
// All layouts and allocation sizes must be validated by the host facade first.
std::vector<Command> build_matvec(const NdpSpec &, const KernelPlan &, Region weights, Region input,
                                  Region output);
} // namespace ndp::detail::command
