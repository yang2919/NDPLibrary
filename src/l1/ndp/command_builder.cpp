// Build GEMV streams with bank-parallel outputs and MAC-resident K-tile accumulation.
#include "command_builder.hpp"

#include <algorithm>
#include <limits>

namespace ndp::detail::command {
size_t stripe_bytes(const NdpSpec &p) {
  return p.channels * p.banks * p.row_bytes;
}

size_t tile_elements(const NdpSpec &p) {
  return std::min(p.row_bytes, p.gb_bytes) / 2;
}

uint64_t shared_address(const NdpSpec &p, Region region, size_t offset) {
  return region.first_row * stripe_bytes(p) + offset;
}

Result<LayoutDesc> weight_layout(const NdpSpec &p, const TensorDesc &desc) {
  auto dense = dense_layout(desc);
  if (!dense) {
    return dense.error();
  }
  if (desc.shape.size() != 2) {
    return Status{ErrorCode::UnsupportedLayout, "NDP banked weights require rank 2"};
  }
  size_t groups = (desc.shape[0] - 1) / (p.channels * p.banks) + 1;
  size_t tiles = (desc.shape[1] - 1) / tile_elements(p) + 1;
  size_t stripe = stripe_bytes(p);
  if (tiles > std::numeric_limits<size_t>::max() / stripe ||
      groups > std::numeric_limits<size_t>::max() / (tiles * stripe)) {
    return Status{ErrorCode::InvalidArgument, "NDP packed weight size overflow"};
  }
  return LayoutDesc{LayoutKind::PimBanked, desc, groups * tiles * stripe, 32,
                    "ndp.fp16.stripe.v1.c" + std::to_string(p.channels) + ".b" +
                        std::to_string(p.banks) + ".r" + std::to_string(p.row_bytes) + ".k" +
                        std::to_string(tile_elements(p))};
}

namespace {
uint64_t all(size_t count) {
  return count == 64 ? UINT64_MAX : (uint64_t{1} << count) - 1;
}

// Dense allocations have contiguous shared addresses, including across bank rows.
void transfer(std::vector<Command> &stream, const NdpSpec &p, Region region, size_t offset,
              size_t gpr, size_t bytes, bool to_gpr) {
  Address shared{Space::Shared, shared_address(p, region, offset)}, reg{Space::Gpr, gpr};
  stream.emplace_back(to_gpr ? Transfer{shared, reg, bytes} : Transfer{reg, shared, bytes});
}
} // namespace

std::vector<Command> build_matvec(const NdpSpec &p, const KernelPlan &plan, Region weights,
                                  Region input, Region output) {
  const auto &shape = plan.operation.inputs[plan.operation.kind == OpKind::Linear ? 1 : 0].shape;
  size_t n = shape[0], k = shape[1], parallel = p.channels * p.banks;
  size_t tile = tile_elements(p), tiles = (k - 1) / tile + 1;
  size_t vector_bytes = tile * 2, zero_offset = vector_bytes;
  size_t result_offset = zero_offset + parallel * 2;
  std::vector<Command> commands;
  // Row zero is reserved, immutable zero memory. Populate initial MAC values from it.
  transfer(commands, p, {0, stripe_bytes(p)}, 0, zero_offset, parallel * 2, true);
  for (size_t group = 0; group < (n - 1) / parallel + 1; ++group) {
    commands.emplace_back(InitializeMac{all(p.channels), all(p.banks), zero_offset});
    for (size_t t = 0; t < tiles; ++t) {
      size_t elements = std::min(tile, k - t * tile);
      // Clear the complete vector tile so a short final tile cannot use stale lanes.
      transfer(commands, p, {0, stripe_bytes(p)}, 0, 0, vector_bytes, true);
      transfer(commands, p, input, t * tile * 2, 0, elements * 2, true);
      for (size_t channel = 0; channel < p.channels; ++channel) {
        commands.emplace_back(
            Transfer{{Space::Gpr, 0}, {Space::GlobalBuffer, channel * p.gb_bytes}, vector_bytes});
      }
      commands.emplace_back(MacAllBanks{all(p.channels), weights.first_row + group * tiles + t, 0,
                                        (elements + 15) / 16});
    }
    commands.emplace_back(ReadMac{all(p.channels), result_offset});
    transfer(commands, p, output, group * parallel * 2, result_offset,
             std::min(parallel, n - group * parallel) * 2, false);
  }
  return commands;
}

PnmCompute build_pnm(const NdpSpec &p, const KernelPlan &plan, std::span<const Region> inputs,
                     Region output) {
  const auto &op = plan.operation;
  PnmCompute cmd{};
  for (size_t i = 0; i < inputs.size(); ++i) {
    cmd.inputs.push_back(
        {{Space::Shared, shared_address(p, inputs[i], 0)}, plan.inputs[i].physical_bytes / 2});
  }
  cmd.output = {{Space::Shared, shared_address(p, output, 0)}, plan.output.physical_bytes / 2};
  cmd.width = op.inputs[0].shape.back();
  cmd.rows = cmd.output.elements / cmd.width;
  cmd.factor = op.attributes.factor;
  cmd.epsilon = op.attributes.epsilon;
  switch (op.kind) {
  case OpKind::Matmul:
  case OpKind::Matvec:
  case OpKind::Linear:
    cmd.kernel = PnmKernel::Matmul;
    cmd.inner = op.inputs[0].shape.back();
    cmd.rows = op.inputs[0].shape[0];
    cmd.width = op.kind == OpKind::Matvec ? 1 : plan.output.logical.shape.back();
    cmd.transpose_b = op.kind == OpKind::Linear;
    if (op.kind != OpKind::Matmul) {
      cmd.factor = 1;
    }
    break;
  case OpKind::Add:
    cmd.kernel = PnmKernel::Add;
    break;
  case OpKind::Mul:
    cmd.kernel = PnmKernel::Mul;
    break;
  case OpKind::RMSNorm:
    cmd.kernel = PnmKernel::RMSNorm;
    break;
  case OpKind::Softmax:
    cmd.kernel = PnmKernel::Softmax;
    break;
  case OpKind::SiLU:
    cmd.kernel = PnmKernel::SiLU;
    break;
  case OpKind::Scale:
    cmd.kernel = PnmKernel::Scale;
    break;
  case OpKind::Transpose:
    cmd.kernel = PnmKernel::Transpose;
    break;
  case OpKind::Reshape:
    cmd.kernel = PnmKernel::Copy;
    break;
  case OpKind::Gather:
    cmd.kernel = PnmKernel::Gather;
    cmd.width = cmd.output.elements;
    cmd.rows = 1;
    cmd.indices = op.attributes.indices;
    break;
  }
  return cmd;
}
} // namespace ndp::detail::command
