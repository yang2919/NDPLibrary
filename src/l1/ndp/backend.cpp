// Bind Session allocations and plans to the command-driven NDP emulator.
#include "ndp/l1/ndp/backend.hpp"

#include "command_builder.hpp"
#include "transport.hpp"

#include <algorithm>
#include <map>
#include <new>
#include <stdexcept>

namespace ndp {
namespace {
using namespace detail::command;

Status invalid(const char *message) {
  return {ErrorCode::InvalidArgument, message};
}

class NdpBackend final : public IBackend {
public:
  NdpBackend(NdpSpec p, std::unique_ptr<ICommandTransport> device)
      : spec_(p), device_(std::move(device)), occupied_(p.rows, false) {
    occupied_[0] = true;
  }

  DeviceCaps capabilities() const override {
    return {"ndp-command-fp16", (spec_.rows - 1) * stripe_bytes(spec_), spec_.scratchpad_bytes};
  }

  Result<AllocationHandle> allocate(size_t bytes) override {
    if (closed_) {
      return Status{ErrorCode::SessionClosed, "Backend closed"};
    }
    if (!bytes) {
      return invalid("Zero-byte allocation");
    }
    size_t count = (bytes - 1) / stripe_bytes(spec_) + 1, run = 0;
    for (size_t row = 1; row < spec_.rows; ++row) {
      run = occupied_[row] ? 0 : run + 1;
      if (run == count) {
        if (next_ == UINT64_MAX) {
          return Status{ErrorCode::OutOfMemory, "Allocation handles exhausted"};
        }
        size_t first = row + 1 - count;
        regions_.emplace(next_, Region{first, bytes});
        std::fill(occupied_.begin() + first, occupied_.begin() + row + 1, true);
        return next_++;
      }
    }
    return Status{ErrorCode::OutOfMemory, "No contiguous NDP row stripes available"};
  }

  void release(AllocationHandle handle) noexcept override {
    auto it = regions_.find(handle);
    if (it == regions_.end()) {
      return;
    }
    size_t count = (it->second.bytes - 1) / stripe_bytes(spec_) + 1;
    std::fill(occupied_.begin() + it->second.first_row,
              occupied_.begin() + it->second.first_row + count, false);
    regions_.erase(it);
  }

  Result<void> upload(AllocationHandle h, std::span<const std::byte> data) override {
    auto valid = validate(h, data.size());
    if (!valid) {
      return valid;
    }
    return write(regions_.at(h), data);
  }

  Result<void> download(AllocationHandle h, std::span<std::byte> data) override {
    auto valid = validate(h, data.size());
    if (!valid) {
      return valid;
    }
    return read(regions_.at(h), data);
  }

  Result<LayoutDesc> query_layout(const TensorDesc &desc, LayoutKind kind) const override {
    if (kind == LayoutKind::DenseRowMajor) {
      return dense_layout(desc);
    }
    if (kind == LayoutKind::PimBanked) {
      return weight_layout(spec_, desc);
    }
    return Status{ErrorCode::UnsupportedLayout, "Unsupported NDP layout"};
  }

  Result<void> transform(AllocationHandle src, const LayoutDesc &from, AllocationHandle dst,
                         const LayoutDesc &to) override {
    auto a = query_layout(from.logical, from.kind), b = query_layout(to.logical, to.kind);
    if (!a) {
      return a.error();
    }
    if (!b) {
      return b.error();
    }
    if (from != a.value() || to != b.value() || from.logical != to.logical || src == dst) {
      return invalid("Invalid NDP transform descriptors or alias");
    }
    auto sv = validate(src, from.physical_bytes), dv = validate(dst, to.physical_bytes);
    if (!sv) {
      return sv;
    }
    if (!dv) {
      return dv;
    }
    Command cmd;
    Address source{Space::Shared, shared_address(spec_, regions_.at(src), 0)};
    Address destination{Space::Shared, shared_address(spec_, regions_.at(dst), 0)};
    if (from.kind == to.kind) {
      cmd = CopyBytes{source, destination, from.physical_bytes};
    } else {
      cmd = PnmRelayout{source,
                        destination,
                        from.logical.shape[0],
                        from.logical.shape[1],
                        spec_.channels * spec_.banks,
                        tile_elements(spec_),
                        spec_.row_bytes,
                        to.kind == LayoutKind::PimBanked};
    }
    return device_->submit(std::span<const Command>(&cmd, 1));
  }

  Result<KernelPlan> query_operation(const OperationSpec &op) const override {
    if (op.attributes.output_dtype != DType::F16)
      return Status{ErrorCode::UnsupportedDType, "Backend has no FP32 partial kernel"};
    auto out = infer_output(op);
    if (!out) {
      return out.error();
    }
    if (op.engine == Engine::PNM) {
      KernelPlan plan;
      plan.operation = op;
      plan.kernel_id = "ndp.pnm.functional.v1";
      for (const auto &input : op.inputs) {
        plan.inputs.push_back(dense_layout(input).value());
      }
      plan.output = dense_layout(out.value()).value();
      if (op.kind == OpKind::Matmul || op.kind == OpKind::Matvec || op.kind == OpKind::Linear) {
        if (spec_.scratchpad_bytes < 4) {
          return Status{ErrorCode::OutOfMemory, "PNM requires four scratchpad bytes"};
        }
        plan.workspace_bytes = 4;
      }
      return plan;
    }
    if (op.kind != OpKind::Matvec &&
        !(op.kind == OpKind::Linear && op.inputs[0].shape[0] == 1 && op.inputs.size() == 2)) {
      return Status{ErrorCode::UnsupportedOperation,
                    "PIM supports Matvec and bias-free single-row Linear"};
    }
    size_t wi = op.kind == OpKind::Linear ? 1 : 0;
    auto weight = weight_layout(spec_, op.inputs[wi]);
    if (!weight) {
      return weight.error();
    }
    KernelPlan plan;
    plan.operation = op;
    plan.kernel_id = "ndp.pim.matvec.fp16.v1";
    plan.inputs = wi ? std::vector<LayoutDesc>{dense_layout(op.inputs[0]).value(), weight.value()}
                     : std::vector<LayoutDesc>{weight.value(), dense_layout(op.inputs[1]).value()};
    plan.output = dense_layout(out.value()).value();
    plan.tile_k = tile_elements(spec_);
    return plan;
  }

  Result<void> execute(const KernelPlan &plan, std::span<const AllocationHandle> inputs,
                       AllocationHandle output) override {
    auto expected = query_operation(plan.operation);
    if (!expected) {
      return expected.error();
    }
    if (plan != expected.value() || inputs.size() != plan.inputs.size()) {
      return invalid("Invalid NDP kernel plan");
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto valid = validate(inputs[i], plan.inputs[i].physical_bytes);
      if (!valid) {
        return valid;
      }
      if (inputs[i] == output) {
        return invalid("NDP output must not alias inputs");
      }
    }
    auto valid = validate(output, plan.output.physical_bytes);
    if (!valid) {
      return valid;
    }
    if (plan.operation.engine == Engine::PNM) {
      std::vector<Region> regions;
      for (auto h : inputs) {
        regions.push_back(regions_.at(h));
      }
      Command cmd = build_pnm(spec_, plan, regions, regions_.at(output));
      return device_->submit(std::span<const Command>(&cmd, 1));
    }
    size_t wi = plan.operation.kind == OpKind::Linear ? 1 : 0;
    const auto stream = build_matvec(spec_, plan, regions_.at(inputs[wi]),
                                     regions_.at(inputs[1 - wi]), regions_.at(output));
    return device_->submit(stream);
  }

  Result<void> reserve_workspace(size_t bytes) override {
    if (closed_) {
      return Status{ErrorCode::SessionClosed, "Backend closed"};
    }
    if (bytes > spec_.scratchpad_bytes) {
      return Status{ErrorCode::OutOfMemory, "Scratchpad exceeded"};
    }
    return {};
  }

  Result<void> fill(AllocationHandle handle, std::byte value) override {
    auto it = regions_.find(handle);
    if (it == regions_.end()) {
      return invalid("Invalid allocation");
    }
    auto valid = validate(handle, it->second.bytes);
    if (!valid) {
      return valid;
    }
    Command cmd =
        FillBytes{{Space::Shared, shared_address(spec_, it->second, 0)}, it->second.bytes, value};
    return device_->submit(std::span<const Command>(&cmd, 1));
  }

  Result<void> copy_region(AllocationHandle src, size_t so, AllocationHandle dst, size_t dest,
                           size_t bytes) override {
    if (closed_) {
      return Status{ErrorCode::SessionClosed, "Backend closed"};
    }
    auto a = regions_.find(src), b = regions_.find(dst);
    if (a == regions_.end() || b == regions_.end() || !bytes || so > a->second.bytes ||
        bytes > a->second.bytes - so || dest > b->second.bytes || bytes > b->second.bytes - dest) {
      return invalid("Invalid copy range");
    }
    Command cmd = CopyBytes{{Space::Shared, shared_address(spec_, a->second, so)},
                            {Space::Shared, shared_address(spec_, b->second, dest)},
                            bytes};
    return device_->submit(std::span<const Command>(&cmd, 1));
  }

  Result<void> shutdown() override {
    closed_ = true;
    return device_->shutdown();
  }

private:
  Result<void> validate(AllocationHandle h, size_t bytes) const {
    if (closed_) {
      return Status{ErrorCode::SessionClosed, "Backend closed"};
    }
    auto it = regions_.find(h);
    if (it == regions_.end() || bytes != it->second.bytes) {
      return invalid("Invalid NDP allocation or size");
    }
    return {};
  }

  Result<void> write(Region region, std::span<const std::byte> bytes) {
    return device_->upload({Space::Shared, shared_address(spec_, region, 0)}, bytes);
  }

  Result<void> read(Region region, std::span<std::byte> bytes) {
    return device_->download({Space::Shared, shared_address(spec_, region, 0)}, bytes);
  }

  const NdpSpec spec_;
  std::unique_ptr<ICommandTransport> device_;
  std::vector<bool> occupied_;
  std::map<AllocationHandle, Region> regions_;
  AllocationHandle next_ = 1;
  bool closed_ = false;
};
} // namespace

Result<std::unique_ptr<IBackend>> make_ndp_backend(const NdpSpec &spec) {
  auto device = detail::command::make_emulator_transport(spec);
  if (!device) {
    return device.error();
  }
  return detail::command::make_command_backend(spec, std::move(device.value()));
}

Result<std::unique_ptr<IBackend>>
detail::command::make_command_backend(const NdpSpec &spec,
                                      std::unique_ptr<ICommandTransport> device) {
  if (spec.rows < 2 ||
      spec.gpr_bytes < detail::command::tile_elements(spec) * 2 + spec.channels * spec.banks * 4) {
    return invalid(
        "NDP backend requires a reserved zero row and vector/accumulator/result GPR space");
  }
  try {
    return std::unique_ptr<IBackend>(new NdpBackend(spec, std::move(device)));
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "NDP backend allocation failed"};
  } catch (const std::length_error &) {
    return Status{ErrorCode::OutOfMemory, "NDP backend allocation too large"};
  }
}
} // namespace ndp
