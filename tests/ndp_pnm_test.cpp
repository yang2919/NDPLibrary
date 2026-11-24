// Verify functional PNM kernels, resident PIM handoffs, and Session buffer/lifetime contracts.
#include "l1/ndp/emulator.hpp"
#include "ndp/l1/ndp/backend.hpp"
#include "test_support.hpp"

#include <array>
using namespace ndp;
using namespace ndp::detail::command;

namespace {
Session session() {
  auto b = make_ndp_backend({2, 2, 256, 64, 64, 256, 128});
  CHECK(b);
  auto s = Session::create(std::move(b.value()));
  CHECK(s);
  return std::move(s.value());
}

void kernels_and_chain() {
  auto s = session();
  auto made = Session::create(make_functional_backend(1 << 20));
  CHECK(made);
  auto &r = made.value();
  auto compare = [&](Result<Tensor> a, Result<Tensor> b) {
    CHECK(a && b);
    reference::close(reference::download(s, a.value()), reference::download(r, b.value()));
  };
  auto a = reference::upload(s, {2, 3}, {1, -2, 3, 4, 5, -6});
  auto ar = reference::upload(r, {2, 3}, {1, -2, 3, 4, 5, -6});
  auto b = reference::upload(s, {3, 2}, {1, 2, 3, 4, 5, 6});
  auto br = reference::upload(r, {3, 2}, {1, 2, 3, 4, 5, 6});
  auto v = reference::upload(s, {3}, {1, 2, 3}), vr = reference::upload(r, {3}, {1, 2, 3});
  compare(s.matmul(a, b), r.matmul(ar, br));
  compare(s.matmul(a, b, {}, 0.25f), r.matmul(ar, br, {}, 0.25f));
  compare(s.linear(a, a, {}, nullptr), r.linear(ar, ar, {}, nullptr));
  auto bias = reference::upload(s, {2}, {2, 3}), biasr = reference::upload(r, {2}, {2, 3});
  compare(s.linear(a, a, {}, &bias), r.linear(ar, ar, {}, &biasr));
  compare(s.add(a, v), r.add(ar, vr));
  compare(s.mul(a, a), r.mul(ar, ar));
  compare(s.rms_norm(a, v), r.rms_norm(ar, vr));
  compare(s.softmax(a), r.softmax(ar));
  compare(s.silu(a), r.silu(ar));
  compare(s.transpose(a), r.transpose(ar));
  compare(s.reshape(a, {3, 2}), r.reshape(ar, {3, 2}));
  compare(s.gather(a, {5, 0, 2}, {3}), r.gather(ar, {5, 0, 2}, {3}));
  compare(s.scale(a, -2), r.scale(ar, -2));
  auto high = reference::upload(s, {1, 1}, {400}), highr = reference::upload(r, {1, 1}, {400});
  compare(s.matmul(high, high, {}, 0.25f), r.matmul(highr, highr, {}, 0.25f));
  // No host download or wait between PIM production and the PNM consumer.
  auto producer = s.matvec(a, v, {Engine::PIM});
  CHECK(producer);
  auto consumer = s.silu(producer.value());
  CHECK(consumer);
  auto refproducer = r.matvec(ar, vr, {Engine::PNM});
  CHECK(refproducer);
  compare(consumer, r.silu(refproducer.value()));
  // Queue from temporary inputs: captured storage must survive until execution.
  auto pending = [&]() {
    auto x = reference::upload(s, {3}, {1, 2, 3});
    return s.scale(x, 2);
  }();
  CHECK(pending);
  CHECK(reference::download(s, pending.value()) == std::vector<float>({2, 4, 6}));
  auto buffer = s.allocate({{6}});
  CHECK(buffer);
  auto zero = s.snapshot(buffer.value(), {{6}});
  CHECK(zero);
  CHECK(reference::download(s, zero.value()) == std::vector<float>(6, 0));
  CHECK(s.copy_into({{a, buffer.value(), 0, 0, 6}}));
  auto snapshot = s.snapshot(buffer.value(), {{2, 3}});
  CHECK(snapshot);
  CHECK(s.clear({buffer.value()}));
  CHECK(reference::download(s, snapshot.value()) == std::vector<float>({1, -2, 3, 4, 5, -6}));
  auto cleared = s.snapshot(buffer.value(), {{6}});
  CHECK(cleared);
  CHECK(reference::download(s, cleared.value()) == std::vector<float>(6, 0));
  CHECK(s.close());
  CHECK(r.close());
}

void resource_failures() {
  auto backend = make_ndp_backend({1, 1, 3, 32, 32, 128, 16});
  CHECK(backend);
  auto made = Session::create(std::move(backend.value()));
  CHECK(made);
  auto &s = made.value();
  auto a = reference::upload(s, {1}, {1}), b = reference::upload(s, {1}, {2});
  CHECK(s.synchronize());
  auto failed = s.scale(a, 2);
  CHECK(failed);
  auto host = s.to_host(failed.value());
  CHECK(!host);
  CHECK(!s.synchronize());
  CHECK(!s.close());
  auto tiny = make_ndp_backend({1, 1, 8, 32, 32, 128, 1});
  CHECK(tiny);
  auto query = tiny.value()->query_operation(
      {OpKind::Matmul, Engine::PNM, {{{1, 1}, DType::F16}, {{1, 1}, DType::F16}}, {}});
  CHECK(!query && query.error().code == ErrorCode::OutOfMemory);
}

void raw_device() {
  auto made = Emulator::create({1, 1, 4, 32, 32, 128, 16});
  CHECK(made);
  auto &d = *made.value();
  auto put = [&](Address address, std::vector<float> values) {
    std::vector<uint16_t> bits;
    for (auto v : values) {
      bits.push_back(reference::encode(v));
    }
    CHECK(d.upload(address, std::as_bytes(std::span(bits))));
  };
  put({Space::Shared, 0}, std::vector<float>(16, 1));
  put({Space::Gpr, 0}, std::vector<float>(16, 1));
  put({Space::Gpr, 32}, {0});
  PnmCompute scale{
      PnmKernel::Scale, {{{Space::Gpr, 64}, 1}}, {{Space::Shared, 64}, 1}, 1, 1, 1, false, 2};
  std::array<Command, 6> stream{InitializeMac{1, 1, 32},
                                Transfer{{Space::Gpr, 0}, {Space::GlobalBuffer, 0}, 32},
                                MacAllBanks{1, 0, 0, 1},
                                ReadMac{1, 64},
                                scale,
                                CopyBytes{{Space::Shared, 64}, {Space::Scratchpad, 8}, 2}};
  CHECK(d.submit(stream));
  std::array<uint16_t, 1> out{};
  CHECK(d.download({Space::Scratchpad, 8}, std::as_writable_bytes(std::span(out))));
  CHECK(reference::decode(out[0]) == 32);
  for (int mode = 0; mode < 5; ++mode) {
    auto bad = scale;
    if (mode == 0) {
      bad.output.address.offset = UINT64_MAX;
    }
    if (mode == 1) {
      bad.output.address = bad.inputs[0].address;
    }
    if (mode == 2) {
      bad.width = SIZE_MAX;
    }
    if (mode == 3) {
      bad.kernel = static_cast<PnmKernel>(99);
    }
    if (mode == 4) {
      bad.inputs.clear();
    }
    std::array<Command, 2> invalid{FillBytes{{Space::Shared, 64}, 2, std::byte{0}}, bad};
    CHECK(!d.submit(invalid));
    CHECK(d.download({Space::Shared, 64}, std::as_writable_bytes(std::span(out))));
    CHECK(reference::decode(out[0]) == 32);
  }
  // Overlap is explicitly memmove for byte copies, unlike compute outputs.
  put({Space::Shared, 64}, {1, 2, 3});
  std::array<Command, 1> copy{CopyBytes{{Space::Shared, 64}, {Space::Shared, 66}, 4}};
  CHECK(d.submit(copy));
  std::array<uint16_t, 3> copied{};
  CHECK(d.download({Space::Shared, 64}, std::as_writable_bytes(std::span(copied))));
  CHECK(reference::decode(copied[1]) == 1 && reference::decode(copied[2]) == 2);
}
} // namespace

int main() {
  kernels_and_chain();
  raw_device();
  resource_failures();
  std::cout << "Functional PNM and Session contracts passed\n";
}
