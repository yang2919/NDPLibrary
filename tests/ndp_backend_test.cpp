// Verify Session GEMV lowering, banked packing, allocation reuse, and command schedules.
#include "l1/ndp/command_builder.hpp"
#include "ndp/l1/ndp/backend.hpp"
#include "test_support.hpp"

#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
using namespace ndp;
using namespace ndp::detail::command;

namespace {
NdpSpec spec() {
  return {2, 2, 96, 64, 64, 256, 128};
}

void session_gemv(NdpSpec profile) {
  auto backend = make_ndp_backend(profile);
  CHECK(backend);
  auto created = Session::create(std::move(backend.value()));
  CHECK(created);
  auto &session = created.value();
  for (auto dims : {std::pair<size_t, size_t>{1, 1}, {3, 17}, {7, 65}, {9, 129}}) {
    auto [n, k] = dims;
    std::vector<float> weights(n * k);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < k; ++j) {
        weights[i * k + j] = (int((i * 3 + j) % 9) - 4) * 0.125f;
      }
    }
    auto w = reference::upload(session, {n, k}, weights);
    for (int iteration = 0; iteration < 3; ++iteration) {
      std::vector<float> x(k), expected(n, 0);
      for (size_t j = 0; j < k; ++j) {
        x[j] = (int((j + iteration) % 7) - 3) * 0.25f;
      }
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k; ++j) {
          expected[i] += weights[i * k + j] * x[j];
        }
      }
      for (auto &value : expected) {
        value = reference::round(value);
      }
      auto input = reference::upload(session, {k}, x);
      auto y = session.matvec(w, input, {Engine::PIM});
      CHECK(y);
      CHECK(reference::download(session, y.value()) == expected);
      auto matrix = session.reshape(input, {1, k});
      CHECK(matrix);
      auto linear = session.linear(matrix.value(), w, {Engine::PIM});
      CHECK(linear);
      CHECK(reference::download(session, linear.value()) == expected);
      auto pnm = session.matvec(w, input, {Engine::PNM});
      CHECK(pnm);
      CHECK(reference::download(session, pnm.value()) == expected);
    }
  }
  CHECK(session.close());
}

void packing_and_validation() {
  auto made = make_ndp_backend(spec());
  CHECK(made);
  auto &backend = *made.value();
  TensorDesc desc{{7, 65}, DType::F16};
  auto dense = backend.query_layout(desc, LayoutKind::DenseRowMajor);
  auto packed = backend.query_layout(desc, LayoutKind::PimBanked);
  CHECK(dense && packed);
  CHECK(packed.value().physical_bytes == 2 * 3 * 4 * 64);
  auto a = backend.allocate(dense.value().physical_bytes);
  auto b = backend.allocate(packed.value().physical_bytes);
  auto c = backend.allocate(dense.value().physical_bytes);
  CHECK(a && b && c);
  std::vector<uint16_t> source(7 * 65);
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = reference::encode(float(i % 31));
  }
  CHECK(backend.upload(a.value(), std::as_bytes(std::span(source))));
  CHECK(backend.transform(a.value(), dense.value(), b.value(), packed.value()));
  std::vector<std::byte> actual(packed.value().physical_bytes),
      expected(actual.size(), std::byte{0});
  CHECK(backend.download(b.value(), actual));
  // Independent [output group][K tile][channel/bank][row lane] reference.
  for (size_t r = 0; r < 7; ++r) {
    for (size_t k = 0; k < 65; ++k) {
      size_t offset = (((r / 4) * 3 + k / 32) * 4 + r % 4) * 64 + (k % 32) * 2;
      auto value = source[r * 65 + k];
      expected[offset] = std::byte(value & 255);
      expected[offset + 1] = std::byte(value >> 8);
    }
  }
  CHECK(actual == expected); // Includes all padded rows and lanes.
  CHECK(backend.transform(b.value(), packed.value(), c.value(), dense.value()));
  std::vector<uint16_t> restored(source.size());
  CHECK(backend.download(c.value(), std::as_writable_bytes(std::span(restored))));
  CHECK(source == restored);
  auto forged = packed.value();
  forged.format += ".invalid";
  CHECK(!backend.transform(a.value(), dense.value(), b.value(), forged));
  auto op = OperationSpec{OpKind::Matvec, Engine::PIM, {desc, {{65}, DType::F16}}, {}};
  auto plan = backend.query_operation(op);
  CHECK(plan);
  auto x = backend.allocate(130), y = backend.allocate(14);
  CHECK(x && y);
  std::array<AllocationHandle, 2> inputs{b.value(), x.value()};
  auto changed = plan.value();
  changed.tile_k++;
  CHECK(!backend.execute(changed, inputs, y.value()));
  CHECK(!backend.execute(plan.value(), inputs, b.value()));
  backend.release(a.value());
  auto reused = backend.allocate(dense.value().physical_bytes);
  CHECK(reused);
  CHECK(!backend.download(a.value(), std::as_writable_bytes(std::span(restored))));
  CHECK(backend.shutdown());
  CHECK(!backend.allocate(2));
  backend.release(reused.value()); // Release remains valid after shutdown.
}

void commands_and_capacity() {
  auto p = spec();
  auto backend = make_ndp_backend(p);
  CHECK(backend);
  auto plan = backend.value()->query_operation(
      {OpKind::Matvec, Engine::PIM, {{{7, 65}, DType::F16}, {{65}, DType::F16}}, {}});
  CHECK(plan);
  auto stream = build_matvec(p, plan.value(), {4, 1536}, {12, 130}, {13, 14});
  size_t macs = 0, reads = 0, inits = 0, loads = 0;
  for (const auto &cmd : stream) {
    if (auto mac = std::get_if<MacAllBanks>(&cmd)) {
      CHECK(mac->row == 4 + macs);
      CHECK(mac->bursts == (macs % 3 == 2 ? 1 : 2));
      ++macs;
    }
    if (std::holds_alternative<InitializeMac>(cmd)) {
      ++inits;
    }
    if (std::holds_alternative<ReadMac>(cmd)) {
      ++reads;
    }
    if (auto transfer = std::get_if<Transfer>(&cmd);
        transfer && transfer->destination.space == Space::GlobalBuffer) {
      ++loads;
    }
    CHECK(!std::holds_alternative<LaunchPnmProgram>(cmd));
  }
  CHECK(macs == 6 && loads == 6 * p.channels && reads == 2 && inits == 2);
  p.rows = 4;
  auto small = make_ndp_backend(p);
  CHECK(small);
  auto first = small.value()->allocate(1);
  CHECK(first);
  auto rest = small.value()->allocate(512);
  CHECK(rest);
  CHECK(!small.value()->allocate(1));
  small.value()->release(first.value());
  CHECK(small.value()->allocate(1));
  p.gpr_bytes = 64;
  CHECK(!make_ndp_backend(p));
  p = spec();
  p.rows = 1;
  CHECK(!make_ndp_backend(p));
}
} // namespace

void config_file(const std::filesystem::path &valid) {
  auto spec = load_ndp_spec(valid);
  CHECK(spec);
  CHECK(spec.value().channels == 2);
  CHECK(make_ndp_backend(spec.value()));
  auto path =
      std::filesystem::temp_directory_path() /
      ("ndp-config-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

  struct Cleanup {
    std::filesystem::path path;

    ~Cleanup() {
      std::error_code ec;
      std::filesystem::remove(path, ec);
    }
  } cleanup{path};

  for (const auto &text :
       {"version = 2\n", "version = 1\nversion = 1\n", "unknown = 1\n", "version = -1\n"}) {
    {
      std::ofstream file(path);
      file << text;
    }
    CHECK(!load_ndp_spec(path));
  }
  CHECK(!load_ndp_spec(path / "missing"));
}

int main(int argc, char **argv) {
  CHECK(argc == 2);
  config_file(argv[1]);
  session_gemv(spec());
  auto smaller_gb = spec();
  smaller_gb.gb_bytes = 32;
  session_gemv(smaller_gb);
  packing_and_validation();
  commands_and_capacity();
  std::cout << "NDP backend contracts passed\n";
}
