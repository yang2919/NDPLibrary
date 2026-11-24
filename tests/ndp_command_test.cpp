// Verify handwritten device streams, numeric state, addressing, and failure boundaries.
#include "l1/ndp/emulator.hpp"
#include "test_support.hpp"

#include <array>
#include <cfenv>
#include <limits>
using namespace ndp;
using namespace ndp::detail::command;

namespace {
void upload(Emulator &e, Address a, const std::vector<float> &values) {
  std::vector<std::byte> bytes;
  for (float v : values) {
    auto h = reference::encode(v);
    bytes.push_back(std::byte(h & 255));
    bytes.push_back(std::byte(h >> 8));
  }
  CHECK(e.upload(a, bytes));
}

std::vector<float> read(Emulator &e, Address a, size_t n) {
  std::vector<std::byte> bytes(n * 2);
  CHECK(e.download(a, bytes));
  std::vector<float> values;
  for (size_t i = 0; i < n; ++i) {
    values.push_back(reference::decode(std::to_integer<unsigned>(bytes[2 * i]) |
                                       (std::to_integer<unsigned>(bytes[2 * i + 1]) << 8)));
  }
  return values;
}

Result<void> run(Emulator &e, std::initializer_list<Command> cmds) {
  return e.submit(std::span(cmds.begin(), cmds.size()));
}

Profile small() {
  return {2, 2, 2, 64, 64, 256, 128};
}

void arithmetic_and_mapping() {
  auto result = Emulator::create(small());
  CHECK(result);
  auto &e = *result.value();
  // Distinct channels, banks, rows, and columns detect permutations in physical addressing.
  for (size_t c = 0; c < 2; ++c) {
    for (size_t b = 0; b < 2; ++b) {
      for (size_t r = 0; r < 2; ++r) {
        std::vector<float> w(32);
        for (size_t lane = 0; lane < 32; ++lane) {
          w[lane] = 1 + c * 8 + b * 4 + r * 2 + lane / 16;
        }
        auto address = e.bank_address(c, b, r, 0);
        CHECK(address);
        upload(e, {Space::Shared, address.value()}, w);
      }
    }
  }
  upload(e, {Space::Gpr, 0}, std::vector<float>(32, 1));
  upload(e, {Space::Gpr, 64}, {1, 2, 3, 4});
  CHECK(run(e, {InitializeMac{3, 3, 64}, Transfer{{Space::Gpr, 0}, {Space::GlobalBuffer, 0}, 64},
                Transfer{{Space::Gpr, 0}, {Space::GlobalBuffer, 64}, 64}, MacAllBanks{3, 0, 0, 2},
                ReadMac{3, 96}}));
  CHECK(read(e, {Space::Gpr, 96}, 4) == std::vector<float>({49, 178, 307, 436}));
  // Repeated submissions preserve accumulators; only channel 1 advances.
  CHECK(run(e, {MacAllBanks{2, 1, 1, 1}, ReadMac{3, 96}}));
  CHECK(read(e, {Space::Gpr, 96}, 4) == std::vector<float>({49, 178, 499, 692}));
  CHECK(run(e, {Transfer{{Space::Gpr, 96}, {Space::Shared, 0}, 8},
                Transfer{{Space::Shared, 0}, {Space::Scratchpad, 16}, 8}}));
  CHECK(read(e, {Space::Scratchpad, 16}, 4) == std::vector<float>({49, 178, 499, 692}));
  CHECK(run(e, {InitializeMac{2, 3, 64}, ReadMac{3, 96}}));
  CHECK(read(e, {Space::Gpr, 96}, 4) == std::vector<float>({49, 178, 1, 2}));
}

void failure_boundaries() {
  auto created = Emulator::create(small());
  CHECK(created);
  auto &e = *created.value();
  upload(e, {Space::Shared, 0}, {7});
  upload(e, {Space::Gpr, 0}, {9});
  auto result =
      run(e, {Transfer{{Space::Shared, 0}, {Space::Gpr, 0}, 2},
              LaunchPnmProgram{1, {Space::Shared, 0}, 2, {Space::Scratchpad, 0}, 16, 100}});
  CHECK(!result && result.error().code == ErrorCode::InvalidArgument);
  CHECK(read(e, {Space::Gpr, 0}, 1)[0] == 9); // Preflight rejects the whole stream.
  CHECK(!run(e, {Transfer{{Space::Shared, 0}, {Space::Gpr, 0}, 2}, MacAllBanks{3, 0, 1, 2}}));
  CHECK(read(e, {Space::Gpr, 0}, 1)[0] == 9);
  result = run(e, {Transfer{{Space::Shared, 0}, {Space::Gpr, 0}, 2}, ReadMac{3, 64},
                   Transfer{{Space::Shared, 0}, {Space::Gpr, 2}, 2}});
  CHECK(!result && result.error().code == ErrorCode::InvalidState);
  CHECK(read(e, {Space::Gpr, 0}, 2) ==
        std::vector<float>({7, 0})); // Runtime fault retains earlier writes only.
  upload(e, {Space::Gpr, 64}, {0, 0, 0, 0});
  CHECK(run(e, {InitializeMac{3, 3, 64}}));
  CHECK(!run(e, {MacAllBanks{3, 0, 0, 1}})); // GB must be explicitly loaded.
  CHECK(run(e, {Transfer{{Space::Gpr, 0}, {Space::GlobalBuffer, 0}, 32}}));
  CHECK(!run(e, {MacAllBanks{3, 0, 0, 1}})); // No partial channel update on invalid GB state.
  CHECK(run(e, {ReadMac{3, 96}}));
  CHECK(read(e, {Space::Gpr, 96}, 4) == std::vector<float>(4, 0));
  CHECK(!run(e, {Transfer{{Space::Gpr, 2}, {Space::GlobalBuffer, 0}, 32}}));
  CHECK(!run(e, {Transfer{{Space::Gpr, 0}, {Space::GlobalBuffer, 32}, 64}}));  // Crosses channels.
  CHECK(!run(e, {Transfer{{Space::Gpr, 0}, {Space::GlobalBuffer, 128}, 32}})); // No channel 2.
  CHECK(run(e, {Transfer{{Space::Shared, 0}, {Space::GlobalBuffer, 64}, 32}}));
  CHECK(run(e, {MacAllBanks{3, 0, 0, 1}}));
  CHECK(!run(e, {ReadMac{4, 0}}));
  CHECK(!run(e, {ReadMac{0, 0}}));
  CHECK(!run(e, {InitializeMac{1, 4, 0}}));
  CHECK(!run(e, {Transfer{{Space::Gpr, 0}, {Space::Gpr, 2}, 2}}));
  std::array<std::byte, 2> bytes{};
  CHECK(!e.upload({Space::Shared, UINT64_MAX}, bytes));
  CHECK(!e.download({static_cast<Space>(99), 0}, bytes));
  CHECK(!e.bank_address(2, 0, 0, 0));
  CHECK(!e.bank_address(0, 0, 0, 64));
  CHECK(std::fesetround(FE_DOWNWARD) == 0);
  CHECK(!run(e, {ReadMac{3, 0}}));
  CHECK(std::fesetround(FE_TONEAREST) == 0);
  auto p = small();
  p.channels = 65;
  CHECK(!Emulator::create(p));
  p = small();
  p.rows = std::numeric_limits<size_t>::max();
  CHECK(!Emulator::create(p));
}

void spill_and_numeric_edges() {
  auto created = Emulator::create({1, 1, 1, 64, 64, 256, 64});
  CHECK(created);
  auto &e = *created.value();
  upload(e, {Space::Shared, 0}, std::vector<float>(32, 1));
  upload(e, {Space::Gpr, 0}, std::vector<float>(32, 1.0f / 16));
  upload(e, {Space::Gpr, 64}, {2048});
  CHECK(run(e, {InitializeMac{1, 1, 64}, Transfer{{Space::Gpr, 0}, {Space::GlobalBuffer, 0}, 64},
                MacAllBanks{1, 0, 0, 1}, ReadMac{1, 96}}));
  CHECK(read(e, {Space::Gpr, 96}, 1)[0] == 2048); // 2049 rounds to even in FP16.
  CHECK(run(e, {MacAllBanks{1, 0, 1, 1}, ReadMac{1, 96}}));
  CHECK(read(e, {Space::Gpr, 96}, 1)[0] == 2050); // Reading did not narrow the live accumulator.
  CHECK(run(e, {InitializeMac{1, 1, 64}, MacAllBanks{1, 0, 0, 1}, ReadMac{1, 96},
                InitializeMac{1, 1, 96}, MacAllBanks{1, 0, 1, 1}, ReadMac{1, 96}}));
  CHECK(read(e, {Space::Gpr, 96}, 1)[0] == 2048); // Spilling/reloading introduces rounding.
  for (float value : {std::ldexp(1.0f, -24), 65504.0f, INFINITY, NAN}) {
    upload(e, {Space::Gpr, 64}, {value});
    CHECK(run(e, {InitializeMac{1, 1, 64}, ReadMac{1, 96}}));
    auto actual = read(e, {Space::Gpr, 96}, 1)[0];
    CHECK(std::isnan(value) ? std::isnan(actual) : actual == value);
  }
}

void highest_channel_bit() {
  auto created = Emulator::create({64, 1, 1, 32, 32, 128, 32});
  CHECK(created);
  auto &e = *created.value();
  const uint64_t last = uint64_t{1} << 63;
  upload(e, {Space::Shared, e.bank_address(63, 0, 0, 0).value()}, std::vector<float>(16, 2));
  upload(e, {Space::Gpr, 0}, std::vector<float>(16, 3));
  upload(e, {Space::Gpr, 32}, {0});
  CHECK(run(e, {InitializeMac{last, 1, 32},
                Transfer{{Space::Gpr, 0}, {Space::GlobalBuffer, 63 * 32}, 32},
                MacAllBanks{last, 0, 0, 1}, ReadMac{last, 64}}));
  CHECK(read(e, {Space::Gpr, 64}, 1)[0] == 96);
}
} // namespace

int main() {
  arithmetic_and_mapping();
  failure_boundaries();
  spill_and_numeric_edges();
  highest_channel_bit();
  std::cout << "NDP command contracts passed\n";
}
