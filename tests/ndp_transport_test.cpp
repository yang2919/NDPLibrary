// Check wire compatibility, malformed input rejection, and persistent execution.
#include "l1/ndp/codec.hpp"
#include "l1/ndp/transport.hpp"
#include "test_support.hpp"

#include <array>
#include <bit>
#include <cctype>
#include <limits>
using namespace ndp;
using namespace ndp::detail::command;

namespace {
std::vector<std::byte> hex(const char *text) {
  std::vector<std::byte> result;
  int high = -1;
  for (; *text; ++text) {
    if (std::isspace(static_cast<unsigned char>(*text))) {
      continue;
    }
    int n = *text >= 'a' ? *text - 'a' + 10 : *text - '0';
    CHECK(n >= 0 && n < 16);
    if (high < 0) {
      high = n;
    } else {
      result.push_back(std::byte(high * 16 + n));
      high = -1;
    }
  }
  CHECK(high == -1);
  return result;
}

std::vector<std::byte> encode_one(Command command) {
  auto encoded = wire::encode(std::span<const Command>(&command, 1));
  CHECK(encoded);
  return std::move(encoded.value());
}

void golden_records() {
  auto golden = hex("4e445031 0200 0100 32000000 01000000 0900 0000 1a000000 "
                    "01 0807060504030201 02 1817161514131211 2827262524232221");
  auto copy = CopyBytes{
      {Space::Shared, 0x0102030405060708}, {Space::Gpr, 0x1112131415161718}, 0x2122232425262728};
  CHECK(encode_one(copy) == golden);
  auto decoded = wire::decode(golden);
  CHECK(decoded && decoded.value().size() == 1);
  const auto &c = std::get<CopyBytes>(decoded.value()[0]);
  CHECK(c.source.space == Space::Shared && c.source.offset == copy.source.offset);
  CHECK(c.destination.space == Space::Gpr && c.destination.offset == copy.destination.offset);
  CHECK(c.bytes == copy.bytes);

  // Literal payloads pin every field, including vector ordering and float bits.
  struct Fixture {
    Command command;
    const char *payload;
  };

  const std::vector<Fixture> fixtures{
      {Transfer{{Space::Shared, 1}, {Space::Gpr, 2}, 3},
       "01 0100000000000000 02 0200000000000000 0300000000000000"},
      {InitializeMac{1, 2, 3}, "0100000000000000 0200000000000000 0300000000000000"},
      {Transfer{{Space::Gpr, 2}, {Space::GlobalBuffer, 3}, 4},
       "02 0200000000000000 04 0300000000000000 0400000000000000"},
      {MacAllBanks{1, 2, 3, 4},
       "0100000000000000 0200000000000000 0300000000000000 0400000000000000"},
      {ReadMac{1, 2}, "0100000000000000 0200000000000000"},
      {LaunchPnmProgram{1, {Space::Gpr, 2}, 3, {Space::Scratchpad, 4}, 5, 6},
       "0100000000000000 02 0200000000000000 0300000000000000 "
       "03 0400000000000000 0500000000000000 0600000000000000"},
      {PnmCompute{PnmKernel::Gather,
                  {{{Space::Shared, 16}, 3}, {{Space::Gpr, 32}, 4}},
                  {{Space::Scratchpad, 48}, 5},
                  6,
                  7,
                  8,
                  true,
                  1.5f,
                  0.25f,
                  {9, 10}},
       "0800 02000000 01 1000000000000000 0300000000000000 "
       "02 2000000000000000 0400000000000000 03 3000000000000000 0500000000000000 "
       "0600000000000000 0700000000000000 0800000000000000 01 0000c03f 0000803e "
       "02000000 0900000000000000 0a00000000000000"},
      {PnmRelayout{{Space::Shared, 1}, {Space::Scratchpad, 2}, 3, 4, 5, 6, 7, true},
       "01 0100000000000000 03 0200000000000000 0300000000000000 0400000000000000 "
       "0500000000000000 0600000000000000 0700000000000000 01"},
      {CopyBytes{{Space::Gpr, 1}, {Space::Scratchpad, 2}, 3},
       "02 0100000000000000 03 0200000000000000 0300000000000000"},
      {FillBytes{{Space::Shared, 1}, 2, std::byte{0xaa}},
       "01 0100000000000000 0200000000000000 aa"}};
  std::vector<Command> batch;
  for (size_t i = 0; i < fixtures.size(); ++i) {
    auto encoded = encode_one(fixtures[i].command);
    CHECK(encoded[16] == std::byte(i == 2 ? 1 : i + 1) && encoded[17] == std::byte{0});
    auto payload = hex(fixtures[i].payload);
    CHECK(std::vector<std::byte>(encoded.begin() + 24, encoded.end()) == payload);
    auto roundtrip = wire::decode(encoded);
    CHECK(roundtrip && roundtrip.value()[0].index() == fixtures[i].command.index());
    CHECK(wire::encode(roundtrip.value()).value() == encoded);
    // Keep envelope lengths consistent while truncating each payload field.
    for (size_t n = 24; n < encoded.size(); ++n) {
      auto truncated = std::vector<std::byte>(encoded.begin(), encoded.begin() + n);
      truncated[8] = std::byte(n);
      truncated[20] = std::byte(n - 24);
      CHECK(!wire::decode(truncated));
    }
    batch.push_back(fixtures[i].command);
  }
  auto bytes = wire::encode(batch);
  CHECK(bytes);
  auto result = wire::decode(bytes.value());
  CHECK(result && result.value().size() == batch.size());
  CHECK(wire::encode(result.value()).value() == bytes.value());
  const auto &pnm = std::get<PnmCompute>(result.value()[6]);
  CHECK(pnm.inputs.size() == 2 && pnm.inputs[1].address.space == Space::Gpr);
  CHECK(pnm.inputs[1].address.offset == 32 && pnm.inputs[1].elements == 4);
  CHECK(pnm.output.address.space == Space::Scratchpad && pnm.output.address.offset == 48);
  CHECK(pnm.output.elements == 5 && pnm.rows == 6 && pnm.width == 7 && pnm.inner == 8);
  CHECK(pnm.transpose_b && pnm.factor == 1.5f && pnm.epsilon == 0.25f);
  CHECK(pnm.indices == std::vector<uint64_t>({9, 10}));

  const std::array kernels{PnmKernel::Matmul,   PnmKernel::Add,     PnmKernel::Mul,
                           PnmKernel::RMSNorm,  PnmKernel::Softmax, PnmKernel::SiLU,
                           PnmKernel::Copy,     PnmKernel::Gather,  PnmKernel::Scale,
                           PnmKernel::Transpose};
  for (size_t i = 0; i < kernels.size(); ++i) {
    auto command = pnm;
    command.kernel = kernels[i];
    command.factor = std::bit_cast<float>(uint32_t{0x80000000});
    command.epsilon = std::bit_cast<float>(uint32_t{0x7fc01234});
    auto encoded = encode_one(command);
    CHECK(encoded[24] == std::byte(i + 1));
    auto roundtrip = wire::decode(encoded);
    CHECK(roundtrip);
    const auto &value = std::get<PnmCompute>(roundtrip.value()[0]);
    CHECK(value.kernel == kernels[i]);
    CHECK(std::bit_cast<uint32_t>(value.factor) == 0x80000000);
    CHECK(std::bit_cast<uint32_t>(value.epsilon) == 0x7fc01234);
  }
}

void malformed_records() {
  auto bytes = encode_one(CopyBytes{{Space::Shared, 0}, {Space::Gpr, 1}, 2});
  for (size_t n = 0; n < bytes.size(); ++n) {
    CHECK(!wire::decode(std::span(bytes).first(n)));
  }
  for (size_t offset : {0u, 4u, 6u, 8u, 12u, 16u, 18u, 20u, 24u, 33u}) {
    auto invalid = bytes;
    invalid[offset] = std::byte{0xff};
    auto result = wire::decode(invalid);
    CHECK(!result && result.error().code == ErrorCode::InvalidArgument);
  }
  auto trailing = bytes;
  trailing.push_back(std::byte{0});
  CHECK(!wire::decode(trailing));
  trailing[8] = std::byte(trailing.size());
  CHECK(!wire::decode(trailing));
  trailing[20] = std::byte{27}; // Valid framing, extra byte inside the command.
  CHECK(!wire::decode(trailing));
  CHECK(!wire::decode(bytes, {bytes.size() - 1, 1}));
  CHECK(!wire::decode(bytes, {bytes.size(), 0}));
  Command command = FillBytes{{Space::Shared, 0}, 1, std::byte{0}};
  auto view = std::span<const Command>(&command, 1);
  CHECK(!wire::encode(view, {16, 1}));
  CHECK(!wire::encode(view, {1024, 0}));
  CHECK(!wire::encode(view, {0, 1}));
  command = FillBytes{{static_cast<Space>(99), 0}, 1, std::byte{0}};
  CHECK(!wire::encode(view));
  PnmCompute compute{PnmKernel::Scale, {}, {{Space::Shared, 0}, 1}};
  auto pnm = encode_one(compute);
  auto invalid = pnm;
  invalid[24] = std::byte{99}; // Kernel ID.
  CHECK(!wire::decode(invalid));
  invalid = pnm;
  for (size_t i = 26; i < 30; ++i) {
    invalid[i] = std::byte{0xff};
  }
  CHECK(!wire::decode(invalid)); // Input vector count cannot fit its payload.
  invalid = pnm;
  invalid[71] = std::byte{2}; // Boolean after output and dimensions.
  CHECK(!wire::decode(invalid));
  invalid = pnm;
  for (size_t i = invalid.size() - 4; i < invalid.size(); ++i) {
    invalid[i] = std::byte{0xff};
  }
  CHECK(!wire::decode(invalid));
  compute.kernel = static_cast<PnmKernel>(99);
  command = compute;
  CHECK(!wire::encode(view));
  auto empty = wire::encode({});
  CHECK(empty && empty.value().size() == wire::header_bytes);
  CHECK(wire::decode(empty.value()).value().empty());
}

void execute_decoded() {
  auto made = make_emulator_transport({1, 1, 4, 32, 32, 128, 128});
  CHECK(made);
  auto &transport = *made.value();
  auto data = hex("003c 0040 0042 0044"); // FP16 [1, 2, 3, 4].
  CHECK(transport.upload({Space::Shared, 0}, data));
  const std::vector<Command> commands{PnmCompute{PnmKernel::Add,
                                                 {{{Space::Shared, 0}, 2}, {{Space::Shared, 4}, 2}},
                                                 {{Space::Shared, 8}, 2},
                                                 1,
                                                 2},
                                      Transfer{{Space::Shared, 8}, {Space::Gpr, 0}, 4},
                                      CopyBytes{{Space::Gpr, 0}, {Space::Scratchpad, 0}, 4}};
  auto encoded = wire::encode(commands);
  CHECK(encoded);
  auto decoded = wire::decode(encoded.value());
  CHECK(decoded && transport.submit(decoded.value()));
  std::vector<std::byte> result(4);
  CHECK(transport.download({Space::Scratchpad, 0}, result));
  CHECK(result == hex("0044 0046")); // [4, 6], independent of codec round-trip.
  // Whole-batch preflight must prevent the first fill when a later range is bad.
  const std::vector<Command> bad{FillBytes{{Space::Scratchpad, 0}, 4, std::byte{0}},
                                 CopyBytes{{Space::Shared, UINT64_MAX}, {Space::Gpr, 0}, 4}};
  auto bad_wire = wire::encode(bad);
  CHECK(bad_wire);
  auto bad_decoded = wire::decode(bad_wire.value());
  CHECK(bad_decoded); // Structural validity does not imply valid device ranges.
  CHECK(!transport.submit(bad_decoded.value()));
  CHECK(transport.download({Space::Scratchpad, 0}, result));
  CHECK(result == hex("0044 0046"));
  // Execute decoded PIM commands against persistent device state as well.
  std::vector<std::byte> weights, vector;
  for (size_t i = 0; i < 16; ++i) {
    weights.push_back(std::byte{0});
    weights.push_back(std::byte{0x3c});
    vector.push_back(std::byte{0});
    vector.push_back(std::byte{0x40});
  }
  CHECK(transport.upload({Space::Shared, 32}, weights));
  CHECK(transport.upload({Space::Gpr, 0}, vector));
  CHECK(transport.upload({Space::Gpr, 64}, hex("0000")));
  const std::vector<Command> pim{InitializeMac{1, 1, 64},
                                 Transfer{{Space::Gpr, 0}, {Space::GlobalBuffer, 0}, 32},
                                 MacAllBanks{1, 1, 0, 1}, ReadMac{1, 96}};
  auto pim_wire = wire::encode(pim);
  CHECK(pim_wire);
  auto pim_decoded = wire::decode(pim_wire.value());
  CHECK(pim_decoded && transport.submit(pim_decoded.value()));
  result.resize(2);
  CHECK(transport.download({Space::Gpr, 96}, result));
  CHECK(result == hex("0050")); // 16 products of 1 * 2 = 32.
  CHECK(transport.shutdown());
  CHECK(transport.shutdown());
  auto closed = transport.submit(commands);
  CHECK(!closed && closed.error().code == ErrorCode::SessionClosed);
  CHECK(!transport.upload({Space::Shared, 0}, data));
  CHECK(!transport.download({Space::Shared, 0}, result));
  CHECK(!make_emulator_transport({0, 1, 4, 32, 32, 128, 128}));
}
} // namespace

int main() {
  golden_records();
  malformed_records();
  execute_decoded();
  std::cout << "Command transport and codec tests passed\n";
}
