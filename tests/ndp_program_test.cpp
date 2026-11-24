// Exercise ELF loading, zero-filled segments, handle lifetimes, and the non-executing launch ABI.
#include "l1/ndp/emulator.hpp"
#include "test_support.hpp"

#include <array>
using namespace ndp;
using namespace ndp::detail::command;

namespace {
void put(std::vector<std::byte> &data, size_t offset, uint64_t value, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    data.at(offset + i) = std::byte((value >> (8 * i)) & 255);
  }
}

std::vector<std::byte> elf(uint64_t base = 0x10000000) {
  std::vector<std::byte> data(264);
  put(data, 0, 0x464c457f, 4);
  put(data, 4, 2, 1);
  put(data, 5, 1, 1);
  put(data, 6, 1, 1);
  put(data, 16, 2, 2);
  put(data, 18, 243, 2);
  put(data, 20, 1, 4);
  put(data, 24, base, 8);
  put(data, 32, 64, 8);
  put(data, 52, 64, 2);
  put(data, 54, 56, 2);
  put(data, 56, 1, 2);
  put(data, 64, 1, 4);
  put(data, 68, 5, 4);
  put(data, 72, 256, 8);
  put(data, 80, base, 8);
  put(data, 88, base, 8);
  put(data, 96, 8, 8);
  put(data, 104, 16, 8);
  put(data, 112, 256, 8);
  // addi a0, zero, 0; ret. The fixture is loaded, never claimed to have executed.
  put(data, 256, 0x00000513, 4);
  put(data, 260, 0x00008067, 4);
  return data;
}

void loader() {
  ProgramStore store;
  auto image = elf();
  auto handle = store.load(image, {});
  CHECK(handle);
  auto program = store.get(handle.value());
  CHECK(program);
  CHECK(program->entry == 0x10000000 && program->segments.size() == 1);
  CHECK(program->segments[0].flags == 5 && program->segments[0].bytes.size() == 16);
  CHECK(program->segments[0].bytes[0] == std::byte{0x13});
  for (size_t i = 8; i < 16; ++i) {
    CHECK(program->segments[0].bytes[i] == std::byte{0});
  }
  image[256] = std::byte{0};
  CHECK(program->segments[0].bytes[0] == std::byte{0x13});
  CHECK(!store.load(elf(), {})); // Resident fixed-address programs may not overlap.
  auto other = store.load(elf(0x10001000), {});
  CHECK(other);
  CHECK(!store.unload(handle.value())); // Launch-style reference pins the program.
  program.reset();
  CHECK(store.unload(handle.value()));
  CHECK(!store.get(handle.value()));
  auto again = store.load(elf(), {});
  CHECK(again && again.value() > handle.value());
  CHECK(!store.unload(handle.value()));
  CHECK(store.unload(again.value()));
  auto req = ProgramRequirements{};
  req.extensions.push_back("zvfh");
  CHECK(!store.load(elf(), req));
  req = {};
  req.minimum_vlen_bits = 256;
  CHECK(!store.load(elf(), req));
  req = {};
  req.vector_major = 0;
  CHECK(!store.load(elf(), req));
  ProgramProfile small;
  small.total_loaded_bytes = 8;
  ProgramStore limited(small);
  CHECK(!limited.load(elf(), {}));
  small.total_loaded_bytes = 16;
  ProgramStore total(small);
  CHECK(total.load(elf(), {}));
  CHECK(!total.load(elf(0x10001000), {}));
}

void malformed() {
  auto image = elf();
  for (size_t size = 0; size < image.size(); ++size) {
    ProgramStore store;
    CHECK(!store.load(std::span(image).first(size), {}));
  }

  struct Change {
    size_t offset;
    uint64_t value;
    size_t size;
  };

  for (auto change : {Change{4, 1, 1},
                      {5, 2, 1},
                      {16, 3, 2},
                      {18, 62, 2},
                      {32, UINT64_MAX, 8},
                      {56, 65535, 2},
                      {64, 2, 4},
                      {64, 3, 4},
                      {64, 7, 4},
                      {68, 7, 4},
                      {72, UINT64_MAX, 8},
                      {96, 17, 8},
                      {104, UINT64_MAX, 8},
                      {112, 3, 8},
                      {24, 0x10000008, 8},
                      {24, 0x10000001, 8},
                      {80, 0x40000000, 8},
                      {48, 8, 4},
                      {48, 1, 4}}) {
    auto bad = elf();
    put(bad, change.offset, change.value, change.size);
    ProgramStore store;
    CHECK(!store.load(bad, {}));
  }
  auto overlap = elf();
  put(overlap, 56, 2, 2);
  std::copy(overlap.begin() + 64, overlap.begin() + 120, overlap.begin() + 120);
  ProgramStore store;
  CHECK(!store.load(overlap, {}));
  auto relocation = elf();
  relocation.resize(384);
  put(relocation, 40, 320, 8);
  put(relocation, 58, 64, 2);
  put(relocation, 60, 1, 2);
  put(relocation, 324, 4, 4);
  CHECK(!store.load(relocation, {}));
}

void launch() {
  auto created = Emulator::create({1, 1, 4, 64, 64, 128, 128});
  CHECK(created);
  auto &device = *created.value();
  auto handle = device.load_program(elf(), {});
  CHECK(handle);
  std::vector<std::byte> args(16);
  put(args, 0, 1, 4);
  put(args, 4, 16, 4);
  CHECK(device.upload({Space::Gpr, 0}, args));
  LaunchPnmProgram cmd{handle.value(), {Space::Gpr, 0}, 16, {Space::Scratchpad, 0}, 64, 1000};
  {
    auto state = device.prepare_launch(cmd);
    CHECK(state);
    CHECK(state.value().pc == 0x10000000 && state.value().ra == 0x1ffff000);
    CHECK(state.value().sp == 0xc0000040 && state.value().a0 == 0x80000000 &&
          state.value().a1 == 16);
    CHECK(!device.unload_program(handle.value()));
  }
  std::array<Command, 2> commands{Transfer{{Space::Gpr, 0}, {Space::Shared, 0}, 16}, cmd};
  auto result = device.submit(commands);
  CHECK(!result && result.error().code == ErrorCode::UnsupportedOperation);
  std::array<std::byte, 16> destination{};
  CHECK(device.download({Space::Shared, 0}, destination));
  CHECK(destination[0] == std::byte{0}); // No earlier transfer executes during rejected submission.
  for (int test = 0; test < 6; ++test) {
    auto bad = cmd;
    if (test == 0) {
      bad.program++;
    }
    if (test == 1) {
      bad.arguments.offset = 1;
    }
    if (test == 2) {
      bad.stack_bytes = 15;
    }
    if (test == 3) {
      bad.stack.offset = UINT64_MAX;
    }
    if (test == 4) {
      bad.instruction_limit = 0;
    }
    if (test == 5) {
      bad.argument_bytes = 128;
    }
    CHECK(!device.prepare_launch(bad));
  }
  put(args, 0, 2, 4);
  CHECK(device.upload({Space::Gpr, 0}, args));
  CHECK(!device.prepare_launch(cmd));
  put(args, 0, 1, 4);
  CHECK(device.upload({Space::Shared, 16}, args));
  cmd.arguments = {Space::Shared, 16};
  CHECK(device.prepare_launch(cmd).value().a0 == 0x40000010);
  CHECK(device.unload_program(handle.value()));
  CHECK(!device.prepare_launch(cmd));
}
} // namespace

int main() {
  loader();
  malformed();
  launch();
  std::cout << "PNM program contracts passed\n";
}
