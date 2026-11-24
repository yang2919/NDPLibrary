// Native protocol test service. Launch captures entry and relocated bytes for N1.
#include "l1/ndp/program.hpp"

#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {
using namespace ndp::detail::command;
using Bytes = std::vector<std::byte>;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

struct Reader {
  std::span<const std::byte> rest;
  auto take(size_t n) {
    require(n <= rest.size(), "short record");
    auto bytes = rest.first(n);
    rest = rest.subspan(n);
    return bytes;
  }
  uint64_t word(size_t n) {
    auto bytes = take(n);
    uint64_t value = 0;
    for (size_t i = 0; i < n; ++i) value |= uint64_t(bytes[i]) << (8 * i);
    return value;
  }
  void end() const { require(rest.empty(), "trailing bytes"); }
};

void append(Bytes &out, uint64_t value, size_t n) {
  for (size_t i = 0; i < n; ++i) out.push_back(std::byte(value >> (8 * i)));
}

Bytes number(uint64_t value) {
  Bytes result;
  append(result, value, 8);
  return result;
}

struct Slot { uint32_t access, alignment; uint64_t extent; };
struct Manifest { Bytes arguments; std::map<uint32_t, Slot> slots; };
struct Buffer { uint64_t base, size, access; };

class Service {
  static ProgramProfile capture_profile() {
    ProgramProfile profile;
    profile.extensions.insert(profile.extensions.end(), {"zvfh", "pim"});
    return profile;
  }
  ProgramStore programs{capture_profile()};
  std::map<uint64_t, Manifest> manifests;
  std::map<uint64_t, Buffer> buffers;
  uint64_t next_buffer = 1;
  bool connected = false;

  Bytes load(Reader &r) {
    auto profile = r.word(4);
    require(profile == 1 || profile == 2, "unsupported test profile");
    auto size = r.word(4), count = r.word(4);
    require(size >= 8 && size <= 65536 && count <= size / 8, "invalid manifest size");
    auto bytes = r.take(size);
    Reader header{bytes};
    require(header.word(4) == 1 && header.word(4) == size, "argument ABI mismatch");
    Manifest manifest{{bytes.begin(), bytes.end()}, {}};
    for (size_t i = 0; i < count; ++i) {
      auto patch = r.word(4), access = r.word(4), alignment = r.word(4), extent = r.word(8);
      require(patch >= 8 && patch % 8 == 0 && patch <= size - 8, "bad pointer slot");
      require((access == 1 || access == 2) && alignment && !(alignment & (alignment-1)) && extent,
              "invalid slot requirements");
      Reader pointer{bytes.subspan(patch, 8)};
      require(pointer.word(8) == 0, "nonzero pointer placeholder");
      require(manifest.slots.emplace(patch, Slot{uint32_t(access), uint32_t(alignment), extent}).second,
              "duplicate manifest slot");
    }
    ProgramRequirements requirements;
    requirements.extensions.push_back("zvfh");
    if (profile == 2) requirements.extensions.push_back("pim");
    auto id = programs.load(r.rest, requirements);
    if (!id) throw std::runtime_error(id.error().message);
    try { manifests.emplace(id.value(), std::move(manifest)); }
    catch (...) { (void)programs.unload(id.value()); throw; }
    return number(id.value());
  }

  Bytes launch(Reader &r) {
    auto id = r.word(8);
    auto program = programs.get(id);
    require(bool(program), "unknown program");
    const auto &manifest = manifests.at(id);
    auto size = r.word(4), count = r.word(4);
    require(size == manifest.arguments.size() && count == manifest.slots.size(), "launch signature mismatch");
    auto input = r.take(size);
    require(std::equal(input.begin(), input.end(), manifest.arguments.begin()), "invalid argument template");
    Bytes args(input.begin(), input.end());
    std::set<uint32_t> bound;
    for (size_t i = 0; i < count; ++i) {
      auto patch = r.word(4), access = r.word(4), handle = r.word(8), offset = r.word(8), extent = r.word(8);
      require(manifest.slots.contains(patch) && buffers.contains(handle) && bound.insert(patch).second,
              "invalid slot or buffer handle");
      const auto &slot = manifest.slots.at(patch);
      const auto &buf = buffers.at(handle);
      require(access == slot.access && (access & buf.access) == access && extent == slot.extent,
              "wrong access or extent");
      require(offset <= buf.size && extent <= buf.size-offset && (buf.base+offset)%slot.alignment == 0,
              "buffer range or alignment mismatch");
      for (size_t j = 0; j < 8; ++j) args[patch+j] = std::byte((buf.base+offset) >> (8*j));
    }
    r.end();
    Bytes capture = number(program->entry);
    capture.insert(capture.end(), args.begin(), args.end());
    return capture;
  }

public:
  Bytes call(Reader r) {
    auto op = r.word(1);
    if (!connected) {
      require(op == 'H' && r.word(1) == 3, "expected protocol v3 handshake");
      r.end();
      connected = true;
      return {std::byte{3}, std::byte{0}}; // version, capture executor
    }
    if (op == 'P') return load(r);
    if (op == 'J') return launch(r);
    if (op == 'A') {
      auto size = r.word(8), access = r.word(4);
      r.end();
      require(size && size <= 0x40000000 && access && access <= 3, "invalid allocation");
      uint64_t base = 0x40000000;
      // First fit in the mock shared bank window; released ranges are reusable.
      std::map<uint64_t, uint64_t> ranges;
      for (const auto &[id,b] : buffers) ranges.emplace(b.base, b.size);
      for (auto [start, length] : ranges) {
        if (base + size <= start) break;
        base = (start + length + 4095) & ~uint64_t(4095);
      }
      require(base + size <= 0x80000000 && next_buffer != UINT64_MAX, "no buffer capacity");
      auto id = next_buffer++;
      buffers.emplace(id, Buffer{base,size,access});
      return number(id);
    }
    if (op == 'F' || op == 'U') {
      auto id = r.word(8);
      r.end();
      if (op == 'F') require(buffers.erase(id) == 1, "unknown buffer");
      else {
        auto status = programs.unload(id);
        if (!status) throw std::runtime_error(status.error().message);
        manifests.erase(id);
      }
      return {};
    }
    throw std::runtime_error("unknown operation");
  }
};
} // namespace

int main() {
  Service service;
  for (;;) {
    std::byte header[4];
    if (!std::cin.read(reinterpret_cast<char *>(header), 4))
      return std::cin.eof() && !std::cin.gcount() ? 0 : 1;
    auto size = Reader{header}.word(4);
    if (!size || size > (16u << 20)) return 1;
    Bytes input(size), result;
    if (!std::cin.read(reinterpret_cast<char *>(input.data()), size)) return 1;
    bool failed = false;
    try { result = service.call({input}); }
    catch (const std::exception &e) {
      failed = true;
      for (const char *p = e.what(); *p; ++p) result.push_back(std::byte(*p));
    }
    Bytes prefix;
    append(prefix, result.size()+1, 4);
    prefix.push_back(std::byte(failed));
    std::cout.write(reinterpret_cast<const char *>(prefix.data()), prefix.size());
    if (!result.empty()) std::cout.write(reinterpret_cast<const char *>(result.data()), result.size());
    if (!std::cout.flush()) return 1;
  }
}
