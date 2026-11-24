// Export real Session uploads, layout results and NDP1 commands for independent CENT replay.
#include "l1/ndp/codec.hpp"
#include "l1/ndp/transport.hpp"
#include "ndp/common/fp16.hpp"
#include "ndp/l2/session.hpp"

#include <charconv>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>

using namespace ndp;
using namespace ndp::detail::command;

namespace {
template <class T> T take(Result<T> r) {
  if (!r) {
    throw std::runtime_error(r.error().message);
  }
  return std::move(r.value());
}

void done(Result<void> r) {
  if (!r) {
    throw std::runtime_error(r.error().message);
  }
}

class Recorder final : public ICommandTransport {
  std::unique_ptr<ICommandTransport> inner;
  std::filesystem::path directory;
  std::ofstream log;
  size_t serial = 0;

  std::string save(std::span<const std::byte> bytes) {
    auto name = std::to_string(serial++) + ".bin";
    std::ofstream out(directory / name, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    return name;
  }

  void record(const char *kind, Address a, std::span<const std::byte> bytes) {
    auto name = save(bytes);
    log << "{\"kind\":\"" << kind << "\",\"space\":" << int(a.space) + 1
        << ",\"offset\":" << a.offset << ",\"bytes\":" << bytes.size() << ",\"file\":\"" << name
        << "\"}\n";
  }

public:
  Recorder(const NdpSpec &p, std::filesystem::path dir)
      : inner(take(make_emulator_transport(p))), directory(std::move(dir)),
        log(directory / "events.jsonl") {
    log.exceptions(std::ios::badbit | std::ios::failbit);
  }

  Result<void> upload(Address a, std::span<const std::byte> bytes) override {
    auto r = inner->upload(a, bytes);
    if (r) {
      record("upload", a, bytes);
    }
    return r;
  }

  Result<void> download(Address a, std::span<std::byte> bytes) override {
    auto r = inner->download(a, bytes);
    if (r) {
      record("download", a, bytes);
    }
    return r;
  }

  Result<void> submit(std::span<const Command> commands) override {
    // Poison the destination before production packing to test complete zero padding.
    const PnmRelayout *layout =
        commands.size() == 1 ? std::get_if<PnmRelayout>(&commands[0]) : nullptr;
    size_t packed = 0;
    if (layout && layout->pack) {
      packed = ((layout->rows + layout->banks - 1) / layout->banks) *
               ((layout->columns + layout->tile_elements - 1) / layout->tile_elements) *
               layout->banks * layout->row_bytes;
      std::vector<uint16_t> poison(packed / 2, 0x3c00);
      auto r = upload(layout->destination, std::as_bytes(std::span(poison)));
      if (!r) {
        return r;
      }
    }
    auto encoded = wire::encode(commands);
    if (!encoded) {
      return encoded.error();
    }
    auto r = inner->submit(commands);
    if (!r) {
      return r;
    }
    log << "{\"kind\":\"submit\",\"file\":\"" << save(encoded.value()) << "\"}\n";
    if (packed) {
      std::vector<std::byte> snapshot(packed);
      auto read = inner->download(layout->destination, snapshot);
      if (!read) {
        return read;
      }
      record("packed", layout->destination, snapshot);
    }
    return {};
  }

  Result<void> shutdown() override {
    log.flush();
    return inner->shutdown();
  }
};

uint64_t number(const char *s) {
  uint64_t v;
  auto end = s + std::char_traits<char>::length(s);
  auto parsed = std::from_chars(s, end, v);
  if (parsed.ec != std::errc{} || parsed.ptr != end) {
    throw std::runtime_error("Invalid integer");
  }
  return v;
}

Tensor upload(Session &s, std::vector<uint64_t> shape, size_t count, uint64_t seed,
              const std::string &pattern) {
  std::vector<uint16_t> bits(count);
  uint64_t state = seed + 1;
  for (size_t i = 0; i < count; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float value;
    if (pattern == "ones") {
      value = 1;
    } else if (pattern == "impulse") {
      value = i == seed % count ? 1 : 0;
    } else if (pattern == "random") {
      value = (int((state >> 32) % 2001) - 1000) / 4096.f;
    } else if (pattern == "exact") {
      value = (int((i * 7 + seed) % 9) - 4) / 16.f;
    } else {
      throw std::runtime_error("Unknown pattern");
    }
    bits[i] = float_to_fp16(value);
  }
  return take(s.from_host({{shape}, std::as_bytes(std::span(bits))}));
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 12) {
      throw std::runtime_error("Usage: export DIR C B R V N K SEED PATTERN OP M");
    }
    auto C = number(argv[2]), B = number(argv[3]), R = number(argv[4]), V = number(argv[5]);
    auto N = number(argv[6]), K = number(argv[7]), seed = number(argv[8]), M = number(argv[11]);
    std::string pattern = argv[9], op = argv[10];
    if (!C || C > 64 || !B || B > 64 || !R || R > 4096 || R % 32 || !V || V > 4096 || V % 32 ||
        !N || N > 16384 || !K || K > 16384 || !M || M > 64 || N * K > 16000000 ||
        (op != "gemv" && op != "matmul" && op != "linear")) {
      throw std::runtime_error("Fixture limits exceeded");
    }
    size_t P = C * B, S = P * R, T = std::min(R, V) / 2;
    size_t packed_rows = ((N + P - 1) / P) * ((K + T - 1) / T);
    size_t rows = 32 + 2 * packed_rows + 4 * ((2 * (N * K + M * K + M * N) + S - 1) / S);
    if (rows * S > 64 * 1024 * 1024) {
      throw std::runtime_error("Fixture memory budget exceeded");
    }
    NdpSpec spec{C, B, rows, R, V, 2 * T + 4 * P, 128};
    std::filesystem::path dir = argv[1];
    if (std::filesystem::exists(dir / "manifest.json")) {
      throw std::runtime_error("Fixture already exists");
    }
    std::filesystem::create_directories(dir);
    std::ofstream meta(dir / "manifest.json");
    meta.exceptions(std::ios::badbit | std::ios::failbit);
    meta << "{\"version\":1,\"C\":" << C << ",\"B\":" << B << ",\"R\":" << R << ",\"V\":" << V
         << ",\"rows\":" << rows << ",\"gpr\":" << spec.gpr_bytes << ",\"scratch\":128,\"N\":" << N
         << ",\"K\":" << K << ",\"M\":" << M << ",\"seed\":" << seed << ",\"pattern\":\"" << pattern
         << "\",\"operation\":\"" << op << "\"}\n";
    auto backend = take(make_command_backend(spec, std::make_unique<Recorder>(spec, dir)));
    auto session = take(Session::create(std::move(backend)));
    {
      // A live sentinel forces nonzero bases and detects writes into unrelated storage.
      auto guard = upload(session, {S / 2}, S / 2, 11, "exact");
      auto guard_before = take(session.to_host(guard)).bytes;
      auto weights = upload(
          session, op == "matmul" ? std::vector<uint64_t>{K, N} : std::vector<uint64_t>{N, K},
          N * K, seed, pattern);
      std::optional<Tensor> bias;
      if (op == "linear") {
        bias = upload(session, {N}, N, seed + 3, pattern);
      }
      if (op == "gemv") {
        done(take(session.materialize_layout(
                      weights, take(session.query_layout(weights.desc(), LayoutKind::PimBanked))))
                 .wait());
      }
      done(session.synchronize());
      auto baseline = session.memory_stats();
      std::vector<std::byte> first;
      for (size_t request = 0; request < 3; ++request) {
        {
          auto input =
              upload(session, op == "gemv" ? std::vector<uint64_t>{K} : std::vector<uint64_t>{M, K},
                     op == "gemv" ? K : M * K, seed + (request == 1 ? 19 : 0), pattern);
          auto output =
              op == "gemv"     ? take(session.matvec(weights, input,
                                                     {Engine::PIM, LayoutPolicy::RequirePrepared}))
              : op == "matmul" ? take(session.matmul(input, weights))
                               : take(session.linear(input, weights, {}, bias ? &*bias : nullptr));
          auto host = take(session.to_host(output));
          if (request == 0) {
            first = host.bytes;
          }
          if (request == 2 && host.bytes != first) {
            throw std::runtime_error("A-B-A output mismatch");
          }
          if (op == "gemv") {
            // PIM result -> PNM Add -> PIM vector: no detached host round trip.
            auto doubled = take(session.add(output, output));
            (void)take(session.to_host(doubled));
            auto reverse = upload(session, {1, N}, N, 13, "exact");
            (void)take(session.to_host(take(session.matvec(reverse, doubled, {Engine::PIM}))));
          }
        }
        done(session.synchronize());
        auto current = session.memory_stats();
        if (current.live_bytes != baseline.live_bytes ||
            current.allocations != baseline.allocations || current.staging_bytes) {
          throw std::runtime_error("Temporary allocations leaked");
        }
      }
      if (take(session.to_host(guard)).bytes != guard_before) {
        throw std::runtime_error("Guard overwritten");
      }
    }
    done(session.synchronize());
    if (session.memory_stats().live_bytes || session.memory_stats().staging_bytes) {
      throw std::runtime_error("Allocation leak");
    }
    done(session.close());
    std::cout << "fixture exported\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
