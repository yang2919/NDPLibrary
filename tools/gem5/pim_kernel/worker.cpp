// Serve one external client while keeping libpim tensors alive between launches.
#include <gem5/pim/pim.h>

#ifdef NDP_PIM_TARGET_FP16
constexpr unsigned lpddr6_layout_id = 2; // Supplier layout generator STD_IDS.
static_assert(PIM_LAYOUT_STD == lpddr6_layout_id && PIM_LAYOUT_NUM_FMT == PIM_FMT_FP16 &&
              PIM_LAYOUT_ACC_FMT == PIM_FMT_FP32, "expected LPDDR6 FP16/FP32 target layout");
#endif

#if PIM_LAYOUT_NUM_FMT == PIM_FMT_FP16
#include "ndp/common/fp16.hpp"
#include "ndp/l1/gem5_pim/dense_kernel.hpp"
#endif

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
using Bytes = std::vector<char>;
using Tensor = std::unique_ptr<pim_tensor_t, decltype(&pim_tensor_destroy)>;
constexpr uint32_t max_frame = 16 * 1024 * 1024;
static_assert(std::endian::native == std::endian::little && sizeof(float) == 4);

void check(pim_status_t status) {
  if (status != PIM_OK)
    throw std::runtime_error(std::string(pim_strerror(status)) + ": " + pim_error_detail());
}

uint32_t word(std::span<const char> request, size_t at) {
  if (at > request.size() || request.size() - at < 4)
    throw std::runtime_error("missing uint32 field");
  uint32_t result;
  std::memcpy(&result, request.data() + at, 4);
  return result;
}

struct Buffer {
  Tensor tensor;
  size_t elements;
  Tensor rounded{nullptr, pim_tensor_destroy}; // FP16 result, ready for a subsequent GB input.
  bool fp32 = false;
  bool ready = false;
  std::vector<uint16_t> storage; // Mutable FP16 region used by L2 DeviceBuffer.
  pim_tensor_t *logical() const {
    if (!tensor) throw std::runtime_error("expected a libpim tensor");
    return rounded ? rounded.get() : tensor.get();
  }
};

#if PIM_LAYOUT_NUM_FMT == PIM_FMT_FP16
std::vector<float> read_values(Buffer &buffer) {
  if (!buffer.ready || !buffer.tensor) throw std::runtime_error("uninitialized input");
  std::vector<float> values(buffer.elements);
  check(pim_copy_from(buffer.logical(), values.data()));
  return values;
}

void write_values(Buffer &buffer, const std::vector<float> &values) {
  buffer.ready = false;
  check(pim_copy_to(buffer.logical(), values.data()));
  buffer.ready = true;
}
#endif

class Session {
  std::map<uint32_t, Buffer> buffers;
  uint32_t next = 1;

public:
  Bytes dispatch(std::span<const char> r) {
    const char op = r.front();
#if PIM_LAYOUT_NUM_FMT == PIM_FMT_FP16
    if (op == 'V' && r.size() == 1) return {7}; // Bit 0: attention/regions. Bit 1: decoder arithmetic.
#endif
    if (op == 'I' && r.size() == 1) {
      const uint32_t geometry[] = {PIM_LAYOUT_CHANNELS * PIM_LAYOUT_UNITS * PIM_LAYOUT_BANKS_PER_UNIT,
                                   PIM_LAYOUT_ROW_BYTES, PIM_LAYOUT_REG_ROW0};
      Bytes result(sizeof(geometry));
      std::memcpy(result.data(), geometry, sizeof(geometry));
      return result;
    }
    if (op == 'C' && r.size() == 10) {
      const auto rows = word(r, 2), cols = word(r, 6);
      const uint64_t count = uint64_t(rows) * cols;
      if (!count || count > (max_frame - 5) / 4 || next == 0)
        throw std::runtime_error("unsupported tensor extent");
#if PIM_LAYOUT_NUM_FMT == PIM_FMT_FP16
      if (r[1] == 'B') {
        Buffer buffer{Tensor(nullptr, pim_tensor_destroy), size_t(count)};
        buffer.storage.resize(count);
        Bytes reply(4);
        std::memcpy(reply.data(), &next, 4);
        buffers.emplace(next++, std::move(buffer));
        return reply;
      }
#endif
      pim_kind_t kind;
      switch (r[1]) {
      case 'W': kind = PIM_MATRIX_AB; break;
      case 'X': kind = PIM_VECTOR_GB; break;
      case 'P':
      case 'Y': kind = PIM_VECTOR_SB; break;
      default: throw std::runtime_error("unknown tensor role");
      }
      pim_tensor_t *raw = nullptr;
      check(pim_tensor_create(&raw, kind, rows, cols, static_cast<pim_dtype_t>(
          kind == PIM_VECTOR_SB ? PIM_LAYOUT_ACC_FMT : PIM_LAYOUT_NUM_FMT)));
      Buffer buffer{Tensor(raw, pim_tensor_destroy), size_t(count)};
      buffer.fp32 = r[1] == 'P';
#if PIM_LAYOUT_NUM_FMT == PIM_FMT_FP16
      if (kind == PIM_VECTOR_SB && r[1] != 'P') {
        pim_tensor_t *result = nullptr;
        check(pim_tensor_create(&result, PIM_VECTOR_GB, rows, cols, PIM_FP16));
        buffer.rounded.reset(result);
      }
#endif
      Bytes reply(4);
      std::memcpy(reply.data(), &next, 4);
      buffers.emplace(next++, std::move(buffer));
      return reply;
    }
    auto &b = buffers.at(word(r, 1));
    constexpr bool half = PIM_LAYOUT_NUM_FMT == PIM_FMT_FP16;
    const size_t wire_bytes = b.fp32 ? 4 : half ? 2 : 4;
    if (!b.storage.empty() && op != 'F' && op != 'B' && op != 'R')
      throw std::runtime_error("mutable storage requires a region operation");
    if (op == 'U' && r.size() == 5 + wire_bytes * b.elements) {
      std::vector<float> values(b.elements);
#if PIM_LAYOUT_NUM_FMT == PIM_FMT_FP16
      for (size_t i = 0; i < values.size(); ++i) {
        if (b.fp32) std::memcpy(&values[i], r.data() + 5 + i*4, 4);
        else {
          uint16_t bits;
          std::memcpy(&bits, r.data() + 5 + i * 2, 2);
          values[i] = ndp::fp16_to_float(bits);
        }
      }
#else
      std::memcpy(values.data(), r.data() + 5, r.size() - 5);
#endif
      b.ready = false;
      check(pim_copy_to(b.logical(), values.data()));
      b.ready = true;
    } else if (op == 'T' && r.size() == 9) {
      auto &src = buffers.at(word(r, 5));
      if (&src == &b || !src.ready || src.elements != b.elements)
        throw std::runtime_error("invalid representation copy");
      std::vector<float> values(src.elements);
      check(pim_copy_from(src.logical(), values.data()));
      b.ready = false;
      check(pim_copy_to(b.logical(), values.data()));
      b.ready = true;
#if PIM_LAYOUT_NUM_FMT == PIM_FMT_FP16
    } else if (op == 'F' && r.size() == 9) {
      auto value = word(r, 5);
      if (value > 255) throw std::runtime_error("invalid fill byte");
      if (b.storage.empty()) {
        write_values(b, std::vector<float>(b.elements, ndp::fp16_to_float(value * 257)));
      } else {
        std::fill(b.storage.begin(), b.storage.end(), uint16_t(value*257));
        b.ready = true;
      }
    } else if (op == 'B' && r.size() == 21) {
      auto &src = buffers.at(word(r, 5));
      size_t from = word(r, 9), to = word(r, 13), count = word(r, 17);
      if (from > src.elements || count > src.elements-from ||
          to > b.elements || count > b.elements-to) throw std::runtime_error("invalid region");
      if (!src.ready || (!b.ready && count != b.elements))
        throw std::runtime_error("uninitialized region");
      if (!src.storage.empty() && !b.storage.empty()) {
        std::memmove(b.storage.data()+to, src.storage.data()+from, count*sizeof(uint16_t));
      } else if (!b.storage.empty()) {
        auto input = read_values(src);
        for (size_t i = 0; i < count; ++i) b.storage[to+i] = ndp::float_to_fp16(input[from+i]);
      } else {
        auto output = count == b.elements ? std::vector<float>(count) : read_values(b);
        if (src.storage.empty()) {
          auto input = read_values(src);
          std::copy_n(input.begin()+from, count, output.begin()+to);
        } else {
          for (size_t i = 0; i < count; ++i) output[to+i] = ndp::fp16_to_float(src.storage[from+i]);
        }
        write_values(b, output);
      }
      b.ready = true;
    } else if (op == 'K' && r.size() == 9 + 4*b.elements) {
      auto input = read_values(buffers.at(word(r, 5)));
      std::vector<float> output(b.elements);
      for (size_t i = 0; i < output.size(); ++i) output[i] = input.at(word(r, 9+4*i));
      write_values(b, output);
    } else if (op == 'A' && r.size() == 37) {
      using namespace ndp::gem5_pim;
      DenseShape shape{DenseOp(word(r, 17)), word(r, 21), word(r, 25), word(r, 29),
                       std::bit_cast<float>(word(r, 33))};
      const auto [kind, m, n, k, factor] = shape;
      bool dot = kind == DenseOp::Matmul || kind == DenseOp::Linear;
      if (kind < DenseOp::Matmul || kind > DenseOp::Add || !m || !n ||
          uint64_t(m)*n != b.elements || !std::isfinite(factor) || dot != bool(k) ||
          (kind == DenseOp::RMSNorm && factor <= 0))
        throw std::runtime_error("invalid dense operation");
      auto operand = [&](size_t field, uint64_t extent) {
        auto id = word(r, field);
        if (!id && !extent) return std::vector<float>{};
        auto &input = buffers.at(id);
        if (&input == &b || !extent || input.elements != extent)
          throw std::runtime_error("invalid dense operand");
        return read_values(input);
      };
      auto left = operand(5, uint64_t(m)*(dot ? k : n));
      auto right = operand(9, kind == DenseOp::RMSNorm ? n : kind == DenseOp::Add ? b.elements : uint64_t(k)*n);
      auto bias = operand(13, kind == DenseOp::Linear && word(r, 13) ? n : 0);
      std::vector<float> out(b.elements);
      dense_compute(shape, left.data(), right.data(), bias.empty() ? nullptr : bias.data(),
                    out.data(), [](float x) { return std::exp(x); });
      write_values(b, out);
    } else if ((op == 'S' && r.size() == 9) || (op == 'M' && r.size() == 13)) {
      auto &left = buffers.at(word(r, 5));
      if (!left.ready || &left == &b || left.elements != b.elements)
        throw std::runtime_error("invalid pointwise input");
      std::vector<float> values(b.elements);
      check(pim_copy_from(left.logical(), values.data()));
      if (op == 'S') {
        for (float &x : values) x = x / (1.0f + std::exp(-x));
      } else {
        auto &right = buffers.at(word(r, 9));
        if (!right.ready || &right == &b || right.elements != b.elements)
          throw std::runtime_error("invalid multiply input");
        std::vector<float> other(b.elements);
        check(pim_copy_from(right.logical(), other.data()));
        for (size_t i = 0; i < values.size(); ++i) values[i] *= other[i];
      }
      b.ready = false;
      check(pim_copy_to(b.logical(), values.data()));
      b.ready = true;
#endif
    } else if (op == 'G' && r.size() == 13) {
      auto &w = buffers.at(word(r, 5));
      auto &x = buffers.at(word(r, 9));
      if (!w.tensor || !w.ready || !x.ready)
        throw std::runtime_error("uninitialized GEMV input");
      if (&b == &w || &b == &x)
        throw std::runtime_error("output aliases GEMV input");
      b.ready = false;
      check(pim_gemv(b.tensor.get(), w.tensor.get(), x.logical()));
      if (b.rounded) {
        // libpim returns FP32 SB results. Materialize the logical FP16 result once.
        std::vector<float> accum(b.elements);
        check(pim_copy_from(b.tensor.get(), accum.data()));
        check(pim_copy_to(b.rounded.get(), accum.data()));
      }
      b.ready = true;
    } else if (op == 'D' && r.size() == 5 && b.ready) {
      std::vector<float> values(b.elements);
      check(pim_copy_from(b.logical(), values.data()));
      Bytes reply(b.elements * wire_bytes);
#if PIM_LAYOUT_NUM_FMT == PIM_FMT_FP16
      for (size_t i = 0; i < values.size(); ++i) {
        if (b.fp32) std::memcpy(reply.data()+4*i, &values[i], 4);
        else {
          const uint16_t bits = ndp::float_to_fp16(values[i]);
          std::memcpy(reply.data() + 2 * i, &bits, 2);
        }
      }
#else
      std::memcpy(reply.data(), values.data(), reply.size());
#endif
      return reply;
    } else if (op == 'R' && r.size() == 5) {
      buffers.erase(word(r, 1));
    } else {
      throw std::runtime_error("invalid request length, operation or unreadable tensor");
    }
    return {};
  }
};

void reply(char status, const Bytes &bytes) {
  const uint32_t length = 1 + bytes.size();
  std::cout.write(reinterpret_cast<const char *>(&length), 4).put(status)
      .write(bytes.data(), bytes.size()).flush();
}

int serve() {
  Session session;
  bool connected = false;
  for (;;) {
    uint32_t length;
    if (!std::cin.read(reinterpret_cast<char *>(&length), 4))
      return std::cin.eof() && std::cin.gcount() == 0 ? 0 : 1;
    if (!length || length > max_frame) return 1;
    Bytes request(length);
    if (!std::cin.read(request.data(), length)) return 1;
    if (!connected) {
      constexpr bool half = PIM_LAYOUT_NUM_FMT == PIM_FMT_FP16;
      if (request != Bytes{'H', half ? 2 : 1}) {
        const std::string message = "worker protocol/profile mismatch";
        reply(1, Bytes(message.begin(), message.end()));
        return 1;
      }
      const std::string profile = half ? "LPDDR6/FP16/FP32" : "LPDDR6/BF16/FP32";
      reply(0, Bytes(profile.begin(), profile.end()));
      connected = true;
      continue;
    }
    if (request == Bytes{'Q'}) {
      reply(0, {});
      return 0;
    }
    try {
      reply(0, session.dispatch(request));
    } catch (const std::exception &error) {
      const std::string message = error.what();
      reply(1, Bytes(message.begin(), message.end()));
    }
  }
}

int run_worker(void (*session_closed)() = nullptr) {
  try {
    check(pim_init(nullptr));
    int status = serve();
    if (!status && session_closed) session_closed();
    pim_finalize();
    return status;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
} // namespace

int main() { return run_worker(); }
