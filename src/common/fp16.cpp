// Convert IEEE binary16 and float32 values with explicit rounding and special-value handling.
#include "ndp/common/fp16.hpp"

#include <bit>
#include <limits>
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

namespace ndp {
uint16_t float_to_fp16(float value) {
  const auto bits = std::bit_cast<uint32_t>(value);
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
  const uint32_t exp = (bits >> 23) & 255, mantissa = bits & 0x7fffff;
  if (exp == 255) {
    return sign | (mantissa ? 0x7e00 : 0x7c00);
  }
  const int half_exp = static_cast<int>(exp) - 127 + 15;
  if (half_exp >= 31) {
    return sign | 0x7c00;
  }
  if (half_exp < -10) {
    return sign;
  }
  uint32_t significand = mantissa;
  int shift = 13;
  uint32_t base = 0;
  if (half_exp <= 0) {
    significand |= 0x800000;
    shift = 14 - half_exp;
  } else {
    base = static_cast<uint32_t>(half_exp) << 10;
  }
  uint32_t rounded = significand >> shift;
  const uint32_t remainder = significand & ((1u << shift) - 1), halfway = 1u << (shift - 1);
  if (remainder > halfway || (remainder == halfway && (rounded & 1))) {
    ++rounded;
  }
  return static_cast<uint16_t>(sign | (base + rounded));
}

float fp16_to_float(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000) << 16;
  uint32_t exp = (bits >> 10) & 31, fraction = bits & 1023;
  if (exp == 31) {
    return std::bit_cast<float>(sign | 0x7f800000 | (fraction ? 0x400000 : 0));
  }
  if (exp == 0) {
    if (!fraction) {
      return std::bit_cast<float>(sign);
    }
    int e = -14;
    while (!(fraction & 1024)) {
      fraction <<= 1;
      --e;
    }
    return std::bit_cast<float>(sign | (static_cast<uint32_t>(e + 127) << 23) |
                                ((fraction & 1023) << 13));
  }
  return std::bit_cast<float>(sign | ((exp + 112) << 23) | (fraction << 13));
}
} // namespace ndp
