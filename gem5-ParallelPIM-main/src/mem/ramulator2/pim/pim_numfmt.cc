/*
 * Number formats of the PIM functional model; see pim_numfmt.hh.
 */

#include "mem/ramulator2/pim/pim_numfmt.hh"

#include <softfloat.h>

#include <cstring>

namespace gem5
{

namespace pim
{

namespace
{

/**
 * SoftFloat's rounding mode and exception flags are globals that the
 * RISC-V floating-point instructions set and read; keep them as they
 * were around every PIM computation.
 */
class SoftFloatScope
{
  public:
    SoftFloatScope()
        : mode(softfloat_roundingMode), flags(softfloat_exceptionFlags)
    {
        softfloat_roundingMode = softfloat_round_near_even;
    }

    ~SoftFloatScope()
    {
        softfloat_roundingMode = mode;
        softfloat_exceptionFlags = flags;
    }

    SoftFloatScope(const SoftFloatScope &) = delete;
    SoftFloatScope &operator=(const SoftFloatScope &) = delete;

  private:
    const uint_fast8_t mode;
    const uint_fast8_t flags;
};

uint32_t
bf16FromF32(uint32_t u)
{
    if ((u & 0x7fffffffu) > 0x7f800000u)
        return (u >> 16) | 0x0040u; // a quiet NaN
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

} // anonymous namespace

unsigned
fmtBytes(NumFmt fmt)
{
    switch (fmt) {
      case NumFmt::Int8:
        return 1;
      case NumFmt::Bf16:
      case NumFmt::Fp16:
      case NumFmt::Int16:
        return 2;
      default:
        return 4;
    }
}

bool
fmtIsFloat(NumFmt fmt)
{
    return fmt == NumFmt::Bf16 || fmt == NumFmt::Fp16 || fmt == NumFmt::Fp32;
}

const char *
fmtName(NumFmt fmt)
{
    constexpr const char *names[NumNumFmts] = {
        "bf16", "fp16", "fp32", "int8", "int16", "int32",
    };
    unsigned i = static_cast<unsigned>(fmt);
    return i < NumNumFmts ? names[i] : "invalid";
}

bool
fmtFromName(const char *name, NumFmt &fmt)
{
    for (unsigned i = 0; i < NumNumFmts; i++) {
        if (std::strcmp(name, fmtName(static_cast<NumFmt>(i))) == 0) {
            fmt = static_cast<NumFmt>(i);
            return true;
        }
    }
    return false;
}

const char *
checkFormats(NumFmt num, NumFmt acc, unsigned colBytes)
{
    if (static_cast<unsigned>(num) >= NumNumFmts ||
        static_cast<unsigned>(acc) >= NumNumFmts)
        return "unknown number format";
    if (fmtIsFloat(num) != fmtIsFloat(acc))
        return "acc_fmt and num_fmt must both be integer or both floating";
    if (fmtBytes(acc) < fmtBytes(num))
        return "acc_fmt must be at least as wide as num_fmt";
    if (colBytes == 0 || colBytes % fmtBytes(num) != 0)
        return "the size of num_fmt must divide the burst size";
    return nullptr;
}

unsigned
elemsPerColumn(NumFmt fmt, unsigned colBytes)
{
    return colBytes / fmtBytes(fmt);
}

uint32_t
loadBits(NumFmt fmt, const uint8_t *p)
{
    uint32_t v = 0;
    for (unsigned i = 0; i < fmtBytes(fmt); i++)
        v |= uint32_t(p[i]) << (8 * i);
    return v;
}

void
storeBits(NumFmt fmt, uint8_t *p, uint32_t bits)
{
    for (unsigned i = 0; i < fmtBytes(fmt); i++)
        p[i] = uint8_t(bits >> (8 * i));
}

int64_t
loadInt(NumFmt fmt, const uint8_t *p)
{
    const unsigned nbits = 8 * fmtBytes(fmt);
    const uint32_t v = loadBits(fmt, p);
    if (nbits == 32)
        return int64_t(int32_t(v));
    const int64_t sign = int64_t(1) << (nbits - 1);
    return int64_t(v ^ uint32_t(sign)) - sign;
}

void
storeInt(NumFmt fmt, uint8_t *p, int64_t v)
{
    storeBits(fmt, p, uint32_t(uint64_t(v)));
}

uint32_t
toF32(NumFmt fmt, uint32_t bits)
{
    switch (fmt) {
      case NumFmt::Bf16:
        return bits << 16;
      case NumFmt::Fp16: {
        SoftFloatScope scope;
        return f16_to_f32(float16_t{uint16_t(bits)}).v;
      }
      default:
        return bits;
    }
}

uint32_t
fromF32(NumFmt fmt, uint32_t f32)
{
    switch (fmt) {
      case NumFmt::Bf16:
        return bf16FromF32(f32);
      case NumFmt::Fp16: {
        SoftFloatScope scope;
        return f32_to_f16(float32_t{f32}).v;
      }
      default:
        return f32;
    }
}

uint32_t
f32Add(uint32_t a, uint32_t b)
{
    SoftFloatScope scope;
    return f32_add(float32_t{a}, float32_t{b}).v;
}

uint32_t
f32Mul(uint32_t a, uint32_t b)
{
    SoftFloatScope scope;
    return f32_mul(float32_t{a}, float32_t{b}).v;
}

void
ewElement(NumFmt fmt, bool mul, const uint8_t *a, const uint8_t *b,
          uint8_t *dst)
{
    if (fmtIsFloat(fmt)) {
        const uint32_t x = toF32(fmt, loadBits(fmt, a));
        const uint32_t y = toF32(fmt, loadBits(fmt, b));
        const uint32_t r = mul ? f32Mul(x, y) : f32Add(x, y);
        storeBits(fmt, dst, fromF32(fmt, r));
    } else {
        const uint64_t x = uint64_t(loadInt(fmt, a));
        const uint64_t y = uint64_t(loadInt(fmt, b));
        storeInt(fmt, dst, int64_t(mul ? x * y : x + y));
    }
}

void
macColumn(NumFmt num, NumFmt accFmt, const uint8_t *x, const uint8_t *y,
          unsigned n, uint8_t *acc)
{
    const unsigned e = fmtBytes(num);
    if (!fmtIsFloat(num)) {
        uint64_t partial = 0;
        for (unsigned i = 0; i < n; i++) {
            partial += uint64_t(loadInt(num, x + i * e)) *
                       uint64_t(loadInt(num, y + i * e));
        }
        storeInt(accFmt, acc, int64_t(uint64_t(loadInt(accFmt, acc)) +
                                      partial));
        return;
    }
    // Round a fp32 value to acc_fmt and back.
    auto inAcc = [accFmt](uint32_t v) {
        return toF32(accFmt, fromF32(accFmt, v));
    };
    uint32_t partial = 0;
    for (unsigned i = 0; i < n; i++) {
        const uint32_t a = inAcc(toF32(num, loadBits(num, x + i * e)));
        const uint32_t b = inAcc(toF32(num, loadBits(num, y + i * e)));
        const uint32_t p = inAcc(f32Mul(a, b));
        partial = i == 0 ? p : inAcc(f32Add(partial, p));
    }
    if (n == 0)
        return;
    const uint32_t sum = f32Add(toF32(accFmt, loadBits(accFmt, acc)), partial);
    storeBits(accFmt, acc, fromF32(accFmt, sum));
}

} // namespace pim
} // namespace gem5
