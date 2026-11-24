/*
 * Tests of the PIM number formats.
 */

#include <gtest/gtest.h>

#include <softfloat.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "mem/ramulator2/pim/pim_numfmt.hh"

using namespace gem5::pim;

namespace
{

uint32_t
f2u(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

float
u2f(uint32_t u)
{
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

/** n elements of fmt from bit patterns. */
std::vector<uint8_t>
elems(NumFmt fmt, std::initializer_list<uint32_t> bits)
{
    std::vector<uint8_t> v(bits.size() * fmtBytes(fmt));
    size_t i = 0;
    for (uint32_t b : bits)
        storeBits(fmt, v.data() + fmtBytes(fmt) * i++, b);
    return v;
}

} // anonymous namespace

TEST(PimNumFmt, SizesNamesAndFamilies)
{
    EXPECT_EQ(fmtBytes(NumFmt::Bf16), 2);
    EXPECT_EQ(fmtBytes(NumFmt::Fp16), 2);
    EXPECT_EQ(fmtBytes(NumFmt::Fp32), 4);
    EXPECT_EQ(fmtBytes(NumFmt::Int8), 1);
    EXPECT_EQ(fmtBytes(NumFmt::Int16), 2);
    EXPECT_EQ(fmtBytes(NumFmt::Int32), 4);
    EXPECT_TRUE(fmtIsFloat(NumFmt::Bf16));
    EXPECT_TRUE(fmtIsFloat(NumFmt::Fp32));
    EXPECT_FALSE(fmtIsFloat(NumFmt::Int8));
    for (unsigned i = 0; i < NumNumFmts; i++) {
        NumFmt f;
        ASSERT_TRUE(fmtFromName(fmtName(NumFmt(i)), f));
        EXPECT_EQ(f, NumFmt(i));
    }
    NumFmt f;
    EXPECT_FALSE(fmtFromName("mxfp8", f));
    EXPECT_STREQ(fmtName(NumFmt(9)), "invalid");
}

TEST(PimNumFmt, ElementsPerColumn)
{
    // B / E elements; bf16 with B = 32 gives 16, fp32 gives 8.
    const unsigned expect32[NumNumFmts] = {16, 16, 8, 32, 16, 8};
    for (unsigned i = 0; i < NumNumFmts; i++) {
        EXPECT_EQ(elemsPerColumn(NumFmt(i), 32), expect32[i]);
        EXPECT_EQ(elemsPerColumn(NumFmt(i), 64), 2 * expect32[i]);
    }
}

TEST(PimNumFmt, FormatPairs)
{
    EXPECT_EQ(checkFormats(NumFmt::Bf16, NumFmt::Fp32, 32), nullptr);
    EXPECT_EQ(checkFormats(NumFmt::Bf16, NumFmt::Bf16, 32), nullptr);
    EXPECT_EQ(checkFormats(NumFmt::Fp16, NumFmt::Fp32, 64), nullptr);
    EXPECT_EQ(checkFormats(NumFmt::Int8, NumFmt::Int32, 32), nullptr);
    EXPECT_EQ(checkFormats(NumFmt::Int16, NumFmt::Int16, 32), nullptr);
    // Narrower accumulator.
    EXPECT_NE(checkFormats(NumFmt::Fp32, NumFmt::Bf16, 32), nullptr);
    EXPECT_NE(checkFormats(NumFmt::Int32, NumFmt::Int16, 32), nullptr);
    // Mixed families.
    EXPECT_NE(checkFormats(NumFmt::Int16, NumFmt::Fp32, 32), nullptr);
    EXPECT_NE(checkFormats(NumFmt::Bf16, NumFmt::Int32, 32), nullptr);
    // The element size must divide the burst.
    EXPECT_NE(checkFormats(NumFmt::Fp32, NumFmt::Fp32, 30), nullptr);
    EXPECT_NE(checkFormats(NumFmt::Bf16, NumFmt::Fp32, 0), nullptr);
    EXPECT_NE(checkFormats(NumFmt(7), NumFmt::Fp32, 32), nullptr);
}

TEST(PimNumFmt, Bf16Conversions)
{
    // Widening is a shift.
    EXPECT_EQ(toF32(NumFmt::Bf16, 0x3f80), f2u(1.0f));
    EXPECT_EQ(toF32(NumFmt::Bf16, 0xc040), f2u(-3.0f));
    // Exact values, and RNE ties: 1 + 2^-8 is halfway between 1 and
    // 1 + 2^-7 and goes to the even 1; 1 + 3 x 2^-8 goes up to 1 + 2^-6.
    EXPECT_EQ(fromF32(NumFmt::Bf16, f2u(1.0f)), 0x3f80);
    EXPECT_EQ(fromF32(NumFmt::Bf16, f2u(1.0f + 0x1p-8f)), 0x3f80);
    EXPECT_EQ(fromF32(NumFmt::Bf16, f2u(1.0f + 3 * 0x1p-8f)), 0x3f82);
    EXPECT_EQ(fromF32(NumFmt::Bf16, 0x3f808001), 0x3f81); // above half
    EXPECT_EQ(fromF32(NumFmt::Bf16, 0x3f807fff), 0x3f80); // below half
    // Overflow rounds to infinity; -0 stays -0.
    EXPECT_EQ(fromF32(NumFmt::Bf16, 0x7f7fffff), 0x7f80);
    EXPECT_EQ(fromF32(NumFmt::Bf16, 0x80000000), 0x8000);
    // A NaN keeps its top bits with the quiet bit set.
    EXPECT_EQ(fromF32(NumFmt::Bf16, 0x7fc00000), 0x7fc0);
    EXPECT_EQ(fromF32(NumFmt::Bf16, 0x7f800001), 0x7fc0);
    EXPECT_EQ(fromF32(NumFmt::Bf16, 0xff812345), 0xffc1);
    // Every bf16 value that is not a NaN round-trips.
    for (uint32_t h = 0; h < 0x10000; h++) {
        if ((h & 0x7f80) == 0x7f80 && (h & 0x7f))
            continue;
        ASSERT_EQ(fromF32(NumFmt::Bf16, toF32(NumFmt::Bf16, h)), h) << h;
    }
}

TEST(PimNumFmt, Fp16Conversions)
{
    EXPECT_EQ(toF32(NumFmt::Fp16, 0x3c00), f2u(1.0f));
    EXPECT_EQ(toF32(NumFmt::Fp16, 0x0001), f2u(0x1p-24f)); // subnormal
    EXPECT_EQ(toF32(NumFmt::Fp16, 0x7c00), 0x7f800000u);
    EXPECT_EQ(fromF32(NumFmt::Fp16, f2u(65504.0f)), 0x7bff);
    // 65520 is halfway to 2^16 and rounds to even, which overflows.
    EXPECT_EQ(fromF32(NumFmt::Fp16, f2u(65519.0f)), 0x7bff);
    EXPECT_EQ(fromF32(NumFmt::Fp16, f2u(65520.0f)), 0x7c00);
    // RNE ties: 1 + 2^-11 goes to 1, 1 + 3 x 2^-11 to 1 + 2^-9.
    EXPECT_EQ(fromF32(NumFmt::Fp16, f2u(1.0f + 0x1p-11f)), 0x3c00);
    EXPECT_EQ(fromF32(NumFmt::Fp16, f2u(1.0f + 3 * 0x1p-11f)), 0x3c02);
    // Subnormal results, and half the smallest subnormal ties to 0.
    EXPECT_EQ(fromF32(NumFmt::Fp16, f2u(0x1p-15f)), 0x0200);
    EXPECT_EQ(fromF32(NumFmt::Fp16, f2u(0x1p-25f)), 0x0000);
    EXPECT_EQ(fromF32(NumFmt::Fp16, f2u(0x1.8p-25f)), 0x0001);
    EXPECT_EQ(fromF32(NumFmt::Fp16, f2u(-0x1p-26f)), 0x8000);
    // A NaN becomes the canonical 0x7e00 (SoftFloat's RISC-V NaN).
    EXPECT_EQ(fromF32(NumFmt::Fp16, 0x7fc00000), 0x7e00);
    EXPECT_EQ(fromF32(NumFmt::Fp16, 0xff812345), 0x7e00);
    for (uint32_t h = 0; h < 0x10000; h++) {
        if ((h & 0x7c00) == 0x7c00 && (h & 0x3ff))
            continue;
        ASSERT_EQ(fromF32(NumFmt::Fp16, toF32(NumFmt::Fp16, h)), h) << h;
    }
}

TEST(PimNumFmt, Fp32AndArithmetic)
{
    EXPECT_EQ(toF32(NumFmt::Fp32, 0x12345678), 0x12345678u);
    EXPECT_EQ(fromF32(NumFmt::Fp32, 0x12345678), 0x12345678u);
    EXPECT_EQ(f32Add(f2u(1.5f), f2u(2.25f)), f2u(3.75f));
    EXPECT_EQ(f32Mul(f2u(1.5f), f2u(-2.0f)), f2u(-3.0f));
    // RNE: 1 + 2^-24 rounds to 1; 1 + 3 x 2^-24 rounds up.
    EXPECT_EQ(f32Add(f2u(1.0f), f2u(0x1p-24f)), f2u(1.0f));
    EXPECT_EQ(f32Add(f2u(1.0f), f2u(3 * 0x1p-24f)), f2u(1.0f + 0x1p-22f));
    // Invalid operations give SoftFloat's RISC-V default NaN.
    EXPECT_EQ(f32Add(0x7f800000, 0xff800000), 0x7fc00000u);
}

TEST(PimNumFmt, SoftFloatStateIsKept)
{
    const uint_fast8_t mode = softfloat_roundingMode;
    softfloat_roundingMode = softfloat_round_minMag;
    softfloat_exceptionFlags = softfloat_flag_overflow;
    // Under RTZ this would give 1 + 2^-23; the model rounds to nearest
    // even.
    EXPECT_EQ(f32Add(f2u(1.0f), f2u(3 * 0x1p-24f)), f2u(1.0f + 0x1p-22f));
    // An invalid operation and an inexact conversion leave no flag.
    f32Add(0x7f800000, 0xff800000);
    fromF32(NumFmt::Fp16, f2u(1.0f + 0x1p-11f));
    EXPECT_EQ(softfloat_roundingMode, softfloat_round_minMag);
    EXPECT_EQ(softfloat_exceptionFlags, softfloat_flag_overflow);
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
}

TEST(PimNumFmt, Integers)
{
    uint8_t b[4];
    storeInt(NumFmt::Int8, b, -1);
    EXPECT_EQ(b[0], 0xff);
    EXPECT_EQ(loadInt(NumFmt::Int8, b), -1);
    storeInt(NumFmt::Int16, b, 0x18000);
    EXPECT_EQ(loadInt(NumFmt::Int16, b), -0x8000);
    storeInt(NumFmt::Int32, b, -5);
    EXPECT_EQ(loadInt(NumFmt::Int32, b), -5);
    // Element-wise results wrap modulo 2^bits.
    auto x = elems(NumFmt::Int8, {0x7f, 0x80, 0x10});
    auto y = elems(NumFmt::Int8, {0x01, 0xff, 0x10});
    uint8_t r[3];
    for (int i = 0; i < 3; i++)
        ewElement(NumFmt::Int8, false, &x[i], &y[i], &r[i]);
    EXPECT_EQ(r[0], 0x80); // 127 + 1 = -128
    EXPECT_EQ(r[1], 0x7f); // -128 + -1 = 127
    EXPECT_EQ(r[2], 0x20);
    for (int i = 0; i < 3; i++)
        ewElement(NumFmt::Int8, true, &x[i], &y[i], &r[i]);
    EXPECT_EQ(r[0], 0x7f);
    EXPECT_EQ(r[1], 0x80); // -128 x -1 = 128 wraps to -128
    EXPECT_EQ(r[2], 0x00); // 256 wraps to 0
}

TEST(PimNumFmt, FloatElementWise)
{
    // bf16: 1 + 2^-8 (exact in fp32) rounds to 1 in bf16.
    auto a = elems(NumFmt::Bf16, {0x3f80, 0x4040});
    auto b = elems(NumFmt::Bf16, {0x3b80, 0xc000});
    uint8_t r[4];
    ewElement(NumFmt::Bf16, false, &a[0], &b[0], &r[0]);
    ewElement(NumFmt::Bf16, true, &a[2], &b[2], &r[2]);
    EXPECT_EQ(loadBits(NumFmt::Bf16, &r[0]), 0x3f80u);
    EXPECT_EQ(loadBits(NumFmt::Bf16, &r[2]), 0xc0c0u); // 3 x -2 = -6
    // fp16 overflow to infinity: 65504 + 32 = 65536.
    auto c = elems(NumFmt::Fp16, {0x7bff});
    auto d = elems(NumFmt::Fp16, {0x5000});
    ewElement(NumFmt::Fp16, false, c.data(), d.data(), r);
    EXPECT_EQ(loadBits(NumFmt::Fp16, r), 0x7c00u);
    // fp32 is computed directly.
    uint8_t e[4], f[4];
    storeBits(NumFmt::Fp32, e, f2u(0.1f));
    storeBits(NumFmt::Fp32, f, f2u(0.2f));
    ewElement(NumFmt::Fp32, false, e, f, r);
    EXPECT_EQ(loadBits(NumFmt::Fp32, r), f2u(0.1f + 0.2f));
}

TEST(PimNumFmt, MacOrderAndOneUpdatePerColumn)
{
    // Products 2^24, 1, 1, -2^24 in ascending order: 2^24 + 1 rounds to
    // 2^24 (RNE), again for the next 1, and -2^24 then gives 0. A sum of
    // 1 + 1 first would give 2.
    auto x = elems(NumFmt::Fp32,
                   {f2u(0x1p24f), f2u(1.0f), f2u(1.0f), f2u(-0x1p24f)});
    auto y = elems(NumFmt::Fp32, {f2u(1.0f), f2u(1.0f), f2u(1.0f),
                                  f2u(1.0f)});
    uint8_t acc[4];
    storeBits(NumFmt::Fp32, acc, f2u(0.5f));
    macColumn(NumFmt::Fp32, NumFmt::Fp32, x.data(), y.data(), 4, acc);
    // The column's sum (0) is added to the accumulator once.
    EXPECT_EQ(u2f(loadBits(NumFmt::Fp32, acc)), 0.5f);
    // Ascending order from the other side: 1 + 1 = 2, then 2^24 + 2 is
    // exact.
    auto x2 = elems(NumFmt::Fp32,
                    {f2u(1.0f), f2u(1.0f), f2u(0x1p24f), f2u(-0x1p24f)});
    storeBits(NumFmt::Fp32, acc, 0);
    macColumn(NumFmt::Fp32, NumFmt::Fp32, x2.data(), y.data(), 4, acc);
    EXPECT_EQ(u2f(loadBits(NumFmt::Fp32, acc)), 2.0f);
    // The column sum is rounded once when added: acc 2^24 + partial 1.
    storeBits(NumFmt::Fp32, acc, f2u(0x1p24f));
    macColumn(NumFmt::Fp32, NumFmt::Fp32, x2.data(), y.data(), 1, acc);
    EXPECT_EQ(u2f(loadBits(NumFmt::Fp32, acc)), 0x1p24f);
}

TEST(PimNumFmt, MacFormats)
{
    // bf16 operands, fp32 accumulator: products are exact.
    auto x = elems(NumFmt::Bf16, {0x3fc0, 0x4000, 0xbf80}); // 1.5 2 -1
    auto y = elems(NumFmt::Bf16, {0x4000, 0x3f00, 0x4040}); // 2 0.5 3
    uint8_t acc[4];
    storeBits(NumFmt::Fp32, acc, f2u(10.0f));
    macColumn(NumFmt::Bf16, NumFmt::Fp32, x.data(), y.data(), 3, acc);
    EXPECT_EQ(u2f(loadBits(NumFmt::Fp32, acc)), 10.0f + 3 + 1 - 3);
    // A bf16 accumulator rounds the products and the sums in bf16:
    // 1 + 2^-8 + 2^-8: each addition ties to the even 1.
    auto ones = elems(NumFmt::Bf16, {0x3f80, 0x3b80, 0x3b80});
    auto unit = elems(NumFmt::Bf16, {0x3f80, 0x3f80, 0x3f80});
    uint8_t acc16[2];
    storeBits(NumFmt::Bf16, acc16, 0);
    macColumn(NumFmt::Bf16, NumFmt::Bf16, ones.data(), unit.data(), 3,
              acc16);
    EXPECT_EQ(loadBits(NumFmt::Bf16, acc16), 0x3f80u);
    // Integers: int8 x int8 into int32, and into a wrapping int16.
    auto i8 = elems(NumFmt::Int8, {0x80, 0x7f, 0x02});  // -128 127 2
    auto j8 = elems(NumFmt::Int8, {0x80, 0x7f, 0xfe});  // -128 127 -2
    uint8_t a32[4], a16[2];
    storeInt(NumFmt::Int32, a32, 1);
    macColumn(NumFmt::Int8, NumFmt::Int32, i8.data(), j8.data(), 3, a32);
    EXPECT_EQ(loadInt(NumFmt::Int32, a32), 1 + 16384 + 16129 - 4);
    storeInt(NumFmt::Int16, a16, 1000);
    macColumn(NumFmt::Int8, NumFmt::Int16, i8.data(), j8.data(), 3, a16);
    EXPECT_EQ(loadInt(NumFmt::Int16, a16), int16_t(1000 + 16384 + 16129 - 4));
}
