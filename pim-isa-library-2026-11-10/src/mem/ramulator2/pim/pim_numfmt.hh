/*
 * Number formats of the Processing-in-Memory (PIM) functional model
 * (docs/pim/04-memctrl-plan.md section 9.2, refined by its Q5 and Q8).
 *
 * num_fmt is the format of the operands in the banks and the global buffer
 * (GB) and of element-wise (EW) results; acc_fmt is the format of the
 * multiply-accumulate (MAC) accumulator. Elements are little endian.
 *
 * - Floating formats (bf16, fp16, fp32) compute in fp32 with SoftFloat,
 *   rounding to nearest even (RNE), and are then rounded to the result's
 *   format. bf16 widens and narrows by a 16-bit shift (RNE when
 *   narrowing; a NaN keeps its top 16 bits with the quiet bit set); fp16
 *   uses SoftFloat's conversions (RISC-V specialization: a NaN narrowed to
 *   fp16 is the canonical 0x7e00).
 * - Integer formats wrap modulo 2^bits.
 * - MAC: the products of a column's elements are summed in acc_fmt
 *   sequentially in ascending element order, and the column's sum is
 *   added to the accumulator once.
 *
 * These are the semantics of the library's C reference model
 * (util/pim/src/ref) and of the shared vectors in util/pim/tests/vectors.
 * SoftFloat's rounding mode and exception flags are global and the RISC-V
 * floating-point instructions read the flags after each operation, so
 * every function here restores both before it returns.
 */

#ifndef __MEM_RAMULATOR2_PIM_PIM_NUMFMT_HH__
#define __MEM_RAMULATOR2_PIM_PIM_NUMFMT_HH__

#include <cstdint>

namespace gem5
{

namespace pim
{

/** Element formats, in the order of PIM_LAYOUT_NUM_FMT. */
enum class NumFmt : uint8_t
{
    Bf16 = 0,
    Fp16,
    Fp32,
    Int8,
    Int16,
    Int32,
};

constexpr unsigned NumNumFmts = 6;

/** Size of one element in bytes. */
unsigned fmtBytes(NumFmt fmt);
bool fmtIsFloat(NumFmt fmt);
/** "bf16", "fp16", ...; "invalid" for any other value. */
const char *fmtName(NumFmt fmt);
/** Parse a name of fmtName(); false if it names no format. */
bool fmtFromName(const char *name, NumFmt &fmt);

/**
 * Check a num_fmt/acc_fmt pair for a column (burst) of colBytes bytes
 * (04-memctrl-plan.md section 8.5): acc_fmt at least as wide as num_fmt
 * and of the same family, and the size of num_fmt dividing colBytes.
 * Returns nullptr if valid, else the rule that fails.
 */
const char *checkFormats(NumFmt num, NumFmt acc, unsigned colBytes);

/** Elements of num_fmt in one column of colBytes bytes (Q8 (a)). */
unsigned elemsPerColumn(NumFmt fmt, unsigned colBytes);

/** Raw bits of one element (zero-extended) and their store. */
uint32_t loadBits(NumFmt fmt, const uint8_t *p);
void storeBits(NumFmt fmt, uint8_t *p, uint32_t bits);

/** Integer element, sign-extended, and its wrapping store. */
int64_t loadInt(NumFmt fmt, const uint8_t *p);
void storeInt(NumFmt fmt, uint8_t *p, int64_t v);

/** fp32 bits of a floating element's bits (exact). */
uint32_t toF32(NumFmt fmt, uint32_t bits);
/** A floating format's bits of fp32 bits, rounded to nearest even. */
uint32_t fromF32(NumFmt fmt, uint32_t f32);

/** fp32 add and multiply, RNE (SoftFloat). */
uint32_t f32Add(uint32_t a, uint32_t b);
uint32_t f32Mul(uint32_t a, uint32_t b);

/** One element-wise result: *dst = a + b or a * b, all in fmt. */
void ewElement(NumFmt fmt, bool mul, const uint8_t *a, const uint8_t *b,
               uint8_t *dst);

/**
 * One MAC column update: acc += sum over i < n of x[i] * y[i], with x and
 * y n consecutive elements of num and acc one element of accFmt.
 */
void macColumn(NumFmt num, NumFmt accFmt, const uint8_t *x, const uint8_t *y,
               unsigned n, uint8_t *acc);

} // namespace pim
} // namespace gem5

#endif // __MEM_RAMULATOR2_PIM_PIM_NUMFMT_HH__
