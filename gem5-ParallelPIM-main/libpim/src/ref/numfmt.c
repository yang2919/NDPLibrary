/*
 * Number formats of the reference model, as gem5's PIM model has them
 * (src/mem/ramulator2/pim/pim_numfmt.cc):
 * bfloat16 (bf16), fp16, fp32 and 8-, 16- and 32-bit integers, little
 * endian. Narrowing to a floating format rounds to nearest even (RNE);
 * integers wrap modulo 2^bits.
 */

#include <string.h>

#include "ref/pim_ref.h"

size_t
pim_ref_fmt_bytes(uint32_t fmt)
{
    switch (fmt) {
        case PIM_REF_INT8:
            return 1;
        case PIM_REF_BF16:
        case PIM_REF_FP16:
        case PIM_REF_INT16:
            return 2;
        default:
            return 4;
    }
}

int
pim_ref_fmt_is_float(uint32_t fmt)
{ return fmt <= PIM_REF_FP32; }

static uint32_t
f2u(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

static float
u2f(uint32_t u)
{
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* fp32 -> bf16: the top 16 bits, rounded to nearest even. */
static uint32_t
f32_to_bf16(float f)
{
    const uint32_t u = f2u(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (u >> 16) | 0x0040u; /* quiet NaN */
    }
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

/* fp16 -> fp32, exact. */
static float
f16_to_f32(uint32_t h)
{
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    if (exp == 0x1f) {
        return u2f(sign | 0x7f800000u | (man << 13));
    }
    if (exp == 0) {
        if (man == 0) {
            return u2f(sign);
        }
        /* Subnormal: normalize. */
        exp = 1;
        while (!(man & 0x400u)) {
            man <<= 1;
            exp--;
        }
        man &= 0x3ffu;
    }
    return u2f(sign | ((exp + 112u) << 23) | (man << 13));
}

/* fp32 -> fp16, rounded to nearest even, with subnormals. */
static uint32_t
f32_to_f16(float f)
{
    const uint32_t u = f2u(f);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const uint32_t a = u & 0x7fffffffu;
    if (a > 0x7f800000u) {
        return 0x7e00u; /* the canonical NaN, as SoftFloat's RISC-V */
    }
    if (a >= 0x477ff000u) {
        return sign | 0x7c00u; /* overflow (or infinity) */
    }
    if (a < 0x38800000u) {
        /* Subnormal or zero: shift the significand with rounding. */
        if (a < 0x33000000u) {
            return sign; /* below half the smallest subnormal */
        }
        const uint32_t exp = a >> 23;
        const uint32_t man = (a & 0x7fffffu) | 0x800000u;
        const uint32_t shift = 126u - exp; /* 14 .. 24 */
        const uint32_t half = 1u << (shift - 1);
        const uint32_t rest = man & ((1u << shift) - 1);
        uint32_t r = man >> shift;
        if (rest > half || (rest == half && (r & 1u))) {
            r++;
        }
        return sign | r;
    }
    /* Normal: rebias and round the 13 dropped bits. */
    uint32_t r = ((a - 0x38000000u) >> 13);
    const uint32_t rest = a & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (r & 1u))) {
        r++;
    }
    return sign | r;
}

float
pim_ref_to_float(uint32_t fmt, uint32_t bits)
{
    switch (fmt) {
        case PIM_REF_BF16:
            return u2f(bits << 16);
        case PIM_REF_FP16:
            return f16_to_f32(bits);
        default:
            return u2f(bits);
    }
}

uint32_t
pim_ref_from_float(uint32_t fmt, float value)
{
    switch (fmt) {
        case PIM_REF_BF16:
            return f32_to_bf16(value);
        case PIM_REF_FP16:
            return f32_to_f16(value);
        default:
            return f2u(value);
    }
}

static uint32_t
load_bits(size_t n, const uint8_t *p)
{
    uint32_t v = 0;
    for (size_t i = 0; i < n; i++) {
        v |= (uint32_t)p[i] << (8 * i);
    }
    return v;
}

static void
store_bits(size_t n, uint8_t *p, uint32_t v)
{
    for (size_t i = 0; i < n; i++) {
        p[i] = (uint8_t)(v >> (8 * i));
    }
}

float
pim_ref_load_f(uint32_t fmt, const uint8_t *p)
{ return pim_ref_to_float(fmt, load_bits(pim_ref_fmt_bytes(fmt), p)); }

void
pim_ref_store_f(uint32_t fmt, uint8_t *p, float v)
{ store_bits(pim_ref_fmt_bytes(fmt), p, pim_ref_from_float(fmt, v)); }

int64_t
pim_ref_load_i(uint32_t fmt, const uint8_t *p)
{
    const size_t n = pim_ref_fmt_bytes(fmt);
    const uint32_t v = load_bits(n, p);
    const unsigned bits = (unsigned)(8 * n);
    const uint32_t sign = 1u << (bits - 1);
    if (bits == 32) {
        return (int64_t)(int32_t)v;
    }
    return (int64_t)((v ^ sign)) - (int64_t)sign;
}

void
pim_ref_store_i(uint32_t fmt, uint8_t *p, int64_t v)
{ store_bits(pim_ref_fmt_bytes(fmt), p, (uint32_t)(uint64_t)v); }
