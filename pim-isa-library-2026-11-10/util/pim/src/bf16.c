/*
 * Number formats of the library (D9.2): fp32 host data to and from the
 * PIM memory's NUM_FMT, and ACC_FMT accumulators. Round to nearest even
 * when narrowing. Written apart from the reference model's conversions
 * (src/ref/numfmt.c), so the tests can compare the two.
 */

#include <math.h>
#include <string.h>

#include "pim_internal.h"

static uint32_t
bits_of(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

static float
float_of(uint32_t u)
{
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

uint16_t
pim_f32_to_bf16(float f)
{
    const uint32_t u = bits_of(f);
    if (isnan(f)) {
        return (uint16_t)((u >> 16) | 0x40u);
    }
    const uint32_t lsb = (u >> 16) & 1u;
    return (uint16_t)((u + 0x7fffu + lsb) >> 16);
}

float
pim_bf16_to_f32(uint16_t b)
{ return float_of((uint32_t)b << 16); }

/* fp16 through frexp/ldexp: exact for every fp16 value. */
static uint16_t
f32_to_f16(float f)
{
    const uint16_t sign = signbit(f) ? 0x8000u : 0;
    if (isnan(f)) {
        return 0x7e00u; /* the canonical NaN */
    }
    double a = fabs((double)f);
    if (a == 0) {
        return sign;
    }
    /* Round to the fp16 grid: spacing 2^(e - 11) for normal numbers,
     * 2^-24 for subnormal ones. */
    int e;
    frexp(a, &e); /* a = m 2^e, 0.5 <= m < 1 */
    int q = e - 11;
    if (q < -24) {
        q = -24;
    }
    const double scaled = ldexp(a, -q);
    double r = nearbyint(scaled); /* round to nearest even */
    a = ldexp(r, q);
    if (a > 65504.0) {
        return sign | 0x7c00u;
    }
    if (a < 0x1p-14) {
        return sign | (uint16_t)(a / 0x1p-24);
    }
    frexp(a, &e);
    const uint16_t exp = (uint16_t)(e - 1 + 15);
    const uint16_t man = (uint16_t)(ldexp(a, -(e - 1)) * 1024.0 - 1024.0);
    return sign | (uint16_t)(exp << 10) | man;
}

static float
f16_to_f32(uint16_t h)
{
    const int exp = (h >> 10) & 0x1f;
    const int man = h & 0x3ff;
    double v;
    if (exp == 0x1f) {
        v = man ? NAN : INFINITY;
    } else if (exp == 0) {
        v = ldexp(man, -24);
    } else {
        v = ldexp(1024 + man, exp - 25);
    }
    return (float)((h & 0x8000) ? -v : v);
}

static void
put(void *dst, uint32_t v, size_t n)
{
    uint8_t *p = (uint8_t *)dst;
    for (size_t i = 0; i < n; i++) {
        p[i] = (uint8_t)(v >> (8 * i));
    }
}

static uint32_t
get(const void *src, size_t n)
{
    const uint8_t *p = (const uint8_t *)src;
    uint32_t v = 0;
    for (size_t i = 0; i < n; i++) {
        v |= (uint32_t)p[i] << (8 * i);
    }
    return v;
}

static void
to_fmt(int fmt, void *dst, float f)
{
    switch (fmt) {
        case PIM_FMT_BF16:
            put(dst, pim_f32_to_bf16(f), 2);
            break;
        case PIM_FMT_FP16:
            put(dst, f32_to_f16(f), 2);
            break;
        case PIM_FMT_FP32:
            put(dst, bits_of(f), 4);
            break;
        case PIM_FMT_INT8:
            put(dst, (uint32_t)(int64_t)nearbyintf(f), 1);
            break;
        case PIM_FMT_INT16:
            put(dst, (uint32_t)(int64_t)nearbyintf(f), 2);
            break;
        default:
            put(dst, (uint32_t)(int64_t)nearbyintf(f), 4);
            break;
    }
}

static float
from_fmt(int fmt, const void *src)
{
    switch (fmt) {
        case PIM_FMT_BF16:
            return pim_bf16_to_f32((uint16_t)get(src, 2));
        case PIM_FMT_FP16:
            return f16_to_f32((uint16_t)get(src, 2));
        case PIM_FMT_FP32:
            return float_of(get(src, 4));
        case PIM_FMT_INT8:
            return (float)(int8_t)get(src, 1);
        case PIM_FMT_INT16:
            return (float)(int16_t)get(src, 2);
        default:
            return (float)(int32_t)get(src, 4);
    }
}

void
pim_f32_to_elem(void *dst, float f)
{ to_fmt(PIM_LAYOUT_NUM_FMT, dst, f); }

float
pim_elem_to_f32(const void *src)
{ return from_fmt(PIM_LAYOUT_NUM_FMT, src); }

float
pim_acc_to_f32(const void *src)
{ return from_fmt(PIM_LAYOUT_ACC_FMT, src); }
