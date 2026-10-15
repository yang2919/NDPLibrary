// Declare explicit conversion helpers between IEEE binary16 storage and float32 values.
#pragma once
#include <cstdint>

namespace ndp {
// IEEE binary16 conversion: round-to-nearest, ties-to-even, gradual underflow.
// NaNs are quieted and canonicalized; signed zero and infinity are preserved.
uint16_t float_to_fp16(float value);
float fp16_to_float(uint16_t bits);
} // namespace ndp
