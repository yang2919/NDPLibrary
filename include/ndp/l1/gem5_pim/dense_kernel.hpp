// FP16 attention arithmetic shared by the reference worker and the freestanding PNM image.
#pragma once
#include <stdint.h>

namespace ndp::gem5_pim {
enum class DenseOp : uint32_t { Matmul = 1, Linear, Transpose, Softmax, RMSNorm, Add };
struct DenseShape {
  DenseOp op;
  uint32_t m, n, k;
  float factor;
};

// Each operation stores its result once, after FP32 accumulation and scaling.
template<class T, class Exp>
void dense_compute(DenseShape s, const T *left, const T *right, const T *bias, T *out, Exp exp) {
  for (uint64_t i = 0; i < s.m; ++i) {
    if (s.op == DenseOp::RMSNorm) {
      float sum = 0;
      for (uint32_t j = 0; j < s.n; ++j) {
        float x = left[i*s.n+j];
        sum = __builtin_fmaf(x, x, sum);
      }
      float inv = 1.0f / __builtin_sqrtf(sum / s.n + s.factor);
      for (uint32_t j = 0; j < s.n; ++j)
        out[i*s.n+j] = float(left[i*s.n+j]) * inv * float(right[j]);
      continue;
    }
    if (s.op == DenseOp::Add) {
      for (uint32_t j = 0; j < s.n; ++j)
        out[i*s.n+j] = float(left[i*s.n+j]) + float(right[i*s.n+j]);
      continue;
    }
    if (s.op == DenseOp::Softmax) {
      float maximum = -__builtin_inff(), sum = 0;
      bool nan = false;
      uint32_t infinities = 0;
      for (uint32_t j = 0; j < s.n; ++j) {
        float x = left[i*s.n+j];
        if (x > maximum) maximum = x;
        nan |= __builtin_isnan(x);
        infinities += x == __builtin_inff();
      }
      if (!nan && !infinities && maximum != -__builtin_inff())
        for (uint32_t j = 0; j < s.n; ++j) sum += exp(float(left[i*s.n+j]) - maximum);
      for (uint32_t j = 0; j < s.n; ++j) {
        float x = left[i*s.n+j];
        out[i*s.n+j] = nan ? __builtin_nanf("") : infinities ? float(x == maximum)/infinities :
                       maximum == -__builtin_inff() ? 0 : exp(x - maximum)/sum;
      }
      continue;
    }
    for (uint64_t j = 0; j < s.n; ++j) {
      if (s.op == DenseOp::Transpose) {
        out[j*s.m+i] = left[i*s.n+j];
        continue;
      }
      float sum = 0;
      for (uint64_t k = 0; k < s.k; ++k) {
        auto rhs = s.op == DenseOp::Linear ? j*s.k+k : k*s.n+j;
        sum = __builtin_fmaf(float(left[i*s.k+k]), float(right[rhs]), sum);
      }
      out[i*s.n+j] = sum*s.factor + (bias ? float(bias[j]) : 0.0f);
    }
  }
}
}
