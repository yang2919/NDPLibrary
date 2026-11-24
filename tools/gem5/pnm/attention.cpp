// PNM entry for the dense primitives used by the L3 causal attention composition.
#include "ndp/l1/gem5_pim/dense_kernel.hpp"
#include <stddef.h>

using namespace ndp::gem5_pim;
struct Arguments {
  uint32_t version, bytes;
  DenseShape shape;
  uint32_t reserved;
  const _Float16 *left, *right, *bias;
  _Float16 *out;
};
static_assert(sizeof(Arguments) == 64 && offsetof(Arguments, left) == 32);

// Softmax subtracts the row maximum, so exp takes only nonpositive values.
static float negative_exp(float x) {
  if (x < -104) return 0;
  float t = x / 256;
  float e = 1 + t*(1 + t*(0.5f + t*(1.f/6 + t*(1.f/24 + t*(1.f/120 + t/720)))));
  for (int i = 0; i < 8; ++i) e *= e;
  return e;
}

extern "C"
#ifdef __riscv
__attribute__((section(".text.entry")))
#endif
int pnm_attention(const Arguments *a, size_t size) {
  if (!a || size != sizeof(*a) || a->bytes != size || a->version != 1 || a->reserved ||
      !a->left || !a->out || !a->shape.m || !a->shape.n) return 1;
  auto op = a->shape.op;
  if (op < DenseOp::Matmul || op > DenseOp::Add) return 1;
  bool dot = op == DenseOp::Matmul || op == DenseOp::Linear;
  bool binary = dot || op == DenseOp::RMSNorm || op == DenseOp::Add;
  if ((op == DenseOp::RMSNorm && a->shape.factor <= 0) || !__builtin_isfinite(a->shape.factor) || binary != bool(a->right) || dot != bool(a->shape.k) || (a->bias && op != DenseOp::Linear))
    return 1;
  dense_compute(a->shape, a->left, a->right, a->bias, a->out, negative_exp);
  return 0;
}
