// Freestanding N4 dense FP16 kernels. The caller enables FP and round-to-nearest-even.
#include <stdint.h>
#include <stddef.h>

struct pointwise_args {
    uint32_t version, bytes, count, operation;
    const _Float16 *left, *right;
    _Float16 *output;
};
_Static_assert(sizeof(struct pointwise_args) == 40 && offsetof(struct pointwise_args, left) == 16,
               "requires the RV64 launch ABI");

static float silu(float x)
{
    if (x > 24.0f) return x;
    if (x < -24.0f) return __builtin_isinf(x) ? __builtin_nanf("") : -0.0f;
    // exp(-abs(x)): reduce to [-0.375, 0], then undo six halvings.
    float t = -__builtin_fabsf(x) / 64.0f;
    float e = 1 + t * (1 + t * (0.5f + t * (1.0f/6 + t * (1.0f/24 +
              t * (1.0f/120 + t * (1.0f/720 + t/5040))))));
    for (unsigned i = 0; i < 6; ++i) e *= e;
    return (x < 0 ? x * e : x) / (1 + e);
}

#ifdef __riscv
__attribute__((section(".text.entry")))
#endif
int pnm_pointwise(const struct pointwise_args *a, size_t size)
{
    if (size != sizeof(*a) || a->version != 1 || a->bytes != size || !a->count ||
        !a->left || !a->output || (a->operation != 1 && a->operation != 2) ||
        ((a->operation == 2) != (a->right != NULL))) return 1;
    for (size_t i = 0; i < a->count; ++i) {
        float x = a->left[i];
        a->output[i] = a->operation == 1 ? silu(x) : x * (float)a->right[i];
    }
    return 0;
}
