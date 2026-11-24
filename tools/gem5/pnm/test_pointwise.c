// Check the freestanding N4 entry against libm for every binary16 input pattern.
#include "pointwise.c"
#include <math.h>
#include <string.h>
#include <stdio.h>

int main(void)
{
    _Float16 x, y, right = -0.75f;
    struct pointwise_args args = {1, 40, 1, 1, &x, NULL, &y};
    unsigned max_ulp = 0;
    for (unsigned i = 0; i <= UINT16_MAX; ++i) {
        uint16_t bits = (uint16_t)i, expected_bits, actual_bits;
        memcpy(&x, &bits, 2);
        args.operation = 1;
        args.right = NULL;
        if (pnm_pointwise(&args, sizeof(args))) return 1;
        _Float16 expected = (double)x / (1 + exp(-(double)x));
        if (isnan(expected)) {
            if (!isnan(y)) return 1;
        } else {
            memcpy(&expected_bits, &expected, 2);
            memcpy(&actual_bits, &y, 2);
            unsigned diff = expected_bits > actual_bits ? expected_bits - actual_bits : actual_bits - expected_bits;
            if (diff > max_ulp) max_ulp = diff;
            if (diff > 1) return 1;
        }
        args.operation = 2;
        args.right = &right;
        if (pnm_pointwise(&args, sizeof(args))) return 1;
        expected = (float)x * (float)right;
        if (isnan(expected) ? !isnan(y) : memcmp(&expected, &y, 2)) return 1;
    }
    printf("65536 FP16 patterns passed, maximum SiLU distance %u ULP\n", max_ulp);
    return 0;
}
