/*
 * An element-wise processing-in-memory (PIM) output reused after the
 * host read it.
 *
 *   pim-reuse [elements]
 *
 * For the SB and the paired kind: c = a + b, read c, c = a * b into the
 * same tensor, read c again. The second command writes rows whose lines
 * the first read left cached, so the library must flush them first
 * (gem5 counts a PIM write that meets a valid copy in pimValidBlk).
 * Prints "PIM-REUSE <kind>: <wrong> of <n> wrong" per kind and exits with
 * 1 if any result is wrong, 2 on a library error.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gem5/pim/pim.h>

static float
q(float v)
{
    /* bf16 operands: round to nearest even, as the library converts. */
    uint32_t u;
    memcpy(&u, &v, 4);
    u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000u;
    memcpy(&v, &u, 4);
    return v;
}

static int
run(pim_kind_t kind, const char *name, size_t n)
{
    pim_tensor_t *a, *b, *c;
    float *va = malloc(n * 4), *vb = malloc(n * 4), *vc = malloc(n * 4);
    if (pim_tensor_create_pair(&a, &b, kind, 1, n, PIM_BF16) != PIM_OK ||
        pim_tensor_create(&c, kind, 1, n, PIM_BF16) != PIM_OK) {
        fprintf(stderr, "%s: %s\n", name, pim_error_detail());
        exit(2);
    }
    for (size_t i = 0; i < n; i++) {
        va[i] = (float)(i % 97) / 8.0f;
        vb[i] = (float)(i % 31) / 4.0f + 1.0f;
    }
    long wrong = 0;
    if (pim_copy_to(a, va) || pim_copy_to(b, vb) || pim_add(c, a, b) ||
        pim_copy_from(c, vc)) {
        fprintf(stderr, "%s: %s\n", name, pim_error_detail());
        exit(2);
    }
    for (size_t i = 0; i < n; i++) {
        wrong += vc[i] != q(q(va[i]) + q(vb[i]));
    }
    if (pim_mul(c, a, b) || pim_copy_from(c, vc)) {
        fprintf(stderr, "%s: %s\n", name, pim_error_detail());
        exit(2);
    }
    for (size_t i = 0; i < n; i++) {
        wrong += vc[i] != q(q(va[i]) * q(vb[i]));
    }
    printf("PIM-REUSE %s: %ld of %zu wrong\n", name, wrong, 2 * n);
    pim_tensor_destroy(a);
    pim_tensor_destroy(b);
    pim_tensor_destroy(c);
    free(va);
    free(vb);
    free(vc);
    return wrong != 0;
}

int
main(int argc, char **argv)
{
    const size_t n = argc > 1 ? strtoul(argv[1], NULL, 0) : 4096;
    if (pim_init(NULL) != PIM_OK) {
        fprintf(stderr, "pim_init: %s\n", pim_error_detail());
        return 2;
    }
    int bad = run(PIM_VECTOR_SB, "sb", n);
    bad |= run(PIM_VECTOR_PAIRED, "paired", n);
    pim_finalize();
    return bad;
}
