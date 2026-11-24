/*
 * eltwise-bench: c = a + b and c = a * b on the PIM memory against the same
 * element-wise operations on the CPU.
 *
 *   eltwise-bench [n]        default: 262144
 *
 * Regions (one stats.txt block each, workload/microbench/common/bench.h):
 *   1 cpu-scalar-add  fp32 scalar loop in the OS memory
 *   2 cpu-rvv-add     fp32 RVV loop (LMUL 8)
 *   3 pim-add         pim_add on paired-bank vectors (bf16): all-bank
 *                     element-wise add, then register-to-bank moves, then
 *                     the completion barrier
 *   4 pim-mul         pim_mul, the same with multiply
 * Completion barrier: with posted PIM requests (riscv-pim.py's default),
 * pim_add's requests are acknowledged when the PIM memory accepts them, so
 * pim_add can return while channels still execute them. Each unit (rank,
 * channel or pseudo channel) executes its requests in order and pim.rd.reg
 * is not posted, so one register read per unit returns only after that
 * unit's work is done: the region ends when the last channel finishes.
 * The CPU regions run first on purpose: on the O3 CPU, an RVV loop after
 * pim_add currently crashes gem5 (register renaming; open bug).
 * Placement and read-back are outside the regions. Every result is checked
 * against double precision; exit status 1 if any is outside tolerance.
 */

#include <math.h>
#include <riscv_vector.h>

#include "bench.h"

__attribute__((noinline, optimize("no-tree-vectorize"))) static void
add_scalar(float *c, const float *a, const float *b, size_t n)
{
    for (size_t i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

__attribute__((noinline)) static void
add_rvv(float *c, const float *a, const float *b, size_t n)
{
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
        __riscv_vse32_v_f32m8(c + i, __riscv_vfadd_vv_f32m8(va, vb, vl), vl);
        i += vl;
    }
}

/* Wait until every unit of every channel has executed its PIM requests. */
static void
pim_barrier(void)
{
    uint8_t acc[PIM_LAYOUT_ACC_BYTES];
    for (uint32_t ch = 0; ch < PIM_LAYOUT_CHANNELS; ch++) {
        for (uint32_t u = 0; u < PIM_LAYOUT_UNITS; u++) {
            const pim_coord_t c = {ch, u, 0, 0, 0, 0, 0};
            pim_rd_reg_mem(pim_acc_addr(&c), acc, sizeof acc);
        }
    }
    pim_fence();
}

int
main(int argc, char **argv)
{
    size_t n = 262144;
    if (argc == 2) {
        n = mb_arg(argv[1]);
    } else if (argc != 1) {
        fprintf(stderr, "usage: eltwise-bench [n]\n");
        return 2;
    }
    float *a = mb_alloc(n * sizeof(float)), *b = mb_alloc(n * sizeof(float));
    float *cs = mb_alloc(n * sizeof(float)), *cv = mb_alloc(n * sizeof(float));
    float *pa = mb_alloc(n * sizeof(float)), *pm = mb_alloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        a[i] = mb_frand();
        b[i] = mb_frand();
    }

    mb_check(pim_init(NULL), "pim_init");
    printf("eltwise-bench n %zu\n", n);
    const pim_dtype_t dt = (pim_dtype_t)PIM_LAYOUT_NUM_FMT;
    pim_tensor_t *A, *B, *S, *P;
    mb_check(pim_tensor_create_pair(&A, &B, PIM_VECTOR_PAIRED, 1, n, dt),
             "create a, b");
    mb_check(pim_tensor_create(&S, PIM_VECTOR_PAIRED, 1, n, dt), "create s");
    mb_check(pim_tensor_create(&P, PIM_VECTOR_PAIRED, 1, n, dt), "create p");
    mb_check(pim_copy_to(A, a), "copy a");
    mb_check(pim_copy_to(B, b), "copy b");

    MB_REGION("cpu-scalar-add") { add_scalar(cs, a, b, n); }
    MB_REGION("cpu-rvv-add") { add_rvv(cv, a, b, n); }
    MB_REGION("pim-add")
    {
        mb_check(pim_add(S, A, B), "pim_add");
        pim_barrier();
    }
    MB_REGION("pim-mul")
    {
        mb_check(pim_mul(P, A, B), "pim_mul");
        pim_barrier();
    }

    mb_check(pim_copy_from(S, pa), "copy sum");
    mb_check(pim_copy_from(P, pm), "copy product");
    size_t bad_pim = 0, bad_cpu = 0;
    for (size_t i = 0; i < n; i++) {
        const double s = (double)a[i] + b[i], p = (double)a[i] * b[i];
        bad_pim += fabs(pa[i] - s) > 3.0 / 256 * (fabs(a[i]) + fabs(b[i]));
        bad_pim += fabs(pm[i] - p) > 3.0 / 256 * fabs(p) + 1e-6;
        bad_cpu += fabs(cs[i] - s) > 1e-6 || fabs(cv[i] - s) > 1e-6;
    }
    printf("pim results outside bf16 tolerance: %zu\n", bad_pim);
    printf("cpu results wrong: %zu\n", bad_cpu);
    pim_tensor_destroy(B);
    pim_tensor_destroy(A);
    pim_tensor_destroy(S);
    pim_tensor_destroy(P);
    pim_finalize();
    return bad_pim || bad_cpu ? 1 : 0;
}
