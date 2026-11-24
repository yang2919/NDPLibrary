/*
 * gemv-bench: y = A x on the PIM memory against the same GEMV on the CPU.
 *
 *   gemv-bench [M N]        default: 2048 1024
 *
 * Regions (one stats.txt block each, workload/microbench/common/bench.h):
 *   1 pim       pim_gemv: A (M x N, bf16) in the PIM banks, x in the global
 *               buffers, fp32 accumulators read back into y
 *   2 cpu-scalar  fp32 scalar loop over A in the OS memory
 *   3 cpu-rvv     fp32 RVV loop (LMUL 1, vfmacc + vfredusum)
 * Placing A and x in the PIM memory (pim_copy_to) and reading y back
 * (pim_copy_from) are outside the regions. Every result is checked against
 * a double-precision loop; exit status 1 if any is outside tolerance.
 */

#include <math.h>
#include <riscv_vector.h>

#include "bench.h"

__attribute__((noinline, optimize("no-tree-vectorize"))) static void
gemv_scalar(float *y, const float *a, const float *x, size_t m, size_t n)
{
    for (size_t i = 0; i < m; i++) {
        float s = 0.0f;
        const float *r = a + i * n;
        for (size_t j = 0; j < n; j++)
            s += r[j] * x[j];
        y[i] = s;
    }
}

/*
 * LMUL 1: on gem5's O3 CPU, accumulating into an LMUL 8 register group
 * (tail undisturbed) can give wrong sums; LMUL 1 is exact (2026-11-12).
 */
__attribute__((noinline)) static void
gemv_rvv(float *y, const float *a, const float *x, size_t m, size_t n)
{
    const size_t vlmax = __riscv_vsetvlmax_e32m1();
    for (size_t i = 0; i < m; i++) {
        const float *r = a + i * n;
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
        for (size_t j = 0; j < n;) {
            size_t vl = __riscv_vsetvl_e32m1(n - j);
            vfloat32m1_t va = __riscv_vle32_v_f32m1(r + j, vl);
            vfloat32m1_t vx = __riscv_vle32_v_f32m1(x + j, vl);
            acc = __riscv_vfmacc_vv_f32m1_tu(acc, va, vx, vl);
            j += vl;
        }
        vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m1_t s = __riscv_vfredusum_vs_f32m1_f32m1(acc, z, vlmax);
        y[i] = __riscv_vfmv_f_s_f32m1_f32(s);
    }
}

int
main(int argc, char **argv)
{
    size_t m = 2048, n = 1024;
    if (argc == 3) {
        m = mb_arg(argv[1]);
        n = mb_arg(argv[2]);
    } else if (argc != 1) {
        fprintf(stderr, "usage: gemv-bench [M N]\n");
        return 2;
    }
    float *a = mb_alloc(m * n * sizeof(float));
    float *x = mb_alloc(n * sizeof(float));
    float *yp = mb_alloc(m * sizeof(float));
    float *ys = mb_alloc(m * sizeof(float));
    float *yv = mb_alloc(m * sizeof(float));
    for (size_t i = 0; i < m * n; i++)
        a[i] = mb_frand();
    for (size_t j = 0; j < n; j++)
        x[j] = mb_frand();

    mb_check(pim_init(NULL), "pim_init");
    printf("gemv-bench M %zu N %zu\n", m, n);
    pim_tensor_t *A, *X, *Y;
    mb_check(pim_tensor_create(&A, PIM_MATRIX_AB, m, n,
                               (pim_dtype_t)PIM_LAYOUT_NUM_FMT), "create A");
    mb_check(pim_tensor_create(&X, PIM_VECTOR_GB, 1, n,
                               (pim_dtype_t)PIM_LAYOUT_NUM_FMT), "create x");
    mb_check(pim_tensor_create(&Y, PIM_VECTOR_SB, 1, m,
                               (pim_dtype_t)PIM_LAYOUT_ACC_FMT), "create y");
    mb_check(pim_copy_to(A, a), "copy A");
    mb_check(pim_copy_to(X, x), "copy x");

    MB_REGION("pim") { mb_check(pim_gemv(Y, A, X), "pim_gemv"); }
    MB_REGION("cpu-scalar") { gemv_scalar(ys, a, x, m, n); }
    MB_REGION("cpu-rvv") { gemv_rvv(yv, a, x, m, n); }

    mb_check(pim_copy_from(Y, yp), "copy y");
    size_t bad_pim = 0, bad_cpu = 0;
    for (size_t i = 0; i < m; i++) {
        double ref = 0, mag = 0;
        for (size_t j = 0; j < n; j++) {
            ref += (double)a[i * n + j] * x[j];
            mag += fabs((double)a[i * n + j] * x[j]);
        }
        bad_pim += fabs(yp[i] - ref) > mag / 128 + 1e-6; /* bf16 inputs */
        bad_cpu += fabs(ys[i] - ref) > 1e-3 * mag + 1e-6;
        bad_cpu += fabs(yv[i] - ref) > 1e-3 * mag + 1e-6;
    }
    printf("pim results outside bf16 tolerance: %zu\n", bad_pim);
    printf("cpu results wrong: %zu\n", bad_cpu);
    pim_tensor_destroy(A);
    pim_tensor_destroy(X);
    pim_tensor_destroy(Y);
    pim_finalize();
    return bad_pim || bad_cpu ? 1 : 0;
}
