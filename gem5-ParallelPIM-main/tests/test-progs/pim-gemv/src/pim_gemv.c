/*
 * pim-gemv: y = A x with the processing-in-memory (PIM) library's user
 * API.
 *
 *   pim-gemv [rows cols [seed]]      default: 37 100 1
 *
 * A (rows x cols, a PIM_MATRIX_AB tensor) and x (cols, a PIM_VECTOR_GB
 * tensor) are filled from a fixed seed with finite values k / 1000 in
 * [-1, 1]; y (rows, a PIM_VECTOR_SB tensor) gets the result in the
 * accumulator format. The program prints every element of y as its bit
 * pattern in hex, then how many elements lie outside bf16 tolerance of a
 * plain double-precision loop over the unrounded inputs. It exits with 2
 * on a library error, 1 if that count is not 0, else 0.
 *
 * Built against libpim_<std>.a it runs in gem5; built against
 * libpim_ref_<std>.a it runs on the host reference backend and gives the
 * reference output (tests/gem5/pim/ref/pim-gemv/).
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gem5/pim/pim.h>

#ifdef PIM_FS_MARKERS
#include <termios.h>
#include <unistd.h>

/*
 * FS builds only (make FS_MARKERS=1): gem5 hypercalls as m5 operations
 * (util/m5's RISC-V encoding 0x7b | func << 25, M5OP_HYPERCALL 0x71, the
 * number in a0). riscv-pim.py --switch-at marker switches to the measured CPU
 * and resets the statistics at 10, after pim_init() has mapped the PIM memory
 * on the fast CPU, and dumps them and ends the simulation at 11, after the
 * output has left the console and before the unmapping. SE builds have no
 * marker, so their binaries are unchanged.
 */
static void
fs_marker(unsigned long n)
{
    /* The console must have sent the output before the run ends. */
    fflush(stdout);
    tcdrain(STDOUT_FILENO);
    /* After the call, which may clobber a0. */
    register unsigned long a0 __asm__("a0") = n;
    __asm__ volatile(".long 0xe200007b" : "+r"(a0) : : "memory");
}
#define FS_MARKER(n) fs_marker(n)
#else
#define FS_MARKER(n) ((void)0)
#endif

static uint64_t rng;

/* A finite value k / 1000, k in [-1000, 1000]. */
static float
frand(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (float)((int)((rng >> 11) % 2001) - 1000) / 1000.0f;
}

static void
check(pim_status_t s, const char *what)
{
    if (s != PIM_OK) {
        fprintf(stderr, "pim-gemv: %s: %s (%s)\n", what, pim_strerror(s),
                pim_error_detail());
        exit(2);
    }
}

static size_t
arg(const char *s, const char *name)
{
    char *end;
    const unsigned long long v = strtoull(s, &end, 0);
    if (*end || v == 0) {
        fprintf(stderr, "pim-gemv: bad %s '%s'\n", name, s);
        exit(2);
    }
    return (size_t)v;
}

int
main(int argc, char **argv)
{
    size_t rows = 37, cols = 100;
    uint64_t seed = 1;
    if (argc != 1 && argc != 3 && argc != 4) {
        fprintf(stderr, "usage: pim-gemv [rows cols [seed]]\n");
        return 2;
    }
    if (argc >= 3) {
        rows = arg(argv[1], "rows");
        cols = arg(argv[2], "cols");
    }
    if (argc == 4) {
        seed = arg(argv[3], "seed");
    }
    rng = 0x9e3779b97f4a7c15ULL ^ seed;

    float *a = (float *)malloc(rows * cols * sizeof(float));
    float *x = (float *)malloc(cols * sizeof(float));
    float *y = (float *)malloc(rows * sizeof(float));
    if (!a || !x || !y) {
        fprintf(stderr, "pim-gemv: out of host memory\n");
        return 2;
    }
    for (size_t i = 0; i < rows * cols; i++) {
        a[i] = frand();
    }
    for (size_t j = 0; j < cols; j++) {
        x[j] = frand();
    }

    pim_tensor_t *A, *X, *Y;
    check(pim_init(NULL), "pim_init");
    FS_MARKER(10);
    check(pim_tensor_create(&A, PIM_MATRIX_AB, rows, cols,
                            (pim_dtype_t)PIM_LAYOUT_NUM_FMT),
          "pim_tensor_create(A)");
    check(pim_tensor_create(&X, PIM_VECTOR_GB, 1, cols,
                            (pim_dtype_t)PIM_LAYOUT_NUM_FMT),
          "pim_tensor_create(x)");
    check(pim_tensor_create(&Y, PIM_VECTOR_SB, 1, rows,
                            (pim_dtype_t)PIM_LAYOUT_ACC_FMT),
          "pim_tensor_create(y)");
    check(pim_copy_to(A, a), "pim_copy_to(A)");
    check(pim_copy_to(X, x), "pim_copy_to(x)");
    check(pim_gemv(Y, A, X), "pim_gemv");
    check(pim_copy_from(Y, y), "pim_copy_from(y)");

    printf("pim-gemv rows %zu cols %zu seed %llu\n", rows, cols,
           (unsigned long long)seed);
    for (size_t m = 0; m < rows; m += 8) {
        printf("y[%zu]", m);
        for (size_t k = m; k < m + 8 && k < rows; k++) {
            uint32_t bits;
            memcpy(&bits, &y[k], sizeof(bits));
            printf(" %08x", (unsigned)bits);
        }
        printf("\n");
    }
    /* bf16 tolerance of the plain loop over the unrounded inputs. */
    size_t outside = 0;
    for (size_t m = 0; m < rows; m++) {
        double ref = 0, mag = 0;
        for (size_t j = 0; j < cols; j++) {
            ref += (double)a[m * cols + j] * x[j];
            mag += fabs((double)a[m * cols + j] * x[j]);
        }
        outside += fabs(y[m] - ref) > mag / 128 + 1e-6;
    }
    printf("outside bf16 tolerance: %zu\n", outside);
    FS_MARKER(11);

    pim_tensor_destroy(A);
    pim_tensor_destroy(X);
    pim_tensor_destroy(Y);
    pim_finalize();
    free(a);
    free(x);
    free(y);
    return outside ? 1 : 0;
}
