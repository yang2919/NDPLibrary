/*
 * pim-eltwise: c = a + b and c = a * b with the processing-in-memory
 * (PIM) library's user API.
 *
 *   pim-eltwise [n_paired n_sb [seed]]      default: 100 50 1
 *
 * For each kind, PIM_VECTOR_PAIRED (n_paired elements, all-bank
 * element-wise commands) and PIM_VECTOR_SB (n_sb elements, single-bank
 * commands): a and b from pim_tensor_create_pair() (a in the even bank 2k,
 * b in the odd bank 2k + 1 of each pair), filled from a fixed seed with
 * finite values k / 1000 in [-1, 1]; then pim_add() and pim_mul() into
 * two result tensors. The program prints every result element's bit
 * pattern in hex, 16 per line, then how many results lie outside bf16
 * tolerance of plain double-precision arithmetic on the unrounded inputs.
 * It exits with 2 on a library error, 1 if that count is not 0, else 0.
 *
 * Built against libpim_<std>.a it runs in gem5; built against
 * libpim_ref_<std>.a it runs on the host reference backend and gives the
 * reference output (tests/gem5/pim/ref/pim-eltwise/).
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
        fprintf(stderr, "pim-eltwise: %s: %s (%s)\n", what, pim_strerror(s),
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
        fprintf(stderr, "pim-eltwise: bad %s '%s'\n", name, s);
        exit(2);
    }
    return (size_t)v;
}

/* The results as element bit patterns, 16 per line. */
static void
print_elems(const char *name, const float *v, size_t n)
{
    static const char hex[] = "0123456789abcdef";
    char line[16 * (2 * 4 + 1) + 64];
    for (size_t i = 0; i < n; i += 16) {
        int len = snprintf(line, sizeof(line), "%s[%zu]", name, i);
        for (size_t k = i; k < i + 16 && k < n; k++) {
            uint8_t e[4] = {0};
            pim_f32_to_elem(e, v[k]);
            line[len++] = ' ';
            for (int b = PIM_LAYOUT_ELEM_BYTES - 1; b >= 0; b--) {
                line[len++] = hex[e[b] >> 4];
                line[len++] = hex[e[b] & 15];
            }
        }
        line[len++] = '\n';
        fwrite(line, 1, (size_t)len, stdout);
    }
}

/* add and mul of one kind; returns the results outside tolerance. */
static size_t
run_kind(pim_kind_t kind, const char *name, size_t n)
{
    const pim_dtype_t dt = (pim_dtype_t)PIM_LAYOUT_NUM_FMT;
    float *av = (float *)malloc(n * sizeof(float));
    float *bv = (float *)malloc(n * sizeof(float));
    float *sum = (float *)malloc(n * sizeof(float));
    float *prod = (float *)malloc(n * sizeof(float));
    if (!av || !bv || !sum || !prod) {
        fprintf(stderr, "pim-eltwise: out of host memory\n");
        exit(2);
    }
    for (size_t i = 0; i < n; i++) {
        av[i] = frand();
        bv[i] = frand();
    }
    pim_tensor_t *a, *b, *cs, *cp;
    check(pim_tensor_create_pair(&a, &b, kind, 1, n, dt),
          "pim_tensor_create_pair");
    check(pim_tensor_create(&cs, kind, 1, n, dt), "pim_tensor_create(sum)");
    check(pim_tensor_create(&cp, kind, 1, n, dt),
          "pim_tensor_create(product)");
    check(pim_copy_to(a, av), "pim_copy_to(a)");
    check(pim_copy_to(b, bv), "pim_copy_to(b)");
    check(pim_add(cs, a, b), "pim_add");
    check(pim_mul(cp, a, b), "pim_mul");
    check(pim_copy_from(cs, sum), "pim_copy_from(sum)");
    check(pim_copy_from(cp, prod), "pim_copy_from(product)");

    char label[64];
    snprintf(label, sizeof(label), "%s.add", name);
    print_elems(label, sum, n);
    snprintf(label, sizeof(label), "%s.mul", name);
    print_elems(label, prod, n);

    size_t outside = 0;
    for (size_t i = 0; i < n; i++) {
        const double s = (double)av[i] + bv[i], p = (double)av[i] * bv[i];
        outside += fabs(sum[i] - s) > 3.0 / 256 * (fabs(av[i]) + fabs(bv[i]));
        outside += fabs(prod[i] - p) > 3.0 / 256 * fabs(p) + 1e-6;
    }
    pim_tensor_destroy(b);
    pim_tensor_destroy(a);
    pim_tensor_destroy(cs);
    pim_tensor_destroy(cp);
    free(av);
    free(bv);
    free(sum);
    free(prod);
    return outside;
}

int
main(int argc, char **argv)
{
    size_t n_paired = 100, n_sb = 50;
    uint64_t seed = 1;
    if (argc != 1 && argc != 3 && argc != 4) {
        fprintf(stderr, "usage: pim-eltwise [n_paired n_sb [seed]]\n");
        return 2;
    }
    if (argc >= 3) {
        n_paired = arg(argv[1], "n_paired");
        n_sb = arg(argv[2], "n_sb");
    }
    if (argc == 4) {
        seed = arg(argv[3], "seed");
    }
    rng = 0x2545f4914f6cdd1dULL ^ seed;

    check(pim_init(NULL), "pim_init");
    FS_MARKER(10);
    printf("pim-eltwise n_paired %zu n_sb %zu seed %llu\n", n_paired, n_sb,
           (unsigned long long)seed);
    size_t outside = run_kind(PIM_VECTOR_PAIRED, "paired", n_paired);
    outside += run_kind(PIM_VECTOR_SB, "sb", n_sb);
    printf("outside bf16 tolerance: %zu\n", outside);
    FS_MARKER(11);
    pim_finalize();
    return outside ? 1 : 0;
}
