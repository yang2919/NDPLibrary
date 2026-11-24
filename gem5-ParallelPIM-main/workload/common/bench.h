/*
 * Shared helpers of the PIM microbenchmarks (workload/microbench).
 *
 * A measured region is bracketed by gem5's m5_reset_stats and
 * m5_dump_stats, so stats.txt holds one statistics block per region, in
 * order, then the end-of-run block. Each region also prints
 * "region <n> <name>" so that collect.py can name the blocks.
 *
 * Note: gem5-side statistics (simTicks, simInsts, pim.reqs) reset per
 * region; the Ramulator 2.1 statistics bridged into stats.txt (channelK.*)
 * are cumulative, so collect.py takes differences between blocks.
 */

#ifndef MICROBENCH_BENCH_H
#define MICROBENCH_BENCH_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <gem5/pim/pim.h>

/* m5ops in the RISC-V encoding (util/m5/src/abi/riscv/m5op.S):
 * opcode 0x7b, function in bits 31:25; reset_stats 0x40, dump_stats 0x41,
 * hypercall 0x71. On the host (the reference build) they do nothing. */
#if defined(__riscv)
static inline void
mb_reset_stats(void)
{
    register uint64_t a0 asm("a0") = 0, a1 asm("a1") = 0;
    asm volatile(".long 0x8000007b" : : "r"(a0), "r"(a1) : "memory");
}

static inline void
mb_dump_stats(void)
{
    register uint64_t a0 asm("a0") = 0, a1 asm("a1") = 0;
    asm volatile(".long 0x8200007b" : : "r"(a0), "r"(a1) : "memory");
}

/* A gem5 hypercall: the configuration's ExitHandler for number n runs
 * (workload/llm/configs/llm_se.py: 4 switches to the detailed CPUs). */
static inline void
mb_hypercall(uint64_t n)
{
    register uint64_t a0 asm("a0") = n;
    asm volatile(".long 0xe200007b" : "+r"(a0) : : "memory");
}
#else
static inline void mb_reset_stats(void) {}
static inline void mb_dump_stats(void) {}
static inline void mb_hypercall(uint64_t n) { (void)n; }
#endif

static int mb_region_count;

/* MB_REGION("name") { code } : one measured region. */
#define MB_REGION(name)                                                    \
    for (int mb_once_ = (printf("region %d %s\n", ++mb_region_count,      \
                                (name)),                                   \
                         fflush(stdout), mb_reset_stats(), 1);            \
         mb_once_; mb_once_ = 0, mb_dump_stats())

static uint64_t mb_rng = 0x2545f4914f6cdd1dULL;

/* A finite value k / 1000, k in [-1000, 1000]. */
static inline float
mb_frand(void)
{
    mb_rng ^= mb_rng << 13;
    mb_rng ^= mb_rng >> 7;
    mb_rng ^= mb_rng << 17;
    return (float)((int)((mb_rng >> 11) % 2001) - 1000) / 1000.0f;
}

static inline void
mb_check(pim_status_t s, const char *what)
{
    if (s != PIM_OK) {
        fprintf(stderr, "%s: %s (%s)\n", what, pim_strerror(s),
                pim_error_detail());
        exit(2);
    }
}

static inline size_t
mb_arg(const char *s)
{
    char *end;
    unsigned long long v = strtoull(s, &end, 0);
    if (*end || !v) {
        fprintf(stderr, "bad argument '%s'\n", s);
        exit(2);
    }
    return (size_t)v;
}

static inline void *
mb_alloc(size_t bytes)
{
    void *p = malloc(bytes);
    if (!p) {
        fprintf(stderr, "out of host memory\n");
        exit(2);
    }
    return p;
}

#endif /* MICROBENCH_BENCH_H */
