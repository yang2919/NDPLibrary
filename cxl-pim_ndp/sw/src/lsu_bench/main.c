// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>

#include "../common/platform.h"
#include "../common/uart_apb.h"
#define RHEX_NIB 8
#include "../common/kio.h"
#include "../common/verify_block.h"

#define RES       VB_RES
#define RES_MAGIC 0x15B00001u

#define STRIDE_W   8u
#define L1_NODES   64u
#define DRAM_NODES 16384u
#define L1_LOADS   131072u
#define DRAM_LOADS DRAM_NODES
#define INDEP_ITER 16384u
#define CRC_BUF    2048u
#define CRC_PASSES 32u
#define ALU_ITER   16384u

static uint64_t RING_L1[L1_NODES * STRIDE_W]     __attribute__((aligned(64)));
static uint64_t RING_DRAM[DRAM_NODES * STRIDE_W] __attribute__((aligned(64)));
static uint64_t BLK[8]                           __attribute__((aligned(64)));
static uint64_t SBLK[8]                          __attribute__((aligned(64)));
static uint8_t  BUF[CRC_BUF]                     __attribute__((aligned(64)));
static uint32_t TAB[256]                         __attribute__((aligned(64)));

static uint32_t lcg_state = 0x2545F491u;
static uint32_t lcg(void) { lcg_state = lcg_state * 1664525u + 1013904223u; return lcg_state; }

static void build_ring(uint64_t *ring, uint32_t nodes) {
    static uint32_t order[DRAM_NODES];
    for (uint32_t i = 0; i < nodes; i++) order[i] = i;
    for (uint32_t i = nodes - 1; i > 0; i--) {
        uint32_t j = lcg() % i;
        uint32_t t = order[i]; order[i] = order[j]; order[j] = t;
    }
    for (uint32_t i = 0; i < nodes; i++) {
        uint32_t from = order[i], to = order[(i + 1) % nodes];
        ring[from * STRIDE_W] = (uint64_t)(uintptr_t)&ring[to * STRIDE_W];
    }
}

static __attribute__((noinline)) uint64_t chase(uint64_t p, uint64_t iters8) {
    asm volatile(
        "1:\n"
        "  ld %[p], 0(%[p])\n  ld %[p], 0(%[p])\n  ld %[p], 0(%[p])\n  ld %[p], 0(%[p])\n"
        "  ld %[p], 0(%[p])\n  ld %[p], 0(%[p])\n  ld %[p], 0(%[p])\n  ld %[p], 0(%[p])\n"
        "  addi %[n], %[n], -1\n"
        "  bnez %[n], 1b\n"
        : [p] "+r"(p), [n] "+r"(iters8) :: "memory");
    return p;
}

static __attribute__((noinline)) uint64_t ld_indep(const uint64_t *b, uint64_t iters) {
    uint64_t s;
    asm volatile(
        "1:\n"
        "  ld t0, 0(%[b])\n  ld t1, 8(%[b])\n  ld t2, 16(%[b])\n  ld t3, 24(%[b])\n"
        "  ld t4, 32(%[b])\n  ld t5, 40(%[b])\n  ld t6, 48(%[b])\n  ld a7, 56(%[b])\n"
        "  addi %[n], %[n], -1\n"
        "  bnez %[n], 1b\n"
        "  add %[s], t0, t1\n  add %[s], %[s], t2\n  add %[s], %[s], t3\n"
        "  add %[s], %[s], t4\n  add %[s], %[s], t5\n  add %[s], %[s], t6\n"
        "  add %[s], %[s], a7\n"
        : [s] "=&r"(s), [n] "+r"(iters)
        : [b] "r"(b)
        : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a7", "memory");
    return s;
}

static __attribute__((noinline)) void st_indep(uint64_t *b, uint64_t v, uint64_t iters) {
    asm volatile(
        "1:\n"
        "  sd %[v], 0(%[b])\n  sd %[v], 8(%[b])\n  sd %[v], 16(%[b])\n  sd %[v], 24(%[b])\n"
        "  sd %[v], 32(%[b])\n  sd %[v], 40(%[b])\n  sd %[v], 48(%[b])\n  sd %[v], 56(%[b])\n"
        "  addi %[n], %[n], -1\n"
        "  bnez %[n], 1b\n"
        : [n] "+r"(iters)
        : [b] "r"(b), [v] "r"(v)
        : "memory");
}

static __attribute__((noinline)) uint64_t alu_ctrl(uint64_t x, uint64_t iters) {
    asm volatile(
        "1:\n"
        "  add %[x], %[x], %[x]\n  add %[x], %[x], %[x]\n  add %[x], %[x], %[x]\n"
        "  add %[x], %[x], %[x]\n  add %[x], %[x], %[x]\n  add %[x], %[x], %[x]\n"
        "  add %[x], %[x], %[x]\n  add %[x], %[x], %[x]\n"
        "  addi %[n], %[n], -1\n"
        "  bnez %[n], 1b\n"
        : [x] "+r"(x), [n] "+r"(iters));
    return x;
}

static __attribute__((noinline)) uint32_t crc32_tab(const uint8_t *p, size_t n, uint32_t crc) {
    for (size_t i = 0; i < n; i++) crc = TAB[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    return crc;
}

static uint32_t crc32_bits(const uint8_t *p, size_t n, uint32_t crc) {
    for (size_t i = 0; i < n; i++) {
        crc ^= p[i];
        for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return crc;
}

typedef uint64_t (*timed_fn)(void);

static uint64_t g_p;
static uint64_t g_sum;
static uint32_t g_crc;
static uint64_t g_alu;

static uint64_t run_chase_l1(void)   { g_p = chase((uint64_t)(uintptr_t)RING_L1, L1_LOADS / 8u); return 0; }
static uint64_t run_chase_dram(void) { g_p = chase((uint64_t)(uintptr_t)RING_DRAM, DRAM_LOADS / 8u); return 0; }
static uint64_t run_ld_indep(void)   { g_sum = ld_indep(BLK, INDEP_ITER); return 0; }
static uint64_t run_st_indep(void)   { st_indep(SBLK, 0x5A5A5A5A5A5A5A5Aull, INDEP_ITER); return 0; }
static uint64_t run_crc(void) {
    uint32_t c = 0xFFFFFFFFu;
    for (uint32_t k = 0; k < CRC_PASSES; k++) c = crc32_tab(BUF, CRC_BUF, c);
    g_crc = c;
    return 0;
}
static uint64_t run_alu(void)        { g_alu = alu_ctrl(1u, ALU_ITER); return 0; }

static uint32_t timed(timed_fn f) {
    f();
    uint64_t t0 = rdcycle();
    f();
    uint64_t t1 = rdcycle();
    return (uint32_t)(t1 - t0);
}

int main(void) {
    VB_HB[0] = 0xCAFE0001;
    uart_init(ARA_UART_DIVISOR);
    VB_HB[0] = 0xCAFE0002;
    uart_puts("\n[lsu_bench] scalar load/store path\n");

    RES[0] = RES_MAGIC;
    RES[1] = 1;
    for (uint32_t k = 2; k < 32u; k++) RES[k] = 0xFFFFFFFFu;
    vb_arm();

    VB_HB[0] = 0xCAFE0003;
    build_ring(RING_L1, L1_NODES);
    build_ring(RING_DRAM, DRAM_NODES);
    for (uint32_t i = 0; i < 8; i++) BLK[i] = (uint64_t)(i + 1) * 0x0101010101010101ull;
    for (uint32_t i = 0; i < CRC_BUF; i++) BUF[i] = (uint8_t)lcg();
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
        TAB[i] = c;
    }

    VB_HB[0] = 0xCAFE0004;
    static const timed_fn fns[6] = {run_chase_l1, run_chase_dram, run_ld_indep,
                                    run_st_indep, run_crc, run_alu};
    static const uint32_t ops[6] = {L1_LOADS, DRAM_LOADS, INDEP_ITER * 8u,
                                    INDEP_ITER * 8u, CRC_BUF * CRC_PASSES, ALU_ITER * 8u};
    static const char *names[6] = {"chase_l1  ", "chase_dram", "ld_indep  ",
                                   "st_indep  ", "crc32     ", "alu_ctrl  "};
    uint32_t cyc[6], mask = 0;
    uint64_t total = 0;

    for (int k = 0; k < 6; k++) {
        cyc[k] = timed(fns[k]);
        total += cyc[k];
        uint32_t ok;
        switch (k) {
        case 0: ok = (g_p == (uint64_t)(uintptr_t)RING_L1); break;
        case 1: ok = (g_p == (uint64_t)(uintptr_t)RING_DRAM); break;
        case 2: ok = (g_sum == 36u * 0x0101010101010101ull); break;
        case 3: {
            ok = 1;
            for (int i = 0; i < 8; i++) ok &= (SBLK[i] == 0x5A5A5A5A5A5A5A5Aull);
            break;
        }
        case 4: {
            uint32_t ref = 0xFFFFFFFFu;
            for (uint32_t p = 0; p < CRC_PASSES; p++) ref = crc32_bits(BUF, CRC_BUF, ref);
            ok = (g_crc == ref);
            break;
        }
        default: ok = (g_alu == 0);
        }
        mask |= ok << k;
        RES[2 + k] = cyc[k];
        RES[8 + k] = ops[k];
        uart_puts(names[k]); uart_puts(" cycles="); print_u64(cyc[k]);
        uart_puts(" per-op x100="); print_u64((uint64_t)cyc[k] * 100u / ops[k]);
        uart_puts(ok ? " ok\n" : " CHECK FAILED\n");
    }
    RES[14] = mask;

    uint32_t pass = (mask == 0x3Fu);
    uart_puts(pass ? "[lsu_bench] PASS\n" : "[lsu_bench] FAIL\n");
    vb_publish(pass, (uint32_t)total, cyc[0], cyc[1], cyc[2], cyc[3], cyc[4]);
    VB_HB[0] = 0xCAFE000F;
    while (1) {}
}
