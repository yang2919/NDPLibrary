// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#ifndef PERF_H
#define PERF_H

#include <stdint.h>

#define PERF_BASE   0xD0010000UL
#define PERF_MAGIC  0x5045524600000001ULL

enum {
    PERF_TAP_SPM      = 0,
    PERF_TAP_DRAM_ARA = 1,
    PERF_TAP_DRAM_ALL = 2,
    PERF_NUM_TAPS     = 3
};

static inline uint64_t perf_rd64(unsigned long off) {
    return *(volatile uint64_t *)(PERF_BASE + off);
}

static inline unsigned long perf_tap_base(int tap) { return 0x40UL * (tap + 1); }

static inline uint64_t perf_magic(void)      { return perf_rd64(0x00); }

static inline uint64_t perf_vinsn_active(void)   { return perf_rd64(0x08); }

static inline uint64_t perf_rd_lat_sum  (int t) { return perf_rd64(perf_tap_base(t) + 0x00); }
static inline uint64_t perf_rd_count    (int t) { return perf_rd64(perf_tap_base(t) + 0x08); }
static inline uint64_t perf_rd_outst_max(int t) { return perf_rd64(perf_tap_base(t) + 0x10); }
static inline uint64_t perf_wr_lat_sum  (int t) { return perf_rd64(perf_tap_base(t) + 0x18); }
static inline uint64_t perf_wr_count    (int t) { return perf_rd64(perf_tap_base(t) + 0x20); }
static inline uint64_t perf_wr_outst_max(int t) { return perf_rd64(perf_tap_base(t) + 0x28); }

static inline uint64_t perf_rd_mean_lat_milli(int t) {
    uint64_t n = perf_rd_count(t);
    return n ? (perf_rd_lat_sum(t) * 1000u) / n : 0u;
}

static inline uint64_t perf_wr_mean_lat_milli(int t) {
    uint64_t n = perf_wr_count(t);
    return n ? (perf_wr_lat_sum(t) * 1000u) / n : 0u;
}

typedef struct {
    uint64_t rd_lat_sum, rd_count, wr_lat_sum, wr_count, vinsn_active;
} perf_snap_t;

static inline void perf_snap(perf_snap_t *s, int t) {
    s->rd_lat_sum = perf_rd_lat_sum(t);
    s->rd_count   = perf_rd_count(t);
    s->wr_lat_sum = perf_wr_lat_sum(t);
    s->wr_count   = perf_wr_count(t);
    s->vinsn_active   = perf_vinsn_active();
}

static inline uint64_t perf_delta_rd_mean_lat_milli(const perf_snap_t *a,
                                                    const perf_snap_t *b) {
    uint64_t dn = b->rd_count - a->rd_count;
    return dn ? ((b->rd_lat_sum - a->rd_lat_sum) * 1000u) / dn : 0u;
}

#endif
