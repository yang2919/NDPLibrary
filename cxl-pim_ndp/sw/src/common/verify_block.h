// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#ifndef ARA_VHK158_VERIFY_BLOCK_H
#define ARA_VHK158_VERIFY_BLOCK_H

#include <stdint.h>

#define VB_HB_ADDR      0x40100000UL
#define VB_VERIFY_ADDR  0x40100100UL
#define VB_RES_ADDR     0x40100200UL

#define VB_MAGIC        0xA1A1B1B1u
#define VB_MAGIC_UNSET  0xDEADBEEFu

#define VB_HB_MAIN      0xCAFE0001u
#define VB_HB_UART      0xCAFE0002u
#define VB_HB_INPUTS    0xCAFE0003u
#define VB_HB_KERNEL    0xCAFE0004u
#define VB_HB_REF       0xCAFE0005u
#define VB_HB_DONE      0xCAFE000Fu

static volatile uint32_t * const VB_HB  = (volatile uint32_t *)VB_HB_ADDR;
static volatile uint32_t * const VB_VER = (volatile uint32_t *)VB_VERIFY_ADDR;
static volatile uint32_t * const VB_RES = (volatile uint32_t *)VB_RES_ADDR;

static inline void vb_hb(uint32_t v) { VB_HB[0] = v; }

static inline void vb_arm(void) {
    for (uint32_t k = 1; k < 8u; k++) VB_VER[k] = 0xFFFFFFFFu;
}

static inline void vb_publish(uint32_t pass, uint32_t cycles,
                              uint32_t d3, uint32_t d4, uint32_t d5,
                              uint32_t d6, uint32_t d7) {
    VB_VER[1] = pass;
    VB_VER[2] = cycles;
    VB_VER[3] = d3; VB_VER[4] = d4; VB_VER[5] = d5;
    VB_VER[6] = d6; VB_VER[7] = d7;
    asm volatile("fence" ::: "memory");
    VB_VER[0] = VB_MAGIC;
}

#endif
