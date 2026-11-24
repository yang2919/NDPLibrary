// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#ifndef ARA_VHK158_VTRANS_ASM_H
#define ARA_VHK158_VTRANS_ASM_H

static inline __attribute__((always_inline)) void vt_exp_v1v2(void) {
    asm volatile(".word 0x4E2390AB");
}

static inline __attribute__((always_inline)) void vt_rcp_v1v2(void) {
    asm volatile(".word 0x4E2310AB");
}

static inline __attribute__((always_inline)) void vt_exp_v3v4(void) {
    asm volatile(".word 0x4E4391AB");
}

static inline __attribute__((always_inline)) void vt_rcp_v3v4(void) {
    asm volatile(".word 0x4E4311AB");
}

#endif
