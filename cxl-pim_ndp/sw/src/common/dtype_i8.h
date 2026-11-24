// Copyright 2026 Jae
// SPDX-License-Identifier: Apache-2.0

#ifndef ARA_VHK158_DTYPE_I8_H
#define ARA_VHK158_DTYPE_I8_H

#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

static inline int8_t i8_dotp(const int8_t *a, const int8_t *b, size_t n) {
    size_t vl0 = __riscv_vsetvl_e8m1(n);

    if (n < 8 * vl0) {
        vint8m1_t acc = __riscv_vmv_v_x_i8m1(0, vl0);
        for (size_t j = 0; j < n; ) {
            size_t vl = __riscv_vsetvl_e8m1(n - j);
            acc = __riscv_vmacc_vv_i8m1_tu(acc, __riscv_vle8_v_i8m1(a + j, vl),
                                                __riscv_vle8_v_i8m1(b + j, vl), vl);
            j += vl;
        }
        return __riscv_vmv_x_s_i8m1_i8(
            __riscv_vredsum_vs_i8m1_i8m1(acc, __riscv_vmv_v_x_i8m1(0, vl0), vl0));
    }
    vint8m1_t a0 = __riscv_vmv_v_x_i8m1(0, vl0), a1 = a0, a2 = a0, a3 = a0;
    size_t i = 0;
    for (; i + 4 * vl0 <= n; i += 4 * vl0) {
        a0 = __riscv_vmacc_vv_i8m1_tu(a0, __riscv_vle8_v_i8m1(a + i,           vl0), __riscv_vle8_v_i8m1(b + i,           vl0), vl0);
        a1 = __riscv_vmacc_vv_i8m1_tu(a1, __riscv_vle8_v_i8m1(a + i + vl0,     vl0), __riscv_vle8_v_i8m1(b + i + vl0,     vl0), vl0);
        a2 = __riscv_vmacc_vv_i8m1_tu(a2, __riscv_vle8_v_i8m1(a + i + 2 * vl0, vl0), __riscv_vle8_v_i8m1(b + i + 2 * vl0, vl0), vl0);
        a3 = __riscv_vmacc_vv_i8m1_tu(a3, __riscv_vle8_v_i8m1(a + i + 3 * vl0, vl0), __riscv_vle8_v_i8m1(b + i + 3 * vl0, vl0), vl0);
    }
    for (; i < n; ) {
        size_t vl = __riscv_vsetvl_e8m1(n - i);
        a0 = __riscv_vmacc_vv_i8m1_tu(a0, __riscv_vle8_v_i8m1(a + i, vl), __riscv_vle8_v_i8m1(b + i, vl), vl);
        i += vl;
    }
    a0 = __riscv_vadd_vv_i8m1(a0, a1, vl0); a2 = __riscv_vadd_vv_i8m1(a2, a3, vl0);
    a0 = __riscv_vadd_vv_i8m1(a0, a2, vl0);
    return __riscv_vmv_x_s_i8m1_i8(
        __riscv_vredsum_vs_i8m1_i8m1(a0, __riscv_vmv_v_x_i8m1(0, vl0), vl0));
}

static inline int8_t i8_redsum(const int8_t *v, size_t n) {
    size_t vl0 = __riscv_vsetvl_e8m1(n);

    if (n < 8 * vl0) {
        vint8m1_t acc = __riscv_vle8_v_i8m1(v, vl0);
        for (size_t j = vl0; j < n; ) {
            size_t vl = __riscv_vsetvl_e8m1(n - j);
            acc = __riscv_vadd_vv_i8m1_tu(acc, acc, __riscv_vle8_v_i8m1(v + j, vl), vl);
            j += vl;
        }
        return __riscv_vmv_x_s_i8m1_i8(
            __riscv_vredsum_vs_i8m1_i8m1(acc, __riscv_vmv_v_x_i8m1(0, vl0), vl0));
    }
    vint8m1_t a0 = __riscv_vmv_v_x_i8m1(0, vl0), a1 = a0, a2 = a0, a3 = a0;
    size_t i = 0;
    for (; i + 4 * vl0 <= n; i += 4 * vl0) {
        a0 = __riscv_vadd_vv_i8m1_tu(a0, a0, __riscv_vle8_v_i8m1(v + i,           vl0), vl0);
        a1 = __riscv_vadd_vv_i8m1_tu(a1, a1, __riscv_vle8_v_i8m1(v + i + vl0,     vl0), vl0);
        a2 = __riscv_vadd_vv_i8m1_tu(a2, a2, __riscv_vle8_v_i8m1(v + i + 2 * vl0, vl0), vl0);
        a3 = __riscv_vadd_vv_i8m1_tu(a3, a3, __riscv_vle8_v_i8m1(v + i + 3 * vl0, vl0), vl0);
    }
    for (; i < n; ) {
        size_t vl = __riscv_vsetvl_e8m1(n - i);
        a0 = __riscv_vadd_vv_i8m1_tu(a0, a0, __riscv_vle8_v_i8m1(v + i, vl), vl);
        i += vl;
    }
    a0 = __riscv_vadd_vv_i8m1(a0, a1, vl0); a2 = __riscv_vadd_vv_i8m1(a2, a3, vl0);
    a0 = __riscv_vadd_vv_i8m1(a0, a2, vl0);
    return __riscv_vmv_x_s_i8m1_i8(
        __riscv_vredsum_vs_i8m1_i8m1(a0, __riscv_vmv_v_x_i8m1(0, vl0), vl0));
}

#endif
