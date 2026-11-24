/*
 * Copyright (c) 2026 Jongmin Kim
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Strip-mined RVV vector add, c[i] = a[i] + b[i] on int32 elements.
 *
 *   size_t rvv_vadd_mL(size_t n, const int32_t *a, const int32_t *b,
 *                      int32_t *c);
 *
 * One function per LMUL (L = 1, 2, 4, 8). Each returns the number of
 * strips (loop iterations). Every strip executes exactly one vsetvli,
 * two vle32.v, one vadd.vv and one vse32.v, so the vector instruction
 * mix is known in advance. This file is the only vector code in the
 * rvv-vadd binary; the C driver is compiled without the V extension.
 */

    .option arch, +v
    .text

.macro RVV_VADD lmul
    .globl  rvv_vadd_m\lmul
    .type   rvv_vadd_m\lmul, @function
rvv_vadd_m\lmul:
    li      t2, 0
    beqz    a0, 2f
1:
    vsetvli t0, a0, e32, m\lmul, ta, ma
    vle32.v v8, (a1)
    vle32.v v16, (a2)
    vadd.vv v24, v8, v16
    vse32.v v24, (a3)
    slli    t1, t0, 2
    add     a1, a1, t1
    add     a2, a2, t1
    add     a3, a3, t1
    sub     a0, a0, t0
    addi    t2, t2, 1
    bnez    a0, 1b
2:
    mv      a0, t2
    ret
    .size   rvv_vadd_m\lmul, . - rvv_vadd_m\lmul
.endm

    RVV_VADD 1
    RVV_VADD 2
    RVV_VADD 4
    RVV_VADD 8

    .section .note.GNU-stack, "", @progbits
