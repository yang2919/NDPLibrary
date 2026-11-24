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
 * Driver for the RVV vector add kernel in vadd_kernel.S.
 *
 * Usage: rvv-vadd [N [LMUL]]   (defaults: N = 1000, LMUL = 1)
 *
 * Runs c = a + b once on N int32 elements with the given LMUL, checks
 * the result and the strip count in scalar code, and exits with 0 on
 * success. This file must be compiled without the V extension so that
 * the kernel is the only vector code in the program.
 *
 * The kernel call is the region of interest (ROI), marked with
 * m5_work_begin() and m5_work_end(). These are gem5 pseudo instructions:
 * the binary only runs in gem5, not on hardware or QEMU.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <gem5/m5ops.h>

#define MAX_N 65536

size_t rvv_vadd_m1(size_t, const int32_t *, const int32_t *, int32_t *);
size_t rvv_vadd_m2(size_t, const int32_t *, const int32_t *, int32_t *);
size_t rvv_vadd_m4(size_t, const int32_t *, const int32_t *, int32_t *);
size_t rvv_vadd_m8(size_t, const int32_t *, const int32_t *, int32_t *);

static int32_t a[MAX_N], b[MAX_N], c[MAX_N];

int
main(int argc, char *argv[])
{
    size_t n = argc > 1 ? strtoul(argv[1], NULL, 0) : 1000;
    int lmul = argc > 2 ? atoi(argv[2]) : 1;
    size_t (*kernel)(size_t, const int32_t *, const int32_t *, int32_t *);
    unsigned long vlenb;

    switch (lmul) {
      case 1: kernel = rvv_vadd_m1; break;
      case 2: kernel = rvv_vadd_m2; break;
      case 4: kernel = rvv_vadd_m4; break;
      case 8: kernel = rvv_vadd_m8; break;
      default:
        fprintf(stderr, "rvv-vadd: LMUL must be 1, 2, 4 or 8\n");
        return 2;
    }
    if (n > MAX_N) {
        fprintf(stderr, "rvv-vadd: N must be at most %d\n", MAX_N);
        return 2;
    }

    for (size_t i = 0; i < n; i++) {
        a[i] = (int32_t)i;
        b[i] = (int32_t)(3 * i + 1);
        c[i] = -1;
    }

    m5_work_begin(0, 0);
    size_t strips = kernel(n, a, b, c);
    m5_work_end(0, 0);

    /* vlenb (CSR 0xc22) is VLEN / 8. */
    asm volatile ("csrr %0, 0xc22" : "=r"(vlenb));
    size_t vlmax = lmul * vlenb / sizeof(int32_t);
    size_t expected_strips = (n + vlmax - 1) / vlmax;

    size_t errors = 0;
    for (size_t i = 0; i < n; i++) {
        if (c[i] != a[i] + b[i])
            errors++;
    }

    printf("rvv-vadd: n=%zu lmul=%d vlen=%lu vlmax=%zu strips=%zu "
           "errors=%zu\n", n, lmul, vlenb * 8, vlmax, strips, errors);

    if (errors != 0 || strips != expected_strips) {
        printf("rvv-vadd: fail (expected %zu strips)\n", expected_strips);
        return 1;
    }
    printf("rvv-vadd: pass\n");
    return 0;
}
