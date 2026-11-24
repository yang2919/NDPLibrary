# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0
KERNEL := lmul_test
# Force the scalar reference loops to stay SCALAR (fp64 on the scalar FPU) so a
# vector-unit bug is caught, not masked by GCC auto-vectorizing the golden too.
EXTRA_CFLAGS := -fno-tree-vectorize -fno-tree-slp-vectorize
include ../common/kernel.mk
