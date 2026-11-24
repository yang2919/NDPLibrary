# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0
# fp32 links the ara Cephes exp kernel (exp.o); fp16 uses the inline exp_2xf16
# from exp_hp.h, so only the _sp build needs exp.o.
KERNEL := exp
ARA := $(abspath ..)
EXTRA_OBJS_SP := build/exp.o
VT := 1
include ../common/kernel.mk
build/exp.o: $(ARA)/ara_apps/exp/kernel/exp.c | build
	$(CC) $(CFLAGS) -c -o $@ $<
