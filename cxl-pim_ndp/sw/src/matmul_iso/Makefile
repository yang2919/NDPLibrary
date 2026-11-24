# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0
# Board isolation of the 8x8/m2 vs 16x16/m1 sp_fmatmul path. fp32 links sp-fmatmul,
# fp16 links hp-fmatmul (both from the ara dtype-matmul kernel; no exp needed).
KERNEL := matmul_iso
ARA := $(abspath ..)
MATMUL_DIR := $(ARA)/ara_apps/dtype-matmul/kernel
EXTRA_CFLAGS  := -I$(MATMUL_DIR) -I$(ARA)/ara_apps/common
EXTRA_OBJS_SP := build/sp-fmatmul.o build/util.o
EXTRA_OBJS_HP := build/hp-fmatmul.o build/util.o
include ../common/kernel.mk
build/sp-fmatmul.o: $(MATMUL_DIR)/sp-fmatmul.c | build
	$(CC) $(CFLAGS) -c -o $@ $<
build/hp-fmatmul.o: $(MATMUL_DIR)/hp-fmatmul.c | build
	$(CC) $(CFLAGS) -c -o $@ $<
build/util.o: $(ARA)/ara_apps/common/util.c | build
	$(CC) $(CFLAGS) -c -o $@ $<
