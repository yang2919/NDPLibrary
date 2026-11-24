# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0
# fp32 links sp-fmatmul + Cephes exp; fp16 links hp-fmatmul (asm) + inline exp.
KERNEL := gqa_attention
ARA := $(abspath ..)
MATMUL_DIR := $(ARA)/ara_apps/dtype-matmul/kernel
EXP_DIR    := $(ARA)/ara_apps/exp/kernel
EXTRA_CFLAGS  := -I$(MATMUL_DIR) -I$(EXP_DIR) -I$(ARA)/ara_apps/common
EXTRA_OBJS_SP := build/sp-fmatmul.o build/exp.o build/util.o
EXTRA_OBJS_HP := build/hp-fmatmul.o build/util.o
VT := 1
include ../common/kernel.mk
build/sp-fmatmul.o: $(MATMUL_DIR)/sp-fmatmul.c | build
	$(CC) $(CFLAGS) -c -o $@ $<
build/hp-fmatmul.o: $(MATMUL_DIR)/hp-fmatmul.c | build
	$(CC) $(CFLAGS) -c -o $@ $<
build/exp.o: $(EXP_DIR)/exp.c | build
	$(CC) $(CFLAGS) -c -o $@ $<
build/util.o: $(ARA)/ara_apps/common/util.c | build
	$(CC) $(CFLAGS) -c -o $@ $<
