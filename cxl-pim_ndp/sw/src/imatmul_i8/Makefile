# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0
#
# int8 matmul. Deliberately NOT built on common/kernel.mk: that file is
# fp-parameterized (one main.c -> _sp and _hp via -DHP) and int8 is a third axis
# that does not belong in it. sw/roofline/Makefile is the precedent for a kernel
# carrying its own ARCH and target set.
#
# Links ara's own bp-imatmul kernel, the same way sw/matmul_iso links
# sp-fmatmul -- an -I into the ara tree plus one extra object.

CROSS_COMPILE ?= riscv64-unknown-elf-
CC      := $(CROSS_COMPILE)gcc
OBJCOPY := $(CROSS_COMPILE)objcopy
OBJDUMP := $(CROSS_COMPILE)objdump

# No zfh/zvfh: this kernel touches no floating point at all.
ARCH := rv64gcv
ABI  := lp64d

ARA        := $(abspath ..)
KERNEL_DIR := $(ARA)/ara_apps/dtype-matmul/kernel

# -fno-tree-vectorize for the same reason as common/kernel.mk: GCC 16 -O3
# auto-vectorizes the scalar reference loops and that code miscomputes at
# vl=128. Here it would silently corrupt the golden this kernel is checked
# against, which is the one thing that must stay scalar.
# EXTRA_CFLAGS is the documented knob for soak depth:
#   make EXTRA_CFLAGS=-DSOAK_ITER=200000
CFLAGS  := -march=$(ARCH) -mabi=$(ABI) -mcmodel=medany -mno-relax \
           -O3 -g -ffreestanding -fno-builtin -fno-tree-vectorize -fno-tree-slp-vectorize \
           -I$(KERNEL_DIR) -I$(ARA)/ara_apps/common \
           -Wall -Wextra $(EXTRA_CFLAGS)
LDFLAGS := -march=$(ARCH) -mabi=$(ABI) -mcmodel=medany -mno-relax \
           -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,../common/link.ld

COMMON_HDRS := $(wildcard ../common/*.h)
BUILD := build

.PHONY: all clean
all: $(BUILD)/imatmul_i8.bin

$(BUILD): ; @mkdir -p $@

$(BUILD)/crt0.o: ../common/crt0.S | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<
$(BUILD)/main.o: main.c $(COMMON_HDRS) $(KERNEL_DIR)/bp-imatmul.h | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<
$(BUILD)/bp-imatmul.o: $(KERNEL_DIR)/bp-imatmul.c | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/imatmul_i8.elf: $(BUILD)/main.o $(BUILD)/bp-imatmul.o $(BUILD)/crt0.o ../common/link.ld
	$(CC) $(LDFLAGS) -o $@ $(BUILD)/main.o $(BUILD)/bp-imatmul.o $(BUILD)/crt0.o

$(BUILD)/%.bin: $(BUILD)/%.elf
	$(OBJCOPY) -O binary $< $@
	@echo "==== $@: $$(stat -c %s $@) bytes ($$(($$(stat -c %s $@)/4)) words)"
	@$(OBJDUMP) -d -S $< > $(@:.bin=.dump)

clean: ; rm -rf $(BUILD)
