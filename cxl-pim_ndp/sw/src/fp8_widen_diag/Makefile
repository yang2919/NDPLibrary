# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0
#
# fp8 bring-up diagnostic. Not built on common/kernel.mk for the same reason as
# imatmul_i8 and bf16_diag: that file is fp-parameterized over {fp32, fp16} via
# -DHP, and fp8 is a third axis with no C type, no intrinsic and no mnemonic --
# the whole kernel is inline asm issued at e8.
#
# No zvfbf* here: this kernel touches no bf16. Plain rv64gcv_zfh_zvfh is enough
# because every fp8 operation is an ordinary vfadd/vfmul/vfmacc encoding whose
# format is decided by vtype, not by the march string.

CROSS_COMPILE ?= riscv64-unknown-elf-
CC      := $(CROSS_COMPILE)gcc
OBJCOPY := $(CROSS_COMPILE)objcopy
OBJDUMP := $(CROSS_COMPILE)objdump

ARCH := rv64gcv_zfh_zvfh
ABI  := lp64d

# -fno-tree-vectorize for the same reason as common/kernel.mk: GCC 16 -O3
# auto-vectorizes the scalar golden loop and that code miscomputes at vl=128 --
# here it would corrupt the reference this kernel is checked against.
CFLAGS  := -march=$(ARCH) -mabi=$(ABI) -mcmodel=medany -mno-relax \
           -O3 -g -ffreestanding -fno-builtin -fno-tree-vectorize -fno-tree-slp-vectorize \
           -Wall -Wextra $(EXTRA_CFLAGS)
LDFLAGS := -march=$(ARCH) -mabi=$(ABI) -mcmodel=medany -mno-relax \
           -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,../common/link.ld

COMMON_HDRS := $(wildcard ../common/*.h)
BUILD := build

.PHONY: all clean
all: $(BUILD)/fp8_widen_diag.bin

$(BUILD): ; @mkdir -p $@

$(BUILD)/crt0.o: ../common/crt0.S | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<
$(BUILD)/main.o: main.c $(COMMON_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/fp8_widen_diag.elf: $(BUILD)/main.o $(BUILD)/crt0.o ../common/link.ld
	$(CC) $(LDFLAGS) -o $@ $(BUILD)/main.o $(BUILD)/crt0.o

$(BUILD)/%.bin: $(BUILD)/%.elf
	$(OBJCOPY) -O binary $< $@
	@echo "==== $@: $$(stat -c %s $@) bytes ($$(($$(stat -c %s $@)/4)) words)"
	@$(OBJDUMP) -d -S $< > $(@:.bin=.dump)

clean: ; rm -rf $(BUILD)
