# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0
#
# bf16 bring-up diagnostic. Deliberately NOT built on common/kernel.mk, for the
# same reason imatmul_i8 is not: that file is fp-parameterized over {fp32, fp16}
# via -DHP and dtype.h, and bf16 does not fit that axis. dtype.h's HP branch defines
# ~35 macros (VFADD_VV, VFMUL_VV, VFREDUSUM, VFMAX_VV, ...) and bf16 has none of
# them -- the ratified extensions give it a widening FMA and a convert and no
# arithmetic. A -DBF16 branch could only be written by faking those macros with a
# widen-op-narrow sequence, which is a datapath this hardware does not have as
# one instruction and would report as if it did.
#
# ARCH carries _zvfbfwma for __riscv_vfwmaccbf16_*. Note it does NOT act as a
# guard: GCC accepts the Zvfbfmin convert intrinsics and emits scalar
# fcvt.bf16.s from a (__bf16) cast even without _zvfbfmin spelled out, and Ara
# decodes neither. main.c keeps clear of both by hand.

CROSS_COMPILE ?= riscv64-unknown-elf-
CC      := $(CROSS_COMPILE)gcc
OBJCOPY := $(CROSS_COMPILE)objcopy
OBJDUMP := $(CROSS_COMPILE)objdump

ARCH := rv64gcv_zfh_zvfh_zvfbfwma
ABI  := lp64d

# -fno-tree-vectorize for the same reason as common/kernel.mk: GCC 16 -O3
# auto-vectorizes the scalar golden loop and that code miscomputes at vl=128.
CFLAGS  := -march=$(ARCH) -mabi=$(ABI) -mcmodel=medany -mno-relax \
           -O3 -g -ffreestanding -fno-builtin -fno-tree-vectorize -fno-tree-slp-vectorize \
           -Wall -Wextra $(EXTRA_CFLAGS)
LDFLAGS := -march=$(ARCH) -mabi=$(ABI) -mcmodel=medany -mno-relax \
           -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,../common/link.ld

COMMON_HDRS := $(wildcard ../common/*.h)
BUILD := build

.PHONY: all clean check-insn
all: $(BUILD)/bf16_diag.bin

$(BUILD): ; @mkdir -p $@

$(BUILD)/crt0.o: ../common/crt0.S | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<
$(BUILD)/main.o: main.c $(COMMON_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/bf16_diag.elf: $(BUILD)/main.o $(BUILD)/crt0.o ../common/link.ld
	$(CC) $(LDFLAGS) -o $@ $(BUILD)/main.o $(BUILD)/crt0.o

$(BUILD)/%.bin: $(BUILD)/%.elf
	$(OBJCOPY) -O binary $< $@
	@echo "==== $@: $$(stat -c %s $@) bytes ($$(($$(stat -c %s $@)/4)) words)"
	@$(OBJDUMP) -d -S $< > $(@:.bin=.dump)
	@$(MAKE) --no-print-directory check-insn

# Fails the build if an instruction Ara cannot decode reached the binary. Cheaper
# than finding it as an illegal-instruction trap on the board.
check-insn: $(BUILD)/bf16_diag.dump
	@if grep -qE "fcvt\.(bf16\.s|s\.bf16)|vfwcvtbf16|vfncvtbf16" $(BUILD)/bf16_diag.dump; then \
	  echo "  UNDECODABLE INSTRUCTION in bf16_diag.dump:"; \
	  grep -nE "fcvt\.(bf16\.s|s\.bf16)|vfwcvtbf16|vfncvtbf16" $(BUILD)/bf16_diag.dump | head; \
	  exit 1; \
	fi
	@echo "==== check-insn: no Zvfbfmin / scalar Zfbfmin instruction in the binary"

clean: ; rm -rf $(BUILD)
