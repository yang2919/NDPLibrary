# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0
#
# Common build for a DTYPE-parameterized kernel. A kernel Makefile sets KERNEL
# (and optionally EXTRA_CFLAGS, EXTRA_OBJS_SP, EXTRA_OBJS_HP) then:
#     KERNEL := vfredsum
#     include ../common/kernel.mk
# Produces <KERNEL>_sp.bin (fp32) and <KERNEL>_hp.bin (fp16, -DHP) from one
# main.c. REQUIRES GCC 16 (fp16 vector intrinsics): build with
#     make CROSS_COMPILE=riscv64-unknown-elf-

CROSS_COMPILE ?= riscv64-unknown-elf-
CC      := $(CROSS_COMPILE)gcc
OBJCOPY := $(CROSS_COMPILE)objcopy
OBJDUMP := $(CROSS_COMPILE)objdump

ARCH    := rv64gcv_zfh_zvfh
ABI     := lp64d
# -fno-tree-vectorize: GCC16 -O3 auto-vectorizes the scalar setup/reference loops
# (init, golden, verify); that auto-vectorized code MISCOMPUTES at vl=128 (element
# 0 -> garbage/-inf, Verilator-reproducible) and corrupts kernel inputs. The kernel
# algorithms are explicit __riscv_v* intrinsics and are unaffected, so disabling
# auto-vec is correct (not a perf loss). Without it: exp/gelu/softmax/layernorm/
# vfredsum/attention FAIL or hang; with it they run.
CFLAGS  := -march=$(ARCH) -mabi=$(ABI) -mcmodel=medany -mno-relax \
           -O3 -g -ffreestanding -fno-builtin -fno-tree-vectorize -fno-tree-slp-vectorize \
           -Wall -Wextra $(EXTRA_CFLAGS)
LDFLAGS := -march=$(ARCH) -mabi=$(ABI) -mcmodel=medany -mno-relax \
           -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,../common/link.ld

# Every kernel includes several of these; listing only dtype.h meant an edit to
# exp_hp.h silently did NOT rebuild (2026-08-11: an exp16 clamp fix was missing
# from the binary while the dtype.h fix in the same session was present).
COMMON_HDRS := $(wildcard ../common/*.h)

BUILD := build
.PHONY: all clean
# Without this the default goal would be whatever target appears first, which
# for a kernel that sets KERNEL_NO_DEFAULT is the `build` directory rule.
.DEFAULT_GOAL := all
# KERNEL_NO_DEFAULT := 1 (set before include) suppresses the {sp,hp} pair and
# leaves only CFLAGS/LDFLAGS/crt0/%.bin for a kernel that defines its own build
# matrix. sw/rag needs it: its third axis is the algorithm, so the algo-less
# pair would not compile.
ifneq ($(KERNEL_NO_DEFAULT),1)
all: $(BUILD)/$(KERNEL)_sp.bin $(BUILD)/$(KERNEL)_hp.bin
endif

$(BUILD): ; @mkdir -p $@

$(BUILD)/crt0.o: ../common/crt0.S | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<
$(BUILD)/main_sp.o: main.c $(COMMON_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<
$(BUILD)/main_hp.o: main.c $(COMMON_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -DHP -c -o $@ $<

$(BUILD)/$(KERNEL)_sp.elf: $(BUILD)/main_sp.o $(BUILD)/crt0.o $(EXTRA_OBJS_SP) ../common/link.ld
	$(CC) $(LDFLAGS) -o $@ $(BUILD)/main_sp.o $(BUILD)/crt0.o $(EXTRA_OBJS_SP)
$(BUILD)/$(KERNEL)_hp.elf: $(BUILD)/main_hp.o $(BUILD)/crt0.o $(EXTRA_OBJS_HP) ../common/link.ld
	$(CC) $(LDFLAGS) -o $@ $(BUILD)/main_hp.o $(BUILD)/crt0.o $(EXTRA_OBJS_HP)

$(BUILD)/%.bin: $(BUILD)/%.elf
	$(OBJCOPY) -O binary $< $@
	@echo "==== $@: $$(stat -c %s $@) bytes ($$(($$(stat -c %s $@)/4)) words)"
	@$(OBJDUMP) -d -S $< > $(@:.bin=.dump)

# VT := 1 (set before include) adds <KERNEL>_vt_{sp,hp}.bin — same main.c built
# with BOTH axes (-DUSE_VTRANS -DUSE_SWOPT): the vtrans FU plus the fused /
# double-buffered / QK^T-matmul restructuring. These kernels are the CORRECTNESS
# check for the fully optimized path; sw/roofline carries the 2x2 attribution
# matrix (base / sw / hw / hw+sw).
ifeq ($(VT),1)
all: $(BUILD)/$(KERNEL)_vt_sp.bin $(BUILD)/$(KERNEL)_vt_hp.bin
$(BUILD)/main_vt_sp.o: main.c $(COMMON_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -DUSE_VTRANS -DUSE_SWOPT -c -o $@ $<
$(BUILD)/main_vt_hp.o: main.c $(COMMON_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -DUSE_VTRANS -DUSE_SWOPT -DHP -c -o $@ $<
$(BUILD)/$(KERNEL)_vt_sp.elf: $(BUILD)/main_vt_sp.o $(BUILD)/crt0.o $(EXTRA_OBJS_SP) ../common/link.ld
	$(CC) $(LDFLAGS) -o $@ $(BUILD)/main_vt_sp.o $(BUILD)/crt0.o $(EXTRA_OBJS_SP)
$(BUILD)/$(KERNEL)_vt_hp.elf: $(BUILD)/main_vt_hp.o $(BUILD)/crt0.o $(EXTRA_OBJS_HP) ../common/link.ld
	$(CC) $(LDFLAGS) -o $@ $(BUILD)/main_vt_hp.o $(BUILD)/crt0.o $(EXTRA_OBJS_HP)
endif

clean: ; rm -rf $(BUILD)
