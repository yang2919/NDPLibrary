# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0
#
# Roofline throughput-sweep harness. One source builds fp32 (roofline_sp) and
# fp16 (roofline_hp, -DHP). fp16 REQUIRES GCC 16 (zvfh intrinsics); build via
# build_all_kernels.sh or `make CROSS_COMPILE=riscv64-unknown-elf-`.

CROSS_COMPILE ?= riscv64-unknown-elf-

CC      := $(CROSS_COMPILE)gcc
OBJCOPY := $(CROSS_COMPILE)objcopy
OBJDUMP := $(CROSS_COMPILE)objdump

ARCH_SP := rv64gcv
ARCH_HP := rv64gcv_zfh_zvfh
ABI     := lp64d

ARA_REPO   := $(abspath ..)
KERNEL_DIR := $(ARA_REPO)/ara_apps/dtype-matmul/kernel

CFLAGS_COMMON := -mabi=$(ABI) -mcmodel=medany -mno-relax \
                 -O3 -g -ffreestanding -fno-builtin -fno-tree-vectorize -fno-tree-slp-vectorize \
                 -Wall -Wextra -I$(KERNEL_DIR) -I$(ARA_REPO)/ara_apps/common
LDFLAGS_COMMON := -mabi=$(ABI) -mcmodel=medany -mno-relax \
                  -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,../common/link.ld

CFLAGS_SP := -march=$(ARCH_SP) $(CFLAGS_COMMON)
CFLAGS_HP := -march=$(ARCH_HP) $(CFLAGS_COMMON) -DHP
LDFLAGS_SP := -march=$(ARCH_SP) $(LDFLAGS_COMMON)
LDFLAGS_HP := -march=$(ARCH_HP) $(LDFLAGS_COMMON)

BUILD := build

# 2x2 measurement matrix, per dtype:
#   <none>  base   classic multi-pass kernels, Cephes exp + vfdiv
#   _sw     sw     fused single-pass + QK^T matmul, still Cephes exp + vfdiv
#   _hw     hw     vtrans FU (vfexp/vfrcp), classic multi-pass structure
#   _hwsw   hw+sw  vtrans FU + fused + double-buffered vt loops + QK^T matmul
#
# Plus the CORRECTED baseline pair. `base` and `sw` divide with vfdiv, which is
# the one non-pipelined op in the FPU (fpnew PULP DIVSQRT, MERGED; stalled at
# vmfpu.sv:1497-1501) -- so measuring the FU's vfrcp against it overstates the
# unit. RVV itself provides a reciprocal seed, and Ara implements it
# (FPExtSupport, ara_pkg.sv:163). These two builds replace vfdiv with vfrec7 + 2
# Newton steps (1 at fp16), measured at 1 ULP, and are the baseline every RCP
# claim must be quoted against:
#   _rec7   base+  classic multi-pass, Cephes exp, vfrec7+Newton divide
#   _rec7sw sw+    fused single-pass + QK^T matmul, Cephes exp, vfrec7+Newton
# Only gelu changes: it is the sole vector-divide site (main.c:338). softmax
# divides by a SCALAR row sum (main.c:346), so exp/sfx/attn/lyn are unaffected
# and serve as the built-in control that the flag did nothing else.
VARIANTS := $(BUILD)/roofline_sp $(BUILD)/roofline_sw_sp $(BUILD)/roofline_hw_sp $(BUILD)/roofline_hwsw_sp \
            $(BUILD)/roofline_rec7_sp $(BUILD)/roofline_rec7sw_sp \
            $(BUILD)/roofline_hp $(BUILD)/roofline_sw_hp $(BUILD)/roofline_hw_hp $(BUILD)/roofline_hwsw_hp \
            $(BUILD)/roofline_rec7_hp $(BUILD)/roofline_rec7sw_hp \
            $(BUILD)/roofline_spm_sp $(BUILD)/roofline_spm_hp \
            $(BUILD)/roofline_i8

.PHONY: all clean
all: $(addsuffix .bin,$(VARIANTS)) $(addsuffix .dump,$(VARIANTS))

$(BUILD):
	@mkdir -p $@

# ---- fp32 (sp) ----
$(BUILD)/main_sp.o: main.c ../common/dtype.h ../common/uart_apb.h | $(BUILD)
	$(CC) $(CFLAGS_SP) -c -o $@ $<
$(BUILD)/crt0_sp.o: ../common/crt0.S | $(BUILD)
	$(CC) $(CFLAGS_SP) -c -o $@ $<
$(BUILD)/sp-fmatmul.o: $(KERNEL_DIR)/sp-fmatmul.c | $(BUILD)
	$(CC) $(CFLAGS_SP) -c -o $@ $<
$(BUILD)/util_sp.o: $(ARA_REPO)/ara_apps/common/util.c | $(BUILD)
	$(CC) $(CFLAGS_SP) -c -o $@ $<
$(BUILD)/roofline_sp.elf: $(BUILD)/main_sp.o $(BUILD)/crt0_sp.o $(BUILD)/sp-fmatmul.o $(BUILD)/util_sp.o ../common/link.ld
	$(CC) $(LDFLAGS_SP) -o $@ $(BUILD)/main_sp.o $(BUILD)/crt0_sp.o $(BUILD)/sp-fmatmul.o $(BUILD)/util_sp.o

# ---- fp16 (hp) ----
$(BUILD)/main_hp.o: main.c ../common/dtype.h ../common/uart_apb.h | $(BUILD)
	$(CC) $(CFLAGS_HP) -c -o $@ $<
$(BUILD)/crt0_hp.o: ../common/crt0.S | $(BUILD)
	$(CC) $(CFLAGS_HP) -c -o $@ $<
$(BUILD)/hp-fmatmul.o: $(KERNEL_DIR)/hp-fmatmul.c | $(BUILD)
	$(CC) $(CFLAGS_HP) -c -o $@ $<
$(BUILD)/util_hp.o: $(ARA_REPO)/ara_apps/common/util.c | $(BUILD)
	$(CC) $(CFLAGS_HP) -c -o $@ $<
$(BUILD)/roofline_hp.elf: $(BUILD)/main_hp.o $(BUILD)/crt0_hp.o $(BUILD)/hp-fmatmul.o $(BUILD)/util_hp.o ../common/link.ld
	$(CC) $(LDFLAGS_HP) -o $@ $(BUILD)/main_hp.o $(BUILD)/crt0_hp.o $(BUILD)/hp-fmatmul.o $(BUILD)/util_hp.o

# ---- axis variants: -DUSE_VTRANS = HW axis, -DUSE_SWOPT = SW axis ----
$(BUILD)/main_sw_sp.o: main.c ../common/dtype.h ../common/exp_hp.h | $(BUILD)
	$(CC) $(CFLAGS_SP) -DUSE_SWOPT -c -o $@ $<
$(BUILD)/main_hw_sp.o: main.c ../common/dtype.h ../common/exp_hp.h | $(BUILD)
	$(CC) $(CFLAGS_SP) -DUSE_VTRANS -c -o $@ $<
$(BUILD)/main_hwsw_sp.o: main.c ../common/dtype.h ../common/exp_hp.h | $(BUILD)
	$(CC) $(CFLAGS_SP) -DUSE_VTRANS -DUSE_SWOPT -c -o $@ $<
$(BUILD)/main_sw_hp.o: main.c ../common/dtype.h ../common/exp_hp.h | $(BUILD)
	$(CC) $(CFLAGS_HP) -DUSE_SWOPT -c -o $@ $<
$(BUILD)/main_hw_hp.o: main.c ../common/dtype.h ../common/exp_hp.h | $(BUILD)
	$(CC) $(CFLAGS_HP) -DUSE_VTRANS -c -o $@ $<
$(BUILD)/main_hwsw_hp.o: main.c ../common/dtype.h ../common/exp_hp.h | $(BUILD)
	$(CC) $(CFLAGS_HP) -DUSE_VTRANS -DUSE_SWOPT -c -o $@ $<
$(BUILD)/main_rec7_sp.o: main.c ../common/dtype.h ../common/exp_hp.h | $(BUILD)
	$(CC) $(CFLAGS_SP) -DUSE_REC7 -c -o $@ $<
$(BUILD)/main_rec7sw_sp.o: main.c ../common/dtype.h ../common/exp_hp.h | $(BUILD)
	$(CC) $(CFLAGS_SP) -DUSE_REC7 -DUSE_SWOPT -c -o $@ $<
$(BUILD)/main_rec7_hp.o: main.c ../common/dtype.h ../common/exp_hp.h | $(BUILD)
	$(CC) $(CFLAGS_HP) -DUSE_REC7 -c -o $@ $<
$(BUILD)/main_rec7sw_hp.o: main.c ../common/dtype.h ../common/exp_hp.h | $(BUILD)
	$(CC) $(CFLAGS_HP) -DUSE_REC7 -DUSE_SWOPT -c -o $@ $<
# ---- scratchpad build (-DRL_SPM, base variant only): every record runs as above, then again with its
# operands in the scratchpad; the extra results go to 0x4010_0400 (main.c, RL_SPM block) ----
$(BUILD)/main_spm_sp.o: main.c ../common/dtype.h ../common/exp_hp.h | $(BUILD)
	$(CC) $(CFLAGS_SP) -DRL_SPM -c -o $@ $<
$(BUILD)/main_spm_hp.o: main.c ../common/dtype.h ../common/exp_hp.h | $(BUILD)
	$(CC) $(CFLAGS_HP) -DRL_SPM -c -o $@ $<

$(BUILD)/roofline_%_sp.elf: $(BUILD)/main_%_sp.o $(BUILD)/crt0_sp.o $(BUILD)/sp-fmatmul.o $(BUILD)/util_sp.o ../common/link.ld
	$(CC) $(LDFLAGS_SP) -o $@ $< $(BUILD)/crt0_sp.o $(BUILD)/sp-fmatmul.o $(BUILD)/util_sp.o
$(BUILD)/roofline_%_hp.elf: $(BUILD)/main_%_hp.o $(BUILD)/crt0_hp.o $(BUILD)/hp-fmatmul.o $(BUILD)/util_hp.o ../common/link.ld
	$(CC) $(LDFLAGS_HP) -o $@ $< $(BUILD)/crt0_hp.o $(BUILD)/hp-fmatmul.o $(BUILD)/util_hp.o

# ---- bin/dump ----
$(BUILD)/%.bin: $(BUILD)/%.elf
	$(OBJCOPY) -O binary $< $@
	@echo "==== $@ size: $$(stat -c %s $@) bytes ($$(($$(stat -c %s $@) / 4)) words)"
$(BUILD)/%.dump: $(BUILD)/%.elf
	$(OBJDUMP) -d -S $< > $@

# ---- int8 (i8) ----
# Only mm / dp / red: the other five fp kernels are transcendental and have no
# integer counterpart in this ISA. No zfh/zvfh, and no HW/SW axis variants --
# the vtrans FU (vfexp/vfrcp) is floating point only, so there is no `hw` variant to
# build. Links bp-imatmul, the int8 sibling of sp-fmatmul in the same directory.
# Keep these rules BELOW `all:` -- the first explicit rule in a makefile becomes
# the default goal, and putting them above it made `main_i8.o` the whole build.
CFLAGS_I8  := -march=$(ARCH_SP) $(CFLAGS_COMMON) -DI8
LDFLAGS_I8 := -march=$(ARCH_SP) $(LDFLAGS_COMMON)

$(BUILD)/main_i8.o: main.c $(KERNEL_DIR)/bp-imatmul.h | $(BUILD)
	$(CC) $(CFLAGS_I8) -c -o $@ $<
$(BUILD)/crt0_i8.o: ../common/crt0.S | $(BUILD)
	$(CC) $(CFLAGS_I8) -c -o $@ $<
$(BUILD)/bp-imatmul.o: $(KERNEL_DIR)/bp-imatmul.c | $(BUILD)
	$(CC) $(CFLAGS_I8) -c -o $@ $<
$(BUILD)/util_i8.o: $(ARA_REPO)/ara_apps/common/util.c | $(BUILD)
	$(CC) $(CFLAGS_I8) -c -o $@ $<
$(BUILD)/roofline_i8.elf: $(BUILD)/main_i8.o $(BUILD)/crt0_i8.o $(BUILD)/bp-imatmul.o $(BUILD)/util_i8.o ../common/link.ld
	$(CC) $(LDFLAGS_I8) -o $@ $(BUILD)/main_i8.o $(BUILD)/crt0_i8.o $(BUILD)/bp-imatmul.o $(BUILD)/util_i8.o

clean:
	rm -rf $(BUILD)
