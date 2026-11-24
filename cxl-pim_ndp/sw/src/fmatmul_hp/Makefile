# Copyright 2026 Jae
# SPDX-License-Identifier: Apache-2.0

CROSS_COMPILE ?= riscv64-unknown-elf-

CC      := $(CROSS_COMPILE)gcc
LD      := $(CROSS_COMPILE)gcc
OBJCOPY := $(CROSS_COMPILE)objcopy
OBJDUMP := $(CROSS_COMPILE)objdump

ARCH    := rv64gcv_zfh_zvfh
ABI     := lp64d

ARA_REPO    := $(abspath ..)
KERNEL_DIR  := $(ARA_REPO)/ara_apps/dtype-matmul/kernel

CFLAGS  := -march=$(ARCH) -mabi=$(ABI) -mcmodel=medany -mno-relax \
           -O3 -g -ffreestanding -fno-builtin -fno-tree-vectorize -fno-tree-slp-vectorize -Wall -Wextra \
           -I$(KERNEL_DIR) -I$(ARA_REPO)/ara_apps/common
LDFLAGS := -march=$(ARCH) -mabi=$(ABI) -mcmodel=medany -mno-relax \
           -nostdlib -nostartfiles -Wl,--no-relax \
           -Wl,-T,../common/link.ld

BUILD   := build
OBJS    := $(BUILD)/main.o $(BUILD)/crt0.o $(BUILD)/hp-fmatmul.o $(BUILD)/util.o

TARGET  := fmatmul_hp

.PHONY: all clean
all: $(BUILD)/$(TARGET).bin $(BUILD)/$(TARGET).dump

$(BUILD):
	@mkdir -p $@

$(BUILD)/main.o: main.c ../common/uart_apb.h | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/crt0.o: ../common/crt0.S | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/hp-fmatmul.o: $(KERNEL_DIR)/hp-fmatmul.c | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/util.o: $(ARA_REPO)/ara_apps/common/util.c | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/$(TARGET).elf: $(OBJS) ../common/link.ld
	$(LD) $(LDFLAGS) -o $@ $(OBJS)

$(BUILD)/$(TARGET).bin: $(BUILD)/$(TARGET).elf
	$(OBJCOPY) -O binary $< $@
	@echo "==== $@ size: $$(stat -c %s $@) bytes ($$(($$(stat -c %s $@) / 4)) words)"

$(BUILD)/$(TARGET).dump: $(BUILD)/$(TARGET).elf
	$(OBJDUMP) -d -S $< > $@

clean:
	rm -rf $(BUILD)
