CROSS_COMPILE ?= riscv64-unknown-elf-
CC := $(CROSS_COMPILE)gcc
LD := $(CROSS_COMPILE)gcc
OBJCOPY := $(CROSS_COMPILE)objcopy
OBJDUMP := $(CROSS_COMPILE)objdump
ARCH := rv64gcv_zfh_zvfh
ABI  := lp64d
CFLAGS := -march=$(ARCH) -mabi=$(ABI) -mcmodel=medany -mno-relax \
          -O3 -g -ffreestanding -fno-builtin -Wall -Wextra
LDFLAGS := -march=$(ARCH) -mabi=$(ABI) -mcmodel=medany -mno-relax \
           -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,../common/link.ld
BUILD := build
OBJS  := $(BUILD)/main.o $(BUILD)/crt0.o
TARGET := fp16_vec_lane0
.PHONY: all clean
all: $(BUILD)/$(TARGET).bin $(BUILD)/$(TARGET).dump
$(BUILD): ; @mkdir -p $@
$(BUILD)/main.o: main.c ../common/uart_apb.h | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<
$(BUILD)/crt0.o: ../common/crt0.S | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<
$(BUILD)/$(TARGET).elf: $(OBJS) ../common/link.ld
	$(LD) $(LDFLAGS) -o $@ $(OBJS)
$(BUILD)/$(TARGET).bin: $(BUILD)/$(TARGET).elf
	$(OBJCOPY) -O binary $< $@
	@echo "==== $@ size: $$(stat -c %s $@) bytes ($$(($$(stat -c %s $@) / 4)) words)"
$(BUILD)/$(TARGET).dump: $(BUILD)/$(TARGET).elf
	$(OBJDUMP) -d -S $< > $@
clean: ; rm -rf $(BUILD)
