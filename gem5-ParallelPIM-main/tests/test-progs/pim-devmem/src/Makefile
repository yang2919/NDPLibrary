# Builds the static RISC-V pim-devmem binary: the check of SE mode's
# emulated /dev/mem (PimMemDriver) over a processing-in-memory (PIM)
# memory (check V12).

CROSS_COMPILE ?= riscv64-linux-gnu-
CC = $(CROSS_COMPILE)gcc
CFLAGS = -static -O2 -mabi=lp64d -march=rv64gc
OUTDIR = ../bin/riscv/linux
OUT = $(OUTDIR)/pim-devmem

.PHONY: all clean

all: $(OUT)

$(OUT): pim_devmem.c
	mkdir -p $(OUTDIR)
	$(CC) $(CFLAGS) -o $@ pim_devmem.c

clean:
	rm -f $(OUT)
