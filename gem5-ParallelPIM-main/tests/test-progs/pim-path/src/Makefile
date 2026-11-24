# Builds the static RISC-V pim-path binary of the processing-in-memory
# (PIM) request-path test.
# It reuses the decode test's kernels and header
# (tests/test-progs/pim-decode/src). pim_path.c is compiled for
# rv64gc_zicbom: it uses the cache-block operations but no vector code.

CROSS_COMPILE ?= riscv64-linux-gnu-
CC = $(CROSS_COMPILE)gcc
CFLAGS = -static -O2 -mabi=lp64d -pthread
DECODE = ../../pim-decode/src
OUTDIR = ../bin/riscv/linux
OUT = $(OUTDIR)/pim-path
OBJS = $(OUTDIR)/pim_path.o $(OUTDIR)/pim_kernels.o

.PHONY: all clean

all: $(OUT)

$(OUT): pim_path.c $(DECODE)/pim_kernels.S $(DECODE)/pim_insn.h
	mkdir -p $(OUTDIR)
	$(CC) $(CFLAGS) -march=rv64gc_zicbom -I$(DECODE) -c \
	    -o $(OUTDIR)/pim_path.o pim_path.c
	$(CC) $(CFLAGS) -march=rv64gcv -I$(DECODE) -c \
	    -o $(OUTDIR)/pim_kernels.o $(DECODE)/pim_kernels.S
	$(CC) $(CFLAGS) -o $@ $(OBJS)

clean:
	rm -f $(OUT) $(OBJS)
