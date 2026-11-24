# Builds the static RISC-V pim-decode binary of the processing-in-memory
# (PIM) decode test. Only pim_kernels.S uses the V extension; pim_decode.c
# is compiled for rv64gc so that the compiler cannot emit vector code
# anywhere else.

CROSS_COMPILE ?= riscv64-linux-gnu-
CC = $(CROSS_COMPILE)gcc
CFLAGS = -static -O2 -mabi=lp64d
OUTDIR = ../bin/riscv/linux
OUT = $(OUTDIR)/pim-decode
OBJS = $(OUTDIR)/pim_decode.o $(OUTDIR)/pim_kernels.o

.PHONY: all clean

all: $(OUT)

$(OUT): pim_decode.c pim_kernels.S pim_insn.h
	mkdir -p $(OUTDIR)
	$(CC) $(CFLAGS) -march=rv64gc -c -o $(OUTDIR)/pim_decode.o pim_decode.c
	$(CC) $(CFLAGS) -march=rv64gcv -c -o $(OUTDIR)/pim_kernels.o \
	    pim_kernels.S
	$(CC) $(CFLAGS) -o $@ $(OBJS)

clean:
	rm -f $(OUT) $(OBJS)
