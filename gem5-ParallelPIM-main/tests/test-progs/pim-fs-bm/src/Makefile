# Builds the bare-metal RISC-V pim-fs-bm binary of the processing-in-memory
# (PIM) full-system translation test. It runs through
# configs/deprecated/example/fs.py --bare-metal; the m5ops come from the
# gem5 tree, the PIM encodings from the decode test's pim_insn.h.

CROSS_COMPILE ?= riscv64-linux-gnu-
CC = $(CROSS_COMPILE)gcc
GEM5_ROOT = $(CURDIR)/../../../..
DECODE = ../../pim-decode/src
CFLAGS = -O2 -mabi=lp64d -march=rv64gc -mcmodel=medany -ffreestanding \
    -fno-builtin -nostdlib -nostartfiles -static -I$(DECODE) -I$(GEM5_ROOT)/include
OUTDIR = ../bin/riscv/bare
OUT = $(OUTDIR)/pim-fs-bm
# The same program without any PIM instruction.
OUT_NOPIM = $(OUTDIR)/pim-fs-bm-nopim
# The PIM program under Sv48 paging, with the Sv48 cases F14-F18; run with
# RiscvISA max_paging_mode = "sv48".
OUT_SV48 = $(OUTDIR)/pim-fs-bm-sv48
SRCS = start.S pim_fs_bm.c link.ld $(DECODE)/pim_insn.h

.PHONY: all clean

all: $(OUT) $(OUT_NOPIM) $(OUT_SV48)

$(OUTDIR)/m5op.o: $(GEM5_ROOT)/util/m5/src/abi/riscv/m5op.S
	mkdir -p $(OUTDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OUT): $(SRCS) $(OUTDIR)/m5op.o
	$(CC) $(CFLAGS) -march=rv64gcv -c -o $(OUTDIR)/start.o start.S
	$(CC) $(CFLAGS) -c -o $(OUTDIR)/pim_fs_bm.o pim_fs_bm.c
	$(CC) $(CFLAGS) -T link.ld -Wl,--no-warn-rwx-segments -Wl,--build-id=none -o $@ $(OUTDIR)/start.o \
	    $(OUTDIR)/pim_fs_bm.o $(OUTDIR)/m5op.o

$(OUT_NOPIM): $(SRCS) $(OUTDIR)/m5op.o
	$(CC) $(CFLAGS) -DPIM_FS_BM_NOPIM -c -o $(OUTDIR)/start-nopim.o start.S
	$(CC) $(CFLAGS) -DPIM_FS_BM_NOPIM -c -o $(OUTDIR)/pim_fs_bm-nopim.o \
	    pim_fs_bm.c
	$(CC) $(CFLAGS) -T link.ld -Wl,--no-warn-rwx-segments -Wl,--build-id=none -o $@ \
	    $(OUTDIR)/start-nopim.o $(OUTDIR)/pim_fs_bm-nopim.o $(OUTDIR)/m5op.o

$(OUT_SV48): $(SRCS) $(OUTDIR)/m5op.o
	$(CC) $(CFLAGS) -march=rv64gcv -DPIM_FS_BM_SV48 -c \
	    -o $(OUTDIR)/start-sv48.o start.S
	$(CC) $(CFLAGS) -DPIM_FS_BM_SV48 -c -o $(OUTDIR)/pim_fs_bm-sv48.o \
	    pim_fs_bm.c
	$(CC) $(CFLAGS) -T link.ld -Wl,--no-warn-rwx-segments -Wl,--build-id=none -o $@ \
	    $(OUTDIR)/start-sv48.o $(OUTDIR)/pim_fs_bm-sv48.o $(OUTDIR)/m5op.o

clean:
	rm -f $(OUT) $(OUT_NOPIM) $(OUT_SV48) $(OUTDIR)/*.o
