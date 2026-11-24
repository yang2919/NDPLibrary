# Copyright (c) 2026 Jongmin Kim
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Builds the static RISC-V rvv-vadd binary. Only vadd_kernel.S uses the
# V extension; rvv_vadd.c is compiled for rv64gc so that the compiler
# cannot emit vector code anywhere else in the program. The m5ops that
# mark the region of interest come from the gem5 tree.

CROSS_COMPILE ?= riscv64-linux-gnu-
CC = $(CROSS_COMPILE)gcc
CFLAGS = -static -O2 -mabi=lp64d
GEM5_ROOT = $(CURDIR)/../../../..
OUTDIR = ../bin/riscv/linux
OUT = $(OUTDIR)/rvv-vadd
OBJS = $(OUTDIR)/rvv_vadd.o $(OUTDIR)/vadd_kernel.o $(OUTDIR)/m5op.o

.PHONY: all clean

all: $(OUT)

$(OUT): rvv_vadd.c vadd_kernel.S
	mkdir -p $(OUTDIR)
	$(CC) $(CFLAGS) -march=rv64gc -I$(GEM5_ROOT)/include \
	    -c -o $(OUTDIR)/rvv_vadd.o rvv_vadd.c
	$(CC) $(CFLAGS) -march=rv64gcv -c -o $(OUTDIR)/vadd_kernel.o \
	    vadd_kernel.S
	$(CC) $(CFLAGS) -march=rv64gc -I$(GEM5_ROOT)/include \
	    -c -o $(OUTDIR)/m5op.o $(GEM5_ROOT)/util/m5/src/abi/riscv/m5op.S
	$(CC) $(CFLAGS) -o $@ $(OBJS)

clean:
	rm -f $(OUT) $(OBJS)
