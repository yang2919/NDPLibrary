# Builds pim-reuse, a PIM library program, for one layout:
#
#   make                                  RISC-V, PIM_STD=lpddr6
#   make PIM_STD=hbm4                     another default layout
#   make PIM_LAYOUT=/abs/my_layout.h LAYOUT_NAME=my
#                                         a generated layout header
#                                         (libpim/gen_pim_layout.py -o)
#   make TARGET=host ...                  the host reference build
#
# The program goes to ../bin/<TARGET>/<layout>/pim-reuse. It links the
# archive of the same layout: libpim_<std>.a or libpim_custom.a for RISC-V,
# libpim_ref_<std>.a or libpim_ref_custom.a for the host, which SCons brings
# up to date in libpim/build first. A generated layout builds
# libpim_custom.a, so build one generated layout at a time.

GEM5_ROOT ?= $(abspath ../../../..)
PIM_STD ?= lpddr6
PIM_LAYOUT ?=
TARGET ?= riscv
CROSS_COMPILE ?= riscv64-linux-gnu-
SCONS ?= scons

PIM_DIR = $(GEM5_ROOT)/libpim
INC = $(GEM5_ROOT)/include/gem5/pim
ifeq ($(PIM_LAYOUT),)
LAYOUT_NAME ?= $(PIM_STD)
LIB_LAYOUT = $(PIM_STD)
LAYOUT_DEF = -DPIM_STD=$(PIM_STD)
LAYOUT_HDR = $(INC)/pim_layout_$(PIM_STD).h
SCONS_LAYOUT =
else
LAYOUT_NAME ?= custom
LIB_LAYOUT = custom
LAYOUT_DEF = -DPIM_LAYOUT_HEADER='"$(abspath $(PIM_LAYOUT))"'
LAYOUT_HDR = $(abspath $(PIM_LAYOUT))
SCONS_LAYOUT = PIM_LAYOUT=$(abspath $(PIM_LAYOUT))
endif

ifeq ($(TARGET),host)
CC = gcc
CFLAGS = -O2 -std=c11 -D_DEFAULT_SOURCE -DPIM_BACKEND_REF -ffp-contract=off
LIB_PATH = build/native/libpim_ref_$(LIB_LAYOUT).a
SCONS_ARGS = $(SCONS_LAYOUT)
LDFLAGS =
else
CC = $(CROSS_COMPILE)gcc
CFLAGS = -O2 -std=c11 -march=rv64gcv_zicbom -mabi=lp64d -ffp-contract=off
LIB_PATH = build/riscv/out/libpim_$(LIB_LAYOUT).a
SCONS_ARGS = riscv.CROSS_COMPILE=$(CROSS_COMPILE) $(SCONS_LAYOUT)
LDFLAGS = -static
endif
LIB = $(PIM_DIR)/$(LIB_PATH)

OUTDIR = ../bin/$(TARGET)/$(LAYOUT_NAME)
OUT = $(OUTDIR)/pim-reuse
HDRS = $(LAYOUT_HDR) $(INC)/pim_layout.h $(INC)/pim_addr.h \
       $(INC)/pim_isa.h $(INC)/pim_runtime.h $(INC)/pim.h

.PHONY: all lib clean

all: $(OUT)

# SCons decides whether the archive is up to date.
lib:
	$(SCONS) -C $(PIM_DIR) -Q $(SCONS_ARGS) $(LIB_PATH)

$(LIB): lib

$(OUT): pim_reuse.c $(HDRS) $(LIB)
	mkdir -p $(OUTDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -I$(GEM5_ROOT)/include $(LAYOUT_DEF) \
	    -o $@ pim_reuse.c $(LIB) -lm

clean:
	rm -rf ../bin
