# Build the standalone vector-add ELF using LLVM's assembler and linker.
PNM_CC ?= clang
ISA_FLAGS ?= --target=riscv64-unknown-elf -march=rv64gv_zvfh -mabi=lp64d -fuse-ld=lld
OUT ?= ../../../build/pnm/vector_add.elf

$(OUT): vector_add.S kernel.ld
	mkdir -p "$(dir $(OUT))"
	$(PNM_CC) $(ISA_FLAGS) \
	    -nostdlib -static -Wl,-e,pnm_vector_add -T kernel.ld vector_add.S -o "$@"

PIM_ROOT ?= ../../../pim-isa-library-2026-11-10
GEMV_OUT ?= ../../../build/pnm/pim_gemv.elf
LAYOUT ?= $(abspath ../pim_kernel/lpddr6_fp16_native.h)
PIM_FLAGS = -I"$(PIM_ROOT)/include" -DPIM_LAYOUT_HEADER='"$(LAYOUT)"'
PIM_HEADERS = $(wildcard $(PIM_ROOT)/include/gem5/pim/*.h)

.PHONY: gemv
gemv: $(GEMV_OUT)

$(GEMV_OUT): gemv.c gemv.h kernel.ld $(LAYOUT) $(PIM_HEADERS)
	mkdir -p "$(dir $@)"
	$(PNM_CC) $(ISA_FLAGS) $(PIM_FLAGS) -O2 -ffreestanding -fno-stack-protector \
	    -fno-pic -msmall-data-limit=0 -nostdlib -static -Wl,-e,pnm_pim_gemv \
	    -T kernel.ld gemv.c -o "$@"

REF_SRC = $(PIM_ROOT)/util/pim/src/ref
TEST_OUT ?= ../../../build/pnm/test_gemv
.PHONY: test-gemv
test-gemv: $(TEST_OUT)
	"$(TEST_OUT)"

$(TEST_OUT): gemv.c gemv.h test_gemv.c $(LAYOUT) $(PIM_HEADERS) $(REF_SRC)/model.c $(REF_SRC)/numfmt.c
	mkdir -p "$(dir $@)"
	$(CC) -std=gnu11 -O2 -ffp-contract=off -DPIM_BACKEND_REF $(PIM_FLAGS) \
	    -I"$(PIM_ROOT)/util/pim/src" gemv.c test_gemv.c $(REF_SRC)/model.c $(REF_SRC)/numfmt.c -lm -o "$@"

POINTWISE_OUT ?= ../../../build/pnm/pointwise.elf
.PHONY: pointwise
pointwise: $(POINTWISE_OUT)

$(POINTWISE_OUT): pointwise.c kernel.ld
	mkdir -p "$(dir $@)"
	$(PNM_CC) $(ISA_FLAGS) -O2 -ffreestanding -fno-stack-protector -fno-pic \
	    -msmall-data-limit=0 -nostdlib -static -Wl,-e,pnm_pointwise -T kernel.ld $< -o "$@"

POINTWISE_TEST ?= ../../../build/pnm/test_pointwise
.PHONY: test-pointwise
test-pointwise: $(POINTWISE_TEST)
	"$(POINTWISE_TEST)"

$(POINTWISE_TEST): pointwise.c test_pointwise.c
	mkdir -p "$(dir $@)"
	$(CC) -std=gnu11 -O2 -ffp-contract=off test_pointwise.c -lm -o "$@"

ATTENTION_OUT ?= ../../../build/pnm/attention.elf
.PHONY: attention
attention: $(ATTENTION_OUT)

$(ATTENTION_OUT): attention.cpp kernel.ld ../../../include/ndp/l1/gem5_pim/dense_kernel.hpp
	mkdir -p "$(dir $@)"
	$(PNM_CC) $(ISA_FLAGS) -std=c++17 -I../../../include -O2 -ffreestanding -fno-exceptions \
	    -fno-rtti -fno-stack-protector -fno-pic -msmall-data-limit=0 -nostdlib -static \
	    -Wl,-e,pnm_attention -T kernel.ld $< -o "$@"

ATTENTION_TEST ?= ../../../build/pnm/test_attention
.PHONY: test-attention
test-attention: $(ATTENTION_TEST)
	"$(ATTENTION_TEST)"

$(ATTENTION_TEST): test_attention.cpp attention.cpp ../../../include/ndp/l1/gem5_pim/dense_kernel.hpp
	mkdir -p "$(dir $@)"
	$(CXX) -std=c++17 -O2 -I../../../include test_attention.cpp -o "$@"
