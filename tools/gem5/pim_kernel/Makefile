# Build the resident LPDDR6 worker against the supplier's native reference archive.
NDP := $(abspath ../../..)
PIM_ROOT ?= $(NDP)/gem5-ParallelPIM-main
LIB_DIR ?= $(PIM_ROOT)/libpim
PYTHON ?= python3
SCONS ?= scons
BUILD_DIR ?= $(NDP)/build/extended-gem5
WORKER_SOURCE ?= worker.cpp
SUITE ?= smoke
FIXTURE_DIR := $(BUILD_DIR)/fixture/$(SUITE)
ARCHIVE := build/native/libpim_ref_lpddr6.a
OUT ?= $(BUILD_DIR)/pim-kernel-worker

.PHONY: all reference
all: $(OUT)

reference:
	$(SCONS) -Q -C "$(LIB_DIR)" $(ARCHIVE)

$(LIB_DIR)/$(ARCHIVE): reference

$(OUT): worker.cpp $(LIB_DIR)/$(ARCHIVE)
	mkdir -p "$(dir $(OUT))"
	$(CXX) -std=c++20 -O2 -I"$(PIM_ROOT)/include" -DPIM_STD=lpddr6 -DPIM_BACKEND_REF $< "$(LIB_DIR)/$(ARCHIVE)" -lm -o "$@"

# Keep the BF16 worker as the protocol-v1 reference.
PIM_LAYOUT ?= $(BUILD_DIR)/lpddr6_fp16.h
FP16_HEADER := $(abspath $(PIM_LAYOUT))
FP16_ARCHIVE := build/native/libpim_ref_custom.a
.PHONY: fp16 fp16-reference profile manifest
fp16: $(OUT)-fp16

# One explicit profile is shared by native, RISC-V and simulator builds.
# Keep generation explicit: an existing custom PIM_LAYOUT must not be overwritten.
profile:
	mkdir -p "$(dir $(FP16_HEADER))"
	$(PYTHON) "$(LIB_DIR)/gen_pim_layout.py" --standard lpddr6 --channels 16 \
	    --policy RoRaBaChCo --num-fmt fp16 --acc-fmt fp32 -o "$(FP16_HEADER)"

$(FP16_HEADER):
	@echo "Missing profile: run make profile, or set PIM_LAYOUT to an existing generated header" >&2
	@exit 1

fp16-reference: $(FP16_HEADER)
	$(SCONS) -Q -C "$(LIB_DIR)" PIM_LAYOUT="$(FP16_HEADER)" $(FP16_ARCHIVE)

$(LIB_DIR)/$(FP16_ARCHIVE): fp16-reference

$(OUT)-fp16: $(WORKER_SOURCE) worker.cpp $(if $(filter smoke.cpp,$(WORKER_SOURCE)),$(FIXTURE_DIR)/fixture.hpp) $(NDP)/include/ndp/l1/gem5_pim/dense_kernel.hpp $(FP16_HEADER) $(LIB_DIR)/$(FP16_ARCHIVE) $(NDP)/src/common/fp16.cpp $(NDP)/include/ndp/common/fp16.hpp
	mkdir -p "$(dir $(OUT))"
	$(CXX) -std=c++20 -O2 -DPIM_BACKEND_REF -DPIM_LAYOUT_HEADER='"$(FP16_HEADER)"' -I"$(FIXTURE_DIR)" -I"$(PIM_ROOT)/include" -I"$(NDP)/include" $< "$(NDP)/src/common/fp16.cpp" "$(LIB_DIR)/$(FP16_ARCHIVE)" -lm -o "$@"

.PHONY: fixture smoke riscv-smoke
fixture:
	$(PYTHON) target_fixture.py "$(FIXTURE_DIR)" --suite "$(SUITE)" --layout "$(FP16_HEADER)"

smoke: fixture
	$(MAKE) fp16 WORKER_SOURCE=smoke.cpp OUT="$(BUILD_DIR)/pim-$(SUITE)"
	"$(BUILD_DIR)/pim-$(SUITE)-fp16"

riscv-smoke: fixture
	$(MAKE) riscv-fp16 WORKER_SOURCE=smoke.cpp RISCV_WORKER="$(BUILD_DIR)/riscv/pim-$(SUITE)-fp16"

manifest: fp16
	$(PYTHON) build_manifest.py --pim-root "$(PIM_ROOT)" --layout "$(FP16_HEADER)" \
	    --archive "$(LIB_DIR)/$(FP16_ARCHIVE)" --worker "$(OUT)-fp16" \
	    --compiler "$(CXX)" --kind native-reference --output "$(OUT)-fp16.manifest.json"

# Build the guest ELF in the full supplier fork with its generated FP16 layout.
CROSS_COMPILE ?= riscv64-linux-gnu-
RISCV_ARCHIVE := build/riscv/out/libpim_custom.a
RISCV_WORKER ?= $(BUILD_DIR)/riscv/pim-kernel-worker-fp16

.PHONY: riscv-fp16
riscv-fp16: $(FP16_HEADER)
	@command -v "$(CROSS_COMPILE)g++" >/dev/null || { echo "Missing RISC-V Linux C++ toolchain: $(CROSS_COMPILE)g++" >&2; exit 1; }
	@test -n "$(PIM_LAYOUT)" && test -f "$(PIM_LAYOUT)" || { echo "Set PIM_LAYOUT to the generated LPDDR6 FP16 header" >&2; exit 1; }
	$(SCONS) -Q -C "$(LIB_DIR)" riscv.CROSS_COMPILE="$(CROSS_COMPILE)" PIM_LAYOUT="$(abspath $(PIM_LAYOUT))" $(RISCV_ARCHIVE)
	mkdir -p "$(dir $(RISCV_WORKER))"
	$(CROSS_COMPILE)g++ -std=c++20 -O2 -static -march=rv64gcv_zicbom -mabi=lp64d -ffp-contract=off \
	    -DNDP_PIM_TARGET_FP16 -DPIM_LAYOUT_HEADER='"$(abspath $(PIM_LAYOUT))"' \
	    -I"$(FIXTURE_DIR)" -I"$(PIM_ROOT)/include" -I"$(NDP)/include" $(WORKER_SOURCE) $(EXTRA_SOURCE) "$(NDP)/src/common/fp16.cpp" \
	    "$(LIB_DIR)/$(RISCV_ARCHIVE)" -lm -o "$(RISCV_WORKER)"
	$(PYTHON) build_manifest.py --pim-root "$(PIM_ROOT)" --layout "$(FP16_HEADER)" \
	    --archive "$(LIB_DIR)/$(RISCV_ARCHIVE)" --worker "$(RISCV_WORKER)" \
	    --compiler "$(CROSS_COMPILE)g++" --kind riscv --output "$(RISCV_WORKER).manifest.json"
	$(PYTHON) inspect_elf.py "$(RISCV_WORKER)" --prefix "$(CROSS_COMPILE)"
