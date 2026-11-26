// Isolate input copy, one GEMV and output readback on a single resident matrix.
#include <gem5/m5ops.h>
#include <gem5/pim/pim.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Tensor = std::unique_ptr<pim_tensor_t, decltype(&pim_tensor_destroy)>;

void check(pim_status_t status) {
    if (status != PIM_OK) throw std::runtime_error(pim_error_detail());
}

Tensor allocate(pim_kind_t kind, int rows, int cols, pim_dtype_t dtype) {
    pim_tensor_t* ptr = nullptr;
    check(pim_tensor_create(&ptr, kind, rows, cols, dtype));
    return Tensor(ptr, pim_tensor_destroy);
}

void run(int rows, int cols) {
    auto w = allocate(PIM_MATRIX_AB, rows, cols, PIM_FP16);
    auto x = allocate(PIM_VECTOR_GB, 1, cols, PIM_FP16);
    auto y = allocate(PIM_VECTOR_SB, 1, rows, PIM_FP32);
    check(pim_copy_to(w.get(), std::vector<float>(size_t(rows) * cols, 1.f).data()));
    std::vector<float> input(cols), output(rows);
    for (int trial = 0; trial <= 3; ++trial) {
        const float value = trial % 2 == 0 ? 1.f : -1.f;
        std::fill(input.begin(), input.end(), value);
        if (trial) m5_reset_stats(0, 0);
        check(pim_copy_to(x.get(), input.data()));
        if (trial) m5_dump_reset_stats(0, 0);
        check(pim_gemv(y.get(), w.get(), x.get()));
        if (trial) m5_dump_reset_stats(0, 0);
        check(pim_copy_from(y.get(), output.data()));
        if (trial) m5_dump_stats(0, 0);
        if (!std::all_of(output.begin(), output.end(),
                         [&](float v) { return v == value * cols; }))
            throw std::runtime_error("incorrect GEMV output");
    }
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected small or gate");
        std::string shape(argv[1]);
        if (shape != "small" && shape != "gate") throw std::runtime_error("unknown shape");
        check(pim_init(nullptr));
        run(shape == "small" ? 256 : 25600, shape == "small" ? 2048 : 5120);
        pim_finalize();
        std::cout << "NDP_SUPPLY_OUTPUT_PASS " << shape << " trials=3\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
