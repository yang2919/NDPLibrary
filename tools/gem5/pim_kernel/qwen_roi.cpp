// Measure one synthetic Qwen3 decode layer in a fresh single-device SE process.
#include <gem5/m5ops.h>
#include <gem5/pim/pim.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Row = std::vector<float>;
using Tensor = std::unique_ptr<pim_tensor_t, decltype(&pim_tensor_destroy)>;
constexpr int head = 128;

void check(pim_status_t status) {
    if (status != PIM_OK) throw std::runtime_error(pim_error_detail());
}

Tensor tensor(pim_kind_t kind, int rows, int cols, pim_dtype_t dtype) {
    pim_tensor_t* ptr = nullptr;
    check(pim_tensor_create(&ptr, kind, rows, cols, dtype));
    return Tensor(ptr, pim_tensor_destroy);
}

struct Projection {
    // Matrix rows occupy distinct bank storage in initially zeroed simulator memory.
    Tensor weight, input, output;
    int rows;
    Projection(int m, int k)
        : weight(tensor(PIM_MATRIX_AB, m, k, PIM_FP16)),
          input(tensor(PIM_VECTOR_GB, 1, k, PIM_FP16)),
          output(tensor(PIM_VECTOR_SB, 1, m, PIM_FP32)), rows(m) {}
    Row operator()(const Row& x) {
        Row y(rows);
        check(pim_copy_to(input.get(), x.data()));
        check(pim_gemv(output.get(), weight.get(), input.get()));
        check(pim_copy_from(output.get(), y.data()));
        return y;
    }
};

Row norm(Row x, int width) {
    for (size_t start = 0; start < x.size(); start += width) {
        float squares = 0;
        for (int i = 0; i < width; ++i) squares += x[start+i] * x[start+i];
        float scale = 1 / std::sqrt(squares / width + 1e-6f);
        for (int i = 0; i < width; ++i) x[start+i] *= scale;
    }
    return x;
}

void rope(Row& x, int position) {
    for (int i = 0; i < head/2; ++i) {
        float angle = position * std::pow(1e6f, -2.f*i/head);
        float c = std::cos(angle), s = std::sin(angle);
        for (size_t j = i; j < x.size(); j += head) {
            float a = x[j], b = x[j+head/2];
            x[j] = c*a - s*b;
            x[j+head/2] = s*a + c*b;
        }
    }
}

struct Expert {
    Projection gate, up, down;
    Expert(int hidden, int ff) : gate(ff, hidden), up(ff, hidden), down(hidden, ff) {}
    Row operator()(const Row& x) {
        Row g = gate(x), u = up(x);
        for (size_t i = 0; i < g.size(); ++i) g[i] = g[i] / (1 + std::exp(-g[i])) * u[i];
        return down(g);
    }
};

void run(bool moe, int context, int tp) {
    const int hidden = moe ? 2048 : 5120, heads = (moe ? 32 : 64) / tp, kvheads = (moe ? 4 : 8) / tp;
    const int kv = kvheads * head;
    Projection qproj(heads*head, hidden), kproj(kv, hidden), vproj(kv, hidden);
    Projection oproj(hidden, heads*head);
    std::vector<Expert> experts;
    for (int e = 0; e < (moe ? 128 : 1); ++e) experts.emplace_back(hidden, (moe ? 768 : 25600) / tp);
    std::unique_ptr<Projection> router;
    if (moe) router = std::make_unique<Projection>(128, hidden);
    Row residual(hidden, .125f), keys(context*kv, .03125f), values(context*kv, .0625f);
    Row attended(heads*head), scores(context);
    m5_reset_stats(0, 0);
    Row x = norm(residual, hidden);
    Row q = norm(qproj(x), head), k = norm(kproj(x), head), v = vproj(x);
    rope(q, context-1);
    rope(k, context-1);
    std::copy(k.begin(), k.end(), keys.end()-kv);
    std::copy(v.begin(), v.end(), values.end()-kv);
    for (int h = 0; h < heads; ++h) {
        int offset = (h / (heads/kvheads)) * head;
        for (int t = 0; t < context; ++t)
            scores[t] = std::inner_product(q.begin()+h*head, q.begin()+(h+1)*head,
                                          keys.begin()+t*kv+offset, 0.f) / std::sqrt(float(head));
        float peak = *std::max_element(scores.begin(), scores.end()), sum = 0;
        for (auto& score : scores) sum += (score = std::exp(score-peak));
        for (int t = 0; t < context; ++t)
            for (int d = 0; d < head; ++d)
                attended[h*head+d] += (scores[t]/sum) * values[t*kv+offset+d];
    }
    Row attention = oproj(attended);
    for (int i = 0; i < hidden; ++i) residual[i] += attention[i];
    m5_dump_reset_stats(0, 0);
    x = norm(residual, hidden);
    if (moe) {
        Row logits = (*router)(x);
        std::vector<int> ids(128);
        std::iota(ids.begin(), ids.end(), 0);
        std::partial_sort(ids.begin(), ids.begin()+8, ids.end(), [&](int a, int b) {
            return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
        });
        Row weights(8);
        for (int j = 0; j < 8; ++j) weights[j] = std::exp(logits[ids[j]] - logits[ids[0]]);
        float sum = std::accumulate(weights.begin(), weights.end(), 0.f);
        for (int j = 0; j < 8; ++j) {
            Row y = experts[ids[j]](x);
            for (int i = 0; i < hidden; ++i) residual[i] += weights[j]/sum*y[i];
        }
    } else {
        Row y = experts[0](x);
        for (int i = 0; i < hidden; ++i) residual[i] += y[i];
    }
    m5_dump_stats(0, 0);
    std::cout << "QWEN_ROI_DONE " << (moe ? "moe" : "dense") << " context=" << context
              << " tp=" << tp << " checksum=" << std::accumulate(residual.begin(), residual.end(), 0.f) << '\n';
}

int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 4) throw std::runtime_error("expected dense|moe and context length");
        std::string model(argv[1]);
        int context = std::stoi(argv[2]);
        if ((model != "dense" && model != "moe") || context < 1 || context > 4096)
            throw std::runtime_error("invalid model or context (1..4096)");
        const int tp = argc == 4 ? std::stoi(argv[3]) : 1;
        if (tp < 1 || (model == "moe" ? 4 : 8) % tp != 0)
            throw std::runtime_error("TP must divide the KV head count");
        check(pim_init(nullptr));
        run(model == "moe", context, tp);
        pim_finalize();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
