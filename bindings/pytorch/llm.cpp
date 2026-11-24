// Expose L3 weight and request-cache policies without introducing them into L2.
#include "wrappers.hpp"

namespace ndp::python {
void bind_llm(py::module_ &m) {
  py::class_<WeightHandle>(m, "PreparedWeight")
      .def("tensor",
           [](const WeightHandle &w) {
             w.owner->check("weight.tensor", true);
             return TensorHandle{w.owner, w.native.tensor()};
           })
      .def_property_readonly("resident_bytes", [](const WeightHandle &w) {
        w.owner->check("resident_bytes", true);
        return w.native.resident_bytes();
      });
  m.def(
      "prepare_weight",
      [](std::shared_ptr<Owner> s, const TensorHandle &x, std::vector<Engine> engines,
         size_t budget) {
        check(s, x, "prepare_weight");
        py::gil_scoped_release release;
        return WeightHandle{
            s, take(ndp::prepare_weight(s->native, x.native, {std::move(engines), budget}),
                    "prepare_weight")};
      },
      py::arg("session"), py::arg("tensor"),
      py::arg("engines") = std::vector<Engine>{Engine::PNM, Engine::PIM},
      py::arg("max_resident_bytes") = size_t(-1));
  py::class_<CacheHandle>(m, "KVCache")
      .def(py::init([](std::shared_ptr<Owner> s, uint64_t capacity, uint64_t heads, uint64_t dim) {
             s->check("KVCache");
             return CacheHandle{
                 s, take(KVCache::create(s->native, {capacity, heads, dim}), "KVCache")};
           }),
           py::arg("session"), py::arg("capacity_tokens"), py::arg("kv_heads"), py::arg("head_dim"))
      .def("scheduled_length",
           [](const CacheHandle &c) {
             c.owner->check("scheduled_length");
             return c.native.scheduled_length();
           })
      .def("committed_length",
           [](const CacheHandle &c) {
             c.owner->check("committed_length");
             return c.native.committed_length();
           })
      .def("poisoned",
           [](const CacheHandle &c) {
             c.owner->check("poisoned");
             return c.native.poisoned();
           })
      .def("append",
           [](CacheHandle &c, const TensorHandle &k, const TensorHandle &v) {
             check(c.owner, k, "cache.append");
             check(c.owner, v, "cache.append");
             return EventHandle{c.owner, take(c.native.append(c.owner->native, k.native, v.native),
                                              "cache.append")};
           })
      .def("reset",
           [](CacheHandle &c) {
             c.owner->check("cache.reset");
             return EventHandle{c.owner, take(c.native.reset(c.owner->native), "cache.reset")};
           })
      .def(
          "attention",
          [](CacheHandle &c, const TensorHandle &q, uint64_t start, OpOptions options) {
            check(c.owner, q, "cache.attention");
            return TensorHandle{c.owner,
                                take(c.native.attention(c.owner->native, q.native, start, options),
                                     "cache.attention")};
          },
          py::arg("q"), py::arg("query_start"), py::arg("options") = OpOptions{});
  m.def(
      "rope",
      [](std::shared_ptr<Owner> s, const TensorHandle &x, const TensorHandle &sin,
         const TensorHandle &cos, std::vector<uint64_t> positions, OpOptions options) {
        check(s, x, "rope");
        check(s, sin, "rope");
        check(s, cos, "rope");
        return TensorHandle{s, take(ndp::rope(s->native, x.native, sin.native, cos.native,
                                              std::move(positions), options),
                                    "rope")};
      },
      py::arg("session"), py::arg("x"), py::arg("sin"), py::arg("cos"), py::arg("positions"),
      py::arg("options") = OpOptions{});
  m.def(
      "causal_attention",
      [](std::shared_ptr<Owner> s, const TensorHandle &q, const TensorHandle &k,
         const TensorHandle &v, uint64_t start, OpOptions options) {
        check(s, q, "causal_attention");
        check(s, k, "causal_attention");
        check(s, v, "causal_attention");
        return TensorHandle{
            s, take(ndp::causal_attention(s->native, q.native, k.native, v.native, start, options),
                    "causal_attention")};
      },
      py::arg("session"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("query_start"),
      py::arg("options") = OpOptions{});
}
} // namespace ndp::python
