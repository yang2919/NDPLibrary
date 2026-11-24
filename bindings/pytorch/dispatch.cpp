// Register the inference-only PIM projection and bridge prepared runtime weights.
#include "wrappers.hpp"
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <torch/custom_class.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace ndp::python {
namespace {
struct Packed : torch::CustomClassHolder {
  explicit Packed(WeightHandle w) : weight(std::move(w)) {}
  WeightHandle weight;

  int64_t width(size_t axis) const { return weight.native.tensor().desc().shape.at(axis); }
  std::tuple<int64_t, int64_t> dimensions() const { return {width(1), width(0)}; }
  auto flatten() const {
    return std::make_tuple(std::make_tuple(std::string("inputs"), width(1)),
                           std::make_tuple(std::string("outputs"), width(0)));
  }
};

using Params = c10::intrusive_ptr<Packed>;

TensorHandle upload_vector(const at::Tensor &input, const Params &packed) {
  packed->weight.owner->check("linear_pim");
  TORCH_CHECK(input.dim() >= 1 && input.dim() <= 3 && input.size(-1) == packed->width(1)
              && input.numel() == packed->width(1), "Expected one input vector with matching width");
  return upload(packed->weight.owner, input.reshape({1, packed->width(1)}));
}

at::Tensor project(const TensorHandle &x, const Params &packed, std::vector<int64_t> shape) {
  RECORD_FUNCTION("ndp::pim_projection", std::vector<c10::IValue>());
  auto &session = x.owner->native;
  auto y = take(session.linear(x.native, packed->weight.native.tensor(),
                               {Engine::PIM, LayoutPolicy::RequirePrepared}), "linear_pim");
  shape.back() = packed->width(0);
  return to_torch(take(session.to_host(y), "linear_pim")).reshape(shape);
}

at::Tensor linear(const at::Tensor &input, const Params &packed) {
  return project(upload_vector(input, packed), packed, input.sizes().vec());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv(const at::Tensor &input,
                                                  const Params &q, const Params &k, const Params &v) {
  TORCH_CHECK(q->weight.owner == k->weight.owner && q->weight.owner == v->weight.owner,
              "Q/K/V weights must share a Session");
  TORCH_CHECK(q->width(1) == k->width(1) && q->width(1) == v->width(1),
              "Q/K/V input widths must match");
  auto x = upload_vector(input, q);
  auto shape = input.sizes().vec();
  return {project(x, q, shape), project(x, k, shape), project(x, v, shape)};
}

TORCH_LIBRARY(ndp, m) {
  m.class_<Packed>("LinearPackedParams").def("__obj_flatten__", &Packed::flatten)
      .def("dimensions", &Packed::dimensions)
      // SchemaCheckMode probes arbitrary arguments with len(). Signal an unsized object.
      .def("__len__", [](const c10::intrusive_ptr<Packed> &) -> int64_t {
        TORCH_CHECK_TYPE(false, "LinearPackedParams has no sequence length");
      })
      .def("__eq__", [](const c10::intrusive_ptr<Packed> &self, c10::IValue other) {
        return other.isCustomClass() && other.toObjectRef().getSlot(0).toCapsule().get() == self.get();
      });
  m.def("qkv_pim(Tensor input, __torch__.torch.classes.ndp.LinearPackedParams q, "
        "__torch__.torch.classes.ndp.LinearPackedParams k, __torch__.torch.classes.ndp.LinearPackedParams v) "
        "-> (Tensor, Tensor, Tensor)");
  m.def("linear_pim(Tensor input, __torch__.torch.classes.ndp.LinearPackedParams packed) -> Tensor");
}
TORCH_LIBRARY_IMPL(ndp, CPU, m) {
  m.impl("qkv_pim", TORCH_FN(qkv));
  m.impl("linear_pim", TORCH_FN(linear));
}
} // namespace

void bind_dispatch(py::module_ &m) {
  m.def("_pack_linear", [](const WeightHandle &weight) {
    weight.owner->check("pack_linear");
    return torch::jit::toPyObject(c10::IValue(c10::make_intrusive<Packed>(weight)));
  });
}
} // namespace ndp::python
