// Forward generic L2 operations without implicit waits or per-operation host transfers.
#include "wrappers.hpp"

namespace ndp::python {
void bind_operations(py::module_ &, SessionBinding &cls) {
  cls.def("_validate_tensor", [](const std::shared_ptr<Owner> &s, const TensorHandle &tensor) {
    check(s, tensor, "validate");
    take(s->native.validate(tensor.native), "validate");
  });
  cls.def("linear_partial", [](std::shared_ptr<Owner> s, const TensorHandle &x, const TensorHandle &w, Engine engine) {
    check(s, x, "linear_partial");
    check(s, w, "linear_partial");
    return TensorHandle{s, take(s->native.linear_partial(x.native, w.native, engine), "linear_partial")};
  }, py::arg("x"), py::arg("weight"), py::arg("engine") = Engine::PIM);
  cls.def(
      "matmul",
      [](std::shared_ptr<Owner> s, const TensorHandle &a, const TensorHandle &b,
         OpOptions options, float factor) {
        check(s, a, "matmul");
        check(s, b, "matmul");
        return TensorHandle{s, take(s->native.matmul(a.native, b.native, options, factor), "matmul")};
      },
      py::arg("a"), py::arg("b"), py::arg("options") = OpOptions{}, py::arg("factor") = 1.f);
  cls.def(
      "matvec",
      [](std::shared_ptr<Owner> s, const TensorHandle &a, const TensorHandle &b,
         OpOptions options) {
        check(s, a, "matvec");
        check(s, b, "matvec");
        return TensorHandle{s, take(s->native.matvec(a.native, b.native, options), "matvec")};
      },
      py::arg("a"), py::arg("b"), py::arg("options") = OpOptions{Engine::PIM});
  cls.def(
      "add",
      [](std::shared_ptr<Owner> s, const TensorHandle &a, const TensorHandle &b,
         OpOptions options) {
        check(s, a, "add");
        check(s, b, "add");
        return TensorHandle{s, take(s->native.add(a.native, b.native, options), "add")};
      },
      py::arg("a"), py::arg("b"), py::arg("options") = OpOptions{});
  cls.def(
      "mul",
      [](std::shared_ptr<Owner> s, const TensorHandle &a, const TensorHandle &b,
         OpOptions options) {
        check(s, a, "mul");
        check(s, b, "mul");
        return TensorHandle{s, take(s->native.mul(a.native, b.native, options), "mul")};
      },
      py::arg("a"), py::arg("b"), py::arg("options") = OpOptions{});
  cls.def(
      "linear",
      [](std::shared_ptr<Owner> s, const TensorHandle &x, const TensorHandle &w, OpOptions options,
         const std::optional<TensorHandle> &bias) {
        check(s, x, "linear");
        check(s, w, "linear");
        if (bias) {
          check(s, *bias, "linear");
        }
        return TensorHandle{
            s, take(s->native.linear(x.native, w.native, options, bias ? &bias->native : nullptr),
                    "linear")};
      },
      py::arg("x"), py::arg("weight"), py::arg("options") = OpOptions{},
      py::arg("bias") = py::none());
  cls.def(
      "rms_norm",
      [](std::shared_ptr<Owner> s, const TensorHandle &x, const TensorHandle &gamma, float epsilon,
         OpOptions options) {
        check(s, x, "rms_norm");
        check(s, gamma, "rms_norm");
        return TensorHandle{
            s, take(s->native.rms_norm(x.native, gamma.native, epsilon, options), "rms_norm")};
      },
      py::arg("x"), py::arg("gamma"), py::arg("epsilon") = 1e-5f, py::arg("options") = OpOptions{});
  cls.def(
      "softmax",
      [](std::shared_ptr<Owner> s, const TensorHandle &x, int axis, OpOptions options) {
        check(s, x, "softmax");
        return TensorHandle{s, take(s->native.softmax(x.native, axis, options), "softmax")};
      },
      py::arg("x"), py::arg("axis") = -1, py::arg("options") = OpOptions{});
  cls.def(
      "silu",
      [](std::shared_ptr<Owner> s, const TensorHandle &x, OpOptions options) {
        check(s, x, "silu");
        return TensorHandle{s, take(s->native.silu(x.native, options), "silu")};
      },
      py::arg("x"), py::arg("options") = OpOptions{});
  cls.def(
      "reshape",
      [](std::shared_ptr<Owner> s, const TensorHandle &x, std::vector<uint64_t> shape) {
        check(s, x, "reshape");
        return TensorHandle{s, take(s->native.reshape(x.native, std::move(shape)), "reshape")};
      },
      py::arg("x"), py::arg("shape"));
  cls.def(
      "gather",
      [](std::shared_ptr<Owner> s, const TensorHandle &x, std::vector<uint64_t> indices,
         std::vector<uint64_t> shape) {
        check(s, x, "gather");
        return TensorHandle{
            s, take(s->native.gather(x.native, std::move(indices), std::move(shape)), "gather")};
      },
      py::arg("x"), py::arg("indices"), py::arg("shape"));
  cls.def(
      "scale",
      [](std::shared_ptr<Owner> s, const TensorHandle &x, float factor) {
        check(s, x, "scale");
        return TensorHandle{s, take(s->native.scale(x.native, factor), "scale")};
      },
      py::arg("x"), py::arg("factor"));
  cls.def(
      "transpose",
      [](std::shared_ptr<Owner> s, const TensorHandle &x) {
        check(s, x, "transpose");
        return TensorHandle{s, take(s->native.transpose(x.native), "transpose")};
      },
      py::arg("x"));
  cls.def("query_operation",
          [](Owner &s, const OperationSpec &spec) {
            s.check("query_operation");
            return take(s.native.query_operation(spec), "query_operation");
          })
      .def("query_layout",
           [](Owner &s, const TensorDesc &desc, LayoutKind kind) {
             s.check("query_layout");
             return take(s.native.query_layout(desc, kind), "query_layout");
           })
      .def("prepare_layouts",
           [](std::shared_ptr<Owner> s, const TensorHandle &x,
              const std::vector<LayoutDesc> &layouts) {
             check(s, x, "prepare_layouts");
             py::gil_scoped_release release;
             return TensorHandle{
                 s, take(s->native.prepare_layouts(x.native, layouts), "prepare_layouts")};
           })
      .def("materialize_layout",
           [](std::shared_ptr<Owner> s, const TensorHandle &x, const LayoutDesc &layout) {
             check(s, x, "materialize_layout");
             return EventHandle{
                 s, take(s->native.materialize_layout(x.native, layout), "materialize_layout")};
           });
}
} // namespace ndp::python
