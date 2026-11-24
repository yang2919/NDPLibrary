// Register Python types and preserve structured runtime errors at the adapter boundary.
#include "wrappers.hpp"

namespace ndp::python {
namespace {
PyObject *error_type = nullptr;
}

PYBIND11_MODULE(_C, m) {
#ifdef NDP_PYTHON_TESTING
  m.attr("_testing_enabled") = true;
#endif
  error_type = PyErr_NewException("ndp.NdpError", PyExc_RuntimeError, nullptr);
  m.add_object("NdpError", py::reinterpret_steal<py::object>(error_type));
  py::register_local_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const Error &e) {
      auto instance = py::reinterpret_borrow<py::object>(error_type)(e.what());
      instance.attr("code") = py::cast(e.code);
      instance.attr("operation") = e.operation;
      PyErr_SetObject(error_type, instance.ptr());
    }
  });
  py::enum_<Engine>(m, "Engine").value("PNM", Engine::PNM).value("PIM", Engine::PIM);
  py::enum_<DType>(m, "DType")
      .value("F16", DType::F16)
      .value("F32", DType::F32)
      .value("BF16", DType::BF16);
  py::enum_<LayoutPolicy>(m, "LayoutPolicy")
      .value("AllowMaterialize", LayoutPolicy::AllowMaterialize)
      .value("RequirePrepared", LayoutPolicy::RequirePrepared);
  py::enum_<LayoutKind>(m, "LayoutKind")
      .value("DenseRowMajor", LayoutKind::DenseRowMajor)
      .value("TestTile4x4", LayoutKind::TestTile4x4)
      .value("PimBanked", LayoutKind::PimBanked);
  py::enum_<EventStatus>(m, "EventStatus")
      .value("Pending", EventStatus::Pending)
      .value("Running", EventStatus::Running)
      .value("Succeeded", EventStatus::Succeeded)
      .value("Failed", EventStatus::Failed);
  py::enum_<OpKind>(m, "OpKind")
      .value("Matmul", OpKind::Matmul)
      .value("Matvec", OpKind::Matvec)
      .value("Linear", OpKind::Linear)
      .value("Add", OpKind::Add)
      .value("Mul", OpKind::Mul)
      .value("RMSNorm", OpKind::RMSNorm)
      .value("Softmax", OpKind::Softmax)
      .value("SiLU", OpKind::SiLU)
      .value("Reshape", OpKind::Reshape)
      .value("Gather", OpKind::Gather)
      .value("Scale", OpKind::Scale)
      .value("Transpose", OpKind::Transpose);
  py::enum_<ErrorCode>(m, "ErrorCode")
      .value("InvalidArgument", ErrorCode::InvalidArgument)
      .value("UnsupportedOperation", ErrorCode::UnsupportedOperation)
      .value("LayoutNotPrepared", ErrorCode::LayoutNotPrepared)
      .value("UnsupportedDType", ErrorCode::UnsupportedDType)
      .value("UnsupportedLayout", ErrorCode::UnsupportedLayout)
      .value("OutOfMemory", ErrorCode::OutOfMemory)
      .value("InvalidState", ErrorCode::InvalidState)
      .value("BackendFailure", ErrorCode::BackendFailure)
      .value("DependencyFailed", ErrorCode::DependencyFailed)
      .value("SessionClosed", ErrorCode::SessionClosed)
      .value("SessionMismatch", ErrorCode::SessionMismatch);
  py::class_<TensorDesc>(m, "TensorDesc")
      .def(py::init<std::vector<uint64_t>, DType>(), py::arg("shape"),
           py::arg("dtype") = DType::F16)
      .def_readonly("shape", &TensorDesc::shape)
      .def_readonly("dtype", &TensorDesc::dtype);
  py::class_<OpOptions>(m, "OpOptions")
      .def(py::init<Engine, LayoutPolicy>(), py::arg("engine") = Engine::PNM,
           py::arg("layout_policy") = LayoutPolicy::AllowMaterialize)
      .def_readonly("engine", &OpOptions::engine)
      .def_readonly("layout_policy", &OpOptions::layout_policy);
  py::class_<OpAttributes>(m, "OpAttributes")
      .def(py::init<>())
      .def_readwrite("epsilon", &OpAttributes::epsilon)
      .def_readwrite("axis", &OpAttributes::axis)
      .def_readwrite("indices", &OpAttributes::indices)
      .def_readwrite("factor", &OpAttributes::factor)
      .def_readwrite("shape", &OpAttributes::shape);
  py::class_<OperationSpec>(m, "OperationSpec")
      .def(py::init<OpKind, Engine, std::vector<TensorDesc>, OpAttributes>(), py::arg("kind"),
           py::arg("engine"), py::arg("inputs"), py::arg("attributes") = OpAttributes{})
      .def_readonly("kind", &OperationSpec::kind)
      .def_readonly("engine", &OperationSpec::engine)
      .def_readonly("inputs", &OperationSpec::inputs);
  py::class_<LayoutDesc>(m, "LayoutDesc")
      .def_readonly("kind", &LayoutDesc::kind)
      .def_readonly("logical", &LayoutDesc::logical)
      .def_readonly("physical_bytes", &LayoutDesc::physical_bytes)
      .def_readonly("alignment", &LayoutDesc::alignment)
      .def_readonly("format", &LayoutDesc::format);
  py::class_<KernelPlan>(m, "KernelPlan")
      .def_readonly("operation", &KernelPlan::operation)
      .def_readonly("kernel_id", &KernelPlan::kernel_id)
      .def_readonly("inputs", &KernelPlan::inputs)
      .def_readonly("output", &KernelPlan::output)
      .def_readonly("workspace_bytes", &KernelPlan::workspace_bytes)
      .def_readonly("tile_k", &KernelPlan::tile_k)
      .def_readonly("accumulator", &KernelPlan::accumulator);
  py::class_<RepresentationInfo>(m, "RepresentationInfo")
      .def_readonly("layout", &RepresentationInfo::layout)
      .def_readonly("status", &RepresentationInfo::status);
  SessionBinding cls(m, "_Session");
  bind_session(m, cls);
  bind_tensors(m, cls);
  bind_operations(m, cls);
  bind_llm(m);
  bind_dispatch(m);
}
} // namespace ndp::python
