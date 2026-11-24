// Copy CPU FP16 PyTorch tensors at transfer boundaries and retain native handle owners.
#include "wrappers.hpp"

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <cstring>
#include <limits>
#include <torch/csrc/utils/pybind.h>

namespace ndp::python {
at::Tensor to_torch(HostTensor host) {
  RECORD_FUNCTION("ndp::download", std::vector<c10::IValue>());
  std::vector<int64_t> shape;
  for (auto dim : host.desc.shape) {
    if (dim > uint64_t(INT64_MAX)) {
      throw Error({ErrorCode::InvalidArgument, "Shape exceeds int64"}, "download");
    }
    shape.push_back(static_cast<int64_t>(dim));
  }
  auto out = at::empty(shape, at::TensorOptions().dtype(host.desc.dtype == DType::F32 ? at::kFloat : at::kHalf).device(at::kCPU));
  std::memcpy(out.data_ptr(), host.bytes.data(), host.bytes.size());
  return out;
}

TensorHandle upload(const std::shared_ptr<Owner> &s, const at::Tensor &input) {
  s->check("upload");
  if (!input.device().is_cpu() || input.layout() != at::kStrided ||
      input.is_quantized() || (input.scalar_type() != at::kHalf && input.scalar_type() != at::kFloat) || input.requires_grad()) {
    throw Error({ErrorCode::InvalidArgument,
       "Expected CPU strided float16/float32 Tensor without requires_grad"},
      "upload");
  }
  auto contiguous = input.resolve_conj().resolve_neg().contiguous();
  TensorDesc desc;
  desc.dtype = input.scalar_type() == at::kFloat ? DType::F32 : DType::F16;
  for (auto dim : contiguous.sizes()) {
    desc.shape.push_back(uint64_t(dim));
  }
  auto bytes = take(byte_size(desc), "upload");
  auto data =
      std::span(static_cast<const std::byte *>(contiguous.const_data_ptr()), bytes);
  RECORD_FUNCTION("ndp::upload", std::vector<c10::IValue>());
  // from_host owns a staging copy before this method returns.
  return TensorHandle{s, take(s->native.from_host({desc, data}), "upload")};
}

void bind_tensors(py::module_ &m, SessionBinding &cls) {
  py::class_<EventHandle>(m, "Event")
      .def_property_readonly("status",
                             [](const EventHandle &h) {
                               h.owner->check("status", true);
                               return h.native.status();
                             })
      .def("wait", [](const EventHandle &h) {
        h.owner->check("wait", true);
        py::gil_scoped_release release;
        take(h.native.wait(), "wait");
      });
  py::class_<TensorHandle>(m, "NdpTensor")
      .def_property_readonly("shape",
                             [](const TensorHandle &h) {
                               h.owner->check("shape", true);
                               return h.native.desc().shape;
                             })
      .def_property_readonly("dtype",
                             [](const TensorHandle &h) {
                               h.owner->check("dtype", true);
                               return h.native.desc().dtype;
                             })
      .def_property_readonly("desc",
                             [](const TensorHandle &h) {
                               h.owner->check("desc", true);
                               return h.native.desc();
                             })
      .def("representations",
           [](const TensorHandle &h) {
             h.owner->check("representations", true);
             return h.native.representations();
           })
      .def("ready_event",
           [](const TensorHandle &h) {
             h.owner->check("ready_event", true);
             return EventHandle{h.owner, h.native.ready_event()};
           })
      .def("wait", [](const TensorHandle &h) {
        h.owner->check("wait", true);
        py::gil_scoped_release release;
        take(h.native.wait(), "wait");
      });
  py::class_<BufferHandle>(m, "DeviceBuffer")
      .def_property_readonly("desc",
                             [](const BufferHandle &h) {
                               h.owner->check("desc", true);
                               return h.native.desc();
                             })
      .def("ready_event", [](const BufferHandle &h) {
        h.owner->check("ready_event", true);
        return EventHandle{h.owner, h.native.ready_event()};
      });
  py::class_<TransferHandle>(m, "HostTransfer")
      .def("ready_event",
           [](const TransferHandle &h) {
             h.owner->check("ready_event", true);
             return EventHandle{h.owner, h.native.ready_event()};
           })
      .def("wait", [](const TransferHandle &h) {
        h.owner->check("download.wait", true);
        py::gil_scoped_release release;
        return to_torch(take(h.native.wait(), "download.wait"));
      });
  py::class_<Copy>(m, "TensorCopy")
      .def(py::init<TensorHandle, BufferHandle, size_t, size_t, size_t>(), py::arg("source"),
           py::arg("destination"), py::arg("source_offset"), py::arg("destination_offset"),
           py::arg("count"));
  cls.def("upload", &upload)
      .def(
          "download",
          [](std::shared_ptr<Owner> s, const TensorHandle &h, LayoutKind source) {
            check(s, h, "download");
            py::gil_scoped_release release;
            return to_torch(take(s->native.to_host(h.native, source), "download"));
          },
          py::arg("tensor"), py::arg("source") = LayoutKind::DenseRowMajor)
      .def(
          "download_async",
          [](std::shared_ptr<Owner> s, const TensorHandle &h, LayoutKind source) {
            check(s, h, "download_async");
            return TransferHandle{
                s, take(s->native.to_host_async(h.native, source), "download_async")};
          },
          py::arg("tensor"), py::arg("source") = LayoutKind::DenseRowMajor)
      .def("allocate",
           [](std::shared_ptr<Owner> s, TensorDesc desc) {
             s->check("allocate");
             return BufferHandle{s, take(s->native.allocate(desc), "allocate")};
           })
      .def("copy_into",
           [](std::shared_ptr<Owner> s, const std::vector<Copy> &copies) {
             s->check("copy_into");
             std::vector<TensorCopy> native;
             for (auto &c : copies) {
               check(s, c.source, "copy_into");
               check(s, c.destination, "copy_into");
               native.push_back({c.source.native, c.destination.native, c.source_offset,
                                 c.destination_offset, c.count});
             }
             return EventHandle{s, take(s->native.copy_into(std::move(native)), "copy_into")};
           })
      .def(
          "snapshot",
          [](std::shared_ptr<Owner> s, const BufferHandle &h, TensorDesc desc, size_t offset) {
            check(s, h, "snapshot");
            return TensorHandle{s, take(s->native.snapshot(h.native, desc, offset), "snapshot")};
          },
          py::arg("buffer"), py::arg("desc"), py::arg("offset") = 0)
      .def("clear", [](std::shared_ptr<Owner> s, const std::vector<BufferHandle> &buffers) {
        s->check("clear");
        std::vector<DeviceBuffer> native;
        for (auto &b : buffers) {
          check(s, b, "clear");
          native.push_back(b.native);
        }
        return EventHandle{s, take(s->native.clear(std::move(native)), "clear")};
      });
}
} // namespace ndp::python
