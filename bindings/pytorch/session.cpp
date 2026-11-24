// Select backends, expose lifecycle operations, and translate native runtime metadata.
#include "ndp/l1/ndp/backend.hpp"
#include "ndp/l1/functional/backend.hpp"
#ifdef NDP_PYTHON_GEM5_PIM
#include "ndp/l1/gem5_pim/backend.hpp"
#endif
#include "wrappers.hpp"
#include "ndp/l2/device_group.hpp"

namespace ndp::python {
#ifdef NDP_PYTHON_TESTING
std::unique_ptr<IBackend> make_failure_backend();
#endif
void bind_session(py::module_ &m, SessionBinding &cls) {
  // The Python wrapper retains the original Session objects and their thread guards.
  py::class_<DeviceGroup>(m, "_DeviceGroup")
      .def(py::init([](const std::vector<std::shared_ptr<Owner>> &owners) {
        std::vector<std::shared_ptr<Session>> devices;
        for (const auto &owner : owners) {
          if (!owner) throw Error({ErrorCode::InvalidArgument, "Expected a Session"}, "DeviceGroup");
          owner->check("DeviceGroup");
          devices.emplace_back(owner, &owner->native);
        }
        return take(DeviceGroup::create(std::move(devices)), "DeviceGroup");
      }))
      .def("_reduce_partials", [](const DeviceGroup &group, const std::vector<TensorHandle> &partials,
                                  DeviceId id, const std::shared_ptr<Owner> &destination) {
        if (!destination) throw Error({ErrorCode::InvalidArgument, "Missing destination"}, "reduce_partials");
        destination->check("reduce_partials");
        if (take(group.device(id), "reduce_partials").get() != &destination->native)
          throw Error({ErrorCode::SessionMismatch, "Wrong destination"}, "reduce_partials");
        std::vector<Tensor> tensors;
        for (const auto &partial : partials) {
          partial.owner->check("reduce_partials");
          tensors.push_back(partial.native);
        }
        py::gil_scoped_release release;
        return TensorHandle{destination, take(group.reduce_partials(tensors, id), "reduce_partials")};
      })
      .def("_all_gather_rows", [](const DeviceGroup &group, const std::vector<TensorHandle> &shards) {
        std::vector<Tensor> native;
        for (const auto &shard : shards) {
          shard.owner->check("all_gather_rows");
          native.push_back(shard.native);
        }
        py::gil_scoped_release release;
        auto replicas = take(group.all_gather_rows(native), "all_gather_rows");
        std::vector<TensorHandle> result;
        for (size_t i = 0; i < replicas.size(); ++i)
          result.push_back({shards[i].owner, std::move(replicas[i])});
        return result;
      })
      .def("_broadcast", [](const DeviceGroup &group, const TensorHandle &tensor,
                             const std::vector<std::shared_ptr<Owner>> &owners) {
        tensor.owner->check("broadcast");
        if (owners.size() != group.size()) throw Error({ErrorCode::InvalidArgument, "Missing sessions"}, "broadcast");
        for (size_t i = 0; i < owners.size(); ++i) {
          if (!owners[i]) throw Error({ErrorCode::InvalidArgument, "Missing session"}, "broadcast");
          owners[i]->check("broadcast");
          if (take(group.device(i), "broadcast").get() != &owners[i]->native)
            throw Error({ErrorCode::SessionMismatch, "Wrong session"}, "broadcast");
        }
        py::gil_scoped_release release;
        auto replicas = take(group.broadcast(tensor.native), "broadcast");
        std::vector<TensorHandle> result;
        for (size_t i = 0; i < replicas.size(); ++i)
          result.push_back({owners[i], std::move(replicas[i])});
        return result;
      })
      .def("_copy_to", [](const DeviceGroup &group, const TensorHandle &tensor, DeviceId id,
                          const std::shared_ptr<Owner> &destination) {
        tensor.owner->check("copy_to");
        if (!destination) throw Error({ErrorCode::InvalidArgument, "Expected destination session"}, "copy_to");
        destination->check("copy_to");
        if (take(group.device(id), "copy_to").get() != &destination->native)
          throw Error({ErrorCode::SessionMismatch, "Incorrect destination session"}, "copy_to");
        py::gil_scoped_release release;
        return TensorHandle{destination, take(group.copy_to(tensor.native, id), "copy_to")};
      });
  py::class_<MemoryStats>(m, "MemoryStats")
      .def_readonly("live_bytes", &MemoryStats::live_bytes)
      .def_readonly("peak_bytes", &MemoryStats::peak_bytes)
      .def_readonly("allocations", &MemoryStats::allocations)
      .def_readonly("staging_bytes", &MemoryStats::staging_bytes)
      .def_readonly("workspace_bytes", &MemoryStats::workspace_bytes);
  py::class_<DeviceCaps>(m, "DeviceCaps")
      .def_readonly("name", &DeviceCaps::name)
      .def_readonly("memory_capacity_bytes", &DeviceCaps::memory_capacity_bytes)
      .def_readonly("scratchpad_bytes", &DeviceCaps::scratchpad_bytes);
  cls.def(py::init([](std::string backend, std::optional<std::string> config, size_t capacity,
                      size_t scratchpad, size_t staging, std::optional<std::string> worker, std::optional<int> worker_fd,
                      std::optional<std::string> capture_worker, std::optional<std::string> gemv_elf,
                      std::optional<std::string> pointwise_elf, std::optional<std::string> attention_elf) {
            if ((capture_worker || gemv_elf || pointwise_elf || attention_elf) &&
                (backend != "gem5_pim" || !worker || !capture_worker || !gemv_elf))
              throw Error({ErrorCode::InvalidArgument, "capture requires gem5_pim, worker, capture_worker and gemv_elf"}, "Session");
            if ((worker || worker_fd) && backend != "gem5_pim")
              throw Error({ErrorCode::InvalidArgument, "worker is specific to gem5_pim"}, "Session");
            std::unique_ptr<IBackend> device;
#ifdef NDP_PYTHON_TESTING
            if (backend == "_test_fail_upload")
              device = make_failure_backend();
            else
#endif
                if (backend == "functional" && !config)
              device = make_functional_backend(capacity, scratchpad);
            else if (backend == "ndp") {
              const auto spec = config ? take(load_ndp_spec(*config), "Session") : NdpSpec{};
              device = take(make_ndp_backend(spec), "Session");
            }
            else if (backend == "gem5_pim") {
              if (config || worker.has_value() == worker_fd.has_value() || (worker && worker->empty()))
                throw Error({ErrorCode::InvalidArgument, "gem5_pim requires either worker or worker_fd, and no config"}, "Session");
#ifdef NDP_PYTHON_GEM5_PIM
              py::gil_scoped_release release;
              if (capture_worker)
                device = take(make_gem5_pim_capture_backend(*worker, *capture_worker, *gemv_elf, pointwise_elf.value_or(""), attention_elf.value_or("")), "Session");
              else device = take(worker_fd ? connect_gem5_pim_backend(*worker_fd) : make_gem5_pim_backend(*worker), "Session");
#else
              throw Error({ErrorCode::UnsupportedOperation, "gem5_pim requires POSIX worker support"}, "Session");
#endif
            } else
              throw Error({ErrorCode::InvalidArgument, "Invalid backend or backend configuration"},
                          "Session");
            return std::make_shared<Owner>(
                take(Session::create(std::move(device), {staging}), "Session"));
          }),
          py::arg("backend") = "functional", py::arg("config") = py::none(),
          py::arg("capacity_bytes") = size_t(64 << 20), py::arg("scratchpad_bytes") = 4096,
          py::arg("max_staging_bytes") = size_t(64 << 20), py::kw_only(),
          py::arg("worker") = py::none(), py::arg("worker_fd") = py::none(),
          py::arg("capture_worker") = py::none(), py::arg("gemv_elf") = py::none(), py::arg("pointwise_elf") = py::none(), py::arg("attention_elf") = py::none())
      .def("close",
           [](Owner &s) {
             s.check("close", true);
             s.closed = true;
             py::gil_scoped_release release;
             take(s.native.close(), "close");
           })
      .def("synchronize",
           [](Owner &s) {
             s.check("synchronize");
             py::gil_scoped_release release;
             take(s.native.synchronize(), "synchronize");
           })
      .def("_check", [](Owner &s) { s.check("Session"); })
      .def("memory_stats",
           [](Owner &s) {
             s.check("memory_stats", true);
             return s.native.memory_stats();
           })
      .def("capabilities", [](Owner &s) {
        s.check("capabilities");
        return take(s.native.capabilities(), "capabilities");
      });
}
} // namespace ndp::python
