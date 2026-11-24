// Own native handles and enforce the single-submitting-thread Python contract.
#pragma once
#include "ndp/l2/session.hpp"
#include "ndp/l3/llm.hpp"
#include "ndp/l3/weight.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>

namespace at { class Tensor; }

namespace ndp::python {
namespace py = pybind11;

struct Error : std::runtime_error {
  ErrorCode code;
  std::string operation;

  Error(Status status, const char *op)
      : std::runtime_error(std::string(op) + ": " + status.message), code(status.code),
        operation(op) {}
};

template <class T> T take(Result<T> result, const char *op) {
  if (!result) {
    throw Error(result.error(), op);
  }
  return std::move(result.value());
}

inline void take(Result<void> result, const char *op) {
  if (!result) {
    throw Error(result.error(), op);
  }
}

struct Owner {
  Session native;
  std::thread::id thread = std::this_thread::get_id();
  bool closed = false;

  explicit Owner(Session session) : native(std::move(session)) {}

  void check(const char *op, bool allow_closed = false) const {
    if (thread != std::this_thread::get_id()) {
      throw Error({ErrorCode::InvalidState, "Session must be used on its creating thread"}, op);
    }
    if (closed && !allow_closed) {
      throw Error({ErrorCode::SessionClosed, "Session is closed"}, op);
    }
  }
};

template <class T> struct Handle {
  std::shared_ptr<Owner> owner;
  T native;
};

using TensorHandle = Handle<Tensor>;
using BufferHandle = Handle<DeviceBuffer>;
using EventHandle = Handle<Event>;
using TransferHandle = Handle<HostTransfer>;
using WeightHandle = Handle<PreparedWeight>;
using CacheHandle = Handle<KVCache>;

struct Copy {
  TensorHandle source;
  BufferHandle destination;
  size_t source_offset, destination_offset, count;
};

template <class T>
void check(const std::shared_ptr<Owner> &owner, const Handle<T> &h, const char *op) {
  owner->check(op);
  if (h.owner != owner) {
    throw Error({ErrorCode::SessionMismatch, "Objects belong to different Sessions"}, op);
  }
}

using SessionBinding = py::class_<Owner, std::shared_ptr<Owner>>;
void bind_session(py::module_ &, SessionBinding &);
void bind_tensors(py::module_ &, SessionBinding &);
void bind_operations(py::module_ &, SessionBinding &);
at::Tensor to_torch(HostTensor host);
TensorHandle upload(const std::shared_ptr<Owner> &, const at::Tensor &);
void bind_dispatch(py::module_ &);
void bind_llm(py::module_ &);
} // namespace ndp::python
