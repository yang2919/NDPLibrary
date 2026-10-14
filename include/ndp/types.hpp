#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ndp {
enum class ErrorCode { InvalidArgument, UnsupportedDType, OutOfMemory,
                       BackendFailure, DependencyFailed, SessionClosed, SessionMismatch };
struct Status { ErrorCode code; std::string message; };
template<class T> class Result {
 public:
  Result(T value) : data_(std::move(value)) {}
  Result(Status error) : data_(std::move(error)) {}
  explicit operator bool() const { return std::holds_alternative<T>(data_); }
  T& value() { return std::get<T>(data_); }
  const T& value() const { return std::get<T>(data_); }
  const Status& error() const { return std::get<Status>(data_); }
 private:
  std::variant<T, Status> data_;
};
template<> class Result<void> {
 public:
  Result() = default;
  Result(Status error) : error_(std::move(error)) {}
  explicit operator bool() const { return std::holds_alternative<std::monostate>(error_); }
  const Status& error() const { return std::get<Status>(error_); }
 private:
  std::variant<std::monostate, Status> error_;
};
enum class DType { F16, F32, BF16 };
enum class Engine { PNM, PIM };
enum class MemorySpace { PimDram, Scratchpad };
struct TensorDesc { std::vector<uint64_t> shape; DType dtype = DType::F16; };
Result<size_t> byte_size(const TensorDesc&);
struct MemoryStats { size_t live_bytes = 0, peak_bytes = 0, allocations = 0;
  size_t staging_bytes = 0, peak_staging_bytes = 0; };
} // namespace ndp
