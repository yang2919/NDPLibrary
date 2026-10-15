// Expose completion status and waiting for asynchronous session work.
#pragma once
#include "ndp/types.hpp"

#include <memory>

namespace ndp {
namespace detail {
struct EventState;
struct State;
} // namespace detail
enum class EventStatus { Pending, Running, Succeeded, Failed };

class Event {
public:
  Event(); // Already complete; submitted work gets a pending event.
  EventStatus status() const;
  Result<void> wait() const;

private:
  std::shared_ptr<detail::EventState> state_;
  friend struct detail::State;
};
} // namespace ndp
