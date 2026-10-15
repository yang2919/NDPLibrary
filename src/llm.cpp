// Implement L3 cache policy and decoder math using only the public generic Session API.
#include "ndp/llm.hpp"

#include "ndp/fp16.hpp"

#include <cmath>
#include <utility>

namespace ndp {
namespace detail {
struct KVState {
  KVCacheDesc desc;
  DeviceBuffer k, v;
  uint64_t scheduled = 0, base_committed = 0;

  struct Update {
    Event event;
    uint64_t end;
  };

  std::vector<Update> updates;
};
} // namespace detail

Result<KVCache> KVCache::create(Session &session, KVCacheDesc desc) {
  TensorDesc tensor{{desc.capacity_tokens, desc.kv_heads, desc.head_dim}, desc.dtype};
  auto k = session.allocate(tensor);
  if (!k) {
    return k.error();
  }
  auto v = session.allocate(tensor);
  if (!v) {
    return v.error();
  }
  try {
    return KVCache(
        std::make_shared<detail::KVState>(detail::KVState{desc, k.value(), v.value(), 0, 0, {}}));
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Cache metadata allocation failed"};
  }
}

const KVCacheDesc &KVCache::desc() const {
  return state_->desc;
}

uint64_t KVCache::scheduled_length() const {
  return state_->scheduled;
}

uint64_t KVCache::committed_length() const {
  auto length = state_->base_committed;
  for (const auto &update : state_->updates) {
    if (update.event.status() != EventStatus::Succeeded) {
      break;
    }
    length = update.end;
  }
  return length;
}

bool KVCache::poisoned() const {
  if (state_->k.ready_event().status() == EventStatus::Failed ||
      state_->v.ready_event().status() == EventStatus::Failed) {
    return true;
  }
  for (const auto &update : state_->updates) {
    if (update.event.status() == EventStatus::Failed) {
      return true;
    }
  }
  return false;
}

Result<Event> KVCache::append(Session &session, const Tensor &k, const Tensor &v) {
  if (auto valid = session.validate(state_->k); !valid) {
    return valid.error();
  }
  if (auto valid = session.validate(state_->v); !valid) {
    return valid.error();
  }
  if (poisoned()) {
    return Status{ErrorCode::InvalidState, "Reset failed cache before reuse"};
  }
  const auto &shape = k.desc().shape;
  if (shape.size() != 3 || k.desc() != v.desc() || shape[1] != desc().kv_heads ||
      shape[2] != desc().head_dim || k.desc().dtype != desc().dtype) {
    return Status{ErrorCode::InvalidArgument, "KV append shape/dtype mismatch"};
  }
  if (shape[0] > desc().capacity_tokens - state_->scheduled) {
    return Status{ErrorCode::OutOfMemory, "KV capacity exceeded"};
  }
  try {
    state_->updates.reserve(state_->updates.size() + 1);
    const size_t count = byte_size(k.desc()).value() / 2;
    const size_t offset = state_->scheduled * (count / shape[0]);
    auto event =
        session.copy_into({{k, state_->k, 0, offset, count}, {v, state_->v, 0, offset, count}});
    if (!event) {
      return event.error();
    }
    state_->scheduled += shape[0];
    state_->updates.push_back({event.value(), state_->scheduled});
    return event;
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Cache append metadata failed"};
  }
}

Result<Event> KVCache::reset(Session &session) {
  try {
    std::vector<detail::KVState::Update> reset;
    reset.reserve(1);
    const auto committed = committed_length();
    auto event = session.clear({state_->k, state_->v});
    if (!event) {
      return event.error();
    }
    reset.push_back({event.value(), 0});
    state_->updates = std::move(reset);
    state_->base_committed = committed;
    state_->scheduled = 0;
    return event;
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Cache reset metadata failed"};
  }
}

Result<Tensor> KVCache::attention(Session &session, const Tensor &q, uint64_t start,
                                  OpOptions options) {
  if (auto valid = session.validate(state_->k); !valid) {
    return valid.error();
  }
  if (auto valid = session.validate(q); !valid) {
    return valid.error();
  }
  if (poisoned()) {
    return Status{ErrorCode::InvalidState, "Reset failed cache before attention"};
  }
  const auto &shape = q.desc().shape;
  if (shape.size() != 3 || shape[2] != desc().head_dim || shape[1] % desc().kv_heads ||
      start >= state_->scheduled || shape[0] > state_->scheduled - start) {
    return Status{ErrorCode::InvalidArgument, "Invalid attention range or shape"};
  }
  TensorDesc prefix{{state_->scheduled, desc().kv_heads, desc().head_dim}, desc().dtype};
  auto k = session.snapshot(state_->k, prefix);
  if (!k) {
    return k.error();
  }
  auto v = session.snapshot(state_->v, prefix);
  if (!v) {
    return v.error();
  }
  return causal_attention(session, q, k.value(), v.value(), start, options);
}

// Local Result propagation preserves operation errors without adding a public exception API.
namespace {
struct SubmissionFailure {
  Status error;
};

template <class T> T take(Result<T> result) {
  if (!result) {
    throw SubmissionFailure{result.error()};
  }
  return std::move(result.value());
}
} // namespace

Result<Tensor> causal_attention(Session &s, const Tensor &q, const Tensor &k, const Tensor &v,
                                uint64_t start, OpOptions options) {
  for (auto tensor : {&q, &k, &v}) {
    if (auto valid = s.validate(*tensor); !valid) {
      return valid.error();
    }
  }
  const auto &qs = q.desc().shape, &ks = k.desc().shape;
  if (qs.size() != 3 || ks.size() != 3 || k.desc() != v.desc() || qs[2] != ks[2] || qs[1] % ks[1] ||
      start >= ks[0] || qs[0] > ks[0] - start) {
    return Status{ErrorCode::InvalidArgument, "Invalid causal attention dimensions"};
  }
  if (options.engine != Engine::PNM) {
    return Status{ErrorCode::UnsupportedOperation, "Composed attention requires PNM"};
  }
  try {
    auto output = take(s.allocate(q.desc()));
    const uint64_t width = qs[2], heads = qs[1], kv_heads = ks[1];
    std::vector<TensorCopy> writes;
    for (uint64_t token = 0; token < qs[0]; ++token) {
      const uint64_t length = start + token + 1;
      for (uint64_t head = 0; head < heads; ++head) {
        std::vector<uint64_t> qi, ki;
        for (uint64_t d = 0; d < width; ++d) {
          qi.push_back((token * heads + head) * width + d);
        }
        const auto kv_head = head / (heads / kv_heads);
        for (uint64_t t = 0; t < length; ++t) {
          for (uint64_t d = 0; d < width; ++d) {
            ki.push_back((t * kv_heads + kv_head) * width + d);
          }
        }
        auto query = take(s.gather(q, std::move(qi), {1, width}));
        auto keys = take(s.gather(k, ki, {length, width}));
        auto values = take(s.gather(v, std::move(ki), {length, width}));
        auto transposed = take(s.transpose(keys));
        auto scores = take(s.matmul(query, transposed, options));
        auto scaled = take(s.scale(scores, 1 / std::sqrt(float(width))));
        auto probabilities = take(s.softmax(scaled, -1, options));
        auto attended = take(s.matmul(probabilities, values, options));
        writes.push_back({attended, output, 0, (token * heads + head) * width, width});
      }
    }
    take(s.copy_into(std::move(writes)));
    return s.snapshot(output, q.desc());
  } catch (const SubmissionFailure &failure) {
    return failure.error;
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "Attention composition failed"};
  }
}

Result<Tensor> rope(Session &s, const Tensor &x, const Tensor &sin, const Tensor &cos,
                    std::vector<uint64_t> positions, OpOptions options) {
  for (auto tensor : {&x, &sin, &cos}) {
    if (auto valid = s.validate(*tensor); !valid) {
      return valid.error();
    }
  }
  const auto &shape = x.desc().shape, &table = sin.desc().shape;
  if (shape.size() != 3 || shape[2] % 2 || table.size() != 2 || sin.desc() != cos.desc() ||
      table[1] != shape[2] / 2 || positions.size() != shape[0]) {
    return Status{ErrorCode::InvalidArgument, "Invalid RoPE dimensions"};
  }
  for (auto position : positions) {
    if (position >= table[0]) {
      return Status{ErrorCode::InvalidArgument, "RoPE position out of range"};
    }
  }
  if (options.engine != Engine::PNM) {
    return Status{ErrorCode::UnsupportedOperation, "Composed RoPE requires PNM"};
  }
  try {
    const size_t count = byte_size(x.desc()).value() / 2;
    std::vector<uint64_t> partner, tables;
    for (size_t i = 0; i < count; ++i) {
      partner.push_back(i ^ 1);
      auto token = i / (shape[1] * shape[2]);
      tables.push_back(positions[token] * table[1] + (i % shape[2]) / 2);
    }
    auto paired = take(s.gather(x, std::move(partner), shape));
    auto sine = take(s.gather(sin, tables, shape));
    auto cosine = take(s.gather(cos, std::move(tables), shape));
    auto direct = take(s.mul(x, cosine, options));
    auto cross = take(s.mul(paired, sine, options));
    // A last-axis sign vector broadcasts across all tokens and heads.
    std::vector<uint16_t> signs(shape.back());
    for (size_t d = 0; d < signs.size(); ++d) {
      signs[d] = float_to_fp16(d % 2 ? 1.0f : -1.0f);
    }
    auto sign_tensor = take(s.from_host({{{shape.back()}}, std::as_bytes(std::span(signs))}));
    auto signed_cross = take(s.mul(cross, sign_tensor, options));
    return s.add(direct, signed_cross, options);
  } catch (const SubmissionFailure &failure) {
    return failure.error;
  } catch (const std::bad_alloc &) {
    return Status{ErrorCode::OutOfMemory, "RoPE composition failed"};
  }
}
} // namespace ndp
