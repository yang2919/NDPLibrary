// Compose L3 request caches, rotary embeddings, and causal attention from public L2 APIs.
#pragma once
#include "ndp/session.hpp"

namespace ndp {
namespace detail {
struct KVState;
}

struct KVCacheDesc {
  uint64_t capacity_tokens, kv_heads, head_dim;
  DType dtype = DType::F16;
};

class KVCache {
public:
  static Result<KVCache> create(Session &, KVCacheDesc);
  uint64_t scheduled_length() const;
  uint64_t committed_length() const;
  bool poisoned() const;
  const KVCacheDesc &desc() const;
  Result<Event> append(Session &, const Tensor &k, const Tensor &v);
  Result<Event> reset(Session &);
  Result<Tensor> attention(Session &, const Tensor &q, uint64_t query_start, OpOptions = {});

private:
  explicit KVCache(std::shared_ptr<detail::KVState> state) : state_(std::move(state)) {}

  std::shared_ptr<detail::KVState> state_;
};

Result<Tensor> rope(Session &, const Tensor &, const Tensor &sin, const Tensor &cos,
                    std::vector<uint64_t> positions, OpOptions = {});
// This implementation composes FP16 intermediate tensors; it is not a fused kernel.
Result<Tensor> causal_attention(Session &, const Tensor &q, const Tensor &k, const Tensor &v,
                                uint64_t query_start, OpOptions = {});
} // namespace ndp
