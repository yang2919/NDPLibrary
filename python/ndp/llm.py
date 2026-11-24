"""L3 weight preparation, rotary embeddings, and per-request cache policies."""
from ._C import PreparedWeight, KVCache, prepare_weight, rope, causal_attention
