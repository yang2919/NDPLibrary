"""Bridge a resident Llama block to the Transformers 4.57.6 DynamicCache."""
import weakref

import torch
from transformers.cache_utils import CacheLayerMixin, DynamicCache, DynamicLayer
from transformers.utils.deprecation import deprecate_kwarg

from .llm import KVCache
from .nn import LlamaDecodeBlock, NdpLinear


class _DeviceKVLayer(CacheLayerMixin):
    """Expose sequence metadata while keeping the layer's KV data on NDP."""

    is_sliding = False

    def __init__(self, block):
        self.owner = weakref.ref(block)
        self.kv = None
        self.length = 0
        self.failed = False

    @property
    def is_initialized(self):
        return self.length > 0

    def get_seq_length(self):
        return self.length

    def get_mask_sizes(self, cache_position):
        return self.length + cache_position.shape[0], 0

    def get_max_cache_shape(self):
        return -1

    def invalidate(self):
        self.owner = None
        self.kv = None

    def check(self, block):
        if self.owner is None or self.owner() is not block:
            raise ValueError("Resident KV belongs to a different or closed conversion. Start a fresh request")
        if self.failed:
            raise RuntimeError("Resident KV failed during inference. Start a fresh request")

    def _unsupported(self, *args, **kwargs):
        raise ValueError("Resident KV cannot be exported, updated by a CPU layer, or rearranged. Start a fresh request")

    keys = values = property(_unsupported)
    update = lazy_initialization = _unsupported
    reset = crop = reorder_cache = batch_repeat_interleave = batch_select_indices = _unsupported
    offload = prefetch = _unsupported


class ResidentLlamaBlock(LlamaDecodeBlock):
    """Run one Llama decoder layer with separately owned histories for each request."""

    def __init__(self, source, rotary, *, max_tokens, runtime):
        super().__init__(source, max_tokens=max_tokens, _prefill=True, runtime=runtime,
                         _inv_freq=rotary.inv_freq)
        self.layer_idx = source.self_attn.layer_idx
        self.config = source.self_attn.config
        self.hidden_size = self.width
        self._modules = {name: self._modules[name] for name in source._modules}
        self._histories = weakref.WeakSet()

    def close_requests(self):
        for history in tuple(self._histories):
            history.invalidate()
        self._histories.clear()

    def _cache_layer(self, cache):
        if cache is None:
            return None
        if type(cache) is not DynamicCache or cache.offloading:
            raise ValueError("Resident Llama expects a non-offloaded Transformers DynamicCache")
        if self.layer_idx >= len(cache.layers):
            if cache.layer_class_to_replicate is DynamicLayer:
                return None
        else:
            current = cache.layers[self.layer_idx]
            if type(current) is _DeviceKVLayer:
                current.check(self)
                return current
            if type(current) is DynamicLayer and current.get_seq_length() == 0:
                return None
        raise ValueError("Resident Llama requires a fresh cache at the converted layer")

    def _positions(self, start, rows, cache_position, position_ids):
        if rows < 1 or start + rows > self.max_tokens:
            raise ValueError(f"Resident Llama rows exceed KV capacity {self.max_tokens} at position {start}")
        expected = torch.arange(start, start+rows, device="cpu")
        if ((cache_position is not None and not torch.equal(cache_position, expected))
                or (position_ids is not None and not torch.equal(position_ids, expected.unsqueeze(0)))):
            raise ValueError("Resident Llama requires contiguous cache positions and position IDs")
        return expected

    def check_backbone_input(self, backbone, args, kwargs):
        # Validate before Transformers advances any CPU layer or installs output recorders.
        if backbone.training or backbone.config._attn_implementation != "eager":
            raise ValueError("Resident Llama requires eager attention in eval mode")
        names = ("input_ids", "attention_mask", "position_ids", "past_key_values",
                 "inputs_embeds", "cache_position", "use_cache")
        options = {**dict(zip(names, args)), **kwargs}
        if any(options.get(name, getattr(backbone.config, name, False))
               for name in ("output_attentions", "output_hidden_states")):
            raise ValueError("Resident Llama supports logits and cache metadata as model outputs")
        inputs = options.get("inputs_embeds")
        if inputs is None:
            inputs = options.get("input_ids")
        if inputs is None or inputs.ndim < 2 or inputs.shape[0] != 1:
            raise ValueError("Resident Llama requires one unpadded sequence")
        cache = options.get("past_key_values")
        history = self._cache_layer(cache)
        start = history.length if history is not None else 0
        if cache is not None and cache.get_seq_length() != start:
            raise ValueError("Resident Llama cache positions differ between layers. Start a fresh request")
        rows = inputs.shape[1]
        self._positions(start, rows, options.get("cache_position"), options.get("position_ids"))
        mask = options.get("attention_mask")
        if mask is not None and (mask.shape != (1, start+rows) or not torch.all(mask == 1)):
            raise ValueError("Resident Llama requires an unpadded sequence with standard causal masking")

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        NdpLinear._validate(hidden_states, "hidden states")
        if (hidden_states.ndim != 3 or hidden_states.shape[0] != 1
                or hidden_states.shape[2] != self.width or hidden_states.requires_grad):
            raise ValueError("Resident Llama expects inference input [1, sequence, hidden_size]")
        history = self._cache_layer(past_key_values)
        start = history.length if history is not None else 0
        rows = hidden_states.shape[1]
        positions = self._positions(start, rows, cache_position, position_ids)
        end = start + rows
        if attention_mask is not None:
            if attention_mask.shape != (1, 1, rows, end) or not attention_mask.is_floating_point():
                raise ValueError("Resident Llama expects a standard additive causal mask")
            visible = torch.arange(end, device="cpu") <= positions[:, None]
            mask = attention_mask[0, 0]
            if not torch.all(torch.where(visible, mask == 0, mask <= torch.finfo(mask.dtype).min)):
                raise ValueError("Resident Llama requires an unpadded sequence with standard causal masking")
        if (position_embeddings is None or len(position_embeddings) != 2
                or any(not torch.equal(value, table[start:end].unsqueeze(0))
                       for value, table in zip(position_embeddings, self._host_rope, strict=True))):
            raise ValueError("Resident Llama requires the original split-half RoPE embeddings")

        if history is None:
            history = _DeviceKVLayer(self)
            if past_key_values is not None:
                while len(past_key_values.layers) <= self.layer_idx:
                    past_key_values.layers.append(DynamicLayer())
                past_key_values.layers[self.layer_idx] = history
                self._histories.add(history)
        try:
            if history.kv is None:
                capacity = self.max_tokens if past_key_values is not None else rows
                history.kv = KVCache(self.session, capacity, self.kv_heads, self.dim)
            resident = self.session.upload(hidden_states.reshape(rows, self.width))
            qkv = self._chunk_qkv(resident, start)
            output = self._finish_chunk(resident, history.kv, start, qkv)
            result = self.session.download(output).reshape_as(hidden_states)
        except BaseException:
            history.failed = True
            history.kv = None
            raise
        history.length = end
        return result
