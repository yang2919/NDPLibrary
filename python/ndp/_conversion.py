"""Discover and temporarily replace supported projections in a PyTorch model."""
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from threading import Lock
from types import MethodType

import torch
from torch.nn.modules import module as torch_module

from . import NdpError, Session, nn


@dataclass(frozen=True)
class _PlanEntry:
    path: str
    rule: str | None
    reason: str | None = None


class _ConvertedLinear(nn.Linear):
    # Match torch.nn.Linear's call signature, including calls using input=.
    def forward(self, input):
        return super().forward(input)


_active_models = set()
_conversion_lock = Lock()
_LLAMA_MODELING_MODULE = "transformers.models.llama.modeling_llama"
_QKV_PROJECTIONS = ("q_proj", "k_proj", "v_proj")
_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _storage_address(tensor):
    if tensor is None or tensor.device.type != "cpu" or tensor.layout != torch.strided:
        return None
    try:
        return tensor.untyped_storage().data_ptr() or None
    except (RuntimeError, ValueError):
        return None


def _has_custom_forward(module):
    return ("forward" in vars(module) or module._compiled_call_impl is not None
            or hasattr(module, "_hf_hook"))


def _has_hooks(module):
    return (module._forward_pre_hooks or module._forward_hooks
            or module._backward_pre_hooks or module._backward_hooks
            or module._state_dict_pre_hooks or module._state_dict_hooks
            or module._load_state_dict_pre_hooks or module._load_state_dict_post_hooks)


def _original_forward(cls, module_name):
    forward = cls.forward
    return (getattr(forward, "__module__", None) == module_name
            and getattr(forward, "__qualname__", None) == f"{cls.__name__}.forward")


def _skip_reason(path, module, aliases, storage_uses):
    if type(module) is not torch.nn.Linear:
        return "expected an unmodified torch.nn.Linear"
    if not path:
        return "the root model object cannot be replaced"
    if aliases[id(module)] > 1:
        return "shared module"
    if module.training or module.bias is not None:
        return "expected an eval-mode, bias-free Linear"
    if _has_custom_forward(module) or _has_hooks(module):
        return "custom forward or registered hooks"
    if module._buffers or module._modules or set(module._parameters) != {"weight", "bias"}:
        return "extra module state"
    weight = module.weight
    address = _storage_address(weight)
    if (type(weight) is not torch.nn.Parameter or weight.dtype != torch.float16
            or weight.shape != (module.out_features, module.in_features) or not address):
        return "expected a materialized CPU FP16 Linear weight"
    if storage_uses[address] > 1:
        return "shared weight storage"
    return None


def _llama_qkv_reason(path, attention, selected):
    from transformers import __version__
    from transformers.models.llama.modeling_llama import LlamaAttention

    if __version__ != "4.57.6" or type(attention) is not LlamaAttention:
        return "shared Q/K/V requires Transformers 4.57.6 LlamaAttention"
    if (attention.config._attn_implementation != "eager" or attention.training or _has_custom_forward(attention)
            or not _original_forward(LlamaAttention, _LLAMA_MODELING_MODULE)):
        return "shared Q/K/V requires an unmodified eager attention forward in eval mode"
    prefix = f"{path}." if path else ""
    if any(prefix + name not in selected for name in _QKV_PROJECTIONS):
        return "shared Q/K/V requires all three eligible projections in the target"
    return None


def _shared_llama_forward(attention):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward
    from transformers.utils.deprecation import deprecate_kwarg

    original_forward = attention.forward

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, cache_position=None, **kwargs):
        projections = (self.q_proj, self.k_proj, self.v_proj)
        # The shared operator bypasses the projection modules' forward hooks.
        if (hidden_states.ndim != 3 or hidden_states.shape[:2] != (1, 1)
                or self.config._attn_implementation != "eager" or self.training
                or torch_module._global_forward_pre_hooks or torch_module._global_forward_hooks
                or any(p._forward_pre_hooks or p._forward_hooks for p in projections)):
            return original_forward(hidden_states, position_embeddings, attention_mask,
                                    past_key_values, cache_position, **kwargs)

        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        q, k, v = (p.view(shape).transpose(1, 2) for p in nn.project_qkv(hidden_states, *projections))
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx,
                                          {"sin": sin, "cos": cos, "cache_position": cache_position})
        output, weights = eager_attention_forward(self, q, k, v, attention_mask,
                                                  dropout=0.0, scaling=self.scaling, **kwargs)
        output = output.reshape(*hidden_states.shape[:-1], -1).contiguous()
        return self.o_proj(output), weights

    return MethodType(forward, attention)


def _llama_mlp_reason(path, module, aliases, storage_uses):
    cls = type(module)
    if cls.__module__ != _LLAMA_MODELING_MODULE or cls.__name__ != "LlamaMLP":
        return "resident conversion requires a Llama SwiGLU MLP target"
    if not path:
        return "the root model object cannot be replaced"

    from transformers import __version__
    from transformers.activations import SiLUActivation
    from transformers.models.llama.modeling_llama import LlamaMLP

    if __version__ != "4.57.6" or cls is not LlamaMLP:
        return "resident SwiGLU requires Transformers 4.57.6 LlamaMLP"
    if (module.training or aliases[id(module)] > 1 or _has_custom_forward(module) or _has_hooks(module)
            or not _original_forward(LlamaMLP, _LLAMA_MODELING_MODULE)):
        return "resident SwiGLU requires an unmodified, unshared MLP in eval mode"
    if (module._parameters or module._buffers or set(module._modules) != {*_MLP_PROJECTIONS, "act_fn"}):
        return "unexpected Llama MLP state"
    activation = module.act_fn
    if (type(activation) is not SiLUActivation or _has_custom_forward(activation) or _has_hooks(activation)
            or activation._parameters or activation._buffers or activation._modules
            or not _original_forward(SiLUActivation, "transformers.activations")):
        return "resident SwiGLU requires an unmodified SiLU activation"
    if torch_module._global_forward_pre_hooks or torch_module._global_forward_hooks:
        return "resident SwiGLU cannot bypass global forward hooks"
    for name in _MLP_PROJECTIONS:
        reason = _skip_reason(f"{path}.{name}", getattr(module, name), aliases, storage_uses)
        if reason is not None:
            return f"{name}: {reason}"
    gate, up, down = (getattr(module, name) for name in _MLP_PROJECTIONS)
    if (gate.weight.shape != up.weight.shape or down.in_features != gate.out_features
            or down.out_features != gate.in_features):
        return "incompatible Llama SwiGLU projection dimensions"
    return None


def _llama_backbone_path(path):
    layers, _, _ = path.rpartition(".")
    return layers.rpartition(".")[0]


def _llama_block_reason(path, block, modules, aliases, storage_uses):
    from transformers import __version__
    from transformers.models.llama.modeling_llama import (
        LlamaAttention, LlamaDecoderLayer, LlamaModel, LlamaRMSNorm, LlamaRotaryEmbedding,
    )

    if __version__ != "4.57.6" or type(block) is not LlamaDecoderLayer:
        return "resident decoder requires Transformers 4.57.6 LlamaDecoderLayer"
    if (block.training or aliases[id(block)] > 1 or _has_custom_forward(block) or _has_hooks(block)
            or not _original_forward(LlamaDecoderLayer, _LLAMA_MODELING_MODULE)
            or block._parameters or block._buffers
            or set(block._modules) != {"self_attn", "mlp", "input_layernorm", "post_attention_layernorm"}):
        return "resident decoder requires an unmodified, unshared decoder layer in eval mode"

    parent, _, index = path.rpartition(".")
    backbone = modules.get(_llama_backbone_path(path))
    if (type(backbone) is not LlamaModel or modules.get(parent) is not backbone.layers
            or not index.isdecimal() or _has_custom_forward(backbone)
            or not _original_forward(LlamaModel, _LLAMA_MODELING_MODULE)):
        return "resident decoder requires a layer registered under LlamaModel.layers"
    config = backbone.config
    if config._attn_implementation != "eager" or config.rope_scaling or getattr(config, "sliding_window", None):
        return "resident decoder requires eager full attention and unscaled split-half RoPE"

    attention = block.self_attn
    if (type(attention) is not LlamaAttention or attention.config is not config or attention.layer_idx != int(index)
            or attention.training or _has_custom_forward(attention) or _has_hooks(attention)
            or not _original_forward(LlamaAttention, _LLAMA_MODELING_MODULE)
            or attention._parameters or attention._buffers or set(attention._modules) != {*_QKV_PROJECTIONS, "o_proj"}):
        return "resident decoder requires unmodified Llama attention at the registered layer index"
    if (attention.scaling != attention.head_dim**-0.5
            or attention.num_key_value_groups != config.num_attention_heads // config.num_key_value_heads):
        return "resident decoder requires standard Llama GQA grouping and scaling"
    rotary = backbone.rotary_emb
    if (type(rotary) is not LlamaRotaryEmbedding or rotary.rope_type != "default" or rotary.attention_scaling != 1
            or _has_custom_forward(rotary) or rotary.inv_freq.device.type != "cpu"
            or rotary.inv_freq.shape != (attention.head_dim // 2,)):
        return "resident decoder requires the original CPU rotary embeddings"
    for name in (*_QKV_PROJECTIONS, "o_proj"):
        reason = _skip_reason(f"{path}.self_attn.{name}", getattr(attention, name), aliases, storage_uses)
        if reason is not None:
            return f"self_attn.{name}: {reason}"
    for name in ("input_layernorm", "post_attention_layernorm"):
        norm = getattr(block, name)
        if (type(norm) is not LlamaRMSNorm or _has_custom_forward(norm) or _has_hooks(norm)
                or not _original_forward(LlamaRMSNorm, _LLAMA_MODELING_MODULE)
                or norm._buffers or norm._modules or set(norm._parameters) != {"weight"}):
            return f"{name}: expected an unmodified Llama RMSNorm"
        weight = norm.weight
        address = _storage_address(weight)
        if (not address or storage_uses[address] != 1 or weight.dtype != torch.float16
                or weight.shape != (config.hidden_size,)):
            return f"{name}: expected an unshared CPU FP16 weight"
    return _llama_mlp_reason(f"{path}.mlp", block.mlp, aliases, storage_uses)


def _discover(model, target, mode):
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if model.training:
        raise ValueError("Put the model in eval mode before NDP conversion")
    if not isinstance(target, str):
        raise TypeError("target must be a registered module path")
    if mode not in ("projections", "resident"):
        raise ValueError("NDP conversion supports mode='projections' or mode='resident'")

    modules = dict(model.named_modules(remove_duplicate=False))
    if target not in modules:
        raise ValueError(f"Unknown module path {target!r}")
    aliases = Counter(id(module) for module in modules.values())
    # Inspect paths outside the target too so replacing one side of a tie is safe.
    storage_uses = Counter(
        address for module in modules.values()
        for tensors in (module._parameters, module._buffers)
        for tensor in tensors.values() if (address := _storage_address(tensor)) is not None
    )
    if mode == "resident":
        module = modules[target]
        cls = type(module)
        if cls.__module__ == _LLAMA_MODELING_MODULE and cls.__name__ == "LlamaDecoderLayer":
            reason = _llama_block_reason(target, module, modules, aliases, storage_uses)
            rule = "llama_resident_block"
        else:
            reason = _llama_mlp_reason(target, module, aliases, storage_uses)
            rule = "llama_resident_swiglu"
        yield _PlanEntry(target, None if reason else rule, reason), module
        return

    region = [(path, module) for path, module in modules.items()
              if not target or path == target or path.startswith(target + ".")]
    selected = set()
    for path, module in region:
        if not isinstance(module, torch.nn.Linear) and any(module.children()):
            continue
        reason = _skip_reason(path, module, aliases, storage_uses)
        if reason is None:
            selected.add(path)
        yield _PlanEntry(path, None if reason else "linear", reason), module

    for path, module in region:
        cls = type(module)
        if cls.__module__ == _LLAMA_MODELING_MODULE and cls.__name__ == "LlamaAttention":
            reason = _llama_qkv_reason(path, module, selected)
            yield _PlanEntry(path, None if reason else "llama_shared_qkv", reason), module


def plan_to_ndp(model, *, target, mode="projections"):
    """Return immutable entries with each candidate's path, rule, and skip reason."""
    return tuple(entry for entry, _ in _discover(model, target, mode))


@contextmanager
def to_ndp(model, *, target, mode="projections", backend="ndp", max_tokens=None, **backend_options):
    """Temporarily offload supported projections or a resident Llama module.

    Load the model weights before entering. Run inference on the entering thread
    and keep the model structure, device, and weights fixed within the scope.
    """
    model_id = id(model)
    with _conversion_lock:
        if model_id in _active_models:
            raise RuntimeError("This model already has an active NDP conversion")
        _active_models.add(model_id)
    try:
        entries = tuple(_discover(model, target, mode))
        selected = [(entry, source) for entry, source in entries
                    if entry.rule in ("linear", "llama_resident_swiglu", "llama_resident_block")]
        if not selected:
            reason = entries[0][0].reason if entries else "no eligible Linear modules"
            selection = "projections under" if mode == "projections" else "resident module at"
            raise ValueError(f"No supported {selection} {target!r} ({reason})")
        is_block = selected[0][0].rule == "llama_resident_block"
        if max_tokens is not None and not is_block:
            raise ValueError("max_tokens applies to resident decoder blocks")
        if is_block:
            from ._llama_resident import ResidentLlamaBlock

            backbone = model.get_submodule(_llama_backbone_path(target))
            limit = backbone.config.max_position_embeddings
            if max_tokens is None:
                max_tokens = min(128, limit)
            if type(max_tokens) is not int or not 0 < max_tokens <= limit:
                raise ValueError(f"max_tokens must be in [1, {limit}]")

        with Session(backend=backend, **backend_options) as runtime, ExitStack() as restore:
            prepared = []
            for entry, source in selected:
                kind = {"linear": "projection", "llama_resident_swiglu": "resident SwiGLU",
                        "llama_resident_block": "resident decoder block"}[entry.rule]
                try:
                    if entry.rule == "linear":
                        replacement = _ConvertedLinear.from_float(source, runtime=runtime)
                    elif entry.rule == "llama_resident_swiglu":
                        projections = (getattr(source, name) for name in _MLP_PROJECTIONS)
                        replacement = nn.SwiGLU(*projections, prefill=True, runtime=runtime)
                    else:
                        replacement = ResidentLlamaBlock(source, backbone.rotary_emb, max_tokens=max_tokens, runtime=runtime)
                except NdpError as error:
                    error.args = (f"Preparing {kind} {entry.path!r} failed ({error})",)
                    raise
                prepared.append((entry.path, source, replacement))
            runtime.synchronize()

            for path, source, replacement in prepared:
                parent_path, _, name = path.rpartition(".")
                slots = model.get_submodule(parent_path)._modules
                if slots.get(name) is not source:
                    raise RuntimeError(f"Module {path!r} changed during NDP conversion")
                # The slot is already registered. Restore the exact object without registration hooks.
                restore.callback(slots.__setitem__, name, source)
                slots[name] = replacement
                if is_block:
                    restore.callback(replacement.close_requests)
                    hook = backbone.register_forward_pre_hook(replacement.check_backbone_input, with_kwargs=True)
                    restore.callback(hook.remove)
            for entry, attention in entries:
                if entry.rule == "llama_shared_qkv":
                    if model.get_submodule(entry.path) is not attention or _has_custom_forward(attention):
                        raise RuntimeError(f"Module {entry.path!r} changed during NDP conversion")
                    forward = _shared_llama_forward(attention)
                    attributes = vars(attention)
                    restore.callback(attributes.pop, "forward", None)
                    attributes["forward"] = forward
            yield model
    finally:
        with _conversion_lock:
            _active_models.remove(model_id)
