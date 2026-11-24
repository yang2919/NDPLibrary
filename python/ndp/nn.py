"""PyTorch inference modules with resident NDP weights and CPU tensor boundaries."""
from dataclasses import dataclass
import threading
import weakref
import torch
from . import Engine, OpOptions, LayoutPolicy, Session, DeviceGroup
from .llm import prepare_weight


_thread_state = threading.local()


def _module_runtime(runtime, options):
    if runtime is not None:
        if options:
            raise ValueError("Choose an existing runtime or backend options")
        return runtime
    if options:
        return Session(**{"backend": "ndp", **options})
    # Share the default device on the submitting thread while modules are alive.
    reference = getattr(_thread_state, "default_runtime", None)
    session = reference() if reference else None
    if session is None:
        session = Session(backend="ndp")
        _thread_state.default_runtime = weakref.ref(session)
    return session


class NdpLinear(torch.nn.Module):
    """Keep checkpoint buffers on CPU and prepare a separate resident weight once."""

    def __init__(self, session, weight, bias=None, engine=Engine.PNM):
        super().__init__()
        self.session = session
        self.options = OpOptions(engine, LayoutPolicy.RequirePrepared)
        self._validate(weight, "weight")
        if weight.ndim != 2:
            raise ValueError("weight must have shape [out_features, in_features]")
        if bias is not None:
            self._validate(bias, "bias")
            if bias.shape != (weight.shape[0],):
                raise ValueError("bias shape does not match weight")
            if engine == Engine.PIM:
                raise ValueError("PIM Linear does not support bias")
        self.register_buffer("weight", self._checkpoint_copy(weight))
        self.register_buffer("bias", None if bias is None else self._checkpoint_copy(bias))
        self.training = False
        self._prepared = None
        self.refresh()

    @staticmethod
    def _checkpoint_copy(tensor):
        # Keep version counters even when the caller constructs the module in inference_mode.
        with torch.inference_mode(False), torch.no_grad():
            return tensor.detach().clone()

    @staticmethod
    def _validate(tensor, name):
        if (tensor.device.type != "cpu" or tensor.dtype != torch.float16
                or tensor.layout != torch.strided):
            raise ValueError(f"{name} must be a CPU strided float16 tensor")

    @classmethod
    def from_torch(cls, module, session, engine=Engine.PNM):
        if not isinstance(module, torch.nn.Linear) or module.training:
            raise ValueError("Expected an nn.Linear module in eval mode")
        return cls(session, module.weight, module.bias, engine)

    def refresh(self):
        """Explicitly rebuild device storage after modifying checkpoint buffers."""
        self._prepared = None
        self._validate(self.weight, "weight")
        if self.weight.ndim != 2:
            raise ValueError("weight must have shape [out_features, in_features]")
        if self.weight.is_inference():
            self.weight = self._checkpoint_copy(self.weight)
        if self.bias is not None:
            self._validate(self.bias, "bias")
            if self.bias.shape != (self.weight.shape[0],):
                raise ValueError("bias shape does not match weight")
            if self.options.engine == Engine.PIM:
                raise ValueError("PIM Linear does not support bias")
            if self.bias.is_inference():
                self.bias = self._checkpoint_copy(self.bias)
        resident = self.session.upload(self.weight.detach())
        prepared = prepare_weight(self.session, resident, engines=[self.options.engine])
        bias = None if self.bias is None else self.session.upload(self.bias.detach())
        self._prepared, self._bias = prepared, bias
        self._checkpoint_refs = (weakref.ref(self.weight),
                                 None if self.bias is None else weakref.ref(self.bias))
        self._versions = (self.weight._version, None if self.bias is None else self.bias._version)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._prepared = None
        errors_before = len(error_msgs)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
        # A shape/copy failure must not leave old resident weights usable.
        if len(error_msgs) == errors_before:
            self.refresh()

    def train(self, mode=True):
        if mode:
            raise RuntimeError("NdpLinear supports inference only")
        return super().train(False)

    def _apply(self, fn, recurse=True):
        raise RuntimeError("Convert the source module before NdpLinear.from_torch; device/dtype migration is unsupported")

    def _check_checkpoint(self):
        if self.training:
            raise RuntimeError("NdpLinear supports inference only")
        # Object replacement may preserve the version number, including inference tensors
        # without version counters. Check identity before reading either counter.
        same_objects = (self._prepared is not None
                        and self.weight is not None
                        and self._checkpoint_refs[0]() is self.weight
                        and ((self._checkpoint_refs[1] is None and self.bias is None)
                             or (self._checkpoint_refs[1] is not None and self.bias is not None
                                 and self._checkpoint_refs[1]() is self.bias)))
        if not same_objects:
            raise RuntimeError("Checkpoint buffers changed; call refresh() before forward")
        versions = (self.weight._version, None if self.bias is None else self.bias._version)
        if versions != self._versions:
            raise RuntimeError("Checkpoint buffers changed; call refresh() before forward")

    def forward_ndp(self, x):
        self._check_checkpoint()
        return self.session.linear(x, self._prepared.tensor(), self.options, self._bias)

    def forward(self, x):
        """Convenience CPU boundary; use forward_ndp for resident layer chains."""
        return self.session.download(self.forward_ndp(self.session.upload(x)))


class DecodeLinear(NdpLinear):
    """Project one CPU FP16 token while preserving its batch and sequence axes."""

    def forward(self, x):
        if x.shape != (1, 1, self.weight.shape[1]):
            raise ValueError("DecodeLinear expects [1, 1, in_features]")
        return super().forward(x[0]).unsqueeze(0)


def _is_single_vector(x):
    return x.ndim <= 3 and x.numel() == x.shape[-1]


class Linear(NdpLinear):
    """Run bias-free single-vector projections on PIM and larger inputs on CPU."""

    def __init__(self, runtime, weight):
        super().__init__(runtime, weight, engine=Engine.PIM)

    @classmethod
    def from_float(cls, module, runtime=None, **backend_options):
        if type(module) is not torch.nn.Linear or module.training or module.bias is not None:
            raise ValueError("Expected an eval-mode, bias-free torch.nn.Linear")
        return cls(_module_runtime(runtime, backend_options), module.weight)

    @classmethod
    def from_torch(cls, module, session=None, engine=Engine.PIM, **backend_options):
        if engine != Engine.PIM:
            raise ValueError("Linear uses the PIM engine")
        return cls.from_float(module, runtime=session, **backend_options)

    def refresh(self):
        from ._C import _pack_linear
        self._packed = None
        super().refresh()
        self._packed = _pack_linear(self._prepared)

    def _validate_input(self, x):
        self._check_checkpoint()
        self.session._check()
        self._validate(x, "input")
        if x.requires_grad or x.ndim == 0 or x.shape[-1] != self.weight.shape[1]:
            raise ValueError("Expected an inference input with matching feature width")

    def forward(self, x):
        self._validate_input(x)
        if _is_single_vector(x):
            return torch.ops.ndp.linear_pim(x, self._packed)
        return torch.nn.functional.linear(x, self.weight)


def project_qkv(x, q, k, v):
    """Apply three PIM projections to one uploaded vector in the same Session."""
    for layer in (q, k, v):
        layer._validate_input(x)
        if layer.session is not q.session:
            raise ValueError("Q/K/V weights must share a Session")
    if _is_single_vector(x):
        return torch.ops.ndp.qkv_pim(x, q._packed, k._packed, v._packed)
    return tuple(torch.nn.functional.linear(x, layer.weight) for layer in (q, k, v))

class SwiGLU(torch.nn.Module):
    """Run gate, up, SiLU, multiply and down as one single-token decode module."""

    def __init__(self, gate_proj, up_proj, down_proj, *, runtime=None, **backend_options):
        super().__init__()
        projections = (gate_proj, up_proj, down_proj)
        if any(type(p) is not torch.nn.Linear or p.training or p.bias is not None
               for p in projections):
            raise ValueError("Expected three eval-mode, bias-free torch.nn.Linear projections")
        if up_proj.weight.shape != gate_proj.weight.shape or down_proj.in_features != gate_proj.out_features:
            raise ValueError("Incompatible SwiGLU projection dimensions")
        session = _module_runtime(runtime, backend_options)
        self.gate_proj, self.up_proj, self.down_proj = (
            Linear.from_float(p, runtime=session) for p in projections
        )
        self.eval()

    def forward(self, x):
        self.gate_proj._validate_input(x)
        if not _is_single_vector(x):
            raise ValueError("SwiGLU expects one decode token with rank 1, 2 or 3")
        session = self.gate_proj.session
        resident = session.upload(x.reshape(1, -1))
        result = session.download(self.forward_ndp(resident))
        return result.reshape(*x.shape[:-1], self.down_proj.weight.shape[0])

    def forward_ndp(self, resident):
        session = self.gate_proj.session
        gate = self.gate_proj.forward_ndp(resident)
        up = self.up_proj.forward_ndp(resident)
        product = session.mul(session.silu(gate), up)
        return self.down_proj.forward_ndp(product)


class DecodeSelfAttention(torch.nn.Module):
    """Causal self-attention for one sequence, with resident projections and KV history."""

    def __init__(self, source, *, max_tokens, runtime=None, **backend_options):
        super().__init__()
        if not isinstance(source, torch.nn.MultiheadAttention) or source.training:
            raise ValueError("Expected an eval-mode torch.nn.MultiheadAttention")
        if (source.in_proj_weight is None or source.bias_k is not None
                or source.bias_v is not None or source.add_zero_attn):
            raise ValueError("DecodeSelfAttention requires packed Q/K/V and standard self-attention")
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        session = _module_runtime(runtime, backend_options)
        self.num_heads = source.num_heads
        self.head_dim = source.head_dim
        self.max_tokens = max_tokens
        weights = source.in_proj_weight.chunk(3)
        biases = (None,) * 3 if source.in_proj_bias is None else source.in_proj_bias.chunk(3)

        def projection(w, b):
            return NdpLinear(session, w, b, Engine.PIM if b is None else Engine.PNM)

        self.q_proj, self.k_proj, self.v_proj = (projection(w, b) for w, b in zip(weights, biases))
        self.out_proj = projection(source.out_proj.weight, source.out_proj.bias)
        self.reset_history()
        self.eval()

    def reset_history(self):
        """Start a new sequence. Its KV cache is allocated on the next token."""
        self._cache = None
        self._needs_reset = False

    def _load_from_state_dict(self, *args, **kwargs):
        self.reset_history()
        super()._load_from_state_dict(*args, **kwargs)

    def forward(self, x):
        from .llm import KVCache
        self.q_proj._validate(x, "input")
        width = self.num_heads * self.head_dim
        if x.requires_grad or tuple(x.shape) != (1, 1, width):
            raise ValueError("Expected one CPU FP16 inference token [1, 1, embed_dim]")
        if self._needs_reset:
            raise RuntimeError("Previous token failed; call reset_history() before continuing")
        s = self.q_proj.session
        if self._cache is None:
            self._cache = KVCache(s, self.max_tokens, self.num_heads, self.head_dim)
        start = self._cache.scheduled_length()
        if start >= self.max_tokens:
            raise ValueError("KV history is full; reset_history() starts a new sequence")
        resident = s.upload(x.reshape(1, width))
        q, k, v = (s.reshape(p.forward_ndp(resident), [1, self.num_heads, self.head_dim])
                   for p in (self.q_proj, self.k_proj, self.v_proj))
        try:
            self._cache.append(k, v)
            attended = s.reshape(self._cache.attention(q, start), [1, width])
            return s.download(self.out_proj.forward_ndp(attended)).reshape_as(x)
        except Exception:
            # A failed token may already have modified device history.
            self._needs_reset = True
            raise

class _ResidentRMSNorm(torch.nn.Module):
    def __init__(self, source, session):
        super().__init__()
        NdpLinear._validate(source.weight, "norm weight")
        self.register_buffer("weight", NdpLinear._checkpoint_copy(source.weight))
        self.epsilon = source.variance_epsilon
        self.session = session
        self._prepare()

    def _prepare(self):
        if self.weight.is_inference():
            self.weight = NdpLinear._checkpoint_copy(self.weight)
        self._gamma = self.session.upload(self.weight.detach())
        self._unit = self.session.upload(torch.ones_like(self.weight))
        self._source = weakref.ref(self.weight)
        self._version = self.weight._version

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._prepare()

    def forward(self, resident):
        if self._source() is not self.weight or self._version != self.weight._version:
            raise RuntimeError("Reload the checkpoint after changing normalization weights")
        # Llama casts normalized values to FP16 before multiplying by the weight.
        normalized = self.session.rms_norm(resident, self._unit, self.epsilon)
        rows, width = resident.shape
        gamma = self.session.reshape(self._gamma, [1, width]) if rows == 1 else \
            self.session.gather(self._gamma, list(range(width))*rows, [rows, width])
        return self.session.mul(normalized, gamma)


class LlamaDecodeBlock(torch.nn.Module):
    """Single-sequence FP16 Llama decoder layer with resident GQA history and SwiGLU."""

    def __init__(self, source, *, max_tokens, runtime=None, **backend_options):
        super().__init__()
        attn = source.self_attn
        config = attn.config
        self.width = config.hidden_size
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.dim = attn.head_dim
        if (source.training or getattr(config, "model_type", None) != "llama" or
                config.hidden_act != "silu" or config.rope_scaling or
                getattr(config, "partial_rotary_factor", 1.0) != 1.0 or
                getattr(config, "rope_interleaved", False) or self.dim % 2 or
                self.heads % self.kv_heads or self.width != self.heads*self.dim or
                type(max_tokens) is not int or not 0 < max_tokens <= config.max_position_embeddings):
            raise ValueError("Expected an eval-mode Llama layer with full-width split-half RoPE and SwiGLU")
        kv_width = self.kv_heads * self.dim
        for name, rows in (("q_proj", self.width), ("k_proj", kv_width),
                           ("v_proj", kv_width), ("o_proj", self.width)):
            if getattr(attn, name).weight.shape != (rows, self.width):
                raise ValueError(f"Llama projection dimensions do not match config: {name}")
        if (source.input_layernorm.weight.shape != (self.width,)
                or source.post_attention_layernorm.weight.shape != (self.width,)
                or source.mlp.gate_proj.in_features != self.width
                or source.mlp.down_proj.out_features != self.width):
            raise ValueError("Llama norm or MLP width does not match hidden_size")
        self.max_tokens = max_tokens
        self.session = s = _module_runtime(runtime, backend_options)
        self.input_layernorm = _ResidentRMSNorm(source.input_layernorm, s)
        self.post_attention_layernorm = _ResidentRMSNorm(source.post_attention_layernorm, s)
        self.self_attn = torch.nn.ModuleDict({
            name: Linear.from_float(getattr(attn, name), runtime=s)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        })
        mlp = source.mlp
        self.mlp = SwiGLU(mlp.gate_proj, mlp.up_proj, mlp.down_proj, runtime=s)
        # Llama rotates pairs (d, d + head_dim/2). Fold the minus sign into the table once.
        inv = 1 / config.rope_theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        phase = torch.outer(torch.arange(max_tokens, dtype=torch.float32), inv)
        self._sin = s.upload(torch.cat((-phase.sin(), phase.sin()), dim=-1).half())
        self._cos = s.upload(phase.cos().repeat(1, 2).half())
        self.reset_history()
        self.eval()

    def reset_history(self):
        self._cache = None
        self._needs_reset = False

    def _load_from_state_dict(self, *args, **kwargs):
        self.reset_history()
        super()._load_from_state_dict(*args, **kwargs)

    def _rotate(self, tensor, heads, position):
        s = self.session
        shape = [1, heads, self.dim]
        table_indices = list(range(position*self.dim, (position+1)*self.dim)) * heads
        partners = [h*self.dim + (d + self.dim//2) % self.dim
                    for h in range(heads) for d in range(self.dim)]
        direct = s.mul(s.reshape(tensor, shape), s.gather(self._cos, table_indices, shape))
        cross = s.mul(s.gather(tensor, partners, shape), s.gather(self._sin, table_indices, shape))
        return s.add(direct, cross)

    def _validate_token(self, x):
        NdpLinear._validate(x, "input")
        if tuple(x.shape) != (1, 1, self.width) or x.requires_grad:
            raise ValueError("Expected one inference token [1, 1, hidden_size]")

    def forward_ndp(self, resident, trace=None):
        """Consume and return a [1, hidden_size] FP16 handle in this block's session."""
        from .llm import KVCache
        if list(resident.shape) != [1, self.width]:
            raise ValueError("Expected one resident token [1, hidden_size]")
        if self._needs_reset:
            raise RuntimeError("Previous token failed; reset_history() before continuing")
        s = self.session
        if self._cache is None:
            self._cache = KVCache(s, self.max_tokens, self.kv_heads, self.dim)
        position = self._cache.scheduled_length()
        if position >= self.max_tokens:
            raise ValueError("KV history is full; reset_history() starts a new sequence")
        norm = self.input_layernorm(resident)
        q, k, v = (self.self_attn[name].forward_ndp(norm) for name in ("q_proj", "k_proj", "v_proj"))
        q = self._rotate(q, self.heads, position)
        k = self._rotate(k, self.kv_heads, position)
        try:
            self._cache.append(k, s.reshape(v, [1, self.kv_heads, self.dim]))
            context = s.reshape(self._cache.attention(q, position), [1, self.width])
            attention = self.self_attn["o_proj"].forward_ndp(context)
            residual = s.add(resident, attention)
            mlp = self.mlp.forward_ndp(self.post_attention_layernorm(residual))
            output = s.add(residual, mlp)
            if trace is not None:
                trace.update(attention=attention, mlp=mlp)
            return output
        except Exception:
            self._needs_reset = True
            raise

    def _token(self, x, trace=None):
        self._validate_token(x)
        resident = self.session.upload(x.reshape(1, self.width))
        output = self.forward_ndp(resident, trace)
        try:
            return self.session.download(output).reshape_as(x)
        except Exception:
            self._needs_reset = True
            raise

    def forward(self, x):
        return self._token(x)

class LlamaDecodeStack(torch.nn.Module):
    """Sequential single-token decode across explicitly placed Llama layers."""

    def __init__(self, sources, *, devices, placement, max_tokens):
        super().__init__()
        sources = tuple(sources)
        self.placement = tuple(placement)
        if not sources or len(sources) != len(self.placement):
            raise ValueError("Provide one device ordinal per decoder layer")
        self.devices = devices if isinstance(devices, DeviceGroup) else DeviceGroup(devices)
        sessions = [self.devices.device(device) for device in self.placement]
        if any(layer.self_attn.config.hidden_size != sources[0].self_attn.config.hidden_size
               for layer in sources):
            raise ValueError("Decoder layers must share the same hidden width")
        self.layers = torch.nn.ModuleList(
            LlamaDecodeBlock(layer, runtime=session, max_tokens=max_tokens)
            for layer, session in zip(sources, sessions)
        )
        self.max_tokens = max_tokens
        self.reset_history()
        self.eval()

    def reset_history(self):
        for layer in self.layers:
            layer.reset_history()
        self._position = 0
        self._needs_reset = False

    def _load_from_state_dict(self, *args, **kwargs):
        self.reset_history()
        super()._load_from_state_dict(*args, **kwargs)

    def forward(self, x):
        first = self.layers[0]
        first._validate_token(x)
        if self._needs_reset:
            raise RuntimeError("Previous token failed; reset_history() on the whole stack")
        if self._position >= self.max_tokens:
            raise ValueError("KV history is full; reset_history() starts a new sequence")
        resident = first.session.upload(x.reshape(1, first.width))
        previous = self.placement[0]
        try:
            for device, layer in zip(self.placement, self.layers):
                if device != previous:
                    resident = self.devices.copy_to(resident, device)
                resident = layer.forward_ndp(resident)
                previous = device
            output = self.layers[-1].session.download(resident).reshape_as(x)
        except Exception:
            self._needs_reset = True
            raise
        self._position += 1
        return output


@dataclass(frozen=True)
class DistributedRow:
    """Resident row with explicit feature ranges, replicas or unreduced FP32 partials."""

    devices: DeviceGroup
    tensors: tuple
    width: int
    layout: str
    ranges: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "tensors", tuple(self.tensors))
        object.__setattr__(self, "ranges", tuple(tuple(r) for r in self.ranges))
        if self.width < 1 or len(self.tensors) != len(self.devices):
            raise ValueError("Expected one resident tensor per device")
        if self.layout not in ("sharded", "replicated", "partial"):
            raise ValueError("Unknown distributed row layout")
        if self.layout == "sharded":
            if len(self.ranges) != len(self.tensors):
                raise ValueError("Expected one feature range per shard")
            end = 0
            for start, stop in self.ranges:
                if start != end or stop <= start:
                    raise ValueError("Feature ranges must cover the row in order")
                end = stop
            if end != self.width:
                raise ValueError("Feature ranges differ from global width")
        elif self.ranges:
            raise ValueError("Only sharded rows specify feature ranges")
        from . import DType
        rows = self.rows
        if rows < 1:
            raise ValueError("Expected nonempty matrix rows")
        for i, tensor in enumerate(self.tensors):
            self.devices.device(i)._validate_tensor(tensor)
            width = self.ranges[i][1] - self.ranges[i][0] if self.layout == "sharded" else self.width
            dtype = DType.F32 if self.layout == "partial" else DType.F16
            if list(tensor.shape) != [rows, width] or tensor.dtype != dtype:
                raise ValueError("Local tensor shape or dtype differs from distributed metadata")

    @property
    def rows(self):
        shape = self.tensors[0].shape
        return shape[0] if len(shape) == 2 else 0

    def replicated(self):
        if self.layout == "replicated":
            return self
        if self.layout == "partial":
            root = self.devices.reduce_partials(self.tensors)
            tensors = self.devices.broadcast(root)
        else:
            tensors = self.devices.all_gather_rows(self.tensors)
        return DistributedRow(self.devices, tuple(tensors), self.width, "replicated")


class TensorParallelLinear(torch.nn.Module):
    """Split a single-token PIM Linear by output channels across independent devices."""

    def __init__(self, source, *, devices, split="output", ranges=None):
        super().__init__()
        if type(source) is not torch.nn.Linear or source.training or source.bias is not None:
            raise ValueError("Expected an eval-mode, bias-free torch.nn.Linear")
        NdpLinear._validate(source.weight, "weight")
        self.devices = devices if isinstance(devices, DeviceGroup) else DeviceGroup(devices)
        self.in_features = source.in_features
        self.out_features = source.out_features
        if split not in ("input", "output"):
            raise ValueError("split must be input or output")
        self.split = split
        extent = self.out_features if split == "output" else self.in_features
        base, remainder = divmod(extent, len(self.devices))
        self.ranges = tuple((i*base + min(i, remainder), (i+1)*base + min(i+1, remainder))
                            for i in range(len(self.devices)))
        if ranges is not None:
            self.ranges = tuple(tuple(pair) for pair in ranges)
            cursor = 0
            if len(self.ranges) != len(self.devices):
                raise ValueError("Expected one weight range per device")
            for start, end in self.ranges:
                if type(start) is not int or type(end) is not int or start != cursor or end <= start:
                    raise ValueError("Weight ranges must be consecutive nonempty integer intervals")
                cursor = end
            if cursor != extent:
                raise ValueError("Weight ranges must cover the partition axis")
        if extent < len(self.devices) or self.in_features < 1:
            raise ValueError("Each device needs at least one output channel and one input feature")
        self.register_buffer("weight", NdpLinear._checkpoint_copy(source.weight))
        self.refresh()
        self.eval()

    def refresh(self):
        """Prepare the configured contiguous weight shards."""
        self._shards = None
        NdpLinear._validate(self.weight, "weight")
        if self.weight.shape != (self.out_features, self.in_features):
            raise ValueError("Checkpoint dimensions differ from the projection")
        if self.weight.is_inference():
            self.weight = NdpLinear._checkpoint_copy(self.weight)
        prepared = []
        for i, (start, end) in enumerate(self.ranges):
            weight = self.weight.detach()[start:end] if self.split == "output" else self.weight.detach()[:, start:end]
            session = self.devices.device(i)
            prepared.append(prepare_weight(session, session.upload(weight), engines=[Engine.PIM]))
        self._shards = prepared
        self._source = weakref.ref(self.weight)
        self._version = self.weight._version

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._shards = None
        errors = len(error_msgs)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        if len(error_msgs) == errors:
            self.refresh()

    def train(self, mode=True):
        if mode:
            raise RuntimeError("TensorParallelLinear supports inference only")
        return super().train(False)

    def _check(self):
        if self._shards is None or self._source() is not self.weight or self.weight._version != self._version:
            raise RuntimeError("Checkpoint buffers changed; call refresh()")

    def forward_distributed(self, x):
        """Consume CPU input or resident rows, and retain sharded/partial outputs."""
        self._check()
        if isinstance(x, DistributedRow):
            if x.devices is not self.devices or x.width != self.in_features or x.layout == "partial":
                raise ValueError("Expected matching group, input width and reduced input")
            if self.split == "output":
                local = x.replicated().tensors
            elif x.layout == "sharded":
                if x.ranges != self.ranges:
                    raise ValueError("Input feature ranges differ from weight shards")
                local = x.tensors
            else:
                local = [self.devices.device(i).gather(t, [row*self.in_features+j for row in range(x.rows) for j in range(start, end)],
                    [x.rows, end-start])
                         for i, (t, (start, end)) in enumerate(zip(x.tensors, self.ranges))]
        else:
            NdpLinear._validate(x, "input")
            if (x.requires_grad or x.ndim not in (1, 2, 3) or x.shape[-1] != self.in_features
                    or x.numel() != self.in_features):
                raise ValueError("Expected one inference token of matching width")
            row = x.reshape(1, self.in_features)
            local = [self.devices.device(i).upload(row if self.split == "output" else row[:, start:end])
                     for i, (start, end) in enumerate(self.ranges)]
        results = []
        for i, (tensor, weight) in enumerate(zip(local, self._shards)):
            session = self.devices.device(i)
            engine = Engine.PNM if tensor.shape[0] > 1 else Engine.PIM
            if self.split == "input":
                results.append(session.linear_partial(tensor, weight.tensor(), engine))
            else:
                options = OpOptions(engine, LayoutPolicy.RequirePrepared)
                results.append(session.linear(tensor, weight.tensor(), options))
        return DistributedRow(self.devices, tuple(results), self.out_features,
                              "partial" if self.split == "input" else "sharded",
                              () if self.split == "input" else self.ranges)

    def forward_shards(self, x):
        return list(self.forward_distributed(x).tensors)

    def forward(self, x):
        result = self.forward_distributed(x)
        if result.layout == "partial":
            tensor = self.devices.reduce_partials(result.tensors)
            host = self.devices.device(0).download(tensor)
        else:
            host = torch.cat([self.devices.device(i).download(t) for i, t in enumerate(result.tensors)], dim=-1)
        shape = (x.rows, self.out_features) if isinstance(x, DistributedRow) else (*x.shape[:-1], self.out_features)
        return host.reshape(shape)

def _replicated_token(x, devices, width):
    if isinstance(x, DistributedRow):
        if x.devices is not devices or x.width != width or x.layout == "partial":
            raise ValueError("Expected a reduced input in the same device group")
        return x.replicated()
    NdpLinear._validate(x, "input")
    if x.requires_grad or x.ndim not in (1, 2, 3) or x.numel() != width or x.shape[-1] != width:
        raise ValueError("Expected a single inference token")
    return DistributedRow(devices, tuple(devices.device(i).upload(x.reshape(1, width))
                                        for i in range(len(devices))), width, "replicated")


def _download_partials(result, x):
    root = result.devices.reduce_partials(result.tensors)
    host = result.devices.device(0).download(root)
    shape = (x.rows, result.width) if isinstance(x, DistributedRow) else (*x.shape[:-1], result.width)
    return host.reshape(shape)


class TensorParallelSwiGLU(torch.nn.Module):
    """Keep gate/up channels and their elementwise activations on the same device."""

    def __init__(self, source, *, devices):
        super().__init__()
        activation = getattr(source, "act_fn", None)
        supported = isinstance(activation, torch.nn.SiLU)
        if type(activation).__module__ == "transformers.activations":
            from transformers.activations import SiLUActivation
            supported = isinstance(activation, SiLUActivation)
        if source.training or not supported:
            raise ValueError("Expected an eval-mode SwiGLU MLP")
        gate, up, down = source.gate_proj, source.up_proj, source.down_proj
        if gate.weight.shape != up.weight.shape or down.in_features != gate.out_features:
            raise ValueError("Incompatible SwiGLU projection dimensions")
        self.devices = devices if isinstance(devices, DeviceGroup) else DeviceGroup(devices)
        self.gate_proj = TensorParallelLinear(gate, devices=self.devices)
        self.up_proj = TensorParallelLinear(up, devices=self.devices)
        self.down_proj = TensorParallelLinear(down, devices=self.devices, split="input")
        self.eval()

    def forward_distributed(self, x):
        token = _replicated_token(x, self.devices, self.gate_proj.in_features)
        gate = self.gate_proj.forward_distributed(token)
        up = self.up_proj.forward_distributed(token)
        product = []
        for i, (g, u) in enumerate(zip(gate.tensors, up.tensors)):
            session = self.devices.device(i)
            product.append(session.mul(session.silu(g), u))
        local = DistributedRow(self.devices, tuple(product), gate.width, "sharded", gate.ranges)
        return self.down_proj.forward_distributed(local)

    def forward(self, x):
        return _download_partials(self.forward_distributed(x), x)


class _AttentionHistory:
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens
        self._caches = None
        self._position = 0
        self._needs_reset = False


class _BlockHistory:
    def __init__(self, max_tokens):
        self.attention = _AttentionHistory(max_tokens)
        self._needs_reset = False


class DecodeRequest:
    """Request-owned KV history; weights and device connections belong to the model."""

    def __init__(self, owner, max_tokens):
        self._owner = weakref.ref(owner)
        self._max_tokens = max_tokens
        self._closed = False
        self.reset()

    @property
    def max_tokens(self):
        return self._max_tokens

    @property
    def position(self):
        return self._position

    @property
    def failed(self):
        return self._needs_reset

    @property
    def closed(self):
        return self._closed

    def _check(self):
        if self._closed or self._owner() is None:
            raise RuntimeError("Request is closed or its model no longer exists")

    def reset(self):
        self._check()
        self._layers = [_BlockHistory(self.max_tokens) for _ in self._owner().layers]
        self._position = 0
        self._needs_reset = False

    def close(self):
        self._layers = []
        self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class TensorParallelLlamaAttention(torch.nn.Module):
    """Decode GQA by whole KV groups with local RoPE, cache and attention."""

    def __init__(self, source, *, devices, max_tokens):
        super().__init__()
        config = source.config
        self.width = config.hidden_size
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.dim = source.head_dim
        if (source.training or config.model_type != "llama" or config.rope_scaling or
                getattr(config, "rope_interleaved", False) or
                getattr(config, "partial_rotary_factor", 1.0) != 1.0 or
                self.dim % 2 or self.heads % self.kv_heads or self.width != self.heads*self.dim or
                type(max_tokens) is not int or not 0 < max_tokens <= config.max_position_embeddings):
            raise ValueError("Expected eval-mode Llama GQA with unscaled split-half RoPE")
        self.devices = devices if isinstance(devices, DeviceGroup) else DeviceGroup(devices)
        if len(self.devices) > self.kv_heads:
            raise ValueError("Each device needs at least one whole KV head")
        for name, rows in (("q_proj", self.width), ("k_proj", self.kv_heads*self.dim),
                           ("v_proj", self.kv_heads*self.dim), ("o_proj", self.width)):
            if getattr(source, name).weight.shape != (rows, self.width):
                raise ValueError("Projection dimensions differ from attention config")
        self.max_tokens = max_tokens
        base, extra = divmod(self.kv_heads, len(self.devices))
        self.local_kv_heads = tuple(base + (i < extra) for i in range(len(self.devices)))
        ratio = self.heads // self.kv_heads
        self.local_q_heads = tuple(kv*ratio for kv in self.local_kv_heads)
        kv_ranges, q_ranges, offset = [], [], 0
        for count in self.local_kv_heads:
            kv_ranges.append((offset*self.dim, (offset+count)*self.dim))
            q_ranges.append((offset*self.dim*ratio, (offset+count)*self.dim*ratio))
            offset += count
        self.q_proj = TensorParallelLinear(source.q_proj, devices=self.devices, ranges=q_ranges)
        self.k_proj = TensorParallelLinear(source.k_proj, devices=self.devices, ranges=kv_ranges)
        self.v_proj = TensorParallelLinear(source.v_proj, devices=self.devices, ranges=kv_ranges)
        self.o_proj = TensorParallelLinear(source.o_proj, devices=self.devices, split="input", ranges=q_ranges)
        inv = 1 / config.rope_theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        phase = torch.outer(torch.arange(max_tokens, dtype=torch.float32), inv)
        self._tables = tuple((self.devices.device(i).upload(torch.cat((-phase.sin(), phase.sin()), -1).half()),
                              self.devices.device(i).upload(phase.cos().repeat(1, 2).half()))
                             for i in range(len(self.devices)))
        self.reset_history()
        self.eval()

    def reset_history(self):
        self._caches = None
        self._position = 0
        self._needs_reset = False

    def _load_from_state_dict(self, *args, **kwargs):
        self.reset_history()
        super()._load_from_state_dict(*args, **kwargs)

    def _rotate(self, ordinal, tensor, heads, position):
        session = self.devices.device(ordinal)
        sine, cosine = self._tables[ordinal]
        rows = tensor.shape[0]
        shape = [rows, heads, self.dim]
        indices = [p*self.dim+d for p in range(position, position+rows)
                   for h in range(heads) for d in range(self.dim)]
        partners = [(row*heads+h)*self.dim+(d+self.dim//2) % self.dim
                    for row in range(rows) for h in range(heads) for d in range(self.dim)]
        direct = session.mul(session.reshape(tensor, shape), session.gather(cosine, indices, shape))
        cross = session.mul(session.gather(tensor, partners, shape), session.gather(sine, indices, shape))
        return session.add(direct, cross)

    def forward_distributed(self, x, *, _state=None):
        from .llm import KVCache
        history = self if _state is None else _state
        if history._needs_reset:
            raise RuntimeError("Previous token failed; reset_history() on the attention module")
        if history._position >= history.max_tokens:
            raise ValueError("KV history is full; reset_history() starts a new sequence")
        token = _replicated_token(x, self.devices, self.width)
        rows = token.rows
        if rows > history.max_tokens-history._position:
            raise ValueError("Prefill chunk exceeds remaining KV capacity")
        if history._caches is None:
            history._caches = tuple(KVCache(self.devices.device(i), history.max_tokens, kv, self.dim)
                                 for i, kv in enumerate(self.local_kv_heads))
        try:
            q = self.q_proj.forward_distributed(token)
            k = self.k_proj.forward_distributed(token)
            v = self.v_proj.forward_distributed(token)
            attended = []
            for i, cache in enumerate(history._caches):
                session = self.devices.device(i)
                queries = self._rotate(i, q.tensors[i], self.local_q_heads[i], history._position)
                keys = self._rotate(i, k.tensors[i], self.local_kv_heads[i], history._position)
                values = session.reshape(v.tensors[i], [rows, self.local_kv_heads[i], self.dim])
                cache.append(keys, values)
                attended.append(session.reshape(cache.attention(queries, history._position),
                                                 [rows, self.local_q_heads[i]*self.dim]))
            local = DistributedRow(self.devices, tuple(attended), self.width, "sharded", self.q_proj.ranges)
            partials = self.o_proj.forward_distributed(local)
            for tensor in partials.tensors:
                tensor.wait()
        except Exception:
            history._needs_reset = True
            raise
        history._position += rows
        return partials

    def forward(self, x):
        partials = self.forward_distributed(x)
        try:
            return _download_partials(partials, x)
        except Exception:
            self._needs_reset = True
            raise


class TensorParallelLlamaDecodeBlock(torch.nn.Module):
    """Compose tensor-parallel GQA and SwiGLU with resident root-device residuals."""

    def __init__(self, source, *, devices, max_tokens):
        super().__init__()
        self.devices = devices if isinstance(devices, DeviceGroup) else DeviceGroup(devices)
        self.width = source.self_attn.config.hidden_size
        if (source.training or source.input_layernorm.weight.shape != (self.width,)
                or source.post_attention_layernorm.weight.shape != (self.width,)
                or source.mlp.gate_proj.in_features != self.width
                or source.mlp.down_proj.out_features != self.width):
            raise ValueError("Expected an eval-mode Llama layer with matching norm and MLP widths")
        root = self.devices.device(0)
        self.input_layernorm = _ResidentRMSNorm(source.input_layernorm, root)
        self.post_attention_layernorm = _ResidentRMSNorm(source.post_attention_layernorm, root)
        self.self_attn = TensorParallelLlamaAttention(source.self_attn, devices=self.devices,
                                                     max_tokens=max_tokens)
        self.mlp = TensorParallelSwiGLU(source.mlp, devices=self.devices)
        self.reset_history()
        self.eval()

    def reset_history(self):
        self.self_attn.reset_history()
        self._needs_reset = False

    def _load_from_state_dict(self, *args, **kwargs):
        self.reset_history()
        super()._load_from_state_dict(*args, **kwargs)

    def _replicate(self, tensor):
        return DistributedRow(self.devices,
            tuple(self.devices.broadcast(tensor)),
            self.width, "replicated")

    def _forward_root(self, x, *, _state=None):
        history = self if _state is None else _state
        attention_history = self.self_attn if _state is None else _state.attention
        if history._needs_reset:
            raise RuntimeError("Previous token failed; reset_history() before continuing")
        if attention_history._position >= attention_history.max_tokens:
            raise ValueError("KV history is full; reset_history() starts a new sequence")
        from . import NdpTensor, DType
        root = self.devices.device(0)
        if isinstance(x, NdpTensor):
            root._validate_tensor(x)
            if (len(x.shape) != 2 or x.shape[0] < 1 or x.shape[1] != self.width) or x.dtype != DType.F16:
                raise ValueError("Expected a root-device FP16 row [1, hidden_size]")
            resident = x
        elif isinstance(x, DistributedRow):
            if x.devices is not self.devices or x.width != self.width or x.layout == "partial":
                raise ValueError("Expected a completed row in the same device group")
            resident = x.replicated().tensors[0]
        else:
            NdpLinear._validate(x, "input")
            if x.requires_grad or tuple(x.shape) != (1, 1, self.width):
                raise ValueError("Expected one CPU FP16 inference token [1, 1, hidden_size]")
            resident = root.upload(x.reshape(1, self.width))
        if resident.shape[0] > attention_history.max_tokens-attention_history._position:
            raise ValueError("Prefill chunk exceeds remaining KV capacity")
        try:
            normalized = self._replicate(self.input_layernorm(resident))
            attention = self.self_attn.forward_distributed(normalized) if _state is None else \
                self.self_attn.forward_distributed(normalized, _state=attention_history)
            residual = root.add(resident, self.devices.reduce_partials(attention.tensors))
            normalized = self._replicate(self.post_attention_layernorm(residual))
            mlp = self.mlp.forward_distributed(normalized)
            output = root.add(residual, self.devices.reduce_partials(mlp.tensors))
            output.wait()
            return output
        except Exception:
            # Attention may have committed its KV entry before a later stage fails.
            history._needs_reset = True
            raise

    def forward_ndp(self, resident, *, _state=None):
        """Consume and return a completed FP16 row on device 0."""
        from . import NdpTensor
        if not isinstance(resident, NdpTensor):
            raise TypeError("Expected a resident NDP Tensor")
        return self._forward_root(resident, _state=_state)

    def forward_distributed(self, x):
        """Return completed replicas for another block without a CPU tensor boundary."""
        output = self._forward_root(x)
        try:
            return self._replicate(output)
        except Exception:
            self._needs_reset = True
            raise

    def forward(self, x):
        output = self._forward_root(x)
        try:
            return self.devices.device(0).download(output).reshape(1, 1, self.width)
        except Exception:
            self._needs_reset = True
            raise


class TensorParallelLlamaDecodeStack(torch.nn.Module):
    """Run consecutive TP layers with root-device rows and request-owned history."""

    def __init__(self, sources, *, devices=None, device_groups=None, placement=None, max_tokens):
        super().__init__()
        sources = tuple(sources)
        if not sources:
            raise ValueError("At least one decoder layer is required")
        self.width = sources[0].self_attn.config.hidden_size
        if any(layer.self_attn.config.hidden_size != self.width for layer in sources):
            raise ValueError("All layers must have the same hidden width")
        if device_groups is None:
            if devices is None or placement is not None:
                raise ValueError("Provide devices, or device_groups with placement")
            group = devices if isinstance(devices, DeviceGroup) else DeviceGroup(devices)
            self.device_groups = (group,)
            self.placement = (0,)*len(sources)
        else:
            if devices is not None or placement is None:
                raise ValueError("Use device_groups and placement without devices")
            self.device_groups = tuple(g if isinstance(g, DeviceGroup) else DeviceGroup(g)
                                       for g in device_groups)
            self.placement = tuple(placement)
            if (not self.device_groups or len(self.placement) != len(sources)
                    or any(type(i) is not int or not 0 <= i < len(self.device_groups)
                           for i in self.placement)):
                raise ValueError("Provide one valid group ordinal per layer")
            if set(self.placement) != set(range(len(self.device_groups))):
                raise ValueError("Every device group must own at least one layer")
        sessions, roots = [], []
        for group in self.device_groups:
            roots.append(len(sessions))
            sessions.extend(group.device(i) for i in range(len(group)))
        if len({id(session) for session in sessions}) != len(sessions):
            raise ValueError("Device groups must use disjoint sessions")
        self.devices = self.device_groups[0] if len(self.device_groups) == 1 else DeviceGroup(sessions)
        self._roots = tuple(roots)
        self.layers = torch.nn.ModuleList(TensorParallelLlamaDecodeBlock(
            layer, devices=self.device_groups[ordinal], max_tokens=max_tokens)
            for layer, ordinal in zip(sources, self.placement))
        self.max_tokens = max_tokens
        self._requests = weakref.WeakSet()
        self.reset_history()
        self.eval()

    @property
    def input_session(self):
        return self.device_groups[self.placement[0]].device(0)

    @property
    def output_session(self):
        return self.device_groups[self.placement[-1]].device(0)

    def new_request(self, *, max_tokens=None):
        capacity = self.max_tokens if max_tokens is None else max_tokens
        if type(capacity) is not int or not 0 < capacity <= self.max_tokens:
            raise ValueError("Request capacity must be within the model capacity")
        request = DecodeRequest(self, capacity)
        self._requests.add(request)
        return request

    def _history(self, request):
        if request is None:
            return self
        if (not isinstance(request, DecodeRequest) or request._owner() is not self
                or request not in self._requests):
            raise ValueError("Request belongs to a different model")
        request._check()
        return request

    def reset_history(self, *, request=None):
        if request is not None:
            self._history(request).reset()
            return
        for layer in self.layers:
            layer.reset_history()
        self._position = 0
        self._needs_reset = False

    def _load_from_state_dict(self, *args, **kwargs):
        for request in list(self._requests):
            request.close()
        self.reset_history()
        super()._load_from_state_dict(*args, **kwargs)

    def _ready_history(self, request):
        history = self._history(request)
        if history._needs_reset:
            message = "reset this request" if request is not None else "reset_history() on the whole stack"
            raise RuntimeError(f"Previous token failed; {message}")
        if history._position >= history.max_tokens:
            raise ValueError("KV history is full; reset history to start a new sequence")
        return history

    def forward_ndp(self, resident, *, request=None):
        history = self._ready_history(request)
        rows = resident.shape[0]
        if rows > history.max_tokens-history._position:
            raise ValueError("Prefill chunk exceeds remaining KV capacity")
        try:
            previous = self.placement[0]
            for index, (ordinal, layer) in enumerate(zip(self.placement, self.layers)):
                if ordinal != previous:
                    resident = self.devices.copy_to(resident, self._roots[ordinal])
                resident = layer.forward_ndp(resident) if request is None else \
                    layer.forward_ndp(resident, _state=history._layers[index])
                previous = ordinal
        except Exception:
            history._needs_reset = True
            raise
        history._position += rows
        return resident

    def forward(self, x, *, request=None):
        history = self._ready_history(request)
        NdpLinear._validate(x, "input")
        if x.requires_grad or (x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 1 or x.shape[2] != self.width):
            raise ValueError("Expected one CPU FP16 inference token [1, 1, hidden_size]")
        output = self.forward_ndp(self.input_session.upload(x.reshape(-1, self.width)), request=request)
        try:
            return self.output_session.download(output).reshape_as(x)
        except Exception:
            history._needs_reset = True
            raise


class TensorParallelLlamaForCausalLM(torch.nn.Module):
    """Greedy Llama decode with shared weights and independently owned request histories."""

    def __init__(self, source, *, devices=None, device_groups=None, placement=None, max_tokens):
        super().__init__()
        import copy
        if source.training or source.config.model_type != "llama":
            raise ValueError("Expected an eval-mode Llama causal language model")
        NdpLinear._validate(source.model.embed_tokens.weight, "embedding weight")
        NdpLinear._validate(source.lm_head.weight, "LM head weight")
        width = source.config.hidden_size
        if (source.model.embed_tokens.weight.shape != source.lm_head.weight.shape
                or source.lm_head.weight.shape[1] != width
                or source.model.norm.weight.shape != (width,)):
            raise ValueError("Embedding, LM head and final norm dimensions must agree")
        self.model = TensorParallelLlamaDecodeStack(source.model.layers, devices=devices,
            device_groups=device_groups, placement=placement, max_tokens=max_tokens)
        self.model.embed_tokens = copy.deepcopy(source.model.embed_tokens).requires_grad_(False)
        self.model.norm = _ResidentRMSNorm(source.model.norm, self.model.output_session)
        self.lm_head = copy.deepcopy(source.lm_head).requires_grad_(False)
        self.eos_token_id = source.config.eos_token_id
        self.eval()

    def new_request(self, *, max_tokens=None):
        return self.model.new_request(max_tokens=max_tokens)

    def reset_history(self, *, request=None):
        self.model.reset_history(request=request)

    def _load_from_state_dict(self, *args, **kwargs):
        self.reset_history()
        super()._load_from_state_dict(*args, **kwargs)

    def _validate_ids(self, ids):
        if (not isinstance(ids, torch.Tensor) or ids.device.type != "cpu"
                or ids.dtype != torch.int64 or ids.ndim != 2 or ids.shape[0] != 1
                or ids.shape[1] < 1 or (ids < 0).any()
                or (ids >= self.model.embed_tokens.num_embeddings).any()):
            raise ValueError("Expected nonempty CPU int64 token IDs [1, sequence] in vocabulary")

    def forward(self, input_ids, *, request=None):
        """Decode one token using PIM projections; return CPU FP32 logits."""
        self._validate_ids(input_ids)
        if input_ids.shape[1] != 1:
            raise ValueError("forward accepts one token; use prefill for multiple tokens")
        return self.prefill(input_ids, request=request)

    @torch.inference_mode()
    def prefill(self, input_ids, *, request=None):
        """Append a contiguous prompt chunk; multi-row projections use PNM GEMM."""
        history = self.model._ready_history(request)
        self._validate_ids(input_ids)
        if input_ids.shape[1] > history.max_tokens-history._position:
            raise ValueError("Prefill chunk exceeds remaining KV capacity")
        hidden = self.model.embed_tokens(input_ids)
        try:
            resident = self.model.forward_ndp(
                self.model.input_session.upload(hidden.reshape(-1, self.model.width)), request=request)
            normalized = self.model.output_session.download(self.model.norm(resident)).reshape_as(hidden)
            return self.lm_head(normalized).float()
        except Exception:
            history._needs_reset = True
            raise

    @torch.inference_mode()
    def generate(self, input_ids, *, max_new_tokens=16, eos_token_id=None, request=None, prefill_chunk_size=1):
        """Reset the selected history and return prompt plus greedy tokens; no batching or sampling."""
        history = self.model._history(request)
        if type(prefill_chunk_size) is not int or prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be a positive integer")
        self._validate_ids(input_ids)
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if max_new_tokens and input_ids.shape[1] + max_new_tokens - 1 > history.max_tokens:
            raise ValueError("Prompt and decode steps exceed max_tokens")
        eos = self.eos_token_id if eos_token_id is None else eos_token_id
        eos = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
        if any(type(token) is not int or not 0 <= token < self.model.embed_tokens.num_embeddings for token in eos):
            raise ValueError("EOS IDs must belong to the vocabulary")
        self.reset_history(request=request)
        result = input_ids.clone()
        if not max_new_tokens:
            return result
        for chunk in input_ids.split(prefill_chunk_size, dim=1):
            logits = self.prefill(chunk, request=request)
        for step in range(max_new_tokens):
            token = logits[:, -1].argmax(-1, keepdim=True)
            result = torch.cat((result, token), dim=1)
            if token.item() in eos or step + 1 == max_new_tokens:
                break
            logits = self(token, request=request)
        return result
