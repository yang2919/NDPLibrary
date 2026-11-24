"""Describe the PIM operator's packed metadata and output for fake execution."""
import torch


@torch._library.register_fake_class("ndp::LinearPackedParams")
class _FakeLinearPackedParams:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    @classmethod
    def __obj_unflatten__(cls, flattened):
        return cls(**dict(flattened))

    def __len__(self):
        raise TypeError("LinearPackedParams has no sequence length")

    def dimensions(self):
        return self.inputs, self.outputs


@torch.library.register_fake("ndp::linear_pim")
def _fake_linear(input, packed):
    inputs, outputs = packed.dimensions()
    torch._check(input.device.type == "cpu" and input.layout == torch.strided)
    torch._check(input.dtype == torch.float16 and not input.requires_grad)
    torch._check(1 <= input.ndim <= 3)
    torch._check(input.shape[-1] == inputs)
    torch._check(input.numel() == inputs)
    return input.new_empty((*input.shape[:-1], outputs))


@torch.library.register_fake("ndp::qkv_pim")
def _fake_qkv(input, q, k, v):
    return tuple(_fake_linear(input, weight) for weight in (q, k, v))
