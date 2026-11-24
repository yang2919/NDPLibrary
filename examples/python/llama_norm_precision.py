"""Isolate Llama RMSNorm rounding on identical checkpoint inputs before any downstream drift."""
import argparse
import json

import ndp
import torch
from transformers import LlamaForCausalLM


@torch.inference_mode()
def verify(source, worker):
    statistics = {name: dict(unequal_elements=0, max_absolute_error=0.0)
                  for name in ('fused', 'round_before_weight')}
    elements = 0
    hooks = []
    with ndp.Session(backend='gem5_pim', worker=worker) as session:
        def observe(module, args, expected):
            nonlocal elements
            width = module.weight.numel()
            gamma = session.upload(module.weight.detach())
            unit = session.upload(torch.ones_like(module.weight))
            gamma_row = session.reshape(gamma, [1, width])
            for row, target in zip(args[0].reshape(-1, width), expected.reshape(-1, width)):
                resident = session.upload(row.reshape(1, width))
                fused = session.rms_norm(resident, gamma, module.variance_epsilon)
                rounded = session.mul(session.rms_norm(resident, unit, module.variance_epsilon), gamma_row)
                elements += width
                for name, tensor in (('fused', fused), ('round_before_weight', rounded)):
                    value = session.download(tensor).flatten()
                    if not torch.isfinite(value).all():
                        raise AssertionError('Nonfinite RMSNorm output')
                    stats = statistics[name]
                    stats['unequal_elements'] += int((value != target).sum())
                    stats['max_absolute_error'] = max(stats['max_absolute_error'],
                                                       (value-target).abs().max().item())
        norms = [norm for layer in source.model.layers
                 for norm in (layer.input_layernorm, layer.post_attention_layernorm)]
        norms.append(source.model.norm)
        try:
            hooks = [norm.register_forward_hook(observe) for norm in norms]
            source(torch.tensor([[1, 42, 30, 198]]), use_cache=False)
        finally:
            for hook in hooks:
                hook.remove()
    print(json.dumps(dict(elements=elements, **statistics), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    source = LlamaForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
        dtype=torch.float16, attn_implementation='eager').eval()
    verify(source, args.worker)
