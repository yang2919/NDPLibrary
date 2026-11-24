"""Compare all TP decoder layers, logits and greedy tokens with a local Llama checkpoint."""
import argparse
import json

import ndp
import torch
from transformers import LlamaForCausalLM


@torch.inference_mode()
def verify(source, worker, steps, devices=2):
    prompt = torch.tensor([[1, 42, 30, 198]])
    capacity = prompt.shape[1] + steps
    model = ndp.nn.TensorParallelLlamaForCausalLM(source,
        devices=[dict(backend='gem5_pim', worker=worker)]*devices, max_tokens=capacity)
    # Observe completed resident rows without changing the execution path.
    captured = {}
    originals = []
    for i, layer in enumerate(model.model.layers):
        original = layer.forward_ndp
        originals.append(original)
        def traced(resident, index=i, call=original):
            result = call(resident)
            captured[index] = model.model.devices.device(0).download(result).reshape(1, 1, -1)
            return result
        layer.forward_ndp = traced
    reference_layers = {}
    hooks = []
    for i, layer in enumerate(source.model.layers):
        def capture(_, args, result, index=i):
            reference_layers[index] = result[0] if isinstance(result, tuple) else result
        hooks.append(layer.register_forward_hook(capture))
    layer_max = [0.0]*len(model.model.layers)
    layer_reference_max = [0.0]*len(layer_max)
    layer_error_energy = [0.0]*len(layer_max)
    layer_reference_energy = [0.0]*len(layer_max)
    layer_outside_tolerance = [0]*len(layer_max)
    logits_outside_tolerance = 0
    logits_max = 0.0
    logits_squared = 0.0
    logits_elements = 0
    agreement = 0
    cache = None
    trajectory = []
    try:
        token = prompt[:, :1]
        for position in range(capacity):
            if position < prompt.shape[1]:
                token = prompt[:, position:position+1]
            # Both models see the same prefix: isolates numerical drift from token divergence.
            reference = source(token, past_key_values=cache, use_cache=True)
            cache = reference.past_key_values
            actual = model(token)
            if not torch.isfinite(actual).all() or not torch.isfinite(reference.logits).all():
                raise AssertionError("Nonfinite logits")
            for i, value in captured.items():
                expected = reference_layers[i]
                if not torch.isfinite(value).all() or not torch.isfinite(expected).all():
                    raise AssertionError(f"Nonfinite hidden values at layer {i}")
                difference = value.float()-expected.float()
                layer_max[i] = max(layer_max[i], difference.abs().max().item())
                layer_reference_max[i] = max(layer_reference_max[i], expected.abs().max().item())
                layer_error_energy[i] += difference.square().sum().item()
                layer_reference_energy[i] += expected.float().square().sum().item()
                layer_outside_tolerance[i] += int((difference.abs() > .02+.02*expected.abs()).sum())
            error = actual-reference.logits.float()
            logits_max = max(logits_max, error.abs().max().item())
            logits_outside_tolerance += int((error.abs() > .02+.02*reference.logits.abs()).sum())
            logits_squared += error.square().sum().item()
            logits_elements += error.numel()
            expected_id = reference.logits[:, -1].argmax(-1, keepdim=True)
            actual_id = actual[:, -1].argmax(-1, keepdim=True)
            agreement += int(torch.equal(expected_id, actual_id))
            trajectory.append([expected_id.item(), actual_id.item()])
            token = expected_id
            print(f'token {position+1}/{capacity}: logit max error {error.abs().max().item():.6g}, '
                  f'top1 reference/TP {trajectory[-1]}', flush=True)
    finally:
        for layer, original in zip(model.model.layers, originals):
            layer.forward_ndp = original
        for hook in hooks:
            hook.remove()
    # Independent autoregressive trajectories, with EOS disabled for a fixed test length.
    generated = model.generate(prompt, max_new_tokens=steps, eos_token_id=[])
    reference_generated = source.generate(prompt, max_new_tokens=steps, do_sample=False,
        eos_token_id=None, pad_token_id=0)
    repeated = model.generate(prompt, max_new_tokens=steps, eos_token_id=[])
    torch.testing.assert_close(generated, repeated, atol=0, rtol=0)
    report = dict(devices=devices, layers=len(layer_max), evaluated_tokens=capacity,
        layer_max_absolute_errors=layer_max, layer_reference_max_absolute_values=layer_reference_max,
        layer_relative_l2_errors=[(e/max(r, 1e-30))**.5
                                 for e, r in zip(layer_error_energy, layer_reference_energy)],
        comparison_atol=.02, comparison_rtol=.02,
        layer_elements_outside_tolerance=layer_outside_tolerance,
        logits_elements_outside_tolerance=logits_outside_tolerance,
        logits_elements=logits_elements,
        numerical_allclose=(logits_outside_tolerance == 0 and not any(layer_outside_tolerance)),
        logits_max_absolute_error=logits_max,
        logits_rmse=(logits_squared/logits_elements)**.5,
        teacher_forced_top1_matches=agreement,
        reference_generated_ids=reference_generated.tolist(), generated_ids=generated.tolist(),
        generated_tokens_match=torch.equal(generated, reference_generated),
        deterministic_reset=True)
    print(json.dumps(report, indent=2), flush=True)
    if not torch.isfinite(torch.tensor(layer_max)).all() or not torch.isfinite(torch.tensor(logits_max)):
        raise AssertionError('Nonfinite decoder results')
    # Keep numerical error reporting separate from greedy trajectory acceptance.
    if not report['generated_tokens_match']:
        raise AssertionError('Greedy tokens differ; inspect the reported numerical drift')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--worker', required=True)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--devices', type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    source = LlamaForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
        dtype=torch.float16, attn_implementation='eager').eval()
    verify(source, args.worker, args.steps, args.devices)
