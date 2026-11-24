"""Check a SmolLM2 gate projection on two PIM workers against one unsharded worker."""
import argparse

import torch
from transformers import LlamaForCausalLM
import ndp


@torch.inference_mode()
def verify(model, worker):
    layer = model.model.layers[0]
    weight = layer.mlp.gate_proj
    group = ndp.DeviceGroup([dict(backend="gem5_pim", worker=worker) for _ in range(2)])
    parallel = ndp.nn.TensorParallelLinear(weight, devices=group)
    single = ndp.nn.Linear.from_float(weight, runtime=group.device(0))
    ids = torch.tensor([[1, 42, 30, 198]])
    hidden = model.model.embed_tokens(ids)
    pos = torch.arange(ids.shape[1]).unsqueeze(0)
    mask = torch.full((4, 4), float("-inf"), dtype=torch.float16).triu(1)
    gate_inputs, down_inputs = [], []
    hook = weight.register_forward_pre_hook(lambda _, args: gate_inputs.append(args[0]))
    down_hook = layer.mlp.down_proj.register_forward_pre_hook(lambda _, args: down_inputs.append(args[0]))
    try:
        layer(hidden, attention_mask=mask[None, None], position_embeddings=model.model.rotary_emb(hidden, pos))
    finally:
        hook.remove()
        down_hook.remove()
    for token in gate_inputs[0].split(1, dim=1):
        reference = single(token)
        torch.testing.assert_close(parallel(token), reference, atol=0, rtol=0)
        shards = parallel.forward_shards(token)
        for i, replica in enumerate(group.all_gather_rows(shards)):
            result = group.device(i).download(replica).reshape_as(reference)
            torch.testing.assert_close(result, reference, atol=0, rtol=0)
    row = ndp.nn.TensorParallelLinear(layer.mlp.down_proj, devices=group, split="input")
    down_single = ndp.nn.Linear.from_float(layer.mlp.down_proj, runtime=group.device(0))
    max_error = 0.0
    for token in down_inputs[0].split(1, dim=1):
        expected = down_single(token)
        actual = row(token)
        max_error = max(max_error, (actual-expected).abs().max().item())
        torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
    # Preserve the gate's sharded output through the next input-sharded projection.
    for token in gate_inputs[0].split(1, dim=1):
        resident = parallel.forward_distributed(token)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            partials = row.forward_distributed(resident)
            for tensor in partials.tensors:
                tensor.wait()
        counts = {event.key: event.count for event in profile.key_averages()}
        if counts.get("ndp::upload", 0) or counts.get("ndp::download", 0):
            raise AssertionError("resident Linear chain transferred a PyTorch tensor")
        result = group.device(0).download(group.reduce_partials(partials.tensors))
        expected = down_single(single(token))
        torch.testing.assert_close(result.reshape_as(expected), expected, atol=.02, rtol=.02)
    print(f"Four input-split down projections passed; maximum absolute error {max_error}. "
          "Resident gate-to-down inputs required zero PyTorch transfers.")
    print(f"Four gate projections passed: {weight.in_features} inputs, "
          f"{[s.shape[-1] for s in shards]} output channels, exact host and replica results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--worker", required=True)
    args = parser.parse_args()
    model = LlamaForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
                                            dtype=torch.float16, attn_implementation="eager").eval()
    verify(model, args.worker)
