"""Compare a complete resident Llama decode block with the checkpoint's CPU PyTorch layer."""
import argparse

import torch
from transformers import LlamaForCausalLM
import ndp


@torch.inference_mode()
def verify(model, block, ids):
    source = model.model.layers[0]
    embeddings = model.model.embed_tokens(torch.tensor([ids]))
    positions = torch.arange(len(ids)).unsqueeze(0)
    mask = torch.full((len(ids), len(ids)), float("-inf"), dtype=torch.float16).triu(1)
    reference = {}

    def save(name):
        def hook(module, inputs, output):
            reference[name] = output
        return hook

    hooks = [source.self_attn.o_proj.register_forward_hook(save("attention")),
             source.mlp.register_forward_hook(save("mlp"))]
    try:
        reference["block"] = source(embeddings, attention_mask=mask[None, None],
                                    position_embeddings=model.model.rotary_emb(embeddings, positions))
    finally:
        for hook in hooks:
            hook.remove()
    maxima = dict.fromkeys(reference, 0.)
    tokens = embeddings.split(1, dim=1)
    diagnostic = []
    for i, token in enumerate(tokens):
        trace = {}
        output = block._token(token, trace)
        diagnostic.append(output)
        actual = {name: block.session.download(handle).reshape_as(token) for name, handle in trace.items()}
        actual["block"] = output
        for name, result in actual.items():
            expected = reference[name][:, i:i+1]
            maxima[name] = max(maxima[name], (result - expected).abs().max().item())
            torch.testing.assert_close(result, expected, atol=.02, rtol=.02)
    print("Maximum absolute FP16 errors:", maxima)

    block.reset_history()
    for i, token in enumerate(tokens):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            output = block(token)
        counts = {e.key: e.count for e in profile.key_averages()}
        pair = (counts.get("ndp::upload", 0), counts.get("ndp::download", 0))
        if pair != (1, 1):
            raise AssertionError(f"token {i}: upload/download counts {pair}")
        torch.testing.assert_close(output, diagnostic[i], rtol=0, atol=0)
    print(f"{len(ids)} cached tokens passed; one activation upload and one output download per token")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--worker", required=True)
    parser.add_argument("--capture-worker")
    parser.add_argument("--gemv-elf")
    parser.add_argument("--attention-elf")
    parser.add_argument("--pointwise-elf")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 42, 30, 198])
    args = parser.parse_args()
    options = {k: getattr(args, k) for k in ("capture_worker", "gemv_elf", "attention_elf", "pointwise_elf")
               if getattr(args, k)}
    if options and len(options) != 4:
        parser.error("capture requires a worker and all three ELF paths")
    model = LlamaForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
                                            dtype=torch.float16, attn_implementation="eager").eval()
    block = ndp.nn.LlamaDecodeBlock(model.model.layers[0], max_tokens=len(args.tokens),
                                    backend="gem5_pim", worker=args.worker, **options)
    verify(model, block, args.tokens)
