"""Verify resident two-device SwiGLU and GQA decoding against a local Llama checkpoint."""
import argparse
from contextlib import ExitStack

import ndp
import torch
from transformers import LlamaForCausalLM
from llama_stages import counted_worker


@torch.inference_mode()
def verify(model, worker):
    source = model.model.layers[0]
    hidden = model.model.embed_tokens(torch.tensor([[1, 42, 30, 198]]))
    attention_input = source.input_layernorm(hidden)
    positions = torch.arange(4).unsqueeze(0)
    mask = torch.full((4, 4), float('-inf'), dtype=torch.float16).triu(1)
    expected_attention = source.self_attn(
        attention_input, attention_mask=mask[None, None],
        position_embeddings=model.model.rotary_emb(hidden, positions))[0]
    mlp_input = source.post_attention_layernorm(hidden + expected_attention)
    expected_mlp = source.mlp(mlp_input)
    with ExitStack() as resources:
        endpoints = [resources.enter_context(counted_worker(worker)) for _ in range(2)]
        group = ndp.DeviceGroup([session for session, _ in endpoints])
        attention = ndp.nn.TensorParallelLlamaAttention(source.self_attn, devices=group, max_tokens=4)
        mlp = ndp.nn.TensorParallelSwiGLU(source.mlp, devices=group)
        print(f'Local Q heads: {attention.local_q_heads}; KV heads: {attention.local_kv_heads}')
        for module, inputs, expected in ((attention, attention_input, expected_attention),
                                         (mlp, mlp_input, expected_mlp)):
            outputs = []
            for token in inputs.split(1, dim=1):
                replicas = tuple(session.upload(token.reshape(1, -1)) for session, _ in endpoints)
                resident = ndp.nn.DistributedRow(group, replicas, token.numel(), 'replicated')
                for session, log in endpoints:
                    session.synchronize()
                    log.read()  # Exclude preparation and external input uploads.
                partials = module.forward_distributed(resident)
                for tensor in partials.tensors:
                    tensor.wait()
                for _, log in endpoints:
                    assert not log.read(), 'Local computation transferred activation data through the host'
                output = group.device(0).download(group.reduce_partials(partials.tensors))
                outputs.append(output.reshape_as(token))
                size = token.numel() * 2
                for i, (_, log) in enumerate(endpoints):
                    expected_frames = [f'D {size*2}']
                    if i == 0:
                        expected_frames += [f'U {size}', f'D {size}']
                    assert log.read().splitlines() == expected_frames
            actual = torch.cat(outputs, dim=1)
            torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
            print(f'{type(module).__name__}: max absolute error {(actual-expected).abs().max().item():.9g}; '
                  'zero intermediate U/D frames; FP32 host reduction verified')
            if module is attention:
                attention.reset_history()
                for token, previous in zip(inputs.split(1, dim=1), outputs):
                    torch.testing.assert_close(attention(token), previous, atol=0, rtol=0)


        block = ndp.nn.TensorParallelLlamaDecodeBlock(source, devices=group, max_tokens=4)
        expected = source(hidden, attention_mask=mask[None, None],
                          position_embeddings=model.model.rotary_emb(hidden, positions))
        if isinstance(expected, tuple):
            expected = expected[0]
        for session, log in endpoints:
            session.synchronize()
            log.read()
        outputs = []
        size = hidden.shape[-1] * 2
        for token in hidden.split(1, dim=1):
            outputs.append(block(token))
            # Root retains residuals and runs norms; only normalized rows are replicated.
            expected_frames = [
                [f'U {size}', f'D {size}', f'D {size*2}', f'U {size}',
                 f'D {size}', f'D {size*2}', f'U {size}', f'D {size}'],
                [f'U {size}', f'D {size*2}', f'U {size}', f'D {size*2}'],
            ]
            for i, (_, log) in enumerate(endpoints):
                assert log.read().splitlines() == expected_frames[i]
        actual = torch.cat(outputs, dim=1)
        torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
        print(f'TensorParallelLlamaDecodeBlock: max absolute error '
              f'{(actual-expected).abs().max().item():.9g}; full-block frame counts verified')
        block.reset_history()
        for token, previous in zip(hidden.split(1, dim=1), outputs):
            torch.testing.assert_close(block(token), previous, atol=0, rtol=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    model = LlamaForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
                                            dtype=torch.float16, attn_implementation='eager').eval()
    verify(model, args.worker)
