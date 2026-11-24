"""Verify PNM GEMM prefill packets and continuation with PIM single-token decode."""
import argparse
from contextlib import ExitStack
import json

import ndp
import torch
from transformers import LlamaForCausalLM
from llama_stages import counted_worker


@torch.inference_mode()
def verify(source, worker):
    ids = torch.tensor([[1, 42, 30, 198]])
    with ExitStack() as resources:
        endpoints = [resources.enter_context(counted_worker(worker, kernels=True)) for _ in range(2)]
        model = ndp.nn.TensorParallelLlamaForCausalLM(source,
            devices=[session for session, _ in endpoints], max_tokens=8)
        reference = torch.cat([model(token) for token in ids.split(1, 1)], dim=1)
        next_id = reference[:, -1].argmax(-1, keepdim=True)
        reference_decode = model(next_id)
        model.reset_history()
        for session, log in endpoints:
            session.synchronize()
            log.read()
        actual = model.prefill(ids)
        commands = []
        for _, log in endpoints:
            lines = log.read().splitlines()
            linear = [list(map(int, line.split()[1:])) for line in lines if line.startswith('A 2 ')]
            assert len(linear) == 7*len(source.model.layers)
            assert all(shape[1] == ids.shape[1] for shape in linear)
            assert 'G' not in lines, 'Prefill unexpectedly used PIM GEMV'
            commands.append(dict(pnm_linear=len(linear), rows=ids.shape[1]))
        actual_decode = model(next_id)
        for _, log in endpoints:
            lines = log.read().splitlines()
            assert lines.count('G') == 7*len(source.model.layers)
            assert not any(line.startswith('A 2 ') for line in lines)
        torch.testing.assert_close(actual[:, -1].argmax(-1), next_id[:, 0], atol=0, rtol=0)
        torch.testing.assert_close(actual_decode.argmax(-1), reference_decode.argmax(-1), atol=0, rtol=0)
        model.reset_history()
        chunked = torch.cat([model.prefill(chunk) for chunk in ids.split(2, 1)], dim=1)
        for value in (actual, actual_decode, chunked):
            assert torch.isfinite(value).all()
        print(json.dumps(dict(layers=len(source.model.layers), prompt_tokens=4,
            prefill_commands_per_device=commands, decode_gemv_per_device=7*len(source.model.layers),
            prefill_vs_tokenwise_max_logit_error=(actual-reference).abs().max().item(),
            decode_after_prefill_max_logit_error=(actual_decode-reference_decode).abs().max().item(),
            chunked_vs_full_max_logit_error=(chunked-actual).abs().max().item(),
            greedy_continuation_matches=True), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    source = LlamaForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
        dtype=torch.float16, attn_implementation='eager').eval()
    verify(source, args.worker)
