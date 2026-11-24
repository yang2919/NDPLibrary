"""Interleave independent decode requests on one shared-weight Llama model."""
import argparse
from contextlib import ExitStack
import json

import ndp
import torch
from transformers import LlamaForCausalLM


@torch.inference_mode()
def verify(source, worker):
    model = ndp.nn.TensorParallelLlamaForCausalLM(source,
        devices=[dict(backend='gem5_pim', worker=worker)]*2, max_tokens=4)
    def memory():
        group = model.model.devices
        for i in range(len(group)):
            group.device(i).synchronize()
        return [group.device(i).memory_stats().live_bytes for i in range(len(group))]
    baseline = memory()
    inputs = ([1, 42], [30, 198])
    expected = []
    for tokens in inputs:
        with model.new_request(max_tokens=len(tokens)) as request:
            expected.append([model(torch.tensor([[token]]), request=request) for token in tokens])
    assert memory() == baseline, 'Serial request caches were not released'
    with ExitStack() as resources:
        requests = [resources.enter_context(model.new_request(max_tokens=len(tokens))) for tokens in inputs]
        assert memory() == baseline, 'Creating empty requests allocated device memory'
        for position in range(2):
            for i, request in enumerate(requests):
                logits = model(torch.tensor([[inputs[i][position]]]), request=request)
                torch.testing.assert_close(logits, expected[i][position], atol=0, rtol=0)
        active = memory()
        assert [request.position for request in requests] == [2, 2]
        # Reset only one request while the other retains its two-token history.
        requests[0].reset()
        after_reset = memory()
        assert requests[1].position == 2
        torch.testing.assert_close(model(torch.tensor([[inputs[0][0]]]), request=requests[0]),
                                   expected[0][0], atol=0, rtol=0)
    released = memory()
    assert released == baseline, 'Explicit close retained request-owned memory'
    print(json.dumps(dict(layers=len(source.model.layers), exact_logits_match=True,
        prepared_live_bytes=baseline, two_requests_live_bytes=active,
        after_one_reset_live_bytes=after_reset, after_close_live_bytes=released,
        request_cache_bytes=[a-b for a, b in zip(active, baseline)]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    source = LlamaForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
        dtype=torch.float16, attn_implementation='eager').eval()
    verify(source, args.worker)
