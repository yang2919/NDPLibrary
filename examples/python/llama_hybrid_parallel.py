"""Compare full-model TP with two sequential TP stages and verify boundary traffic."""
import argparse
from contextlib import ExitStack
import json

import ndp
import torch
from transformers import LlamaForCausalLM
from llama_stages import counted_worker


@torch.inference_mode()
def run(source, worker, placement):
    group_count = max(placement)+1
    with ExitStack() as resources:
        endpoints = [resources.enter_context(counted_worker(worker)) for _ in range(2*group_count)]
        groups = [ndp.DeviceGroup([endpoints[2*i][0], endpoints[2*i+1][0]]) for i in range(group_count)]
        model = ndp.nn.TensorParallelLlamaForCausalLM(source, device_groups=groups,
                                                      placement=placement, max_tokens=2)
        for session, log in endpoints:
            session.synchronize()
            log.read()
        results, traffic = [], []
        size = source.config.hidden_size*2
        for token in (1, 42):
            results.append(model(torch.tensor([[token]])))
            devices = []
            for ordinal, (session, log) in enumerate(endpoints):
                frames = [line.split() for line in log.read().splitlines()]
                upload = sum(int(n) for op, n in frames if op == 'U')
                download = sum(int(n) for op, n in frames if op == 'D')
                group, local = divmod(ordinal, 2)
                layers = placement.count(group)
                incoming = sum(a != b and b == group for a, b in zip(placement, placement[1:]))
                outgoing = sum(a != b and a == group for a, b in zip(placement, placement[1:]))
                expected_u = (2*layers + (incoming+(placement[0] == group) if local == 0 else 0))*size
                expected_d = ((6*layers+outgoing+(placement[-1] == group)) if local == 0 else 4*layers)*size
                assert (upload, download) == (expected_u, expected_d), (ordinal, upload, download)
                stats = session.memory_stats()
                devices.append(dict(upload_bytes=upload, download_bytes=download,
                                    live_bytes=stats.live_bytes, peak_bytes=stats.peak_bytes))
            traffic.append(devices)
        return results, traffic


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    source = LlamaForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
        dtype=torch.float16, attn_implementation='eager').eval()
    layers = len(source.model.layers)
    if layers < 2:
        parser.error('At least two decoder layers are required')
    expected, baseline_traffic = run(source, args.worker, [0]*layers)
    placement = [0]*(layers//2) + [1]*(layers-layers//2)
    actual, hybrid_traffic = run(source, args.worker, placement)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    print(json.dumps(dict(placement=placement, baseline=baseline_traffic,
                         hybrid=hybrid_traffic, exact_logits_match=True), indent=2))
