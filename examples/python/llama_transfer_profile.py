"""Measure full-model activation frames and runtime memory with shared broadcast staging."""
import argparse
from contextlib import ExitStack
import json

import ndp
import torch
from transformers import LlamaForCausalLM
from llama_stages import counted_worker


def memory(session):
    stats = session.memory_stats()
    return {name: getattr(stats, name) for name in
            ('live_bytes', 'peak_bytes', 'allocations', 'staging_bytes', 'workspace_bytes')}


@torch.inference_mode()
def verify(source, worker, count):
    with ExitStack() as resources:
        endpoints = [resources.enter_context(counted_worker(worker)) for _ in range(count)]
        group = ndp.DeviceGroup([session for session, _ in endpoints])
        model = ndp.nn.TensorParallelLlamaForCausalLM(source, devices=group, max_tokens=2)
        for session, log in endpoints:
            session.synchronize()
            log.read()  # Weight, RoPE and normalization preparation is outside token accounting.
        preparation = [memory(session) for session, _ in endpoints]
        broadcast = group.broadcast
        outputs, reports = {}, {}
        try:
            for phase in ('repeated_copy', 'shared_staging'):
                group.broadcast = (lambda tensor: [group.copy_to(tensor, i) for i in range(count)]) \
                    if phase == 'repeated_copy' else broadcast
                model.reset_history()
                outputs[phase], reports[phase] = [], []
                for token in (1, 42):
                    outputs[phase].append(model(torch.tensor([[token]])))
                    devices = []
                    for session, log in endpoints:
                        frames = [line.split() for line in log.read().splitlines()]
                        devices.append({
                            'upload_bytes': sum(int(n) for op, n in frames if op == 'U'),
                            'download_bytes': sum(int(n) for op, n in frames if op == 'D'),
                            'upload_frames': sum(op == 'U' for op, _ in frames),
                            'download_frames': sum(op == 'D' for op, _ in frames),
                            'memory': memory(session)})
                    reports[phase].append(devices)
        finally:
            group.broadcast = broadcast
        width_bytes = source.config.hidden_size*2
        layers = len(source.model.layers)
        for baseline, optimized in zip(outputs['repeated_copy'], outputs['shared_staging']):
            torch.testing.assert_close(optimized, baseline, atol=0, rtol=0)
        for phase, tokens in reports.items():
            copies = count-1 if phase == 'repeated_copy' else 1
            for devices in tokens:
                for i, stats in enumerate(devices):
                    expected_u = (2*layers + (i == 0))*width_bytes
                    expected_d = ((4+2*copies)*layers+1)*width_bytes if i == 0 else 4*layers*width_bytes
                    assert stats['upload_bytes'] == expected_u, (phase, i, stats)
                    assert stats['download_bytes'] == expected_d, (phase, i, stats)
        print(json.dumps(dict(devices=count, layers=layers, width_bytes=width_bytes,
            preparation_memory=preparation, phases=reports, exact_logits_match=True), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--worker', required=True)
    parser.add_argument('--devices', type=int, default=3)
    args = parser.parse_args()
    if args.devices < 2:
        parser.error('Use at least two devices')
    source = LlamaForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
        dtype=torch.float16, attn_implementation='eager').eval()
    verify(source, args.worker, args.devices)
