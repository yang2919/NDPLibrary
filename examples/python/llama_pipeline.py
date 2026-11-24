"""Verify overlapped TP stages against sequential generation and record host stage intervals."""
import argparse
import json

import ndp
import torch
from transformers import LlamaForCausalLM


@torch.inference_mode()
def verify(source, worker):
    options = dict(backend='gem5_pim', worker=worker)
    groups = [[options]*2, [options]*2]
    midpoint = len(source.model.layers)//2
    placement = [0]*midpoint + [1]*(len(source.model.layers)-midpoint)
    prompts = [torch.tensor([[1, 42]]), torch.tensor([[30, 198]])]
    reference = ndp.nn.TensorParallelLlamaForCausalLM(source, device_groups=groups,
                                                      placement=placement, max_tokens=3)
    try:
        expected = [reference.generate(prompt, max_new_tokens=2, eos_token_id=[]) for prompt in prompts]
    finally:
        for i in range(len(reference.model.devices)):
            reference.model.devices.device(i).close()
    del reference
    with ndp.pipeline.PipelineGenerationScheduler(source, device_groups=groups,
            placement=placement, max_tokens=3, max_active_requests=2, record_timing=True) as scheduler:
        baseline = scheduler.memory_stats()
        requests = [scheduler.submit(prompt, max_new_tokens=2, eos_token_id=[]) for prompt in prompts]
        scheduler.run()
        for request, expected_ids in zip(requests, expected):
            assert request.status == 'completed', request.error
            torch.testing.assert_close(request.token_ids, expected_ids, atol=0, rtol=0)
        after = scheduler.memory_stats()
        for before_group, after_group in zip(baseline, after):
            assert [s['live_bytes'] for s in before_group] == [s['live_bytes'] for s in after_group]
        timings = scheduler.timings
        overlaps = []
        for i, a in enumerate(timings):
            for b in timings[i+1:]:
                duration = min(a['finished'], b['finished'])-max(a['started'], b['started'])
                if a['stage'] != b['stage'] and a['request_id'] != b['request_id'] and duration > 0:
                    overlaps.append(duration)
        if not overlaps:
            raise AssertionError('No cross-request stage overlap observed in this run')
        print(json.dumps(dict(layers=len(source.model.layers), generated_ids=[r.token_ids.tolist() for r in requests],
            exact_generation_match=True, overlapping_interval_pairs=len(overlaps),
            maximum_host_interval_overlap_seconds=max(overlaps), preparation_memory=baseline,
            after_completion_memory=after, stage_timings=timings), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    source = LlamaForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
        dtype=torch.float16, attn_implementation='eager').eval()
    if len(source.model.layers) < 2:
        parser.error('At least two source layers are required')
    verify(source, args.worker)
