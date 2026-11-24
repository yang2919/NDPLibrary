"""Run queued greedy requests with shared weights and compare independent generation."""
import argparse
from dataclasses import asdict
import json

import ndp
import torch
from transformers import LlamaForCausalLM


@torch.inference_mode()
def verify(source, worker):
    model = ndp.nn.TensorParallelLlamaForCausalLM(source,
        devices=[dict(backend='gem5_pim', worker=worker)]*2, max_tokens=4)
    prompts = [torch.tensor([ids]) for ids in ([1, 42], [30], [198, 504])]
    expected = [model.generate(ids, max_new_tokens=2, eos_token_id=[]) for ids in prompts]
    model.reset_history()
    def memory():
        devices = model.model.devices
        for i in range(len(devices)):
            devices.device(i).synchronize()
        return [devices.device(i).memory_stats().live_bytes for i in range(len(devices))]
    baseline = memory()
    with ndp.serving.GenerationScheduler(model, max_active_requests=2) as scheduler:
        requests = [scheduler.submit(ids, max_new_tokens=2, eos_token_id=[]) for ids in prompts]
        assert memory() == baseline, 'Queued requests allocated device cache'
        events = []
        while scheduler.pending:
            event = scheduler.step()
            events.append(asdict(event))
            if event.status == 'failed':
                raise AssertionError(event.error)
        for request, reference in zip(requests, expected):
            assert request.status == 'completed'
            torch.testing.assert_close(request.token_ids, reference, atol=0, rtol=0)
    released = memory()
    assert released == baseline, 'Completed requests retained cache memory'
    print(json.dumps(dict(layers=len(source.model.layers), max_active_requests=2,
        events=events, generated_ids=[request.token_ids.tolist() for request in requests],
        exact_generation_match=True, prepared_live_bytes=baseline,
        after_completion_live_bytes=released), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    source = LlamaForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
        dtype=torch.float16, attn_implementation='eager').eval()
    verify(source, args.worker)
