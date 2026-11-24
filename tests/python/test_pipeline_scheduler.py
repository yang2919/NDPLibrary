"""Verify thread-owned stages, cross-request overlap and ordered cache cleanup."""
import os
import threading
import unittest
from unittest.mock import patch

import ndp
import torch
from ndp.pipeline import _Stage
from test_tensor_parallel_model import tiny_model


@unittest.skipUnless(os.getenv('NDP_GEM5_PIM_WORKER'), 'Configure the FP16 worker')
class PipelineTests(unittest.TestCase):
    def setup_model(self):
        torch.manual_seed(79)
        source = tiny_model()
        options = dict(backend='gem5_pim', worker=os.environ['NDP_GEM5_PIM_WORKER'])
        return source, [[options]*2, [options]*2]

    def live(self, scheduler):
        return [[item['live_bytes'] for item in group] for group in scheduler.memory_stats()]

    def test_overlap_and_order(self):
        source, groups = self.setup_model()
        reference = ndp.nn.TensorParallelLlamaForCausalLM(source, device_groups=groups,
                                                        placement=[0, 1], max_tokens=4)
        prompts = [torch.tensor([[1, 2]]), torch.tensor([[3, 4]])]
        expected = [reference.generate(prompt, max_new_tokens=2, eos_token_id=[]) for prompt in prompts]
        original = _Stage.run
        barrier = threading.Barrier(2, timeout=10)
        reached, threads = [], set()
        def overlapping(stage, request_id, capacity, hidden):
            threads.add(threading.get_ident())
            key = (stage.index, request_id)
            if key in ((0, 1), (1, 0)) and key not in reached:
                reached.append(key)
                barrier.wait()  # Would fail if stages were executed serially on one thread.
            return original(stage, request_id, capacity, hidden)
        with ndp.pipeline.PipelineGenerationScheduler(source, device_groups=groups,
                placement=[0, 1], max_tokens=4, max_active_requests=2, record_timing=True) as scheduler:
            baseline = self.live(scheduler)
            requests = [scheduler.submit(prompt, max_new_tokens=2, eos_token_id=[]) for prompt in prompts]
            with patch.object(_Stage, 'run', overlapping):
                scheduler.run()
            self.assertEqual(len(threads), 2)
            self.assertEqual(len(reached), 2)
            for request, expected_ids in zip(requests, expected):
                self.assertEqual(request.status, 'completed', request.error)
                torch.testing.assert_close(request.token_ids, expected_ids, atol=0, rtol=0)
            for request in requests:
                trace = [t for t in scheduler.timings if t['request_id'] == request.request_id]
                self.assertEqual([(t['token_index'], t['stage']) for t in trace],
                                 [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])
            self.assertEqual(self.live(scheduler), baseline)

    def test_failure_cancel_and_reuse(self):
        source, groups = self.setup_model()
        original = _Stage.run
        with ndp.pipeline.PipelineGenerationScheduler(source, device_groups=groups,
                placement=[0, 1], max_tokens=4, max_active_requests=2) as scheduler:
            baseline = self.live(scheduler)
            bad = scheduler.submit(torch.tensor([[1]]), max_new_tokens=2, eos_token_id=[])
            good = scheduler.submit(torch.tensor([[2]]), max_new_tokens=2, eos_token_id=[])
            def fail(stage, request_id, capacity, hidden):
                if stage.index == 1 and request_id == bad.request_id:
                    raise RuntimeError('stage failure')
                return original(stage, request_id, capacity, hidden)
            with patch.object(_Stage, 'run', fail):
                scheduler.run()
            self.assertEqual(bad.status, 'failed')
            self.assertIn('stage failure', bad.error)
            self.assertEqual(good.status, 'completed', good.error)
            self.assertEqual(self.live(scheduler), baseline)
            a = scheduler.submit(torch.tensor([[1, 2]]), max_new_tokens=2, eos_token_id=[])
            b = scheduler.submit(torch.tensor([[3]]), max_new_tokens=2, eos_token_id=[])
            queued = scheduler.submit(torch.tensor([[4]]), max_new_tokens=2, eos_token_id=[])
            scheduler.step()
            scheduler.cancel(queued)
            scheduler.cancel(a)
            scheduler.cancel(a)
            scheduler.run()
            self.assertEqual((a.status, queued.status, b.status), ('cancelled', 'cancelled', 'completed'))
            self.assertEqual(self.live(scheduler), baseline)
            later = scheduler.submit(torch.tensor([[1]]), max_new_tokens=1)
            scheduler.step()
        self.assertEqual(later.status, 'cancelled')
        self.assertTrue(all(not executor._threads or all(not t.is_alive() for t in executor._threads)
                            for executor in scheduler._executors))

    def test_configuration_and_initialization_failure(self):
        source, groups = self.setup_model()
        for placement in ([1, 0], [0, 0], [0], [False, 1]):
            with self.assertRaises(ValueError):
                ndp.pipeline.PipelineGenerationScheduler(source, device_groups=groups,
                                                         placement=placement, max_tokens=4)
        broken = [[groups[0][0]], [dict(backend='not_a_backend')]]
        with self.assertRaises(Exception):
            ndp.pipeline.PipelineGenerationScheduler(source, device_groups=broken,
                                                     placement=[0, 1], max_tokens=4)
        self.assertFalse(any(t.name.startswith('ndp-stage-') for t in threading.enumerate()))

    def test_interrupt_and_terminal_requests(self):
        source, groups = self.setup_model()
        with ndp.pipeline.PipelineGenerationScheduler(source, device_groups=groups,
                placement=[0, 1], max_tokens=4) as scheduler:
            zero = scheduler.submit(torch.tensor([[1]]), max_new_tokens=0)
            self.assertEqual(zero.status, 'completed')
            first = scheduler.submit(torch.tensor([[1]]), max_new_tokens=1, eos_token_id=[])
            scheduler.run()
            eos_id = int(first.token_ids[0, -1])
            stopped = scheduler.submit(torch.tensor([[1]]), max_new_tokens=3, eos_token_id=eos_id)
            scheduler.run()
            self.assertEqual(stopped.generated_tokens, 1)
            for kwargs in ({'max_new_tokens': 5}, {'eos_token_id': 19}):
                with self.assertRaises(ValueError):
                    scheduler.submit(torch.tensor([[1]]), **kwargs)
            active = scheduler.submit(torch.tensor([[1, 2]]), max_new_tokens=2)
            with patch('ndp.pipeline.wait', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    scheduler.step()
            self.assertEqual(active.status, 'cancelled')
            self.assertEqual(scheduler.pending, 0)
            self.assertTrue(scheduler._closed)
