"""Check round-robin fairness, generation equivalence and terminal cache reclamation."""
import os
import unittest
from unittest.mock import patch

import ndp
import torch
from test_tensor_parallel_model import tiny_model


@unittest.skipUnless(os.getenv('NDP_GEM5_PIM_WORKER'), 'Configure the FP16 worker')
class SchedulerTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(78)
        options = dict(backend='gem5_pim', worker=os.environ['NDP_GEM5_PIM_WORKER'])
        return ndp.nn.TensorParallelLlamaForCausalLM(tiny_model(), devices=[options]*2, max_tokens=8)

    def memory(self, model):
        sessions = [model.model.devices.device(i) for i in range(len(model.model.devices))]
        for session in sessions:
            session.synchronize()
        return [session.memory_stats().live_bytes for session in sessions]

    def test_round_robin_and_results(self):
        model = self.make_model()
        inputs = [torch.tensor([[1, 2, 3]]), torch.tensor([[4]]), torch.tensor([[5, 6]])]
        reference = [model.generate(ids, max_new_tokens=2, eos_token_id=[]) for ids in inputs]
        model.reset_history()
        baseline = self.memory(model)
        with ndp.serving.GenerationScheduler(model, max_active_requests=2) as scheduler:
            requests = [scheduler.submit(ids, max_new_tokens=2, eos_token_id=[]) for ids in inputs]
            self.assertEqual(self.memory(model), baseline)
            events = []
            while scheduler.pending:
                events.append(scheduler.step())
                self.assertLessEqual(len(scheduler._active), 2)
            self.assertEqual([e.request_id for e in events], [0, 1, 0, 1, 0, 2, 0, 2, 2])
            self.assertEqual([e.phase for e in events[:4]], ['prefill', 'prefill', 'prefill', 'decode'])
            self.assertIsNone(events[0].token_id)
            self.assertIsNotNone(events[1].token_id)
            self.assertIsNone(scheduler.step())
            for result, expected in zip(requests, reference):
                self.assertEqual(result.status, 'completed')
                self.assertIsNone(result._history)
                torch.testing.assert_close(result.token_ids, expected, atol=0, rtol=0)
            self.assertEqual(self.memory(model), baseline)
            new = scheduler.submit(inputs[0], max_new_tokens=1, eos_token_id=[])
            scheduler.run()
            self.assertEqual(new.status, 'completed')
        with self.assertRaises(RuntimeError):
            scheduler.submit(inputs[0])

    def test_cancel_eos_and_failure(self):
        model = self.make_model()
        ids = torch.tensor([[1, 2]])
        expected = model.generate(ids, max_new_tokens=1, eos_token_id=[])
        model.reset_history()
        baseline = self.memory(model)
        with ndp.serving.GenerationScheduler(model, max_active_requests=2) as scheduler:
            zero = scheduler.submit(ids, max_new_tokens=0)
            self.assertTrue(zero.done)
            torch.testing.assert_close(zero.token_ids, ids)
            a = scheduler.submit(ids, max_new_tokens=2, eos_token_id=[])
            b = scheduler.submit(ids, max_new_tokens=2, eos_token_id=[])
            c = scheduler.submit(ids, max_new_tokens=2, eos_token_id=[])
            scheduler.cancel(c)
            self.assertEqual(c.status, 'cancelled')
            scheduler.step()
            scheduler.cancel(a)
            scheduler.cancel(a)
            self.assertEqual(a.status, 'cancelled')
            healthy = scheduler.submit(ids, max_new_tokens=1, eos_token_id=[])
            def fail_norm(resident):
                raise RuntimeError('injected')
            with patch.object(model.model.norm, 'forward', fail_norm):
                event = scheduler.step()
            self.assertEqual((event.request_id, event.status), (b.request_id, 'failed'))
            self.assertIn('injected', b.error)
            self.assertEqual(scheduler.pending, 1)
            scheduler.run()
            self.assertEqual(healthy.status, 'completed')
            torch.testing.assert_close(healthy.token_ids, expected)
            self.assertEqual(self.memory(model), baseline)
            eos = scheduler.submit(ids, max_new_tokens=4, eos_token_id=int(expected[0, -1]))
            scheduler.run()
            self.assertEqual(eos.generated_tokens, 1)
            torch.testing.assert_close(eos.token_ids, expected)
            active = scheduler.submit(ids, max_new_tokens=2)
            waiting = scheduler.submit(ids, max_new_tokens=2)
            scheduler.step()
        self.assertEqual((active.status, waiting.status), ('cancelled', 'cancelled'))
        self.assertEqual(self.memory(model), baseline)

    def test_validation_and_interrupt(self):
        model = self.make_model()
        with ndp.serving.GenerationScheduler(model) as scheduler:
            for kwargs in ({'max_new_tokens': -1}, {'max_new_tokens': 9}, {'eos_token_id': 19}):
                with self.assertRaises(ValueError):
                    scheduler.submit(torch.tensor([[1]]), **kwargs)
            with self.assertRaises(ValueError):
                scheduler.submit(torch.tensor([[19]]), max_new_tokens=1)
            with ndp.serving.GenerationScheduler(model) as other:
                foreign = other.submit(torch.tensor([[1]]), max_new_tokens=1)
                with self.assertRaises(ValueError):
                    scheduler.cancel(foreign)
            request = scheduler.submit(torch.tensor([[1]]), max_new_tokens=1)
            with patch.object(model, 'forward', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    scheduler.step()
            self.assertEqual(request.status, 'cancelled')
            self.assertIsNone(request._history)
            self.assertEqual(scheduler.pending, 0)

            admitted = scheduler.submit(torch.tensor([[1]]), max_new_tokens=1)
            with patch.object(model, 'new_request', side_effect=RuntimeError('admission failed')):
                event = scheduler.step()
            self.assertEqual((event.phase, admitted.status), ('admission', 'failed'))
            reentrant = scheduler.submit(torch.tensor([[1]]), max_new_tokens=1)
            with patch.object(model, 'forward', side_effect=lambda *args, **kwargs: scheduler.step()):
                event = scheduler.step()
            self.assertEqual(reentrant.status, 'failed')
            self.assertIn('reentrant', event.error)
