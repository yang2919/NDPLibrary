"""Check PNM prompt chunks, causal cache continuity and FP32 partial GEMMs."""
import os
import unittest
from unittest.mock import patch

import ndp
import torch
from test_tensor_parallel_model import tiny_model


@unittest.skipUnless(os.getenv('NDP_GEM5_PIM_WORKER'), 'Configure the FP16 worker')
class PrefillTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(80)
        options = dict(backend='gem5_pim', worker=os.environ['NDP_GEM5_PIM_WORKER'])
        return ndp.nn.TensorParallelLlamaForCausalLM(tiny_model(), devices=[options]*2, max_tokens=8)

    def test_chunk_and_decode(self):
        model = self.model()
        ids = torch.tensor([[1, 2, 3, 4]])
        expected = torch.cat([model(token) for token in ids.split(1, 1)], dim=1)
        next_expected = model(torch.tensor([[5]]))
        model.reset_history()
        with model.new_request() as request:
            actual = model.prefill(ids, request=request)
            torch.testing.assert_close(actual, expected, atol=.005, rtol=.03)
            self.assertEqual(request.position, 4)
            torch.testing.assert_close(model(torch.tensor([[5]]), request=request), next_expected,
                                       atol=.005, rtol=.03)
            request.reset()
            chunks = torch.cat([model.prefill(chunk, request=request) for chunk in ids.split(2, 1)], dim=1)
            torch.testing.assert_close(chunks, actual, atol=0, rtol=0)
            request.reset()
            prefix = model.prefill(ids[:, :3], request=request)
            changed = model.prefill(torch.tensor([[6]]), request=request)
            torch.testing.assert_close(prefix, actual[:, :3], atol=0, rtol=0)
            self.assertEqual(changed.shape, (1, 1, 19))
        reference = model.generate(ids, max_new_tokens=3, eos_token_id=[])
        chunked = model.generate(ids, max_new_tokens=3, eos_token_id=[], prefill_chunk_size=3)
        torch.testing.assert_close(chunked, reference)

    def test_capacity_failure_and_fp32(self):
        model = self.model()
        with model.new_request(max_tokens=2) as request:
            with self.assertRaises(ValueError):
                model.prefill(torch.tensor([[1, 2, 3]]), request=request)
            self.assertEqual(request.position, 0)
            self.assertFalse(request.failed)
            with patch.object(model.model.layers[1].mlp, 'forward_distributed', side_effect=RuntimeError('chunk failed')):
                with self.assertRaisesRegex(RuntimeError, 'chunk failed'):
                    model.prefill(torch.tensor([[1, 2]]), request=request)
            self.assertTrue(request.failed)
            self.assertEqual(request._layers[0].attention._position, 2)
            request.reset()
            model.prefill(torch.tensor([[1, 2]]), request=request)
        group = model.model.devices
        parts = []
        for i in range(2):
            session = group.device(i)
            x = session.upload(torch.full((3, 2), 200.).half())
            w = session.upload(torch.full((1, 2), 200. if i == 0 else -200.).half())
            partial = session.linear_partial(x, w, ndp.Engine.PNM)
            self.assertEqual(partial.dtype, ndp.DType.F32)
            self.assertEqual(session.download(partial)[0, 0], 80000. if i == 0 else -80000.)
            parts.append(partial)
        result = group.device(0).download(group.reduce_partials(parts))
        torch.testing.assert_close(result, torch.zeros(3, 1).half(), atol=0, rtol=0)
        shards = [group.device(i).upload(torch.full((3, i+1), float(i)).half()) for i in range(2)]
        for i, replica in enumerate(group.all_gather_rows(shards)):
            torch.testing.assert_close(group.device(i).download(replica),
                                       torch.tensor([[0., 1., 1.]]*3).half(), atol=0, rtol=0)

    def test_schedulers_with_chunks(self):
        model = self.model()
        ids = torch.tensor([[1, 2, 3]])
        expected = model.generate(ids, max_new_tokens=2, eos_token_id=[], prefill_chunk_size=2)
        with ndp.serving.GenerationScheduler(model, max_active_requests=2, prefill_chunk_size=2) as scheduler:
            request = scheduler.submit(ids, max_new_tokens=2, eos_token_id=[])
            events = []
            while scheduler.pending:
                events.append(scheduler.step())
            self.assertEqual([event.input_tokens for event in events], [2, 1, 1])
            torch.testing.assert_close(request.token_ids, expected)
        options = dict(backend='gem5_pim', worker=os.environ['NDP_GEM5_PIM_WORKER'])
        torch.manual_seed(80)
        source = tiny_model()
        with ndp.pipeline.PipelineGenerationScheduler(source, device_groups=[[options]*2, [options]*2],
                placement=[0, 1], max_tokens=8, prefill_chunk_size=2) as scheduler:
            request = scheduler.submit(ids, max_new_tokens=2, eos_token_id=[])
            events = []
            while scheduler.pending:
                events.append(scheduler.step())
            self.assertEqual(request.status, 'completed', request.error)
            self.assertEqual([event.input_tokens for event in events], [2, 2, 1, 1, 1, 1])
            torch.testing.assert_close(request.token_ids, expected)
