"""Verify shared weights with isolated request histories, failures and cache lifetimes."""
import os
import unittest
from unittest.mock import patch

import ndp
import torch
from test_tensor_parallel_model import tiny_model


@unittest.skipUnless(os.getenv('NDP_GEM5_PIM_WORKER'), 'Configure the FP16 worker')
class DecodeRequestTests(unittest.TestCase):
    @torch.inference_mode()
    def test_interleaving_and_lifetime(self):
        torch.manual_seed(76)
        group = ndp.DeviceGroup([dict(backend='gem5_pim', worker=os.environ['NDP_GEM5_PIM_WORKER'])]*2)
        model = ndp.nn.TensorParallelLlamaForCausalLM(tiny_model(), devices=group, max_tokens=4)
        def live():
            for i in range(len(group)):
                group.device(i).synchronize()
            return [group.device(i).memory_stats().live_bytes for i in range(len(group))]
        baseline = live()
        keys = set(model.state_dict())
        sequences = ([1, 2, 3], [4, 5])
        expected = []
        for sequence in sequences:
            model.reset_history()
            expected.append([model(torch.tensor([[t]])) for t in sequence])
        model.reset_history()
        self.assertEqual(live(), baseline)
        with model.new_request(max_tokens=3) as a, model.new_request(max_tokens=2) as b:
            self.assertEqual(live(), baseline)  # Cache allocation is lazy; weights are shared.
            for i in range(3):
                torch.testing.assert_close(model(torch.tensor([[sequences[0][i]]]), request=a),
                                           expected[0][i], atol=0, rtol=0)
                if i < 2:
                    torch.testing.assert_close(model(torch.tensor([[sequences[1][i]]]), request=b),
                                               expected[1][i], atol=0, rtol=0)
            self.assertEqual((a.position, b.position, model.model._position), (3, 2, 0))
            self.assertEqual(set(model.state_dict()), keys)
            with self.assertRaises(ValueError):
                model(torch.tensor([[1]]), request=b)
            self.assertFalse(b.failed)
            a.reset()
            self.assertEqual((a.position, b.position), (0, 2))
            before_close = live()
            a.close()
            self.assertEqual(live(), before_close)
            with self.assertRaises(RuntimeError):
                model(torch.tensor([[1]]), request=a)
            with self.assertRaises(RuntimeError):
                a.reset()
        self.assertEqual(live(), baseline)
        self.assertTrue(a.closed and b.closed)
        for cap in (0, 5, True):
            with self.assertRaises(ValueError):
                model.new_request(max_tokens=cap)

    @torch.inference_mode()
    def test_failure_reload_and_generation(self):
        torch.manual_seed(77)
        options = dict(backend='gem5_pim', worker=os.environ['NDP_GEM5_PIM_WORKER'])
        source = tiny_model()
        model = ndp.nn.TensorParallelLlamaForCausalLM(source, device_groups=[[options], [options]],
                                                      placement=[0, 1], max_tokens=4)
        a, b = model.new_request(), model.new_request()
        token = torch.tensor([[1]])
        first = model(token, request=b)
        with patch.object(model.model.devices, 'copy_to', side_effect=RuntimeError('request failure')):
            with self.assertRaisesRegex(RuntimeError, 'request failure'):
                model(token, request=a)
        self.assertTrue(a.failed)
        self.assertFalse(b.failed)
        with self.assertRaisesRegex(RuntimeError, 'reset this request'):
            model(token, request=a)
        # An independent history continues while the failed request is retained.
        continued = model(torch.tensor([[2]]), request=b)
        model.reset_history()
        torch.testing.assert_close(model(token), first, atol=0, rtol=0)
        torch.testing.assert_close(model(torch.tensor([[2]])), continued, atol=0, rtol=0)
        a.reset()
        torch.testing.assert_close(model(token, request=a), first, atol=0, rtol=0)
        original_b_position = b.position
        ids = torch.tensor([[1, 2]])
        generated = model.generate(ids, max_new_tokens=2, request=a, eos_token_id=[])
        self.assertEqual(b.position, original_b_position)
        torch.testing.assert_close(generated, model.generate(ids, max_new_tokens=2, eos_token_id=[]))
        with patch.object(model.model.norm, 'forward', side_effect=RuntimeError('final norm failure')):
            with self.assertRaisesRegex(RuntimeError, 'final norm failure'):
                model(token, request=b)
        self.assertTrue(b.failed)
        self.assertFalse(a.failed)
        model.load_state_dict(source.state_dict())
        self.assertTrue(a.closed and b.closed)
        fresh = model.new_request()
        foreign = ndp.nn.TensorParallelLlamaForCausalLM(source, devices=model.model.device_groups[0], max_tokens=4)
        with self.assertRaises(ValueError):
            foreign(token, request=fresh)
        self.assertEqual(fresh.position, 0)
        torch.testing.assert_close(model(token, request=fresh), first, atol=0, rtol=0)
        fresh.close()
