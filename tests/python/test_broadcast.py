"""Verify broadcast ownership and one-download staging on actual worker streams."""
from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import unittest

import ndp
import torch


@unittest.skipUnless(os.getenv('NDP_GEM5_PIM_WORKER'), 'Configure the FP16 worker')
class BroadcastTests(unittest.TestCase):
    def test_three_workers(self):
        path = Path(__file__).resolve().parents[2] / 'examples/python/llama_stages.py'
        spec = importlib.util.spec_from_file_location('broadcast_relay', path)
        relay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(relay)
        with ExitStack() as resources, torch.inference_mode():
            endpoints = [resources.enter_context(relay.counted_worker(os.environ['NDP_GEM5_PIM_WORKER']))
                         for _ in range(3)]
            group = ndp.DeviceGroup([session for session, _ in endpoints])
            for dtype in (torch.float16, torch.float32):
                values = torch.arange(7, dtype=dtype).reshape(1, 7)
                original = group.device(1).upload(values)
                for session, log in endpoints:
                    session.synchronize()
                    log.read()
                replicas = group.broadcast(original)
                size = values.numel()*values.element_size()
                for i, (_, log) in enumerate(endpoints):
                    self.assertEqual(log.read().splitlines(), [f'D {size}' if i == 1 else f'U {size}'])
                for i, tensor in enumerate(replicas):
                    torch.testing.assert_close(group.device(i).download(tensor), values, atol=0, rtol=0)
                local = ndp.DeviceGroup([group.device(1)])
                for _, log in endpoints:
                    log.read()
                self.assertEqual(local.broadcast(original)[0].shape, [1, 7])
                self.assertEqual(endpoints[1][1].read(), '')
                with self.assertRaises(ndp.NdpError):
                    ndp.DeviceGroup([group.device(0)]).broadcast(original)
            pair = ndp.DeviceGroup([group.device(0), group.device(1)])
            original = group.device(1).upload(torch.ones(1, 7).half())
            for session, log in endpoints:
                session.synchronize()
                log.read()
            pair.broadcast(original)
            self.assertEqual(endpoints[0][1].read().splitlines(), ['U 14'])
            self.assertEqual(endpoints[1][1].read().splitlines(), ['D 14'])
            self.assertEqual(endpoints[2][1].read(), '')
            # All-gather uses the same broadcast implementation for the assembled row.
            shards = [group.device(i).upload(torch.tensor([[i]], dtype=torch.float16)) for i in range(3)]
            for session, log in endpoints:
                session.synchronize()
                log.read()
            replicas = group.all_gather_rows(shards)
            root_frames = endpoints[0][1].read().splitlines()
            self.assertEqual(root_frames.count('D 6'), 1)
            for i, tensor in enumerate(replicas):
                torch.testing.assert_close(group.device(i).download(tensor), torch.tensor([[0., 1., 2.]]).half())
