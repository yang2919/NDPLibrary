"""Verify external PyTorch projections against a persistent native libpim worker."""
import os
import unittest

import ndp
import torch


class CountingSession(ndp.Session):
    def upload(self, tensor):
        self.uploads += 1
        return super().upload(tensor)

    def download(self, tensor):
        self.downloads += 1
        return super().download(tensor)


class Gem5PimTests(unittest.TestCase):
    def test_worker_option(self):
        for options in ({"backend": "gem5_pim"}, {"backend": "functional", "worker": "unused"},
                        {"backend": "gem5_pim", "worker": "unused", "config": "unused"}):
            with self.subTest(options=options), self.assertRaises(ndp.NdpError):
                ndp.Session(**options)

    @unittest.skipUnless(os.getenv("NDP_GEM5_PIM_WORKER"), "Configure the FP16 worker")
    def test_resident_projections(self):
        def projection(rows):
            weight = torch.tensor(rows, dtype=torch.float16)
            layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False,
                                    dtype=torch.float16).eval()
            with torch.no_grad():
                layer.weight.copy_(weight)
            return layer

        first = projection([[1, 2**-11, 0], [1, 2**-11, 2**-11], [1 + 2**-10, 0, 0]])
        second = projection([[1, -1, 0], [0, 0, 1]])
        with CountingSession(backend="gem5_pim", worker=os.environ["NDP_GEM5_PIM_WORKER"]) as s:
            self.assertEqual(s.capabilities().name, "gem5_pim")
            s.uploads = s.downloads = 0
            a, b = [ndp.nn.NdpLinear.from_torch(layer, s, ndp.Engine.PIM)
                    for layer in (first, second)]
            self.assertEqual(s.uploads, 2)
            for step, value in enumerate((1, 2), start=1):
                x = torch.full((1, 3), value, dtype=torch.float16)
                y = a.forward_ndp(s.upload(x))
                result = s.download(b.forward_ndp(y))
                expected_y = (x.float() @ first.weight.detach().float().T).half()
                expected = (expected_y.float() @ second.weight.detach().float().T).half()
                torch.testing.assert_close(result, expected, atol=0, rtol=0)
                self.assertEqual((s.uploads, s.downloads), (2 + step, step))

    @unittest.skipUnless(all(os.getenv(key) for key in
                            ("NDP_GEM5_PIM_WORKER", "NDP_PNM_CAPTURE_WORKER", "NDP_PNM_GEMV_ELF")),
                         "Configure the reference worker, capture worker and GEMV ELF")
    def test_pnm_capture(self):
        options = dict(backend="gem5_pim", worker=os.environ["NDP_GEM5_PIM_WORKER"],
                       capture_worker=os.environ["NDP_PNM_CAPTURE_WORKER"],
                       gemv_elf=os.environ["NDP_PNM_GEMV_ELF"])
        with ndp.Session(**options) as session:
            layers = [torch.nn.Linear(n, m, bias=False, dtype=torch.float16).eval()
                      for n, m in [(3, 2), (2, 1)]]
            with torch.no_grad():
                for layer in layers:
                    layer.weight.fill_(1)
            first, second = [ndp.nn.NdpLinear.from_torch(layer, session, ndp.Engine.PIM)
                             for layer in layers]
            x = session.upload(torch.ones(1, 3, dtype=torch.float16))
            for _ in range(2):
                y = first.forward_ndp(x)
                self.assertEqual(session.download(y).tolist(), [[3, 3]])
            self.assertEqual(session.download(second.forward_ndp(y)).tolist(), [[6]])

        # A v2 numerical worker must fail the v3 handshake before any submission.
        with self.assertRaises(ndp.NdpError):
            ndp.Session(**{**options, "capture_worker": options["worker"]})

    @unittest.skipUnless(all(os.getenv(key) for key in ("NDP_GEM5_PIM_WORKER",
                         "NDP_PNM_CAPTURE_WORKER", "NDP_PNM_GEMV_ELF", "NDP_PNM_POINTWISE_ELF")),
                         "Configure both N4 programs and workers")
    def test_resident_mixed_mlp(self):
        with CountingSession(backend="gem5_pim", worker=os.environ["NDP_GEM5_PIM_WORKER"],
                             capture_worker=os.environ["NDP_PNM_CAPTURE_WORKER"],
                             gemv_elf=os.environ["NDP_PNM_GEMV_ELF"],
                             pointwise_elf=os.environ["NDP_PNM_POINTWISE_ELF"]) as s:
            s.uploads = s.downloads = 0
            weights = [torch.tensor(rows, dtype=torch.float16) for rows in (
                [[1, 0], [-2, 1], [0, 1]], [[2, -1], [1, 0], [1, 1]], [[1, 2, -1], [0, -1, 1]])]
            layers = []
            for w in weights:
                layer = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=w.dtype).eval()
                with torch.no_grad():
                    layer.weight.copy_(w)
                layers.append(ndp.nn.NdpLinear.from_torch(layer, s, ndp.Engine.PIM))
            self.assertEqual(s.uploads, 3)
            for values in ([[1, 2]], [[-2, 1]]):
                host = torch.tensor(values, dtype=torch.float16)
                before = (s.uploads, s.downloads)
                x = s.upload(host)
                gate, up = (layer.forward_ndp(x) for layer in layers[:2])
                activated = s.silu(gate)
                product = s.mul(activated, up)
                output = s.download(layers[2].forward_ndp(product))
                self.assertEqual((s.uploads, s.downloads), (before[0] + 1, before[1] + 1))
                g, u = (host @ w.T for w in weights[:2])
                act = torch.nn.functional.silu(g.float()).half()
                prod = act * u
                torch.testing.assert_close(output, prod @ weights[2].T, rtol=0, atol=0)
                # Diagnostic reads follow the boundary-count assertion.
                for tensor, expected in zip((gate, up, activated, product), (g, u, act, prod)):
                    torch.testing.assert_close(s.download(tensor), expected, rtol=0, atol=0)

    def test_decode_projection_shape_and_checkpoint(self):
        with ndp.Session() as session:
            layer = torch.nn.Linear(3, 2, bias=False, dtype=torch.float16).eval()
            adapted = ndp.nn.DecodeLinear.from_torch(layer, session, ndp.Engine.PIM)
            x = torch.ones(1, 1, 3, dtype=torch.float16)
            torch.testing.assert_close(adapted(x), layer(x), atol=.001, rtol=.001)
            self.assertEqual(set(adapted.state_dict()), set(layer.state_dict()))
            with self.assertRaisesRegex(ValueError, "DecodeLinear"):
                adapted(x.expand(1, 2, 3))
