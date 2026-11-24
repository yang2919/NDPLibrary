"""Check registered PIM Linear routing, metadata, checkpoint and lifetime contracts."""
import concurrent.futures
import os
import sys
import unittest

import torch
import ndp


class DispatchTests(unittest.TestCase):
    def test_projection(self):
        backends = [{}]
        if worker := os.getenv("NDP_GEM5_PIM_WORKER"):
            backends.append(dict(backend="gem5_pim", worker=worker))
        for backend in backends:
            with self.subTest(backend=backend), ndp.Session(**backend) as runtime:
                source = torch.nn.Linear(3, 2, bias=False, dtype=torch.float16).eval()
                layer = ndp.nn.Linear.from_float(source, runtime=runtime)
                op = torch.ops.ndp.linear_pim.default
                packed = layer._packed
                for shape in ((3,), (1, 3), (1, 1, 3), (2, 3)):
                    x = torch.ones(shape, dtype=torch.float16)
                    expected = source(x).detach()
                    torch.testing.assert_close(layer(x), expected)
                    if x.numel() == 3:
                        y = op(x, packed)
                        torch.testing.assert_close(y, expected)
                        self.assertNotEqual(x.data_ptr(), y.data_ptr())
                self.assertIs(layer._packed, packed)
                self.assertEqual(list(layer.state_dict()), ["weight"])
                for assign in (False, True):
                    layer.load_state_dict({"weight": torch.full_like(source.weight, 2)}, assign=assign)
                    self.assertIsNot(layer._packed, packed)
                    self.assertTrue((layer(x) == 6).all())
                    self.assertTrue((layer(x[0]) == 6).all())
                with self.assertRaisesRegex(RuntimeError, "width"):
                    op(x, packed)
                with concurrent.futures.ThreadPoolExecutor(1) as executor:
                    with self.assertRaisesRegex(RuntimeError, "creating thread"):
                        executor.submit(op, x[0], packed).result()
                layer.weight.zero_()
                with self.assertRaisesRegex(RuntimeError, "refresh"):
                    layer(x)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                op(x[0], packed)

    def test_shared_qkv(self):
        with ndp.Session() as runtime, ndp.Session() as other:
            weights = [torch.arange(rows * 4).reshape(rows, 4).half() for rows in (4, 2, 2)]
            layers = [ndp.nn.Linear(runtime, w) for w in weights]
            x = torch.arange(8).half()[::2]
            for value in (x, -x, torch.stack((x, -x))):
                for actual, layer in zip(ndp.nn.project_qkv(value, *layers), layers):
                    torch.testing.assert_close(actual, layer(value), atol=0, rtol=0)
            torch.library.opcheck(torch.ops.ndp.qkv_pim.default, (x, *(m._packed for m in layers)))
            foreign = ndp.nn.Linear(other, weights[2])
            with self.assertRaisesRegex(ValueError, "Session"):
                ndp.nn.project_qkv(x, layers[0], layers[1], foreign)

    def test_fake(self):
        with ndp.Session() as runtime:
            layer = ndp.nn.Linear(runtime, torch.ones(2, 3).half())
            x = torch.zeros(1, 1, 3).half()
            self.assertTrue(layer._packed == layer._packed)
            self.assertFalse(layer._packed == ndp.nn.Linear(runtime, layer.weight)._packed)
            with self.assertRaises(TypeError):
                len(layer._packed)
            torch.library.opcheck(torch.ops.ndp.linear_pim.default, (x, layer._packed))
        # Metadata remains usable after close. A real dispatch would now fail.
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode() as mode:
            output = torch.ops.ndp.linear_pim(mode.from_tensor(x), layer._packed)
            self.assertEqual(output.shape, (1, 1, 2))

    @unittest.skipIf(sys.version_info >= (3, 14) and torch.__version__.startswith("2.9."),
                     "PyTorch 2.9 disables torch.compile on Python 3.14")
    def test_compiled_operator(self):
        backends = [{}]
        if worker := os.getenv("NDP_GEM5_PIM_WORKER"):
            backends.append(dict(backend="gem5_pim", worker=worker))
        for backend in backends:
            with self.subTest(backend=backend), ndp.Session(**backend) as runtime:
                layer = ndp.nn.Linear(runtime, torch.ones(2, 3).half())
                x = torch.ones(1, 3).half()
                compiled = torch.compile(lambda t, p: torch.ops.ndp.linear_pim(t, p),
                                         fullgraph=True, backend="eager")
                for value in (x, -2 * x):
                    torch.testing.assert_close(compiled(value, layer._packed), layer(value))
