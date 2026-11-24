"""Exercise ownership, transfer, primitive operations, layouts, and module adapters."""
import gc
import os
from pathlib import Path
import threading
import unittest
import weakref

import torch
import ndp

ROOT = Path(os.environ.get("NDP_SOURCE_DIR", Path(__file__).resolve().parents[2]))


def profiles():
    return [{}, {"backend": "ndp", "config": str(ROOT / "configs/ndp-small.ini")},
            {"backend": "ndp"}]


def close(actual, expected, atol=.002, rtol=.003):
    torch.testing.assert_close(actual, expected.half(), atol=atol, rtol=rtol)


class RuntimeTests(unittest.TestCase):
    @unittest.skipUnless(getattr(ndp._C, "_testing_enabled", False), "Requires NDP_PYTHON_TESTING")
    def test_execution_failure_gil_and_context_error(self):
        import time
        s = ndp.Session(backend="_test_fail_upload")
        x = s.upload(torch.ones(1).half())
        ran = threading.Event()
        def mark():
            time.sleep(.025)
            ran.set()
        thread = threading.Thread(target=mark)
        thread.start()
        with self.assertRaises(ndp.NdpError) as cm:
            x.wait()
        self.assertTrue(ran.is_set())
        thread.join()
        self.assertEqual(cm.exception.code, ndp.ErrorCode.BackendFailure)
        self.assertEqual(x.ready_event().status, ndp.EventStatus.Failed)
        with self.assertRaises(ndp.NdpError):
            s.download_async(x).wait()
        with self.assertRaises(ndp.NdpError):
            s.synchronize()
        with self.assertRaises(ndp.NdpError):
            s.close()
        with self.assertRaisesRegex(ValueError, "original") as cm:
            with ndp.Session(backend="_test_fail_upload") as failed:
                failed.upload(torch.ones(1).half())
                raise ValueError("original")
        self.assertTrue(any("shutdown" in note for note in cm.exception.__notes__))
        # Drop all owners while upload is pending on the native worker.
        pending = ndp.Session(backend="_test_fail_upload")
        tensor = pending.upload(torch.ones(1).half())
        box = [pending, tensor]
        del pending, tensor
        thread = threading.Thread(target=box.clear)
        thread.start(); thread.join(timeout=5)
        self.assertFalse(thread.is_alive())

    def test_transfer_bits_views_and_staging(self):
        # Every binary16 encoding, including NaN payloads, survives a pure transfer.
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
        with ndp.Session() as s:
            y = s.download(s.upload(bits.view(torch.float16)))
            self.assertTrue(torch.equal(y.view(torch.int16), bits))
            base = torch.arange(40).half().reshape(5, 8)
            for x in (base[:, 1::2].t(), base[1:3], torch._neg_view(base)):
                expected = x.clone()
                h = s.upload(x)
                self.assertEqual(h.dtype, ndp.DType.F16)
                close(s.download(h), expected, 0, 0)
            x = base.clone()
            h = s.upload(x)
            x.fill_(99)
            del x
            transfer = s.download_async(h)
            del h
            close(transfer.wait(), base, 0, 0)
            transfer.ready_event().wait()
            self.assertEqual(transfer.ready_event().status, ndp.EventStatus.Succeeded)
            for bad in (base.float(), base.clone().requires_grad_(), torch.empty(0).half(),
                        torch.tensor(1).half(), base.to_sparse()):
                with self.assertRaises(ndp.NdpError):
                    s.upload(bad)

    def test_thread_close_and_owner_lifetimes(self):
        s = ndp.Session()
        x = s.upload(torch.ones(8).half())
        calls = [lambda: s.upload(torch.ones(1).half()), s.close, s.synchronize,
                 s.memory_stats, lambda: x.shape, x.wait]
        failures = []
        def other_thread():
            for call in calls:
                try:
                    call()
                except ndp.NdpError as error:
                    failures.append(error.code)
        thread = threading.Thread(target=other_thread)
        thread.start(); thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [ndp.ErrorCode.InvalidState] * len(calls))
        calls.clear()  # Drop bound methods retained by the thread-policy test.
        ref = weakref.ref(s)
        del s
        gc.collect()
        self.assertIsNone(ref())  # Native ownership, not a Python wrapper cycle.
        x.wait()
        # Last native owner may be dropped on a different thread; cleanup is native RAII.
        box = [x]
        del x
        thread = threading.Thread(target=box.clear)
        thread.start(); thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        s = ndp.Session()
        x = s.upload(torch.ones(8).half())
        s.close(); s.close()
        x.wait()
        with self.assertRaises(ndp.NdpError) as cm:
            s.download(x)
        self.assertEqual(cm.exception.code, ndp.ErrorCode.SessionClosed)
        self.assertEqual(cm.exception.operation, "download")
        with self.assertRaisesRegex(ValueError, "original"):
            with ndp.Session():
                raise ValueError("original")

    def test_mismatch_and_memory_budget(self):
        with ndp.Session() as a, ndp.Session() as b:
            x, y = a.upload(torch.ones(2).half()), b.upload(torch.ones(2).half())
            with self.assertRaises(ndp.NdpError) as cm:
                a.add(x, y)
            self.assertEqual(cm.exception.code, ndp.ErrorCode.SessionMismatch)
            with self.assertRaises(ndp.NdpError):
                a.copy_into([ndp.TensorCopy(x, b.allocate(ndp.TensorDesc([2])), 0, 0, 2)])
        with ndp.Session(capacity_bytes=4) as s:
            with self.assertRaises(ndp.NdpError) as cm:
                s.upload(torch.ones(3).half())
            self.assertEqual(cm.exception.code, ndp.ErrorCode.OutOfMemory)
        with ndp.Session(max_staging_bytes=2) as s:
            with self.assertRaises(ndp.NdpError):
                s.upload(torch.ones(3).half())
        with self.assertRaises(ndp.NdpError):
            ndp.Session(backend="ndp", config="missing.ini")

    def test_primitives_and_layouts(self):
        torch.manual_seed(19)
        for profile in profiles():
            with self.subTest(profile=profile), ndp.Session(**profile) as s:
                host = torch.randn(2, 5).half() / 4
                weight = torch.randn(7, 5).half() / 4
                x, w = s.upload(host), s.upload(weight)
                gamma_host = torch.ones(5).half()
                gamma = s.upload(gamma_host)
                operations = [
                    (s.add(x, gamma), host + gamma_host),
                    (s.mul(x, gamma), host * gamma_host),
                    (s.scale(x, .25), host.float() * .25),
                    (s.silu(x), torch.nn.functional.silu(host.float())),
                    (s.softmax(x), torch.softmax(host.float(), -1)),
                    (s.rms_norm(x, gamma), host.float() * torch.rsqrt(host.float().square().mean(-1, keepdim=True)+1e-5)),
                    (s.transpose(x), host.t()), (s.reshape(x, [10]), host.reshape(10)),
                    (s.gather(x, [0, 7, 3], [3]), host.flatten()[[0, 7, 3]]),
                    (s.matmul(x, s.transpose(w)), host.float() @ weight.float().t()),
                    (s.linear(x, w), host.float() @ weight.float().t()),
                ]
                for result, expected in operations:
                    close(s.download(result), expected)
                one = s.upload(host[:1])
                plan = s.query_operation(ndp.OperationSpec(ndp.OpKind.Linear, ndp.Engine.PIM, [one.desc, w.desc]))
                with self.assertRaises(ndp.NdpError) as cm:
                    s.linear(one, w, ndp.OpOptions(ndp.Engine.PIM, ndp.LayoutPolicy.RequirePrepared))
                self.assertEqual(cm.exception.code, ndp.ErrorCode.LayoutNotPrepared)
                prepared = s.prepare_layouts(w, [plan.inputs[1]])
                self.assertEqual(len(w.representations()), 1)
                self.assertEqual(len(prepared.representations()), 2)
                y = s.linear(one, prepared, ndp.OpOptions(ndp.Engine.PIM, ndp.LayoutPolicy.RequirePrepared))
                close(s.download(s.add(y, y)), ((host[:1].float() @ weight.float().t()).half().float()*2))
                close(s.download(s.matvec(prepared, s.reshape(one, [5]))), (weight.float() @ host[0].float()))
                s.materialize_layout(w, plan.inputs[1]).wait()
                close(s.download(w, plan.inputs[1].kind), weight, 0, 0)
                with self.assertRaises(ndp.NdpError):
                    s.linear(x, prepared, ndp.OpOptions(ndp.Engine.PIM))
                with self.assertRaises(ndp.NdpError):
                    s.silu(one, ndp.OpOptions(ndp.Engine.PIM))

    def test_mutable_buffers_and_snapshots(self):
        with ndp.Session() as s:
            source = s.upload(torch.tensor([1, 2, 3]).half())
            buffer = s.allocate(ndp.TensorDesc([6]))
            s.copy_into([ndp.TensorCopy(source, buffer, 0, 2, 3)])
            snapshot = s.snapshot(buffer, ndp.TensorDesc([6]))
            s.clear([buffer]).wait()
            close(s.download(snapshot), torch.tensor([0, 0, 1, 2, 3, 0]).half(), 0, 0)
            close(s.download(s.snapshot(buffer, ndp.TensorDesc([6]))), torch.zeros(6).half(), 0, 0)
            with self.assertRaises(ndp.NdpError):
                s.copy_into([ndp.TensorCopy(source, buffer, 0, 5, 3)])
            with self.assertRaises(ndp.NdpError):
                s.snapshot(buffer, ndp.TensorDesc([8]))

    def test_checkpoint_object_replacement(self):
        for profile in profiles():
            with self.subTest(profile=profile), ndp.Session(**profile) as s:
                module = ndp.nn.NdpLinear(s, torch.ones(1, 2).half(), torch.ones(1).half())
                x = torch.ones(1, 2).half()
                close(module(x), torch.tensor([[3.]]).half(), 0, 0)
                # A new object starts at the same version as the prepared checkpoint.
                old_version = module.weight._version
                module.weight = torch.zeros_like(module.weight)
                self.assertEqual(module.weight._version, old_version)
                with self.assertRaisesRegex(RuntimeError, "refresh"):
                    module(x)
                module.refresh()
                close(module(x), torch.tensor([[1.]]).half(), 0, 0)
                module.bias = torch.zeros_like(module.bias)
                with self.assertRaisesRegex(RuntimeError, "refresh"):
                    module(x)
                module.refresh()
                close(module(x), torch.zeros(1, 1).half(), 0, 0)
                module.bias = None
                with self.assertRaisesRegex(RuntimeError, "refresh"):
                    module(x)
                module.refresh()
                with torch.inference_mode():
                    module.weight = torch.ones(1, 2).half()
                with self.assertRaisesRegex(RuntimeError, "refresh"):
                    module(x)
                module.refresh()
                close(module(x), torch.tensor([[2.]]).half(), 0, 0)

    def test_matmul_output_scaling(self):
        for profile in profiles():
            with self.subTest(profile=profile), ndp.Session(**profile) as s:
                a = s.upload(torch.full((1, 128), 32., dtype=torch.float16))
                b = s.upload(torch.full((128, 1), 32., dtype=torch.float16))
                self.assertTrue(torch.isinf(s.download(s.matmul(a, b))).all())
                for factor in (.25, -.25, 0.):
                    close(s.download(s.matmul(a, b, factor=factor)),
                          torch.tensor([[131072 * factor]]).half(), 0, 0)
                for invalid in (float("nan"), float("inf")):
                    with self.assertRaises(ndp.NdpError):
                        s.matmul(a, b, factor=invalid)

    def test_module_inside_inference_mode(self):
        with torch.inference_mode(), ndp.Session() as s:
            source = torch.nn.Linear(3, 2, bias=False, dtype=torch.float16).eval()
            module = ndp.nn.NdpLinear.from_torch(source, s)
            x = torch.ones(1, 3, dtype=torch.float16)
            close(module(x), source(x))
            module.load_state_dict({"weight": source.weight.clone() * .5}, assign=True)
            close(module(x), source(x) * .5)
            module.weight.add_(1)
            with self.assertRaisesRegex(RuntimeError, "refresh"):
                module(x)

    def test_linear_module_checkpoint_refresh(self):
        with ndp.Session() as s:
            source = torch.nn.Linear(5, 7, dtype=torch.float16).eval()
            module = ndp.nn.NdpLinear.from_torch(source, s)
            x = torch.randn(2, 5).half()
            close(module(x), source(x).detach())
            state = {k: v.clone() * .5 for k, v in module.state_dict().items()}
            module.load_state_dict(state)
            close(module(x), torch.nn.functional.linear(x, state["weight"], state["bias"]))
            module.weight.add_(1)
            with self.assertRaisesRegex(RuntimeError, "refresh"):
                module(x)
            module.refresh()
            close(module(x), torch.nn.functional.linear(x, module.weight, module.bias))
            with self.assertRaises(RuntimeError):
                module.train()
            with self.assertRaises(RuntimeError):
                module.to(torch.float32)
            with self.assertRaises(ndp.NdpError):
                module(x.requires_grad_())
            with self.assertRaises(ValueError):
                ndp.nn.NdpLinear.from_torch(source, s, ndp.Engine.PIM)
            with self.assertRaises(ValueError):
                ndp.nn.NdpLinear.from_torch(source.train(), s)
            # Parent-module loading must also rebuild child resident storage.
            parent = torch.nn.ModuleDict({"linear": module})
            parent.load_state_dict({"linear.weight": module.weight.clone() * .5,
                                    "linear.bias": module.bias.clone() * .5})
            close(module(x.detach()), torch.nn.functional.linear(x.detach(), module.weight, module.bias))
            with self.assertRaises(RuntimeError):
                module.load_state_dict({"weight": torch.ones(1).half(), "bias": module.bias})
            with self.assertRaisesRegex(RuntimeError, "refresh"):
                module(x.detach())
            no_bias = torch.nn.Linear(5, 7, bias=False, dtype=torch.float16).eval()
            pim = ndp.nn.NdpLinear.from_torch(no_bias, s, ndp.Engine.PIM)
            close(pim(x[:1].detach()), no_bias(x[:1]).detach())


if __name__ == "__main__":
    unittest.main()
