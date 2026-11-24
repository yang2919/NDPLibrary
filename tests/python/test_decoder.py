"""Compare the explicit Python decoder with an independent rounded PyTorch reference."""
import importlib.util
import math
from pathlib import Path
import unittest
import torch
import ndp
from test_runtime import ROOT, profiles, close

spec = importlib.util.spec_from_file_location("ndp_decoder_example", ROOT / "examples/python/decoder_block.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class CountingSession(ndp.Session):
    """Count Python transfer boundaries without adding production profiling APIs."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uploads = self.downloads = 0

    def upload(self, tensor):
        self.uploads += 1
        return super().upload(tensor)

    def download(self, tensor, *args):
        self.downloads += 1
        return super().download(tensor, *args)


def linear(x, w):
    return (x.float() @ w.float().t()).half()


def norm(x, gamma):
    f = x.float()
    return (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-5) * gamma.float()).half()


def rotate(x, start):
    angles = torch.arange(start, start + len(x)).float() * .25
    sine = angles.sin().half().reshape(-1, 1, 1)
    cosine = angles.cos().half().reshape(-1, 1, 1)
    # Match the public primitive boundaries: each multiply/add stores FP16.
    direct = (x.float() * cosine.float()).half()
    cross = (x.flip(-1).float() * sine.float()).half()
    cross[..., 0] = -cross[..., 0]
    return (direct.float() + cross.float()).half()


class ReferenceDecoder:
    def __init__(self, weights):
        self.w = weights
        self.keys = torch.empty(0, 1, 2).half()
        self.values = torch.empty(0, 1, 2).half()

    def run(self, x):
        w, start, tokens = self.w, len(self.keys), len(x)
        n = norm(x, w["attention_norm"])
        q = rotate(linear(n, w["q"]).reshape(tokens, 2, 2), start)
        k = rotate(linear(n, w["k"]).reshape(tokens, 1, 2), start)
        v = linear(n, w["v"]).reshape(tokens, 1, 2)
        self.keys = torch.cat([self.keys, k])
        self.values = torch.cat([self.values, v])
        attended = torch.empty_like(q)
        for t in range(tokens):
            length = start + t + 1
            for h in range(2):
                scores = q[t, h].float() @ self.keys[:length, 0].float().t()
                scaled = (scores / math.sqrt(2)).half()
                probabilities = torch.softmax(scaled.float(), -1).half()
                attended[t, h] = (probabilities.float() @ self.values[:length, 0].float()).half()
        residual = (x.float() + linear(attended.reshape(tokens, 4), w["output"]).float()).half()
        ffn = norm(residual, w["ffn_norm"])
        gate = torch.nn.functional.silu(linear(ffn, w["gate"]).float()).half()
        up = linear(ffn, w["up"])
        down = linear((gate.float() * up.float()).half(), w["down"])
        return (residual.float() + down.float()).half(), dict(normalized=n, q=q, k=k, attended=attended, residual=residual)


class DecoderTests(unittest.TestCase):
    def test_prefill_decode_intermediates_and_residency(self):
        for profile in profiles():
            with self.subTest(profile=profile), CountingSession(**profile) as s:
                weights = example.toy_weights()
                layer = example.DecoderLayer(s, weights)
                cache, independent_cache = layer.new_cache(), layer.new_cache()
                reference = ReferenceDecoder(weights)
                initial_uploads = s.uploads
                for step in range(4):
                    tokens = 2 if step == 0 else 1
                    x = example.values(tokens * 4, step + 1).reshape(tokens, 4)
                    inp = s.upload(x)
                    before = (s.uploads, s.downloads)
                    trace = {}
                    engine = ndp.Engine.PNM if step == 0 else ndp.Engine.PIM
                    result = layer.forward_ndp(inp, cache, engine, trace)
                    self.assertEqual((s.uploads, s.downloads), before)
                    expected, intermediates = reference.run(x)
                    close(s.download(result), expected)
                    for name, tensor in trace.items():
                        close(s.download(tensor), intermediates[name])
                    self.assertEqual(cache.committed_length(), 2 + step)
                    self.assertEqual(independent_cache.scheduled_length(), 0)
                    self.assertEqual(s.uploads, initial_uploads + step + 1)
                cache.reset().wait()
                self.assertEqual(cache.committed_length(), 0)
                self.assertFalse(cache.poisoned())
                fresh = example.values(8, 1).reshape(2, 4)
                close(s.download(layer.forward_ndp(s.upload(fresh), cache, ndp.Engine.PNM)),
                      ReferenceDecoder(weights).run(fresh)[0])

    def test_attention_scales_before_fp16_storage(self):
        for profile in profiles():
            with self.subTest(profile=profile), ndp.Session(**profile) as s:
                q = torch.full((1, 1, 128), 32., dtype=torch.float16)
                k = torch.stack([torch.full((1, 128), 32.), torch.full((1, 128), 31.)]).half()
                v = torch.stack([torch.ones(1, 128), torch.zeros(1, 128)]).half()
                scores = (q[0, 0].float() @ k[:, 0].float().t()) / math.sqrt(128)
                expected = (torch.softmax(scores, 0) @ v[:, 0].float()).half()
                out = ndp.llm.causal_attention(s, s.upload(q), s.upload(k), s.upload(v), 1)
                close(s.download(out)[0, 0], expected, 0, 0)

    def test_causality_and_cache_limits(self):
        with ndp.Session() as s:
            q = s.upload(torch.tensor([[[1., 0.]], [[0., 1.]]]).half())
            k = s.upload(torch.tensor([[[1., 0.]], [[0., 1.]]]).half())
            v = s.upload(torch.tensor([[[2., 3.]], [[90., 80.]]]).half())
            out = ndp.llm.causal_attention(s, q, k, v, 0)
            close(s.download(out)[0], torch.tensor([[2., 3.]]).half(), 0, 0)
            cache = ndp.llm.KVCache(s, 2, 1, 2)
            cache.append(k, v).wait()
            with self.assertRaises(ndp.NdpError) as cm:
                cache.append(k, v)
            self.assertEqual(cm.exception.code, ndp.ErrorCode.OutOfMemory)
            self.assertEqual(cache.scheduled_length(), 2)
            cache.reset().wait()
            self.assertEqual(cache.scheduled_length(), 0)


if __name__ == "__main__":
    unittest.main()
