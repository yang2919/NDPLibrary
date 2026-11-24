"""Check decode modules through their public PyTorch tensor and checkpoint interfaces."""
import unittest
import os
from types import SimpleNamespace

import torch
import ndp


def tiny_llama_layer():
    source = torch.nn.Module()
    source.self_attn = torch.nn.Module()
    source.self_attn.config = SimpleNamespace(model_type="llama", hidden_size=8,
        num_attention_heads=2, num_key_value_heads=1, hidden_act="silu",
        rope_theta=10000., max_position_embeddings=8, rope_scaling=None)
    source.self_attn.head_dim = 4
    for name, rows in (("q_proj", 8), ("k_proj", 4), ("v_proj", 4), ("o_proj", 8)):
        setattr(source.self_attn, name, torch.nn.Linear(8, rows, bias=False).half())
    source.mlp = torch.nn.Module()
    for name, dims in (("gate_proj", (8, 12)), ("up_proj", (8, 12)), ("down_proj", (12, 8))):
        setattr(source.mlp, name, torch.nn.Linear(*dims, bias=False).half())
    for name in ("input_layernorm", "post_attention_layernorm"):
        norm = torch.nn.RMSNorm(8).half()
        norm.variance_epsilon = 1e-5
        setattr(source, name, norm)
    return source.eval()


class ModuleTests(unittest.TestCase):
    def test_attention_decode_reset_and_checkpoint(self):
        self._check_attention({})

    @unittest.skipUnless(os.getenv("NDP_GEM5_PIM_WORKER"), "Configure native FP16 worker")
    def test_gem5_attention(self):
        self._check_attention(dict(backend="gem5_pim", worker=os.environ["NDP_GEM5_PIM_WORKER"]))

    @unittest.skipUnless(all(os.getenv(k) for k in ("NDP_GEM5_PIM_WORKER", "NDP_PNM_CAPTURE_WORKER",
                         "NDP_PNM_GEMV_ELF", "NDP_PNM_ATTENTION_ELF")), "Configure attention capture")
    def test_gem5_attention_capture(self):
        self._check_attention(dict(backend="gem5_pim", worker=os.environ["NDP_GEM5_PIM_WORKER"],
                                   capture_worker=os.environ["NDP_PNM_CAPTURE_WORKER"],
                                   gemv_elf=os.environ["NDP_PNM_GEMV_ELF"],
                                   attention_elf=os.environ["NDP_PNM_ATTENTION_ELF"]))

    def _check_attention(self, options):
        torch.manual_seed(17)
        for bias in (True, False):
            with self.subTest(bias=bias), torch.inference_mode():
                source = torch.nn.MultiheadAttention(8, 2, bias=bias, batch_first=True,
                                                     dtype=torch.float16).eval()
                if bias:
                    source.in_proj_bias.uniform_(-0.5, 0.5)
                    source.out_proj.bias.uniform_(-0.5, 0.5)
                layer = ndp.nn.DecodeSelfAttention(source, max_tokens=3, **options)
                tokens = torch.randn(1, 3, 8, dtype=torch.float16)
                for i in range(3):
                    token, prefix = tokens[:, i:i+1], tokens[:, :i+1]
                    expected = source(token, prefix, prefix)[0]
                    torch.testing.assert_close(layer(token), expected, atol=.002, rtol=.002)
                with self.assertRaisesRegex(ValueError, "full"):
                    layer(tokens[:, :1])
                layer.reset_history()
                first = layer(tokens[:, :1])
                torch.testing.assert_close(first, source(tokens[:, :1], tokens[:, :1], tokens[:, :1])[0],
                                           atol=.002, rtol=.002)
                state = layer.state_dict()
                self.assertTrue(all(name.startswith(("q_proj.", "k_proj.", "v_proj.", "out_proj."))
                                    for name in state))
                layer.load_state_dict(state)
                torch.testing.assert_close(layer(tokens[:, :1]), first, atol=0, rtol=0)

    def test_llama_block_checkpoint_resets_history(self):
        with torch.inference_mode():
            source = tiny_llama_layer()
            block = ndp.nn.LlamaDecodeBlock(source.eval(), max_tokens=2)
            x = torch.ones(1, 1, 8).half()
            expected = block(x)
            block(-x)
            with self.assertRaisesRegex(ValueError, "history is full"):
                block(x)
            block.reset_history()
            torch.testing.assert_close(block(x), expected, atol=0, rtol=0)
            self.assertEqual(set(block.state_dict()), set(source.state_dict()))
            source.mlp.down_proj.weight.zero_()
            source.self_attn.o_proj.weight.zero_()
            block.load_state_dict(source.state_dict(), assign=True)
            torch.testing.assert_close(block(x), x, atol=0, rtol=0)

    def test_device_group_copy(self):
        with ndp.Session() as a, ndp.Session() as b, ndp.Session() as foreign:
            group = ndp.DeviceGroup([a, b])
            words = torch.tensor([0, -32768, 1, 31744, 32257], dtype=torch.int16)
            tensor = a.upload(words.view(torch.float16))
            result = group.copy_to(tensor, 1)
            torch.testing.assert_close(b.download(result).view(torch.int16), words)
            with self.assertRaises(ndp.NdpError):
                a.download(result)
            with self.assertRaises(ndp.NdpError):
                group.copy_to(foreign.upload(torch.zeros(1).half()), 0)
            with self.assertRaises(ndp.NdpError):
                ndp.DeviceGroup([a, a])
            for limit in ({"capacity_bytes": 8}, {"max_staging_bytes": 8}):
                with ndp.Session(**limit) as small:
                    with self.assertRaises(ndp.NdpError) as error:
                        ndp.DeviceGroup([a, small]).copy_to(tensor, 1)
                    self.assertEqual(error.exception.code, ndp.ErrorCode.OutOfMemory)
                    self.assertEqual(small.memory_stats().live_bytes, 0)
            torch.testing.assert_close(a.download(group.copy_to(tensor, 0)).view(torch.int16), words)

    def test_stack_failure_and_reset(self):
        from unittest.mock import patch
        with torch.inference_mode():
            sources = [tiny_llama_layer(), tiny_llama_layer()]
            stack = ndp.nn.LlamaDecodeStack(sources, devices=[{}, {}], placement=[0, 1], max_tokens=2)
            x = torch.ones(1, 1, 8).half()
            expected = stack(x)
            stack.reset_history()
            with patch.object(stack.layers[1], "forward_ndp", side_effect=RuntimeError("injected")):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    stack(x)
            with self.assertRaisesRegex(RuntimeError, "reset_history"):
                stack(x)
            stack.reset_history()
            torch.testing.assert_close(stack(x), expected, rtol=0, atol=0)

    def test_swiglu_and_linear(self):
        with torch.inference_mode():
            def projection(n, m):
                return torch.nn.Linear(n, m, bias=False, dtype=torch.float16).eval()

            gate, up, down = projection(4, 8), projection(4, 8), projection(8, 4)
            mlp = ndp.nn.SwiGLU(gate, up, down)
            linear = ndp.nn.Linear.from_float(up)
            for shape in ((4,), (1, 4), (1, 1, 4)):
                x = torch.ones(shape, dtype=torch.float16)
                expected = down(torch.nn.functional.silu(gate(x)) * up(x))
                torch.testing.assert_close(mlp(x), expected, atol=.002, rtol=.002)
                torch.testing.assert_close(linear(x), up(x), atol=.002, rtol=.002)
