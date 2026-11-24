"""Check multi-layer history, checkpoint compatibility and greedy decode lifecycle."""
import copy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ndp
import torch


def tiny_model():
    config = SimpleNamespace(hidden_size=12, num_attention_heads=6, num_key_value_heads=3,
        model_type='llama', rope_scaling=None, max_position_embeddings=32, rope_theta=10000.,
        hidden_act='silu', eos_token_id=None)
    def norm():
        layer = torch.nn.Module()
        layer.register_buffer('weight', torch.ones(12).half())
        layer.variance_epsilon = 1e-5
        return layer
    def linear(outputs, inputs):
        return torch.nn.Linear(inputs, outputs, bias=False, dtype=torch.float16)
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.config, layer.self_attn.head_dim = config, 2
    for name, width in [('q_proj', 12), ('k_proj', 6), ('v_proj', 6), ('o_proj', 12)]:
        setattr(layer.self_attn, name, linear(width, 12))
    layer.mlp = torch.nn.Module()
    layer.mlp.gate_proj, layer.mlp.up_proj = linear(7, 12), linear(7, 12)
    layer.mlp.down_proj, layer.mlp.act_fn = linear(12, 7), torch.nn.SiLU()
    layer.input_layernorm, layer.post_attention_layernorm = norm(), norm()
    source = torch.nn.Module()
    source.config = config
    source.model = torch.nn.Module()
    source.model.layers = torch.nn.ModuleList([layer, copy.deepcopy(layer)])
    source.model.embed_tokens = torch.nn.Embedding(19, 12, dtype=torch.float16)
    source.model.norm = norm()
    source.lm_head = linear(19, 12)
    return source.eval()


@unittest.skipUnless(os.getenv('NDP_GEM5_PIM_WORKER'), 'Configure the FP16 worker')
class TensorParallelModelTests(unittest.TestCase):
    @torch.inference_mode()
    def test_model_history_and_generation(self):
        torch.manual_seed(72)
        source = tiny_model()
        group = ndp.DeviceGroup([dict(backend='gem5_pim', worker=os.environ['NDP_GEM5_PIM_WORKER'])]*2)
        model = ndp.nn.TensorParallelLlamaForCausalLM(source, devices=group, max_tokens=6)
        self.assertEqual(set(model.state_dict()), set(source.state_dict()))
        reference = [ndp.nn.LlamaDecodeBlock(layer, runtime=group.device(0), max_tokens=6)
                     for layer in source.model.layers]
        ids = torch.tensor([[1, 2, 3]])
        for token in ids.split(1, dim=1):
            x = source.model.embed_tokens(token)
            for block in reference:
                x = block(x)
            x = (x.float()*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+1e-5)).half()
            expected = source.lm_head(x*source.model.norm.weight).float()
            torch.testing.assert_close(model(token), expected, atol=.005, rtol=.03)
        self.assertEqual([layer.self_attn._position for layer in model.model.layers], [3, 3])
        model.load_state_dict(source.state_dict(), assign=True)
        self.assertEqual(model.model._position, 0)
        first = model.generate(ids, max_new_tokens=4, eos_token_id=[])
        self.assertEqual(first.shape, (1, 7))
        torch.testing.assert_close(model.generate(ids, max_new_tokens=4, eos_token_id=[]), first)
        torch.testing.assert_close(model.generate(ids, max_new_tokens=0), ids)
        stopped = model.generate(ids, max_new_tokens=4, eos_token_id=first[0, 3].item())
        self.assertEqual(stopped.shape, (1, 4))
        with self.assertRaises(ValueError):
            model.generate(ids, max_new_tokens=5)
        with self.assertRaises(ValueError):
            model(torch.tensor([[19]]))
        model.reset_history()
        last = model.model.layers[-1]
        with patch.object(last.mlp, 'forward_distributed', side_effect=RuntimeError('late failure')):
            with self.assertRaisesRegex(RuntimeError, 'late failure'):
                model(ids[:, :1])
        with self.assertRaisesRegex(RuntimeError, 'whole stack'):
            model(ids[:, :1])
        model.reset_history()
        self.assertEqual([layer.self_attn._position for layer in model.model.layers], [0, 0])
        original = model(ids[:, :1])
        model.reset_history()
        with patch.object(model.model.norm, 'forward', side_effect=RuntimeError('norm failed')):
            with self.assertRaisesRegex(RuntimeError, 'norm failed'):
                model(ids[:, :1])
        with self.assertRaisesRegex(RuntimeError, 'whole stack'):
            model(ids[:, :1])
        model.reset_history()
        torch.testing.assert_close(model(ids[:, :1]), original, atol=0, rtol=0)

    @torch.inference_mode()
    def test_stack_keeps_interlayer_rows_resident(self):
        torch.manual_seed(73)
        source = tiny_model()
        group = ndp.DeviceGroup([dict(backend='gem5_pim', worker=os.environ['NDP_GEM5_PIM_WORKER'])]*2)
        stack = ndp.nn.TensorParallelLlamaDecodeStack(source.model.layers, devices=group, max_tokens=2)
        x = source.model.embed_tokens(torch.tensor([[1]]))
        # Python upload/download calls occur only at the stack boundary; collectives use L2 staging.
        root = group.device(0)
        with patch.object(root, 'upload', wraps=root.upload) as upload, \
                patch.object(root, 'download', wraps=root.download) as download:
            actual = stack(x)
            self.assertEqual(upload.call_count, 1)
            self.assertEqual(download.call_count, 1)
        stack.reset_history()
        torch.testing.assert_close(stack(x), actual, atol=0, rtol=0)
        stack(x)
        with self.assertRaises(ValueError):
            stack(x)

    @torch.inference_mode()
    def test_llama_norm_rounds_before_weight(self):
        from ndp.nn import _ResidentRMSNorm
        torch.manual_seed(74)
        norm = torch.nn.Module()
        norm.register_buffer('weight', torch.linspace(.1, 2.3, 32).half())
        norm.variance_epsilon = 1e-5
        session = ndp.Session(backend='gem5_pim', worker=os.environ['NDP_GEM5_PIM_WORKER'])
        adapter = _ResidentRMSNorm(norm, session)
        x = torch.randn(1, 32).half()
        normalized = x.float()*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+norm.variance_epsilon)
        expected = normalized.half()*norm.weight
        fused = (normalized*norm.weight.float()).half()
        self.assertTrue(torch.any(expected != fused))
        actual = session.download(adapter(session.upload(x)))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        norm.weight.fill_(.75)
        adapter.load_state_dict(norm.state_dict())
        torch.testing.assert_close(session.download(adapter(session.upload(x))),
                                   normalized.half()*norm.weight, atol=0, rtol=0)

    @torch.inference_mode()
    def test_group_placement_and_boundary_failure(self):
        torch.manual_seed(75)
        source = tiny_model()
        options = dict(backend='gem5_pim', worker=os.environ['NDP_GEM5_PIM_WORKER'])
        groups = [ndp.DeviceGroup([options]*2) for _ in range(2)]
        reference = ndp.nn.TensorParallelLlamaForCausalLM(source, devices=groups[0], max_tokens=5)
        hybrid = ndp.nn.TensorParallelLlamaForCausalLM(source, device_groups=groups,
                                                      placement=[1, 0], max_tokens=5)
        self.assertIs(hybrid.model.input_session, groups[1].device(0))
        self.assertIs(hybrid.model.output_session, groups[0].device(0))
        ids = torch.tensor([[1, 2]])
        with patch.object(hybrid.model.devices, 'copy_to', wraps=hybrid.model.devices.copy_to) as copy_to:
            for token in ids.split(1, dim=1):
                torch.testing.assert_close(hybrid(token), reference(token), atol=0, rtol=0)
            self.assertEqual(copy_to.call_count, 2)
        torch.testing.assert_close(hybrid.generate(ids, max_new_tokens=3),
                                   reference.generate(ids, max_new_tokens=3), atol=0, rtol=0)
        hybrid.load_state_dict(source.state_dict())
        self.assertEqual([layer.self_attn._position for layer in hybrid.model.layers], [0, 0])
        with patch.object(hybrid.model.devices, 'copy_to', side_effect=RuntimeError('handoff failed')):
            with self.assertRaisesRegex(RuntimeError, 'handoff failed'):
                hybrid(ids[:, :1])
        self.assertEqual([layer.self_attn._position for layer in hybrid.model.layers], [1, 0])
        with self.assertRaisesRegex(RuntimeError, 'whole stack'):
            hybrid(ids[:, :1])
        hybrid.reset_history()
        reference.reset_history()
        torch.testing.assert_close(hybrid(ids[:, :1]), reference(ids[:, :1]), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            ndp.nn.TensorParallelLlamaDecodeStack(source.model.layers, device_groups=[groups[0], groups[0]],
                                                  placement=[0, 1], max_tokens=2)
        for placement in ([0], [0, 2], [0, True], [0, 0]):
            with self.assertRaises(ValueError):
                ndp.nn.TensorParallelLlamaDecodeStack(source.model.layers, device_groups=groups,
                                                      placement=placement, max_tokens=2)
        with self.assertRaises(ValueError):
            ndp.nn.TensorParallelLlamaDecodeStack(source.model.layers, devices=groups[0],
                                                  device_groups=groups, placement=[0, 1], max_tokens=2)
