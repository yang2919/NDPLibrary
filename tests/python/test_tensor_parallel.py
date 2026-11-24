"""Compare output-channel shards, host gather, and device replicas with unsharded Linear."""
import os
import unittest

import ndp
import torch


class TensorParallelTests(unittest.TestCase):
    def check_projection(self, options):
        torch.manual_seed(37)
        with torch.inference_mode():
            group = ndp.DeviceGroup([options, options])
            source = torch.nn.Linear(7, 5, bias=False, dtype=torch.float16).eval()
            parallel = ndp.nn.TensorParallelLinear(source, devices=group)
            single = ndp.nn.Linear.from_float(source, runtime=group.device(0))
            for rank in (1, 2, 3):
                x = torch.randn(14, dtype=torch.float16)[::2].reshape((1,)*(rank-1) + (7,))
                torch.testing.assert_close(parallel(x), single(x), rtol=0, atol=0)
                shards = parallel.forward_shards(x)
                self.assertEqual([s.shape for s in shards], [[1, 3], [1, 2]])
                for i, replica in enumerate(group.all_gather_rows(shards)):
                    torch.testing.assert_close(group.device(i).download(replica), single(x).reshape(1, 5),
                                               rtol=0, atol=0)
            with self.assertRaises(ndp.NdpError):
                group.all_gather_rows(shards[::-1])
            with self.assertRaises(ndp.NdpError):
                group.all_gather_rows(shards[:1])
            with self.assertRaises(ValueError):
                parallel(torch.ones(2, 7).half())
            self.assertEqual(list(parallel.state_dict()), ["weight"])
            source.weight.zero_()
            parallel.load_state_dict(source.state_dict(), assign=True)
            self.assertEqual(parallel(x).abs().max().item(), 0)

    def test_uneven_projection(self):
        self.check_projection({"backend": "functional"})

    @unittest.skipUnless(os.getenv("NDP_GEM5_PIM_WORKER"), "Configure the FP16 worker")
    def test_two_workers(self):
        self.check_projection({"backend": "gem5_pim", "worker": os.environ["NDP_GEM5_PIM_WORKER"]})

    @unittest.skipUnless(os.getenv("NDP_GEM5_PIM_WORKER"), "Configure the FP16 worker")
    def test_input_split_and_resident_chain(self):
        with torch.inference_mode():
            group = ndp.DeviceGroup([dict(backend="gem5_pim", worker=os.environ["NDP_GEM5_PIM_WORKER"])]*2)
            first = torch.nn.Linear(3, 5, bias=False, dtype=torch.float16).eval()
            second = torch.nn.Linear(5, 2, bias=False, dtype=torch.float16).eval()
            first.weight.fill_(1)
            second.weight.fill_(2)
            column = ndp.nn.TensorParallelLinear(first, devices=group)
            row = ndp.nn.TensorParallelLinear(second, devices=group, split="input")
            resident = column.forward_distributed(torch.ones(1, 3).half())
            partials = row.forward_distributed(resident)
            self.assertEqual(partials.layout, "partial")
            self.assertEqual(partials.tensors[0].dtype, ndp.DType.F32)
            for tensor in partials.replicated().tensors:
                self.assertEqual(tensor.dtype, ndp.DType.F16)
            torch.testing.assert_close(row(resident), torch.full((1, 2), 30.).half())
            torch.testing.assert_close(row(resident.replicated()), row(resident), rtol=0, atol=0)
            with self.assertRaises(ValueError):
                row.forward_distributed(partials)
            with self.assertRaises(ndp.NdpError):
                ndp.nn.DistributedRow(group, resident.tensors[::-1], 5, "sharded", resident.ranges)
            extreme = torch.nn.Linear(4, 1, bias=False, dtype=torch.float16).eval()
            extreme.weight.copy_(torch.tensor([[200, 200, -200, -200]]).half())
            layer = ndp.nn.TensorParallelLinear(extreme, devices=group, split="input")
            x = torch.full((1, 4), 200.).half()
            partials = layer.forward_distributed(x)
            for i, expected in enumerate((80000., -80000.)):
                value = group.device(i).download(partials.tensors[i])
                self.assertEqual(value.dtype, torch.float32)
                self.assertEqual(value.item(), expected)
            copied = group.copy_to(partials.tensors[0], 1)
            self.assertEqual(group.device(1).download(copied).item(), 80000.)
            self.assertEqual(layer(x).item(), 0.)

    @unittest.skipUnless(os.getenv("NDP_GEM5_PIM_WORKER"), "Configure the FP16 worker")
    def test_mlp_and_gqa_history(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        torch.manual_seed(71)
        with torch.inference_mode():
            group = ndp.DeviceGroup([dict(backend="gem5_pim", worker=os.environ["NDP_GEM5_PIM_WORKER"])]*2)
            def linear(n, k):
                return torch.nn.Linear(k, n, bias=False, dtype=torch.float16)
            mlp = torch.nn.Module()
            mlp.gate_proj, mlp.up_proj, mlp.down_proj = linear(7, 12), linear(7, 12), linear(12, 7)
            mlp.act_fn = torch.nn.SiLU()
            mlp.eval()
            parallel_mlp = ndp.nn.TensorParallelSwiGLU(mlp, devices=group)
            x = torch.randn(1, 12).half()
            expected = mlp.down_proj(mlp.act_fn(mlp.gate_proj(x))*mlp.up_proj(x))
            torch.testing.assert_close(parallel_mlp(x), expected, atol=.002, rtol=.02)
            attention = torch.nn.Module()
            attention.config = SimpleNamespace(hidden_size=12, num_attention_heads=6,
                num_key_value_heads=3, model_type="llama", rope_scaling=None,
                max_position_embeddings=8, rope_theta=10000.)
            attention.head_dim = 2
            attention.q_proj, attention.k_proj = linear(12, 12), linear(6, 12)
            attention.v_proj, attention.o_proj = linear(6, 12), linear(12, 12)
            attention.eval()
            parallel = ndp.nn.TensorParallelLlamaAttention(attention, devices=group, max_tokens=3)
            self.assertEqual(parallel.local_q_heads, (4, 2))
            self.assertEqual(parallel.local_kv_heads, (2, 1))
            tokens = torch.randn(3, 12).half()
            keys, values, outputs = [], [], []
            for position, token in enumerate(tokens):
                def rotate(t):
                    t = t.reshape(-1, 2)
                    phase = torch.tensor(float(position))
                    return t * phase.cos().half() + torch.stack((-t[:, 1], t[:, 0]), -1) * phase.sin().half()
                q = rotate(attention.q_proj(token))
                keys.append(rotate(attention.k_proj(token)))
                values.append(attention.v_proj(token).reshape(3, 2))
                k = torch.stack(keys).repeat_interleave(2, dim=1).transpose(0, 1)
                v = torch.stack(values).repeat_interleave(2, dim=1).transpose(0, 1)
                probabilities = torch.softmax((q.float().unsqueeze(1)*k.float()).sum(-1)/2**.5, -1)
                context = (probabilities.unsqueeze(-1)*v.float()).sum(1).half().flatten()
                expected = attention.o_proj(context)
                output = parallel(token)
                torch.testing.assert_close(output, expected, atol=.003, rtol=.03)
                outputs.append(output)
            with self.assertRaises(ValueError):
                parallel(tokens[0])
            parallel.reset_history()
            torch.testing.assert_close(parallel(tokens[0]), outputs[0], atol=0, rtol=0)
            parallel.load_state_dict(attention.state_dict())
            self.assertEqual(parallel._position, 0)
            torch.testing.assert_close(parallel(tokens[0]), outputs[0], atol=0, rtol=0)
            parallel.reset_history()
            with patch.object(parallel.o_proj, 'forward_distributed', side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    parallel(tokens[0])
            with self.assertRaisesRegex(RuntimeError, 'reset_history'):
                parallel(tokens[0])
            parallel.reset_history()
            torch.testing.assert_close(parallel(tokens[0]), outputs[0], atol=0, rtol=0)
            with self.assertRaises(ValueError):
                ndp.nn.TensorParallelLinear(attention.q_proj, devices=group, ranges=[(0, 5), (6, 12)])
            layer = torch.nn.Module()
            layer.self_attn, layer.mlp = attention, mlp
            attention.config.hidden_act = "silu"
            for name in ('input_layernorm', 'post_attention_layernorm'):
                norm = torch.nn.Module()
                norm.register_buffer('weight', torch.ones(12).half())
                norm.variance_epsilon = 1e-5
                setattr(layer, name, norm)
            layer.eval()
            block = ndp.nn.TensorParallelLlamaDecodeBlock(layer, devices=group, max_tokens=3)
            single = ndp.nn.LlamaDecodeBlock(layer, runtime=group.device(0), max_tokens=3)
            reference = []
            for token in tokens:
                x = token.reshape(1, 1, 12)
                output = block(x)
                reference.append(output)
                torch.testing.assert_close(output, single(x), atol=.003, rtol=.03)
            with self.assertRaises(ValueError):
                block(tokens[0].reshape(1, 1, 12))
            block.load_state_dict(layer.state_dict())
            resident = block.forward_distributed(tokens[0].reshape(1, 1, 12))
            self.assertEqual(resident.layout, 'replicated')
            for i, tensor in enumerate(resident.tensors):
                torch.testing.assert_close(group.device(i).download(tensor).reshape(1, 1, 12),
                                           reference[0], atol=0, rtol=0)
            block.reset_history()
            with patch.object(block.mlp, 'forward_distributed', side_effect=RuntimeError('MLP failed')):
                with self.assertRaisesRegex(RuntimeError, 'MLP failed'):
                    block(tokens[0].reshape(1, 1, 12))
            self.assertEqual(block.self_attn._position, 1)
            with self.assertRaisesRegex(RuntimeError, 'reset_history'):
                block(tokens[0].reshape(1, 1, 12))
            block.reset_history()
            torch.testing.assert_close(block(tokens[0].reshape(1, 1, 12)), reference[0], atol=0, rtol=0)
            block.reset_history()
            single.reset_history()
            torch.testing.assert_close(block(resident), single(reference[0]), atol=.003, rtol=.03)
