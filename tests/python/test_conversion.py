"""Check scoped projection discovery, preparation, and restoration."""
from importlib.util import find_spec
import unittest
from unittest.mock import patch

import torch

import ndp


def projection():
    layer = torch.nn.Linear(4, 4, bias=False, dtype=torch.float16)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(4, dtype=torch.float16))
    return layer


class ConversionTests(unittest.TestCase):
    def test_preview_reports_ineligible_projections_and_aliases(self):
        target = torch.nn.ModuleDict({name: projection() for name in
                                      ("eligible", "shared", "tied", "view", "hooked", "parametrized")})
        external = torch.nn.ModuleDict({"shared": target["shared"], "tied": projection(), "view": projection()})
        external["tied"].weight = target["tied"].weight
        external["view"].weight = torch.nn.Parameter(target["view"].weight.detach().view(4, 4))
        target["hooked"].register_forward_hook(lambda *args: None)
        torch.nn.utils.parametrize.register_parametrization(
            target["parametrized"], "weight", torch.nn.Identity())
        target["biased"] = torch.nn.Linear(4, 4, dtype=torch.float16)
        target["fp32"] = torch.nn.Linear(4, 4, bias=False)
        model = torch.nn.ModuleDict({"target": target, "external": external}).eval()

        with patch.object(ndp._conversion, "Session") as session:
            report = {entry.path: entry for entry in ndp.plan_to_ndp(model, target="target")}
            with self.assertRaisesRegex(ValueError, "shared module"):
                with ndp.to_ndp(model, target="target.shared"):
                    self.fail("An aliased module must be skipped")
            session.assert_not_called()
        self.assertEqual(report["target.eligible"].rule, "linear")
        self.assertEqual(report["target.shared"].reason, "shared module")
        for name in ("tied", "view"):
            self.assertEqual(report[f"target.{name}"].reason, "shared weight storage")
        self.assertIn("hooks", report["target.hooked"].reason)
        self.assertIsNone(report["target.parametrized"].rule)
        self.assertIn("bias-free", report["target.biased"].reason)
        self.assertIn("CPU FP16", report["target.fp32"].reason)

        originals = dict(target.items())
        with ndp.to_ndp(model, target="target"):
            self.assertIsInstance(target["eligible"], ndp.nn.Linear)
            for name, original in originals.items():
                if name != "eligible":
                    self.assertIs(target[name], original)
        self.assertEqual(dict(target.items()), originals)

    @torch.inference_mode()
    def test_shared_session_routing_state_dict_and_normal_exit(self):
        model = torch.nn.Sequential(torch.nn.Sequential(projection(), projection()), projection()).eval()
        original = tuple(model[0])
        outside = model[1]
        checkpoint = model.state_dict()
        with ndp.to_ndp(model, target="0", mode="projections") as converted:
            self.assertIs(converted, model)
            self.assertIs(model[1], outside)
            self.assertIsInstance(model[0][0], ndp.nn.Linear)
            self.assertIsInstance(model[0][1], ndp.nn.Linear)
            replacement = model[0][0]
            self.assertEqual(replacement.in_features, original[0].in_features)
            self.assertEqual(replacement.out_features, original[0].out_features)
            self.assertIs(replacement.session, model[0][1].session)
            self.assertEqual(list(model.state_dict()), list(checkpoint))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, checkpoint[name], atol=0, rtol=0)
            with self.assertRaisesRegex(RuntimeError, "already has an active"):
                with ndp.to_ndp(model, target="1"):
                    self.fail("An active model cannot be converted twice")
            for tokens, expected_calls in ((1, 2), (3, 0)):
                inputs = torch.ones((1, tokens, 4), dtype=torch.float16)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                    output = converted(inputs)
                calls = {event.key: event.count for event in trace.key_averages()}
                self.assertEqual(calls.get("ndp::pim_projection", 0), expected_calls)
                torch.testing.assert_close(output, inputs, atol=0, rtol=0)
            torch.testing.assert_close(replacement(input=inputs), inputs, atol=0, rtol=0)
        self.assertEqual(tuple(model[0]), original)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            replacement(torch.ones((1, 1, 4), dtype=torch.float16))

    def test_prepare_failure_leaves_original_modules_and_allows_retry(self):
        model = torch.nn.Sequential(projection(), projection()).eval()
        original = tuple(model)
        prepare = ndp.nn.Linear.from_float
        sessions = []

        def fail_second(source, *, runtime):
            self.assertEqual(tuple(model), original)
            sessions.append(runtime)
            if source is original[1]:
                raise ndp.NdpError("injected preparation failure")
            return prepare(source, runtime=runtime)

        with patch.object(ndp.nn.Linear, "from_float", side_effect=fail_second):
            with self.assertRaisesRegex(ndp.NdpError, "projection '1'.*injected preparation failure"):
                with ndp.to_ndp(model, target=""):
                    self.fail("Failed preparation must prevent entry")
        self.assertEqual(len(sessions), 2)
        self.assertIs(sessions[0], sessions[1])
        self.assertEqual(tuple(model), original)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            sessions[0].capabilities()
        with ndp.to_ndp(model, target=""):
            self.assertIsInstance(model[1], ndp.nn.Linear)
        self.assertEqual(tuple(model), original)

    def test_inference_error_restores_modules_and_closes_session(self):
        model = torch.nn.Sequential(projection()).eval()
        original = model[0]
        with self.assertRaisesRegex(ValueError, "matching feature width"), torch.inference_mode():
            with ndp.to_ndp(model, target="0", backend="functional"):
                runtime = model[0].session
                model(torch.ones((1, 1, 3), dtype=torch.float16))
        self.assertIs(model[0], original)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            runtime.capabilities()

    def test_invalid_target_or_mode(self):
        model = torch.nn.Sequential(projection()).eval()
        with self.assertRaisesRegex(ValueError, "Unknown module path"):
            ndp.plan_to_ndp(model, target="0.weight")
        with self.assertRaisesRegex(ValueError, "mode='projections'"):
            ndp.plan_to_ndp(model, target="0", mode="unknown")
        with self.assertRaisesRegex(ValueError, "eval mode"):
            ndp.plan_to_ndp(model.train(), target="0")


@unittest.skipUnless(find_spec("transformers"), "optional Transformers dependency")
class LlamaConversionTests(unittest.TestCase):
    def setUp(self):
        from transformers import __version__

        if __version__ != "4.57.6":
            self.skipTest("Llama conversion targets Transformers 4.57.6")
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
                             attn_implementation="eager")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.model = LlamaForCausalLM(config).half().eval()
        self.attention = self.model.model.layers[0].self_attn
        self.target = "model.layers.0.self_attn"

    @torch.inference_mode()
    def test_shared_projection_preserves_attention_cache_and_state_dict(self):
        model, attention = self.model, self.attention
        original = attention.forward
        source_modules = tuple(attention.children())
        checkpoint = model.state_dict()
        report = {entry.path: entry for entry in ndp.plan_to_ndp(model, target=self.target)}
        self.assertEqual(report[self.target].rule, "llama_shared_qkv")

        def trace():
            prefill = model(torch.tensor([[1, 3, 4]]), use_cache=True, output_attentions=True)
            decode = model(torch.tensor([[5]]), past_key_values=prefill.past_key_values, use_cache=True,
                           output_attentions=True)
            self.assertEqual(decode.past_key_values.get_seq_length(), 4)
            cache = decode.past_key_values.layers[0]
            return (prefill.logits, prefill.attentions[0], decode.logits, decode.attentions[0], cache.keys, cache.values)

        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            with ndp.to_ndp(model, target=self.target, backend="functional"):
                self.assertIs(model.get_submodule(self.target), attention)
                self.assertEqual(list(model.state_dict()), list(checkpoint))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, checkpoint[name], atol=0, rtol=0)
                with patch.object(attention, "forward", original):
                    expected = trace()
                with patch.object(ndp.nn, "project_qkv", wraps=ndp.nn.project_qkv) as project:
                    actual = trace()
                    project.assert_called_once()
                    project.reset_mock()
                    calls = []
                    with attention.q_proj.register_forward_hook(lambda *_: calls.append(True)):
                        trace()
                    project.assert_not_called()
                    self.assertEqual(calls, [True, True])
                for observed, reference in zip(actual, expected, strict=True):
                    torch.testing.assert_close(observed, reference, atol=0, rtol=0)
                raise RuntimeError("inference failed")
        self.assertNotIn("forward", vars(attention))
        self.assertEqual(attention.forward, original)
        self.assertEqual(tuple(attention.children()), source_modules)

    @torch.inference_mode()
    def test_partial_targets_and_unsupported_attention_keep_individual_projection(self):
        model, attention, target = self.model, self.attention, self.target
        original_q = attention.q_proj
        original_k = attention.k_proj
        with ndp.to_ndp(model, target=f"{target}.q_proj"):
            self.assertNotIn("forward", vars(attention))
            self.assertIsInstance(attention.q_proj, ndp.nn.Linear)
            self.assertIs(attention.k_proj, original_k)

        calls = []
        with original_q.register_forward_hook(lambda *_: calls.append(True)):
            report = {entry.path: entry for entry in ndp.plan_to_ndp(model, target=target)}
            self.assertIn("all three eligible", report[target].reason)
            with ndp.to_ndp(model, target=target):
                self.assertNotIn("forward", vars(attention))
                self.assertIs(attention.q_proj, original_q)
                self.assertIsInstance(attention.k_proj, ndp.nn.Linear)
                model(torch.tensor([[3]]))
            self.assertEqual(calls, [True])

        original_forward = type(attention).forward

        def custom_forward(module, *args, **kwargs):
            return original_forward(module, *args, **kwargs)

        with patch.object(type(attention), "forward", custom_forward):
            report = {entry.path: entry for entry in ndp.plan_to_ndp(model, target=target)}
            self.assertIn("unmodified", report[target].reason)
            with ndp.to_ndp(model, target=target):
                self.assertNotIn("forward", vars(attention))
                self.assertIsInstance(attention.q_proj, ndp.nn.Linear)

        model.config._attn_implementation = "sdpa"
        report = {entry.path: entry for entry in ndp.plan_to_ndp(model, target=target)}
        self.assertIn("eager", report[target].reason)
        with ndp.to_ndp(model, target=target):
            self.assertNotIn("forward", vars(attention))
            self.assertIsInstance(attention.q_proj, ndp.nn.Linear)
            model(torch.tensor([[3]]))

    @torch.inference_mode()
    def test_resident_swiglu_prefill_and_decode(self):
        model = self.model
        target = "model.layers.0.mlp"
        original = model.get_submodule(target)
        checkpoint = model.state_dict()
        entry, = ndp.plan_to_ndp(model, target=target, mode="resident")
        self.assertEqual(entry.rule, "llama_resident_swiglu")

        with ndp.to_ndp(model, target=target, mode="resident"):
            mlp = model.get_submodule(target)
            self.assertIsInstance(mlp, ndp.nn.SwiGLU)
            self.assertEqual(list(model.state_dict()), list(checkpoint))
            for name, weight in model.state_dict().items():
                torch.testing.assert_close(weight, checkpoint[name], atol=0, rtol=0)
            session = mlp.gate_proj.session
            for rows, engine in ((1, ndp.Engine.PIM), (3, ndp.Engine.PNM)):
                x = torch.linspace(-1, 1, rows * 16, dtype=torch.float16).reshape(1, rows, 16)
                with (patch.object(session, "linear", wraps=session.linear) as project,
                      patch.object(session, "silu", wraps=session.silu) as silu,
                      patch.object(session, "mul", wraps=session.mul) as mul,
                      torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler):
                    output = mlp(x)
                self.assertEqual([call.args[2].engine for call in project.call_args_list], [engine] * 3)
                self.assertEqual([call.args[0].shape[0] for call in project.call_args_list], [rows] * 3)
                silu.assert_called_once()
                mul.assert_called_once()
                transfers = {event.key: event.count for event in profiler.key_averages()}
                self.assertEqual((transfers.get("ndp::upload"), transfers.get("ndp::download")), (1, 1))
                self.assertEqual(output.shape, x.shape)
                torch.testing.assert_close(output, original(x), atol=0.002, rtol=0.002)
        self.assertIs(model.get_submodule(target), original)

    def test_resident_requires_complete_mlp_and_backend_support(self):
        model = self.model
        target = "model.layers.0.mlp"
        original = model.get_submodule(target)
        attention = "model.layers.0.self_attn"
        with self.assertRaisesRegex(ValueError, "requires a Llama SwiGLU MLP target"):
            with ndp.to_ndp(model, target=attention, mode="resident"):
                self.fail("Resident conversion must cover its requested target")

        with original.act_fn.register_forward_hook(lambda *_: None):
            entry, = ndp.plan_to_ndp(model, target=target, mode="resident")
            self.assertIn("unmodified SiLU", entry.reason)

        query = ndp.Session.query_operation

        def reject_gemm(session, spec):
            if spec.kind == ndp.OpKind.Linear and spec.engine == ndp.Engine.PNM:
                raise ndp.NdpError("backend has no PNM Linear")
            return query(session, spec)

        with (patch.object(ndp.Session, "query_operation", reject_gemm),
              patch.object(ndp.Session, "upload") as upload):
            with self.assertRaisesRegex(ndp.NdpError, r"model\.layers\.0\.mlp.*PNM Linear.*2 row"):
                with ndp.to_ndp(model, target=target, mode="resident"):
                    self.fail("Unsupported prefill must prevent conversion")
            upload.assert_not_called()
        self.assertIs(model.get_submodule(target), original)

    @torch.inference_mode()
    def test_resident_block_chunked_prefill_greedy_and_independent_requests(self):
        from transformers.cache_utils import DynamicCache

        model = self.model
        target = "model.layers.0"
        original = model.get_submodule(target)
        state_dict = model.state_dict()
        prompt = torch.tensor([[1, 3, 4, 5]])
        other = torch.tensor([[1, 7, 8]])
        expected = model(prompt, use_cache=True).logits
        expected_other = model(other, use_cache=True).logits
        expected_ids = model.generate(prompt, max_new_tokens=2, do_sample=False, pad_token_id=0)
        entry, = ndp.plan_to_ndp(model, target=target, mode="resident")
        self.assertEqual(entry.rule, "llama_resident_block")

        with ndp.to_ndp(model, target=target, mode="resident", backend="functional", max_tokens=8):
            self.assertIsInstance(model.get_submodule(target), ndp.nn.LlamaDecodeBlock)
            self.assertEqual(list(model.state_dict()), list(state_dict))
            for name, weight in model.state_dict().items():
                torch.testing.assert_close(weight, state_dict[name], atol=0, rtol=0)
            uncached = model(prompt, use_cache=False)
            self.assertIsNone(uncached.past_key_values)
            torch.testing.assert_close(uncached.logits, expected, atol=0.02, rtol=0.02)

            def step(ids, cache):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                    result = model(ids, past_key_values=cache, use_cache=True)
                counts = {event.key: event.count for event in profiler.key_averages()}
                self.assertEqual((counts.get("ndp::upload"), counts.get("ndp::download")), (1, 1))
                return result.logits

            first = DynamicCache()
            second = DynamicCache(config=model.config)
            a_begin = step(prompt[:, :2], first)
            b_begin = step(other[:, :2], second)
            a_rest = step(prompt[:, 2:], first)
            b_rest = step(other[:, 2:], second)
            torch.testing.assert_close(torch.cat((a_begin, a_rest), dim=1), expected, atol=0.02, rtol=0.02)
            torch.testing.assert_close(torch.cat((b_begin, b_rest), dim=1), expected_other, atol=0.02, rtol=0.02)
            self.assertEqual([first.get_seq_length(i) for i in range(2)], [4, 4])
            self.assertEqual([second.get_seq_length(i) for i in range(2)], [3, 3])
            self.assertIsNot(first.layers[0].kv, second.layers[0].kv)
            with self.assertRaisesRegex(ValueError, "cannot be exported"):
                _ = first[0]
            for _ in range(2):
                ids = model.generate(prompt, max_new_tokens=2, do_sample=False, pad_token_id=0)
                torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)
        self.assertIs(model.get_submodule(target), original)
        self.assertIsNone(first.layers[0].kv)
        self.assertIsNone(second.layers[0].kv)
        with self.assertRaisesRegex(ValueError, "closed conversion"):
            with ndp.to_ndp(model, target=target, mode="resident"):
                model(torch.tensor([[6]]), past_key_values=first, use_cache=True)

    @torch.inference_mode()
    def test_resident_block_rejects_invalid_calls_before_earlier_layers_advance(self):
        from transformers.cache_utils import StaticCache

        model = self.model
        target = "model.layers.1"
        prompt = torch.tensor([[1, 3]])
        token = torch.tensor([[4]])
        cpu_cache = model(prompt, use_cache=True).past_key_values
        with ndp.to_ndp(model, target=target, mode="resident", max_tokens=4):
            with self.assertRaisesRegex(ValueError, "fresh cache"):
                model(token, past_key_values=cpu_cache, use_cache=True)
            self.assertEqual(cpu_cache.get_seq_length(), 2)
            with self.assertRaisesRegex(ValueError, "DynamicCache"):
                model(token, past_key_values=StaticCache(config=model.config, max_cache_len=4))

            cache = model(prompt, use_cache=True).past_key_values
            for inputs, kwargs, pattern in (
                    (token, {"attention_mask": torch.tensor([[0, 1, 1]])}, "unpadded"),
                    (token, {"cache_position": torch.tensor([3])}, "contiguous"),
                    (token, {"output_attentions": True}, "outputs"),
                    (torch.tensor([[5, 6, 7]]), {}, "KV capacity"),
                    (token.repeat(2, 1), {}, "one unpadded sequence")):
                with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                    model(inputs, past_key_values=cache, use_cache=True, **kwargs)
                self.assertEqual([cache.get_seq_length(i) for i in range(2)], [2, 2])
            model(token, past_key_values=cache, use_cache=True)
            self.assertEqual([cache.get_seq_length(i) for i in range(2)], [3, 3])

            session = model.get_submodule(target).session
            with patch.object(session, "download", side_effect=RuntimeError("injected download failure")):
                with self.assertRaisesRegex(RuntimeError, "injected download failure"):
                    model(token, past_key_values=cache, use_cache=True)
            self.assertIsNone(cache.layers[1].kv)
            with self.assertRaisesRegex(RuntimeError, "Start a fresh request"):
                model(token, past_key_values=cache, use_cache=True)
            self.assertEqual(model(prompt, use_cache=True).past_key_values.get_seq_length(1), 2)
