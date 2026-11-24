"""Compare cached Llama decode logits with selected first-block projections on NDP."""
import argparse
from contextlib import ExitStack, nullcontext

import torch
from transformers import LlamaForCausalLM

import ndp


def projection_owners(block, target):
    attention = ("q_proj", "k_proj", "v_proj", "o_proj")
    owners = {}
    if target in ("attention", "block"):
        owners.update(dict.fromkeys(attention, block.self_attn))
    if target == "block":
        owners.update(dict.fromkeys(("gate_proj", "up_proj", "down_proj"), block.mlp))
    elif target == "down":
        owners["down_proj"] = block.mlp
    return owners


def replace_module(scope, parent, name, replacement):
    """Restore the previous module when the verification scope exits."""
    original = getattr(parent, name)
    setattr(parent, name, replacement)
    scope.callback(setattr, parent, name, original)


class SharedQKVAttention(torch.nn.Module):
    """Run the first eager Llama attention layer with one explicit Q/K/V input."""
    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, cache_position=None, **kwargs):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward

        layer = self.attention
        q, k, v = ndp.nn.project_qkv(hidden_states, layer.q_proj, layer.k_proj, layer.v_proj)
        shape = (*hidden_states.shape[:-1], -1, layer.head_dim)
        q, k, v = (value.view(shape).transpose(1, 2) for value in (q, k, v))
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, layer.layer_idx,
                                          {"sin": sin, "cos": cos, "cache_position": cache_position})
        output, weights = eager_attention_forward(layer, q, k, v, attention_mask,
                                                   scaling=layer.scaling, **kwargs)
        output = output.reshape(*hidden_states.shape[:-1], -1).contiguous()
        return layer.o_proj(output), weights


@torch.inference_mode()
def verify_routing(model, session, tokens, reuse_qkv=False):
    """Count completed model calls by phase at the CPU/NDP tensor boundary."""
    prompt = torch.tensor([tokens[:-2]])
    if prompt.shape[1] < 2:
        raise ValueError("Routing verification requires at least two prefill tokens")
    owners = projection_owners(model.model.layers[0], "block")
    events = ("ndp::upload", "ndp::pim_projection", "ndp::download")

    def measure(phase, fn, expected):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            result = fn()
        counts = {item.key: item.count for item in profiler.key_averages()}
        observed = tuple(counts.get(key, 0) for key in events)
        print(f"{phase}: uploads={observed[0]} PIM projections={observed[1]} downloads={observed[2]}")
        if observed != expected:
            raise AssertionError(f"{phase}: expected {expected}, observed {observed}")
        return result

    with ExitStack() as scope:
        def prepare():
            for name, owner in owners.items():
                original = getattr(owner, name)
                replace_module(scope, owner, name, ndp.nn.Linear.from_float(original, runtime=session))

        measure("weight preparation", prepare, (7, 0, 0))
        def trajectory(uploads):
            from torch.utils._python_dispatch import TorchDispatchMode

            trace = []
            class Capture(TorchDispatchMode):
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    result = func(*args, **(kwargs or {}))
                    if func == torch.ops.ndp.qkv_pim.default:
                        trace.extend(result)
                    elif func == torch.ops.ndp.linear_pim.default:
                        trace.append(result)
                    return result

            with Capture() if reuse_qkv else nullcontext():
                result = measure("multi-token prefill", lambda: model(prompt, use_cache=True),
                                 (0, 0, 0))
                trace.append(result.logits)
                for step, token in enumerate(tokens[-2:], 1):
                    cache = result.past_key_values
                    def decode():
                        return model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
                    result = measure(f"cached decode {step}", decode, (uploads, 7, 7))
                    trace.append(result.logits)
                result = measure("fresh single-token prompt", lambda: model(prompt[:, :1], use_cache=True),
                                 (uploads, 7, 7))
                trace.append(result.logits)
            return trace

        baseline = trajectory(7)
        if reuse_qkv:
            block = model.model.layers[0]
            replace_module(scope, block, "self_attn", SharedQKVAttention(block.self_attn))
            optimized = trajectory(5)
            # Four logits and seven projection outputs for each of three single-vector calls.
            if len(baseline) != 25 or len(optimized) != 25:
                raise AssertionError("Incomplete projection or logits trace")
            for ref, actual in zip(baseline, optimized):
                torch.testing.assert_close(actual, ref, rtol=0, atol=0)
            print("QKV_REUSE_PASS: all 21 projection outputs and four logits tensors match exactly")


@torch.inference_mode()
def verify(model, session, tokens, target="down"):
    prompt = torch.tensor([tokens[:-2]])
    decode_ids = tokens[-2:]
    block = model.model.layers[0]
    owners = projection_owners(block, target)
    original = {name: getattr(parent, name) for name, parent in owners.items()}
    adapted = {name: ndp.nn.Linear.from_float(layer, runtime=session)
               for name, layer in original.items()}

    packed = {name: layer._packed for name, layer in adapted.items()}

    def run(projections):
        trace = {name: [] for name in original}
        trace["logits"] = []
        with ExitStack() as scope:
            for name, layer in projections.items():
                replace_module(scope, owners[name], name, layer)

            # Each run builds its own cache from the same prompt.
            cache = model(prompt, use_cache=True).past_key_values
            for name, layer in projections.items():
                hook = layer.register_forward_hook(
                    lambda module, args, output, key=name: trace[key].append(output.detach().clone()))
                scope.callback(hook.remove)
            if target == "block":
                trace["block_output"] = []
                hook = block.register_forward_hook(
                    lambda module, args, output: trace["block_output"].append(output.detach().clone()))
                scope.callback(hook.remove)
            if target in ("attention", "block"):
                # The next RMSNorm receives the sum of the attention output and residual.
                trace["attention_residual"] = []
                hook = block.post_attention_layernorm.register_forward_pre_hook(
                    lambda module, args: trace["attention_residual"].append(args[0].detach().clone()))
                scope.callback(hook.remove)
            for token in decode_ids:
                result = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
                trace["logits"].append(result.logits.detach())
                cache = result.past_key_values
        return trace

    cpu = run(original)
    comparisons = {"CPU": cpu}
    if target == "block":
        baseline = dict(original)
        baseline.update({name: adapted[name] for name in projection_owners(block, "attention")})
        comparisons["attention-only"] = run(baseline)
    actual = run(adapted)
    if any(layer._packed is not packed[name] for name, layer in adapted.items()):
        raise AssertionError("A projection replaced its prepared weight during inference")
    print(f"Reused {len(packed)} prepared weights")
    failures = []
    for label, reference in comparisons.items():
        print(f"Reference: {label}")
        for step in range(len(decode_ids)):
            for name, outputs in reference.items():
                observed, expected = actual[name][step], outputs[step]
                try:
                    torch.testing.assert_close(observed, expected, atol=0.02, rtol=0.02)
                except AssertionError as error:
                    failures.append(f"{label} comparison, step {step + 1} {name}: {error}")
                error = (observed.float() - expected.float()).abs().max().item()
                print(f"step {step + 1} {name} shape={tuple(observed.shape)} max absolute error {error}")
            cpu_token = reference["logits"][step].argmax(-1).item()
            ndp_token = actual["logits"][step].argmax(-1).item()
            print(f"step {step + 1} greedy token reference={cpu_token} NDP={ndp_token}")
            if cpu_token != ndp_token:
                failures.append(f"Greedy token mismatch at step {step + 1}")
    if failures:
        raise AssertionError("\n".join(failures))


@torch.inference_mode()
def verify_greedy(model, session, checkpoint, limit, reuse_qkv=False):
    """Compare independent greedy trajectories, using fresh KV caches for every run."""
    from pathlib import Path
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(checkpoint) / "tokenizer.json"))
    prompts = ("The capital of France is", "Water freezes at", "The first three prime numbers are",
               "Once upon a time", "Hello")
    owners = projection_owners(model.model.layers[0], "block")
    adapted = {name: ndp.nn.Linear.from_float(getattr(parent, name), runtime=session)
               for name, parent in owners.items()}
    eos = model.generation_config.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos])

    def generate(ids, steps):
        inputs = torch.tensor([ids])
        cache = None
        tokens, scores = [], []
        for _ in range(steps):
            result = model(inputs, past_key_values=cache, use_cache=True)
            logits = result.logits[0, -1].float()
            token = logits.argmax().item()
            scores.append(logits)
            tokens.append(token)
            if token in stops:
                break
            inputs = torch.tensor([[token]])
            cache = result.past_key_values
        return tokens, scores

    def run_ndp(ids, steps):
        with ExitStack() as scope:
            for name, parent in owners.items():
                replace_module(scope, parent, name, adapted[name])
            if reuse_qkv:
                block = model.model.layers[0]
                replace_module(scope, block, "self_attn", SharedQKVAttention(block.self_attn))
            return generate(ids, steps)

    failures = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        cpu, ref = generate(ids, limit)
        ndp_ids, actual = run_ndp(ids, limit)
        print(f"{prompt!r} CPU={cpu} NDP={ndp_ids}")
        first = next((i for i, (a, b) in enumerate(zip(cpu, ndp_ids)) if a != b), None)
        comparable = first + 1 if first is not None else len(cpu)
        error = max((a - b).abs().max().item()
                    for a, b in zip(ref[:comparable], actual[:comparable]))
        print(f"  common-prefix max logits error={error}")
        if first is not None:
            decision_error = (ref[first] - actual[first]).abs().max().item()
            print(f"  first differing step {first + 1} max logits error={decision_error}")
            # Rebuild each cache from the prompt to reproduce the first differing decision.
            for label, replay, expected in (("CPU", generate, cpu), ("NDP", run_ndp, ndp_ids)):
                chosen, scores = replay(ids, first + 1)
                if chosen != expected[:first + 1]:
                    raise AssertionError(f"{label} replay did not reproduce its trajectory")
                values, indices = scores[-1].topk(2)
                print(f"  step {first + 1} {label} top2={indices.tolist()} "
                      f"scores={values.tolist()} margin={(values[0] - values[1]).item()}")
            failures.append(f"{prompt!r} at generated token {first + 1}")
        elif cpu != ndp_ids:
            failures.append(f"{prompt!r} generated lengths differ")
    if failures:
        raise AssertionError("Greedy divergence: " + ", ".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Local Hugging Face LlamaForCausalLM checkpoint")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-routing", action="store_true", help="Measure all seven first-block projections")
    mode.add_argument("--verify-greedy", action="store_true", help="Compare five independent greedy prompts")
    parser.add_argument("--reuse-qkv", action="store_true", help="Share Q/K/V input during routing or greedy verification")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    mode.add_argument("--target", choices=("down", "attention", "block"), default="down")
    parser.add_argument("--worker", help="Native FP16 libpim worker; default uses functional backend")
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 3, 4])
    args = parser.parse_args()
    if args.reuse_qkv and not (args.verify_routing or args.verify_greedy):
        parser.error("--reuse-qkv requires --verify-routing or --verify-greedy")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not args.verify_greedy and len(args.tokens) < 3:
        parser.error("provide at least one prompt token and two decode tokens")
    model = LlamaForCausalLM.from_pretrained(
        args.checkpoint, local_files_only=True, torch_dtype=torch.float16,
        attn_implementation="eager").eval()
    options = dict(backend="gem5_pim", worker=args.worker) if args.worker else {}
    with ndp.Session(**options) as session:
        if args.verify_greedy:
            verify_greedy(model, session, args.checkpoint, args.max_new_tokens, args.reuse_qkv)
        elif args.verify_routing:
            verify_routing(model, session, args.tokens, args.reuse_qkv)
        else:
            verify(model, session, args.tokens, args.target)
