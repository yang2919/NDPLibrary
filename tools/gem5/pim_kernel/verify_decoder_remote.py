"""Validate the production PyTorch prefill/decode adapter over one remote session."""
import argparse
import json
from pathlib import Path
import select
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import ndp
import torch


def tiny_source(kv_heads=1, layers=1):
    """A deterministic Llama-shaped checkpoint; no model download."""
    torch.manual_seed(1729)
    width = 4 * kv_heads
    config = SimpleNamespace(hidden_size=width, num_attention_heads=2 * kv_heads, num_key_value_heads=kv_heads,
        model_type='llama', rope_scaling=None, max_position_embeddings=8,
        rope_theta=10000., hidden_act='silu', eos_token_id=None)

    def norm():
        result = torch.nn.Module()
        result.register_buffer('weight', torch.ones(width, dtype=torch.float16))
        result.variance_epsilon = 1e-5
        return result

    def linear(inputs, outputs):
        return torch.nn.Linear(inputs, outputs, bias=False, dtype=torch.float16)

    def make_layer():
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.config, layer.self_attn.head_dim = config, 2
        for name, outputs in [('q_proj', width), ('k_proj', width // 2), ('v_proj', width // 2), ('o_proj', width)]:
            setattr(layer.self_attn, name, linear(width, outputs))
        layer.input_layernorm, layer.post_attention_layernorm = norm(), norm()
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_proj, layer.mlp.up_proj = linear(width, 6), linear(width, 6)
        layer.mlp.down_proj, layer.mlp.act_fn = linear(6, width), torch.nn.SiLU()
        return layer

    source = torch.nn.Module()
    source.config = config
    source.model = torch.nn.Module()
    source.model.layers = torch.nn.ModuleList([make_layer() for _ in range(layers)])
    source.model.embed_tokens = torch.nn.Embedding(8, width, dtype=torch.float16)
    source.model.norm = norm()
    source.lm_head = linear(width, 8)
    return source.eval()


@torch.inference_mode()
def verify(host, port):
    source = tiny_source()
    ids = torch.tensor([[1, 2, 3]])
    # A local single-device block is the integration oracle. Arithmetic/adapter
    # code is partly shared; this is not an independent Llama implementation.
    with ndp.Session() as local:
        reference = ndp.nn.LlamaDecodeBlock(source.model.layers[0], runtime=local, max_tokens=3)
        outputs = []
        for token in ids.split(1, 1):
            hidden = reference(source.model.embed_tokens(token))
            normalized = (hidden.float() * torch.rsqrt(
                hidden.float().square().mean(-1, keepdim=True) + 1e-5)).half()
            outputs.append(source.lm_head(normalized * source.model.norm.weight).float())
        expected = torch.cat(outputs, dim=1)
    errors = {}

    def compare(name, actual, golden):
        if not torch.isfinite(actual).all():
            raise AssertionError(f'{name}: nonfinite logits')
        torch.testing.assert_close(actual, golden, atol=.005, rtol=.03)
        errors[name] = (actual - golden).abs().max().item()

    with ndp.Session.connect(host, port) as remote:
        model = ndp.nn.TensorParallelLlamaForCausalLM(source, devices=[remote], max_tokens=3)
        with model.new_request() as request:
            prefix = model.prefill(ids[:, :2], request=request)
            assert request.position == 2
            last = model(ids[:, 2:], request=request)
            assert request.position == 3
            chunked = torch.cat([prefix, last], dim=1)
            compare('prefill_decode_vs_functional', chunked, expected)
            print('NDP_REMOTE_DECODER_PREFILL_DECODE_PASS', flush=True)
            try:
                model(ids[:, :1], request=request)
            except ValueError:
                pass
            else:
                raise AssertionError('cache capacity overflow was accepted')
            assert request.position == 3
            request.reset()
            assert request.position == 0
            tokenwise = torch.cat([model(token, request=request)
                                   for token in ids.split(1, 1)], dim=1)
            compare('tokenwise_vs_functional', tokenwise, expected)
            compare('prefill_vs_tokenwise', chunked, tokenwise)
            request.reset()
            first = model(ids[:, :1], request=request)
            torch.testing.assert_close(first, tokenwise[:, :1], atol=0, rtol=0)
            assert request.position == 1
            print('NDP_REMOTE_DECODER_RESET_PASS', flush=True)
    print(json.dumps({'max_absolute_logit_error': errors, 'tokens': 7,
                      'layers': 1, 'devices': 1, 'query_heads': 2, 'kv_heads': 1}), flush=True)
    print('NDP_SE_DECODER_CLIENT_PASS', flush=True)


@torch.inference_mode()
def verify_isolation(host, port):
    """Compare interleaved request histories against isolated runs on one endpoint."""
    source = tiny_source()
    sequences = ([1, 2, 3], [4, 5, 6, 7])
    errors = {}

    def ids(tokens):
        return torch.tensor([tokens], dtype=torch.int64)

    def compare(name, actual, expected, exact=True):
        if not torch.isfinite(actual).all():
            raise AssertionError(f'{name}: nonfinite logits')
        torch.testing.assert_close(actual, expected,
                                   atol=0 if exact else .005, rtol=0 if exact else .03)
        errors[name] = (actual-expected).abs().max().item()

    with ndp.Session.connect(host, port) as session:
        model = ndp.nn.TensorParallelLlamaForCausalLM(source, devices=[session], max_tokens=4)
        expected = []
        for sequence in sequences:
            with model.new_request(max_tokens=len(sequence)) as isolated:
                expected.append(torch.cat([model(ids([token]), request=isolated)
                                           for token in sequence], dim=1))
        if torch.equal(expected[0][:, :1], expected[1][:, :1]):
            raise AssertionError('distinct prompts must produce distinct reference logits')
        print('NDP_REMOTE_ISOLATION_BASELINES_PASS', flush=True)
        with model.new_request(max_tokens=3) as a, model.new_request(max_tokens=4) as b:
            compare('a_prefill', model.prefill(ids([1, 2]), request=a), expected[0][:, :2], False)
            compare('b_first', model(ids([4]), request=b), expected[1][:, :1])
            compare('a_decode', model(ids([3]), request=a), expected[0][:, 2:], False)
            assert (a.position, b.position) == (3, 1)
            try:
                model(ids([1]), request=a)
            except ValueError:
                pass
            else:
                raise AssertionError('request A capacity overflow accepted')
            assert (a.position, b.position) == (3, 1)
            assert not a.failed and not b.failed
            compare('b_after_a_rejection', model(ids([5]), request=b), expected[1][:, 1:2])
            a.reset()
            assert (a.position, b.position) == (0, 2)
            compare('a_after_reset', model(ids([1]), request=a), expected[0][:, :1])
            compare('b_after_a_reset', model(ids([6]), request=b), expected[1][:, 2:3])
            a.close()
            assert a.closed and not b.closed
            try:
                model(ids([2]), request=a)
            except RuntimeError:
                pass
            else:
                raise AssertionError('closed request A accepted a token')
            compare('b_after_a_close', model(ids([7]), request=b), expected[1][:, 3:4])
            assert b.position == 4 and not b.failed
        assert a.closed and b.closed
    print(json.dumps({'max_absolute_logit_error': errors, 'tokens': 15,
                      'devices': 1, 'shared_model': True,
                      'interleaved_requests': 2, 'parallel_execution': False}), flush=True)
    print('NDP_SE_REQUEST_ISOLATION_CLIENT_PASS', flush=True)


def native_test(worker, suite="decoder"):
    """Run only the native relay/worker, never gem5, for local development."""
    with tempfile.TemporaryDirectory() as directory:
        relay = subprocess.Popen([sys.executable, str(Path(__file__).with_name('se_stream.py')),
            '--directory', directory, '--worker', str(worker.resolve()), '--port', '0'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            if not select.select([relay.stdout], [], [], 15)[0]:
                raise RuntimeError('native relay readiness timeout')
            ready = relay.stdout.readline()
            if 'NDP_SE_HOST_READY' not in ready:
                raise RuntimeError(relay.communicate(timeout=5))
            verifier = verify_isolation if suite == 'isolation' else verify
            verifier('127.0.0.1', int(ready.split('port=')[1].split()[0]))
            output, error = relay.communicate(timeout=15)
            if relay.returncode != 0 or 'NDP_SE_SESSION_CLOSED' not in output:
                raise RuntimeError((output, error))
            print('NDP_NATIVE_DECODER_RELAY_PASS', flush=True)
        finally:
            if relay.poll() is None:
                relay.terminate()
                try:
                    relay.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    relay.kill()
                    relay.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=45240)
    parser.add_argument('--native-worker', type=Path,
                        help='Start a native relay for development; never launches gem5')
    parser.add_argument('--suite', choices=['decoder', 'isolation'], default='decoder')
    args = parser.parse_args()
    if args.native_worker:
        native_test(args.native_worker, args.suite)
    else:
        (verify_isolation if args.suite == 'isolation' else verify)(args.host, args.port)
