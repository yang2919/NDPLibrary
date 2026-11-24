"""Compare two-worker decoder execution with the same checkpoint on one worker."""
from contextlib import ExitStack
from unittest.mock import patch

import ndp
import torch

from verify_decoder_remote import tiny_source


def compare(actual, expected):
    torch.testing.assert_close(actual, expected, atol=.005, rtol=.03)


def run_schedule(model, chunks):
    outputs = []
    with model.new_request() as request:
        for tokens in chunks:
            outputs.append(model.prefill(torch.tensor([tokens]), request=request))
        assert request.position == sum(map(len, chunks))
    return torch.cat(outputs, dim=1)


def observe(model, placement):
    """Read activations at normalization inputs without changing the forward path."""
    stack = model.model
    layer = stack.layers[1 if placement else 0]
    session = layer.devices.device(0)
    captures = []
    def capture(module, inputs):
        captures.append(session.download(inputs[0]).clone())
    with ExitStack() as hooks:
        if placement:
            hooks.enter_context(layer.input_layernorm.register_forward_pre_hook(capture))
        else:
            # The block reduces attention first and FFN second for each chunk.
            reduce = layer.devices.reduce_partials
            reductions = 0
            def observe_reduce(parts):
                nonlocal reductions
                result = reduce(parts)
                if reductions % 2 == 0:
                    captures.append(session.download(result).clone())
                reductions += 1
                return result
            hooks.enter_context(patch.object(layer.devices, "reduce_partials", observe_reduce))
        hooks.enter_context(stack.norm.register_forward_pre_hook(capture))
        logits = run_schedule(model, ([1, 2], [3]))
    assert len(captures) == 4
    return [*captures, logits]


@torch.inference_mode()
def verify_decoder(group):
    source = tiny_source(kv_heads=2)
    make_model = ndp.nn.TensorParallelLlamaForCausalLM
    single = make_model(source, devices=[group.device(0)], max_tokens=5)
    parallel = make_model(source, devices=group, max_tokens=5)
    for actual, expected in zip(observe(parallel, False), observe(single, False)):
        compare(actual, expected)
    print('NDP_TP_DECODER_BLOCK_PASS', flush=True)

    full = run_schedule(parallel, ([1, 2, 3, 4], [5]))
    compare(full, run_schedule(parallel, ([1, 2], [3, 4], [5])))
    other = run_schedule(parallel, ([6], [7]))
    with parallel.new_request() as a, parallel.new_request() as b:
        compare(parallel.prefill(torch.tensor([[1, 2]]), request=a), full[:, :2])
        compare(parallel(torch.tensor([[6]]), request=b), other[:, :1])
        assert (a.position, b.position) == (2, 1)
        a.close()
        compare(parallel(torch.tensor([[7]]), request=b), other[:, 1:])
        assert b.position == 2
    print('NDP_TP_DECODER_STATE_PASS', flush=True)

    source = tiny_source(kv_heads=2, layers=2)
    single = make_model(source, devices=[group.device(0)], max_tokens=3)
    placed = make_model(source, device_groups=[[group.device(0)], [group.device(1)]],
                        placement=[0, 1], max_tokens=3)
    for actual, expected in zip(observe(placed, True), observe(single, True)):
        compare(actual, expected)
    print('NDP_LAYER_TRANSFER_PASS', flush=True)
