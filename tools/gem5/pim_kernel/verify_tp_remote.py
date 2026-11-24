"""Verify output shards and host reduction across two independent PIM workers."""
import argparse
from contextlib import ExitStack

import ndp
import torch


@torch.inference_mode()
def verify(group):
    source = torch.nn.Linear(4, 3, bias=False, dtype=torch.float16).eval()
    source.weight.copy_(torch.tensor([[1, 2, 3, 4], [2, -1, 1, 0], [0, 1, 0, -1]]))
    column = ndp.nn.TensorParallelLinear(source, devices=group)
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.float16)
    torch.testing.assert_close(column(x), source(x), atol=0, rtol=0)

    source = torch.nn.Linear(4, 2, bias=False, dtype=torch.float16).eval()
    source.weight.copy_(torch.tensor([[200, 200, -200, -200], [1, 2, 3, 4]]))
    row = ndp.nn.TensorParallelLinear(source, devices=group, split="input")
    parts = row.forward_distributed(torch.full((1, 4), 200., dtype=torch.float16))
    for rank, expected in enumerate(([80000., 600.], [-80000., 1400.])):
        actual = group.device(rank).download(parts.tensors[rank])
        torch.testing.assert_close(actual, torch.tensor([expected]), atol=0, rtol=0)
    result = group.device(0).download(group.reduce_partials(parts.tensors))
    torch.testing.assert_close(result, torch.tensor([[0., 2000.]], dtype=torch.float16),
                               atol=0, rtol=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ports", type=int, nargs=2, default=(45240, 45241))
    parser.add_argument("--native-worker", help="Start two local worker processes")
    parser.add_argument("--suite", choices=("linear", "decoder"), default="linear")
    args = parser.parse_args()
    if not args.native_worker and args.ports[0] == args.ports[1]:
        parser.error("Each worker requires a distinct port")
    with ExitStack() as stack:
        sessions = []
        for port in args.ports:
            session = (ndp.Session(backend="gem5_pim", worker=args.native_worker)
                       if args.native_worker else ndp.Session.connect("127.0.0.1", port))
            sessions.append(stack.enter_context(session))
        group = ndp.DeviceGroup(sessions)
        if args.suite == "decoder":
            from tp_decoder_checks import verify_decoder
            verify_decoder(group)
        else:
            verify(group)
    print(f"NDP_TP_{args.suite.upper()}_CLIENT_PASS devices=2")
