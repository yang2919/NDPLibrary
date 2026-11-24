"""Verify two FP16 projections against an independently started K4 endpoint."""
import argparse

import ndp
import torch


def verify(host, port, label="K4"):
    with torch.inference_mode(), ndp.Session.connect(host, port) as session:
        weights = torch.tensor([[1, 2, 3], [-1, 0, 2]], dtype=torch.float16)
        projection = ndp.nn.Linear(session, weights)
        for values in ([1, 1, 1], [2, 0, -1]):
            x = torch.tensor([values], dtype=torch.float16)
            y = projection(x)
            torch.testing.assert_close(y, x @ weights.T, rtol=0, atol=0)
            print("GEMV", y.tolist(), flush=True)
    print(f"{label}_CLIENT_PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=45240)
    parser.add_argument("--label", choices=["K4", "NDP_SE", "NDP_FS"], default="K4")
    args = parser.parse_args()
    verify(args.host, args.port, args.label)
