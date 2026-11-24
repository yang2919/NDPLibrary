"""Compare PNM and PIM linear execution using one resident prepared weight."""
import argparse
import torch
import ndp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndp-config", help="Select the NDP command backend; omit to use the reference backend")
    args = parser.parse_args()
    options = {"backend": "ndp", "config": args.ndp_config} if args.ndp_config else {}
    with ndp.Session(**options) as session:
        weight = session.upload(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16))
        weight = ndp.llm.prepare_weight(session, weight)
        x = session.upload(torch.tensor([[1, 2, 3]], dtype=torch.float16))
        for engine in (ndp.Engine.PNM, ndp.Engine.PIM):
            y = session.linear(x, weight.tensor(), ndp.OpOptions(engine, ndp.LayoutPolicy.RequirePrepared))
            print(engine, session.download(y))


if __name__ == "__main__":
    main()
