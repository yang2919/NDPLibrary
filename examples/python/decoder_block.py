"""Implement a decoder layer directly with NDP operations, from prefill through decode."""
import argparse
import torch
import ndp


def values(count, seed):
    return (((torch.arange(count) * 7 + seed) % 11 - 5) / 16).half()


def toy_weights():
    """Match examples/toy_decoder.hpp; these are illustrative weights, not a model."""
    shapes = {"q": (4, 4), "k": (2, 4), "v": (2, 4), "output": (4, 4),
              "gate": (6, 4), "up": (6, 4), "down": (4, 6)}
    weights = {key: values(rows * cols, seed).reshape(rows, cols)
               for seed, (key, (rows, cols)) in enumerate(shapes.items(), 1)}
    weights["attention_norm"] = torch.tensor([1, .875, 1.125, 1], dtype=torch.float16)
    weights["ffn_norm"] = torch.tensor([.875, 1, 1, 1.125], dtype=torch.float16)
    return weights


class DecoderLayer:
    """A fixed toy GQA/SwiGLU layer with explicit request-owned cache state."""

    def __init__(self, session, weights, capacity=8):
        self.s = session
        self.w = {}
        for name, host in weights.items():
            raw = session.upload(host)
            self.w[name] = raw if name.endswith("norm") else ndp.llm.prepare_weight(session, raw).tensor()
        positions = torch.arange(capacity, dtype=torch.float32) * .25
        self.sin = session.upload(positions.sin().half().reshape(capacity, 1))
        self.cos = session.upload(positions.cos().half().reshape(capacity, 1))

    def new_cache(self, capacity=8):
        return ndp.llm.KVCache(self.s, capacity, kv_heads=1, head_dim=2)

    def forward_ndp(self, x, cache, engine, trace=None):
        s, w = self.s, self.w
        tokens, width = x.shape
        start = cache.scheduled_length()
        positions = list(range(start, start + tokens))
        projection = ndp.OpOptions(engine, ndp.LayoutPolicy.RequirePrepared)

        # Normalize and project Q/K/V; all intermediate values stay on the device.
        normalized = s.rms_norm(x, w["attention_norm"])
        q = s.linear(normalized, w["q"], projection)
        k = s.linear(normalized, w["k"], projection)
        v = s.linear(normalized, w["v"], projection)

        # Split heads and apply interleaved rotary embeddings at absolute positions.
        q = s.reshape(q, [tokens, 2, 2])
        k = s.reshape(k, [tokens, 1, 2])
        v = s.reshape(v, [tokens, 1, 2])
        q = ndp.llm.rope(s, q, self.sin, self.cos, positions)
        k = ndp.llm.rope(s, k, self.sin, self.cos, positions)

        # Append this request's K/V and attend only to the causal prefix.
        cache.append(k, v)
        attended = cache.attention(q, start)
        merged = s.reshape(attended, [tokens, width])
        residual = s.add(x, s.linear(merged, w["output"], projection))

        # Apply SwiGLU FFN and the second residual using the same primitive API.
        n = s.rms_norm(residual, w["ffn_norm"])
        gate = s.silu(s.linear(n, w["gate"], projection))
        up = s.linear(n, w["up"], projection)
        down = s.linear(s.mul(gate, up), w["down"], projection)
        output = s.add(residual, down)
        if trace is not None:
            # Handles only: tests may download them; normal serving does not.
            trace.update(normalized=normalized, q=q, k=k, attended=attended, residual=residual)
        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndp-config")
    args = parser.parse_args()
    options = {"backend": "ndp", "config": args.ndp_config} if args.ndp_config else {}
    with ndp.Session(**options) as session:
        layer = DecoderLayer(session, toy_weights())
        cache = layer.new_cache()
        for step in range(4):
            tokens = 2 if step == 0 else 1
            x = session.upload(values(tokens * 4, step + 1).reshape(tokens, 4))
            engine = ndp.Engine.PNM if step == 0 else ndp.Engine.PIM
            y = layer.forward_ndp(x, cache, engine)
            print("prefill" if step == 0 else "decode", session.download(y),
                  "committed tokens:", cache.committed_length())
        cache.reset().wait()


if __name__ == "__main__":
    main()
