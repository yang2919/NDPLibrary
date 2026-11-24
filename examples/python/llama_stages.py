"""Verify two resident SmolLM2 layers and count v2 activation frames on each worker."""
import argparse
from contextlib import ExitStack, contextmanager
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import tempfile


def relay(worker_path, log_path, kernels=False):
    """Observe activation sizes on the actual v2 stream in a separate process."""
    stream = sys.stdin.buffer
    with subprocess.Popen([worker_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as worker, \
            open(log_path, "w", buffering=1) as log:
        while header := stream.read(4):
            size, = struct.unpack("<I", header)
            request = stream.read(size)
            worker.stdin.write(header + request)
            worker.stdin.flush()
            reply_header = worker.stdout.read(4)
            reply_size, = struct.unpack("<I", reply_header)
            reply = worker.stdout.read(reply_size)
            if reply[0] == 0:
                if request[:1] == b"U":
                    print("U", len(request) - 5, file=log)
                elif request[:1] == b"D":
                    print("D", len(reply) - 1, file=log)
                elif kernels and request[:1] == b"A":
                    print("A", *struct.unpack_from("<IIII", request, 17), file=log)
                elif kernels and request[:1] == b"G":
                    print("G", file=log)
            sys.stdout.buffer.write(reply_header + reply)
            sys.stdout.buffer.flush()
            if request == b"Q":
                return


@contextmanager
def counted_worker(path, *, kernels=False):
    import ndp
    with tempfile.TemporaryDirectory() as directory:
        executable = Path(directory) / "worker"
        log = Path(directory) / "transfers"
        command = [sys.executable, str(Path(__file__).resolve()), "--relay", str(Path(path).resolve()), str(log)]
        if kernels:
            command.append("kernels")
        executable.write_text("#!/bin/sh\nexec " + shlex.join(command) + "\n")
        executable.chmod(0o700)
        with ndp.Session(backend="gem5_pim", worker=str(executable)) as session, log.open() as frames:
            yield session, frames


def run(layers, tokens, worker, placement):
    import ndp
    with ExitStack() as resources:
        endpoints = [resources.enter_context(counted_worker(worker)) for _ in set(placement)]
        stack = ndp.nn.LlamaDecodeStack(layers, devices=[s for s, _ in endpoints],
                                        placement=placement, max_tokens=len(tokens))
        outputs = []
        for session, log in endpoints:
            session.synchronize()
            log.read()  # Exclude weight and RoPE uploads during preparation.
        for token in tokens:
            outputs.append(stack(token))
            size = token.numel() * token.element_size()
            for device, (_, log) in enumerate(endpoints):
                frames = log.read().splitlines()
                if frames != [f"U {size}", f"D {size}"]:
                    raise AssertionError(f"device {device}: unexpected worker transfers {frames}")
        return outputs


def main():
    import torch
    from transformers import LlamaForCausalLM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--worker", required=True)
    args = parser.parse_args()
    model = LlamaForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
                                            dtype=torch.float16, attn_implementation="eager").eval()
    with torch.inference_mode():
        tokens = model.model.embed_tokens(torch.tensor([[1, 42, 30, 198]])).split(1, dim=1)
        layers = model.model.layers[:2]
        baseline = run(layers, tokens, args.worker, [0, 0])
        split = run(layers, tokens, args.worker, [0, 1])
    for expected, actual in zip(baseline, split):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    print("Four tokens matched exactly. Per token: one U/D pair per worker,",
          tokens[0].numel() * tokens[0].element_size(), "activation bytes per frame.")


if __name__ == "__main__":
    if len(sys.argv) in (4, 5) and sys.argv[1] == "--relay":
        relay(sys.argv[2], sys.argv[3], len(sys.argv) == 5)
    else:
        main()
