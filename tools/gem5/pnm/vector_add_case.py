"""Prepare real ELF launch fixtures via N1, or check a target's N2 execution evidence."""
import argparse
import json
import struct
import subprocess
from pathlib import Path

from client import Client


def prepare(worker, elf, output, vlen):
    if vlen < 128 or vlen > 65536 or vlen & (vlen - 1):
        raise ValueError("VLEN must be a power of two in [128, 65536]")
    image = elf.read_bytes()
    vlmax = vlen // 16
    for n in sorted({5, vlmax - 1, vlmax, vlmax + 1, 2 * vlmax + 1}):
        folder = output / f"n{n}"
        folder.mkdir(parents=True, exist_ok=True)
        args = struct.pack("<IIIIQQQ", 1, 40, n, 0, 0, 0, 0)
        slots = [(16, 1, 2, 2*n), (24, 1, 2, 2*n), (32, 2, 2, 2*n)]
        with subprocess.Popen([worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
            try:
                client = Client(process.stdout, process.stdin)
                program = client.load(image, 1, args, slots)
                handles = [client.allocate(2*n + 4, mode) for _, mode, _, _ in slots]
                bindings = [(slot, mode, h, 2, size) for (slot, mode, _, size), h in zip(slots, handles)]
                pc, patched = client.launch(program, args, bindings)
                client.unload(program)
                for h in handles:
                    client.release(h)
            finally:
                process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("capture worker failed")
        if pc != 0x10000000:
            raise ValueError("unexpected ELF entry for fixture linker script")
        a = [1, 1, -2, 0.5, 4]
        b = [2**-11, 3*2**-11, 1.5, -0.5, 2]
        def halves(values):
            return b"".join(struct.pack("<e", values[i % len(values)]) for i in range(n))
        guard = b"\x55\xaa"
        for name, data in [("a", halves(a)), ("b", halves(b)), ("y", b"\xff" * (2*n)),
                           ("expected", halves([x+y for x, y in zip(a, b)]))]:
            (folder / f"{name}.bin").write_bytes(guard + data + guard)
        (folder / "args.bin").write_bytes(patched)
        addresses = struct.unpack_from("<QQQ", patched, 16)
        state = dict(pc=pc, sp=0xc0010000, ra=0x1ffff000, a0=0x41000000, a1=len(patched),
                     vlen_bits=vlen, elements=n,
                     buffer_bases=[p-2 for p in addresses])
        (folder / "state.json").write_text(json.dumps(state, indent=2) + "\n")
        (folder / "vector_add.elf").write_bytes(image)
        print(f"Captured n={n} PC={pc:#x} args={len(patched)}")


def verify(folder, evidence):
    state = json.loads((folder / "state.json").read_text())
    report = json.loads((evidence / "completion.json").read_text())
    for key, expected in dict(issuer="pnm", entry_pc=state["pc"], return_pc=state["ra"],
                              status=0, quiescent=True, vlen_bits=state["vlen_bits"]).items():
        if key not in report or type(report[key]) is not type(expected) or report[key] != expected:
            raise ValueError(f"completion mismatch: {key}")
    if (folder / "expected.bin").read_bytes() != (evidence / "output.bin").read_bytes():
        raise ValueError("FP16 output or guard mismatch")
    print(f"N2 vector add passed for n={state['elements']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare")
    for name in ("worker", "elf", "output"):
        prep.add_argument(name, type=Path)
    prep.add_argument("--vlen", type=int, default=128)
    check = modes.add_parser("verify")
    for name in ("case", "evidence"):
        check.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.worker.resolve(), args.elf, args.output, args.vlen)
    else:
        verify(args.case, args.evidence)
