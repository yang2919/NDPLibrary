"""Prepare the N3 two-launch target fixture or check its returned FP16 output and trace summary."""
import argparse
import json
from pathlib import Path
import struct
import subprocess

from client import Client

WINDOW = 0x800000000
WEIGHT_ROW = 7
CASES = [((1, 1, 1), (6, 1)), ((2, 0, -1), (-1, -4))]
GUARD = b"\x55\xaa"


def prepare(worker, elf, directory):
    directory.mkdir(parents=True, exist_ok=True)
    image = elf.read_bytes()
    template = struct.pack("<IIIIQQQ", 1, 40, 2, 3, 0, 0, 0)
    slots = [(16, 1, 4, 16), (24, 1, 2, 6), (32, 2, 2, 4)]
    with subprocess.Popen([worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
        try:
            client = Client(process.stdout, process.stdin)
            program = client.load(image, 2, template, slots)
            buffers = [client.allocate(16, 1), client.allocate(6, 1), client.allocate(8, 2)]
            bindings = [(patch, mode, handle, 2 if mode == 2 else 0, extent)
                        for (patch, mode, _, extent), handle in zip(slots, buffers)]
            pc, args = client.launch(program, template, bindings)
            if client.launch(program, template, bindings) != (pc, args):
                raise ValueError("second launch changed entry or buffer addresses")
            client.unload(program)
            for handle in buffers:
                client.release(handle)
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("capture worker failed")
    if pc != 0x10000000:
        raise ValueError("unexpected entry for fixture linker script")
    descriptor, vector, output = struct.unpack_from("<QQQ", args, 16)
    (directory / "pim_gemv.elf").write_bytes(image)
    (directory / "args.bin").write_bytes(args)
    (directory / "matrix.bin").write_bytes(struct.pack("<4I", 1, 1, WEIGHT_ROW, 1))
    # LPDDR6 lane 0 and lane 1 of unit 0, row 7, column 0.
    writes = {"matrix.bin": descriptor, "args.bin": 0x41000000}
    for lane, row in enumerate(((1, 2, 3), (-1, 0, 2))):
        name = f"weight{lane}.bin"
        (directory / name).write_bytes(struct.pack("<3e", *row))
        writes[name] = WINDOW + (WEIGHT_ROW << 18) + (lane << 16)
    for i, (values, expected) in enumerate(CASES, 1):
        (directory / f"x{i}.bin").write_bytes(struct.pack("<3e", *values))
        (directory / f"expected{i}.bin").write_bytes(GUARD + struct.pack("<2e", *expected) + GUARD)
    (directory / "y.bin").write_bytes(GUARD + b"\xff" * 4 + GUARD)
    state = dict(pc=pc, sp=0xc0010000, ra=0x1ffff000, a0=writes["args.bin"], a1=len(args),
                 frm=0, vstart=0, vlen_bits=128, writes=writes, x=vector, guarded_y=output-2)
    (directory / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    print(f"Captured two launches, PC={pc:#x}, descriptor={descriptor:#x}, output={output:#x}")


def verify(directory, evidence):
    state = json.loads((directory / "state.json").read_text())
    for i in (1, 2):
        report = json.loads((evidence / f"completion{i}.json").read_text())
        expected = dict(issuer="pnm", entry_pc=state["pc"], return_pc=state["ra"],
                        status=0, quiescent=True, vlen_bits=128)
        for key, value in expected.items():
            if type(report.get(key)) is not type(value) or report[key] != value:
                raise ValueError(f"launch {i}: invalid {key}")
        # At e8,m8 and VLEN=128 a six-byte tile fits in one vector command per unit.
        if report.get("wr_gb") != 8 or report.get("ab_mac") != 8:
            raise ValueError(f"launch {i}: unexpected PNM-issued GB/MAC instruction counts")
        if (directory / f"expected{i}.bin").read_bytes() != (evidence / f"output{i}.bin").read_bytes():
            raise ValueError(f"launch {i}: guarded FP16 output mismatch")
    print("N3 target output and completion summaries matched for both launches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for field in ("worker", "elf", "directory"):
        p.add_argument(field, type=Path)
    p = sub.add_parser("verify")
    for field in ("directory", "evidence"):
        p.add_argument(field, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.worker.resolve(), args.elf, args.directory)
    else:
        verify(args.directory, args.evidence)
