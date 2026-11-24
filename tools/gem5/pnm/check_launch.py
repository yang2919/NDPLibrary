"""Decode an N0 launch body, validate bindings and patch its device argument block."""
import argparse
import struct
from pathlib import Path


def relocate(data, program, allocations, slots):
    if len(data) < 16:
        raise ValueError("short launch header")
    handle, size, count = struct.unpack_from("<QII", data)
    if handle != program or size != 40 or count != len(slots) or len(data) != 16 + size + 32 * count:
        raise ValueError("invalid program or launch length")
    args = bytearray(data[16:16 + size])
    if struct.unpack_from("<II", args) != (1, size):
        raise ValueError("argument ABI mismatch")
    pending = dict(slots)
    for patch, access, buffer, offset, extent in struct.iter_unpack("<IIQQQ", data[16 + size:]):
        if patch not in pending or buffer not in allocations:
            raise ValueError("unknown or duplicate binding")
        mode, required, alignment = pending.pop(patch)
        base, capacity, allowed = allocations[buffer]
        if access != mode or access & ~allowed or extent != required:
            raise ValueError("binding permissions or extent mismatch")
        if offset > capacity or extent > capacity - offset or (base + offset) % alignment:
            raise ValueError("binding outside allocation or misaligned")
        if any(args[patch:patch + 8]):
            raise ValueError("nonzero pointer placeholder")
        struct.pack_into("<Q", args, patch, base + offset)
    return args


def check(directory):
    for name, program, first, base, sizes, alignments, scalars in [
        ("add", 1, 11, 0x40001000, (10, 10, 10), (2, 2, 2), (5, 0)),
        ("gemv", 2, 21, 0x40004000, (16, 6, 4), (4, 2, 2), (2, 3)),
    ]:
        allocations, slots = {}, {}
        for i, (size, alignment) in enumerate(zip(sizes, alignments)):
            mode = 2 if i == 2 else 1
            slots[16 + 8 * i] = (mode, size, alignment)
            allocations[first + i] = (base + 4096 * i, size + (2 if name == "add" and i == 0 else 0), mode)
        data = (directory / f"{name}.launch").read_bytes()
        args = relocate(data, program, allocations, slots)
        if struct.unpack_from("<II", args, 8) != scalars or args != (directory / f"{name}.args").read_bytes():
            raise ValueError("golden argument mismatch")
        for index, value in [(0, 99), (16, 2), (32, 1), (60, 3), (64, 99), (72, 255), (88, 16)]:
            bad = bytearray(data)
            bad[index] = value
            try:
                relocate(bad, program, allocations, slots)
            except ValueError:
                continue
            raise AssertionError(f"accepted corrupt byte at {index}")
        print(f"{name}: {len(data)} launch bytes, {len(args)} relocated bytes, seven corruptions rejected")
    if struct.unpack("<IIII", (directory / "gemv.matrix").read_bytes()) != (1, 1, 7, 1):
        raise ValueError("matrix slice fixture mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    check(parser.parse_args().directory)
