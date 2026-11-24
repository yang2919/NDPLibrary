"""Emit the two N0 launch examples for an independent service or simulator decoder."""
import argparse
import struct
from pathlib import Path

HEADER = struct.Struct("<QII")
BINDING = struct.Struct("<IIQQQ")


def emit(directory, name, program, args, operands):
    directory.mkdir(parents=True, exist_ok=True)
    message = HEADER.pack(program, len(args), len(operands)) + args
    relocated = bytearray(args)
    for patch, access, handle, offset, extent, allocation_base in operands:
        message += BINDING.pack(patch, access, handle, offset, extent)
        struct.pack_into("<Q", relocated, patch, allocation_base + offset)
    (directory / f"{name}.launch").write_bytes(message)
    (directory / f"{name}.args").write_bytes(relocated)


def main(directory):
    # All pointer fields start at zero; the service supplies device addresses.
    args = struct.pack("<IIIIQQQ", 1, 40, 5, 0, 0, 0, 0)
    emit(directory, "add", 1, args, [
        (16, 1, 11, 2, 10, 0x40001000),
        (24, 1, 12, 0, 10, 0x40002000),
        (32, 2, 13, 0, 10, 0x40003000),
    ])
    args = struct.pack("<IIIIQQQ", 1, 40, 2, 3, 0, 0, 0)
    emit(directory, "gemv", 2, args, [
        (16, 1, 21, 0, 16, 0x40004000),
        (24, 1, 22, 0, 6, 0x40005000),
        (32, 2, 23, 0, 4, 0x40006000),
    ])
    # One all-bank row slice, beginning at row 7 in the example allocator.
    (directory / "gemv.matrix").write_bytes(struct.pack("<IIII", 1, 1, 7, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
