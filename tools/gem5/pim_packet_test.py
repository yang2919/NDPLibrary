"""Send one FP16 dot product using the NIP1 frame and NDP1 v2 command formats."""
import argparse
import os
import socket
import struct
import sys
import zlib

HEADER = struct.Struct("<4sHHQQII")
EPOCH = 1
# channels, banks/channel, rows/bank, row bytes, GB bytes/channel, GPR bytes, scratchpad bytes
GEOMETRY = (1, 1, 8, 64, 64, 256, 128)


def address(space, offset):
    return struct.pack("<BQ", space, offset)


def command(opcode, body):
    return struct.pack("<HHI", opcode, 0, len(body)) + body


def read_exact(sock, count):
    data = bytearray()
    while len(data) < count:
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise ValueError("connection closed before full reply")
        data.extend(chunk)
    return bytes(data)


def exchange(sock, sequence, payload, expected):
    sock.sendall(HEADER.pack(b"NIP1", 1, 4, EPOCH, sequence, len(payload),
                             zlib.crc32(payload)) + payload)
    magic, version, kind, epoch, seq, size, crc = HEADER.unpack(read_exact(sock, HEADER.size))
    if (magic, version, epoch, seq) != (b"NIP1", 1, EPOCH, sequence) or size > 8 * 1024 * 1024:
        raise ValueError(f"request {sequence}: invalid response header")
    reply = read_exact(sock, size)
    if zlib.crc32(reply) != crc:
        raise ValueError(f"request {sequence}: response CRC mismatch")
    if kind == 6 and len(reply) >= 3:
        op, error = struct.unpack_from("<BH", reply)
        raise ValueError(f"request {sequence}, op {op}, error {error}: {reply[3:]!r}")
    if kind != 5 or reply != expected:
        raise ValueError(f"request {sequence}: expected {expected.hex()}, got kind {kind}, {reply.hex()}")


def run(sock):
    sock.settimeout(10)
    hello = struct.pack("<BHHH7Q", 1, 2, 1, 1, *GEOMETRY)
    exchange(sock, 1, hello, hello)

    # Row 1 starts at Shared byte 64. Store 16 weights of 2 and 16 inputs of 3.
    for seq, offset, value in [(2, 64, 2), (3, 128, 3)]:
        data = struct.pack("<e", value) * 16
        exchange(sock, seq, b"\x02" + address(1, offset) + struct.pack("<Q", len(data)) + data, b"\x02")

    commands = [
        command(10, address(2, 0) + struct.pack("<QB", 2, 0)),  # Zero the MAC seed in GPR.
        command(2, struct.pack("<QQQ", 1, 1, 0)),               # Initialize channel 0, bank 0.
        command(1, address(1, 128) + address(4, 0) + struct.pack("<Q", 32)),  # Shared to GB.
        command(4, struct.pack("<QQQQ", 1, 1, 0, 1)),           # One burst from weight row 1.
        command(5, struct.pack("<QQ", 1, 32)),                  # MAC result to GPR byte 32.
        command(9, address(2, 32) + address(1, 192) + struct.pack("<Q", 2)),
    ]
    records = b"".join(commands)
    batch = struct.pack("<4sHHII", b"NDP1", 2, 1, 16 + len(records), len(commands)) + records
    exchange(sock, 4, b"\x04" + batch, b"\x04")
    exchange(sock, 5, b"\x03" + address(1, 192) + struct.pack("<Q", 2),
             b"\x03" + struct.pack("<e", 96))
    exchange(sock, 6, b"\x05", b"\x05")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--socket", help="connect to a listening Unix stream socket")
    target.add_argument("--receiver", help="launch the repository reference receiver executable")
    args = parser.parse_args()
    if args.socket:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(10)
            sock.connect(args.socket)
            run(sock)
    else:
        client, child = socket.socketpair()
        with client, child:
            argv = [args.receiver, "--serve", "3", str(EPOCH), "10000", *map(str, GEOMETRY)]
            pid = os.posix_spawn(args.receiver, argv, os.environ,
                                 file_actions=[(os.POSIX_SPAWN_DUP2, child.fileno(), 3)])
            child.close()
            try:
                run(client)
            finally:
                client.close()
                _, status = os.waitpid(pid, 0)
            if status:
                raise ValueError(f"reference receiver exited with status {os.waitstatus_to_exitcode(status)}")
    print("PASS: PIM dot product = 96 (FP16); all six replies matched.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"FAIL: {error}")
