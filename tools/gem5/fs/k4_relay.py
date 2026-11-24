"""Exchange one binary worker frame per stock gem5 guest exit event."""
import socket
import struct

LIMIT = 16 * 1024 * 1024


def read_frame(stream):
    header = stream.read(4)
    if len(header) != 4:
        raise EOFError("external client disconnected")
    length, = struct.unpack("<I", header)
    if not 1 <= length <= LIMIT:
        raise ValueError("invalid external frame length")
    body = stream.read(length)
    if len(body) != length:
        raise EOFError("truncated external request")
    return header + body


def relay_events(directory, readfile, port):
    ready = directory / "k4-ready.txt"
    if not ready.is_file() or ready.read_bytes() != b"K4 guest ready\n":
        raise RuntimeError("guest exited before K4 readiness; inspect " + str(directory / "board.pc.com_1.device"))
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", port))
        listener.listen(1)
        print("K4_HOST_READY port={}".format(listener.getsockname()[1]), flush=True)
        client, _ = listener.accept()
        with client, client.makefile("rb") as stream:
            yield from exchange_events(directory, readfile, client, stream)


def exchange_events(directory, readfile, client, stream):
    response = directory / "k4-response.bin"
    client.settimeout(300)
    while True:
        request = read_frame(stream)
        readfile.write_bytes(request)
        yield False  # Resume guest until the reply has been exported.
        reply = response.read_bytes()
        response.unlink()
        if len(reply) < 5 or len(reply) > LIMIT + 4 or struct.unpack_from("<I", reply)[0] != len(reply) - 4:
            raise ValueError("invalid reply from guest")
        client.sendall(reply)
        if request[4:] == b"Q":
            if reply != b"\x01\0\0\0\0":
                raise ValueError("guest close failed")
            print("K4_HOST_SESSION_CLOSED", flush=True)
            yield True
            return
