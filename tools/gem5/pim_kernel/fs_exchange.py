"""Relay one complete external frame at each paused Linux guest rendezvous."""
import socket
import struct

LIMIT = 16 * 1024 * 1024


def receive(stream, n):
    data = stream.read(n)
    if len(data) != n:
        raise EOFError(f"peer closed after {len(data)}/{n} bytes")
    return data


class Exchange:
    def __init__(self, directory, mailbox, client):
        self.mailbox = mailbox
        self.reply = directory / 'ndp-reply.bin'
        self.client = client
        self.stream = client.makefile('rb')
        self.pending = None
        self.completed = []
        self.disconnected = False
        self.partial_body = None

    def step(self, final=False):
        try:
            if self.pending is not None:
                data = self.reply.read_bytes()
                self.reply.unlink()
                if len(data) < 5 or len(data) > LIMIT + 4 or int.from_bytes(data[:4], 'little') != len(data)-4:
                    raise ValueError('invalid guest reply')
                if data[4]:
                    raise ValueError(f'guest rejected {self.pending}: {data[5:]}')
                self.completed.append(self.pending)
                self.pending = None
                self.client.sendall(data)
            if final:
                return
            header = receive(self.stream, 4)
            size, = struct.unpack('<I', header)
            if not 1 <= size <= LIMIT:
                raise ValueError('invalid client length')
            body = self.stream.read(size)
            if len(body) != size:
                self.partial_body = (body[:1], len(body), size)
                raise EOFError("peer closed during request body")
            self.mailbox.write_bytes(header + body)
            self.pending = chr(body[0])
        except (EOFError, ConnectionError):
            self.disconnected = True
            self.mailbox.write_bytes(b'')  # Guest observes EOF only between dispatches.

    def close(self):
        self.stream.close()
        self.client.close()
