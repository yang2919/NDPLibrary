"""Synchronous host endpoint for the N1 capture-worker protocol."""
import struct


class Client:
    def __init__(self, source, sink):
        self.source = source
        self.sink = sink
        if self.request(b"H\x03") != b"\x03\x00":
            raise RuntimeError("expected a v3 capture executor")

    def request(self, body):
        if not 1 <= len(body) <= 16 << 20:
            raise ValueError("invalid request size")
        self.sink.write(struct.pack("<I", len(body)) + body)
        self.sink.flush()
        header = self.source.read(4)
        if len(header) != 4:
            raise EOFError("worker disconnected")
        size, = struct.unpack("<I", header)
        if not 1 <= size <= 16 << 20:
            raise ValueError("invalid reply size")
        reply = self.source.read(size)
        if len(reply) != size:
            raise EOFError("incomplete worker response")
        if reply[0]:
            raise RuntimeError(reply[1:].decode())
        return reply[1:]

    def load(self, elf, profile, template, slots):
        record = b"P" + struct.pack("<III", profile, len(template), len(slots)) + template
        declarations = b"".join(struct.pack("<IIIQ", *slot) for slot in slots)
        return struct.unpack("<Q", self.request(record + declarations + elf))[0]

    def allocate(self, size, access):
        return struct.unpack("<Q", self.request(b"A" + struct.pack("<QI", size, access)))[0]

    def launch(self, program, template, bindings):
        record = struct.pack("<QII", program, len(template), len(bindings)) + template
        operands = b"".join(struct.pack("<IIQQQ", *binding) for binding in bindings)
        capture = self.request(b"J" + record + operands)
        if len(capture) != 8 + len(template):
            raise ValueError("invalid capture receipt")
        return struct.unpack_from("<Q", capture)[0], capture[8:]

    def unload(self, handle):
        self.request(b"U" + struct.pack("<Q", handle))

    def release(self, handle):
        self.request(b"F" + struct.pack("<Q", handle))
