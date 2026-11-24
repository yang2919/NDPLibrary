"""Check FS worker replies, dependent GEMVs, and controlled client disconnects."""
import argparse
import socket
import struct
from fs_exchange import receive


def verify(stream, send, case):
    def frame(body):
        return struct.pack('<I', len(body)) + body

    def request(body):
        send(frame(body))
        size, = struct.unpack('<I', receive(stream, 4))
        if not 1 <= size <= 16 * 1024 * 1024:
            raise ValueError('invalid reply length')
        reply = receive(stream, size)
        if reply[0]:
            raise RuntimeError(reply[1:])
        return reply[1:]

    def allocate(role, rows, cols):
        return struct.unpack('<I', request(struct.pack('<ccII', b'C', role, rows, cols)))[0]

    def upload(handle, values):
        request(struct.pack(f'<cI{len(values)}e', b'U', handle, *values))

    def gemv(out, weight, vector):
        return struct.pack('<cIII', b'G', out, weight, vector)

    def expect(actual, expected):
        if actual != expected:
            raise ValueError(f'expected {expected!r}, received {actual!r}')

    expect(request(b'H\x02'), b'LPDDR6/FP16/FP32')
    w = allocate(b'W', 2, 3)
    x = allocate(b'X', 1, 3)
    y = allocate(b'Y', 1, 2)
    upload(w, [1, 2, 3, -1, 0, 2])
    if case in ('partial', 'lost'):
        upload(x, [1, 1, 1])
        packet = frame(gemv(y, w, x))
        send(packet[:-1] if case == 'partial' else packet)
        return  # Close without reading a GEMV reply.

    second = allocate(b'W', 1, 2)
    out = allocate(b'Y', 1, 1)
    upload(second, [2, -1])
    for values, expected, chained in [([1, 1, 1], (6, 1), 11), ([2, 0, -1], (-1, -4), 2)]:
        upload(x, values)
        request(gemv(y, w, x))
        request(gemv(out, second, y))
        expect(struct.unpack('<e', request(struct.pack('<cI', b'D', out))), (chained,))
        expect(struct.unpack('<2e', request(struct.pack('<cI', b'D', y))), expected)
    for handle in (out, second, y, x, w):
        request(struct.pack('<cI', b'R', handle))
    expect(request(b'Q'), b'')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('chain', 'partial', 'lost'), default='chain')
    parser.add_argument('--port', type=int, default=45240)
    args = parser.parse_args()
    with socket.create_connection(('127.0.0.1', args.port)) as client, client.makefile('rb') as stream:
        verify(stream, client.sendall, args.case)
    print('NDP_FS_CLIENT_PASS' if args.case == 'chain' else 'NDP_FS_DISCONNECT_INJECTED', args.case)
