"""Run a native libpim worker as a separate process and reuse its tensor handles."""
import argparse
import struct
import subprocess
from pathlib import Path

MAX_FRAME = 16 * 1024 * 1024


class Session:
    def __init__(self, executable, profile='bf16'):
        if profile not in ('bf16', 'fp16'):
            raise ValueError('unsupported operand profile')
        self.element_format = 'e' if profile == 'fp16' else 'f'
        version = 2 if profile == 'fp16' else 1
        self.process = subprocess.Popen([executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            if self._request(bytes((ord('H'), version))) != f'LPDDR6/{profile.upper()}/FP32'.encode():
                raise RuntimeError('unexpected worker profile')
        except BaseException:
            self._stop()
            raise

    def _request(self, payload):
        if len(payload) > MAX_FRAME:
            raise ValueError('request exceeds frame limit')
        self.process.stdin.write(struct.pack('<I', len(payload)) + payload)
        self.process.stdin.flush()
        header = self.process.stdout.read(4)
        if len(header) != 4:
            raise RuntimeError('worker disconnected before reply')
        size, = struct.unpack('<I', header)
        if not 1 <= size <= MAX_FRAME:
            raise RuntimeError('invalid reply size')
        reply = self.process.stdout.read(size)
        if len(reply) != size:
            raise RuntimeError('truncated reply')
        if reply[0]:
            raise RuntimeError(reply[1:].decode('utf-8', errors='replace'))
        return reply[1:]

    def create(self, role, rows, cols):
        return struct.unpack('<I', self._request(struct.pack('<ccII', b'C', role.encode(), rows, cols)))[0]

    def upload(self, handle, values):
        self._request(struct.pack('<cI', b'U', handle) + struct.pack(f'<{len(values)}{self.element_format}', *values))

    def gemv(self, output, weight, vector):
        self._request(struct.pack('<cIII', b'G', output, weight, vector))

    def download(self, handle):
        data = self._request(struct.pack('<cI', b'D', handle))
        size = struct.calcsize('<' + self.element_format)
        return struct.unpack(f'<{len(data) // size}{self.element_format}', data)

    def release(self, handle):
        self._request(struct.pack('<cI', b'R', handle))

    def _stop(self):
        self.process.stdin.close()
        self.process.stdout.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                self._request(b'Q')
        finally:
            self._stop()
        if exc_type is None and self.process.returncode:
            raise RuntimeError(f'worker exited with {self.process.returncode}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('worker', type=Path)
    parser.add_argument('--profile', choices=('fp16', 'bf16'), default='bf16')
    args = parser.parse_args()
    with Session(args.worker.resolve(), args.profile) as session:
        weight = session.create('W', 2, 3)
        vector = session.create('X', 1, 3)
        output = session.create('Y', 1, 2)
        session.upload(weight, [1, 2, 3, -1, 0, 2])
        for values, expected in [([1, 1, 1], (6, 1)), ([2, 0, -1], (-1, -4))]:
            session.upload(vector, values)
            session.gemv(output, weight, vector)
            result = session.download(output)
            if result != expected:
                raise RuntimeError(f'expected {expected}, received {result}')
            print(f'GEMV {values} -> {result}')
        for handle in (output, vector, weight):
            session.release(handle)


if __name__ == '__main__':
    main()
