"""Check persistent TCP relay replies and cleanup using a native worker only."""
import argparse
from pathlib import Path
import select
import socket
import struct
import subprocess
import sys
import tempfile

from target_fixture import write_fixture


def test(worker, failure=None):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_fixture(root / 'fixture')
        process = subprocess.Popen([
            sys.executable, str(Path(__file__).with_name('se_stream.py')),
            '--directory', str(root), '--worker', str(worker), '--port', '0',
            '--timeout', '5'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            if not select.select([process.stdout], [], [], 10)[0]:
                raise AssertionError('relay readiness timeout')
            ready = process.stdout.readline()
            if 'NDP_SE_HOST_READY' not in ready:
                raise AssertionError(process.communicate(timeout=5))
            port = int(ready.split('port=')[1].split()[0])
            with socket.create_connection(('127.0.0.1', port), timeout=5) as client:
                if failure == 'length':
                    client.sendall(struct.pack('<I', 0xffffffff))
                elif failure == 'disconnect':
                    client.sendall(b'\x02\0')
                else:
                    requests = (root / 'fixture/requests.bin').read_bytes()
                    actual = bytearray()
                    with client.makefile('rb') as stream:
                        pos = 0
                        while pos < len(requests):
                            size, = struct.unpack_from('<I', requests, pos)
                            request = requests[pos:pos+4+size]
                            pos += len(request)
                            client.sendall(request[:2])
                            client.sendall(request[2:])
                            header = stream.read(4)
                            length, = struct.unpack('<I', header)
                            actual.extend(header + stream.read(length))
                    assert actual == (root / 'fixture/expected.bin').read_bytes()
            stdout, stderr = process.communicate(timeout=10)
            if failure:
                assert process.returncode != 0
                assert 'NDP_SE_SESSION_CLOSED' not in stdout
            else:
                assert process.returncode == 0, stderr
                assert 'NDP_SE_SESSION_CLOSED' in stdout
            assert not (root / 'request.fifo').exists()
            assert not (root / 'reply.fifo').exists()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('worker', type=Path)
    worker = parser.parse_args().worker.resolve()
    for failure in (None, 'length', 'disconnect'):
        test(worker, failure)
    print('PASS: native persistent session, split frames, invalid length and disconnect cleanup')
