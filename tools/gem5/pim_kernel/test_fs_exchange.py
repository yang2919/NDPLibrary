"""Exercise the FS mailbox relay with a native worker and a disconnected peer."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile

from fs_exchange import Exchange, receive
from verify_fs_remote import verify


def exercise(worker, case):
    peer, server = socket.socketpair()
    peer.settimeout(10)
    server.settimeout(10)

    def client():
        with peer, peer.makefile('rb') as stream:
            verify(stream, peer.sendall, case)

    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(1) as pool:
        directory = Path(tmp)
        relay = Exchange(directory, directory / 'request', server)
        future = pool.submit(client)
        with subprocess.Popen([worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
            try:
                while True:
                    relay.step()
                    packet = relay.mailbox.read_bytes()
                    if not packet:
                        break
                    proc.stdin.write(packet)
                    proc.stdin.flush()
                    header = receive(proc.stdout, 4)
                    size, = struct.unpack('<I', header)
                    relay.reply.write_bytes(header + receive(proc.stdout, size))
                    if packet[4:] == b'Q':
                        break
                relay.step(final=True)
                future.result(timeout=5)  # Client finishes before guest teardown.
                proc.stdin.close()
                assert proc.wait(timeout=5) == 0
            finally:
                relay.close()
                if proc.poll() is None:
                    proc.kill()
        future.result()
        if case == 'partial':
            assert relay.partial_body == (b'G', 12, 13)
        assert relay.disconnected == (case != 'chain')
        assert relay.completed.count('G') == {'partial': 0, 'lost': 1, 'chain': 4}[case]
        print('NDP_NATIVE_FS_EXCHANGE_PASS', case)


if __name__ == '__main__':
    for scenario in ('chain', 'partial', 'lost'):
        exercise(sys.argv[1], scenario)
