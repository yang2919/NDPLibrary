"""Bridge one external TCP session to a persistent SE or native worker stream."""
import argparse
import os
from pathlib import Path
import select
import socket
import struct
import subprocess
import time

from check_smoke import verify_trace

LIMIT = 16 * 1024 * 1024


def read_exact(fd, size, timeout, process=None):
    deadline = time.monotonic() + timeout
    result = bytearray()
    while len(result) < size:
        if process is not None and process.poll() is not None:
            # Drain a buffered final reply even after a clean worker exit.
            if not select.select([fd], [], [], 0)[0]:
                raise EOFError('worker exited before a complete reply')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('stream deadline expired; completion is uncertain')
        if not select.select([fd], [], [], min(remaining, 0.25))[0]:
            continue
        chunk = os.read(fd, size - len(result))
        if not chunk:
            raise EOFError('stream disconnected')
        result.extend(chunk)
    return bytes(result)


def frame(fd, timeout, process=None):
    header = read_exact(fd, 4, timeout, process)
    size, = struct.unpack('<I', header)
    if not 1 <= size <= LIMIT:
        raise ValueError('invalid frame length')
    return header + read_exact(fd, size, timeout, process)


def write_all(fd, data, timeout):
    deadline = time.monotonic() + timeout
    while data:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('write deadline expired; completion is uncertain')
        if select.select([], [fd], [], remaining)[1]:
            try:
                count = os.write(fd, data)
            except BlockingIOError:
                continue
            data = data[count:]


def accept_client(server, process, timeout):
    """Wait for the user separately from request deadlines; notice worker exit."""
    deadline = time.monotonic() + timeout if timeout else None
    server.settimeout(0.25)
    while True:
        if process.poll() is not None:
            raise RuntimeError('worker exited while waiting for client; inspect worker.log')
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('no external client connected before connect deadline')
        try:
            return server.accept()[0]
        except socket.timeout:
            continue


def serve(directory, worker, port, timeout, gem5=None, cpu='atomic', bind='127.0.0.1', connect_timeout=0):
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / name for name in ('request.fifo', 'reply.fifo')]
    fds = []
    process = None
    try:
        for path in paths:
            os.mkfifo(path, 0o600)
            fds.append(os.open(path, os.O_RDWR | os.O_NONBLOCK))
        with (directory / 'worker.log').open('wb') as log:
            if gem5:
                command = [str(gem5), f'--outdir={directory / "sim"}',
                           '--debug-flags=PIM', '--debug-file=pim.trace',
                           str(Path(__file__).with_name('se_stream_config.py')),
                           '--binary', str(worker), '--input', str(paths[0]),
                           '--output', str(paths[1]), '--cpu', cpu,
                           '--layout', str(directory.parent / 'lpddr6_fp16.h')]
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            else:
                # Open blocking child descriptors separately from the relay ends.
                with paths[0].open('rb', buffering=0) as inp, paths[1].open('wb', buffering=0) as out:
                    process = subprocess.Popen([str(worker)], stdin=inp, stdout=out, stderr=log)
            with socket.socket() as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((bind, port))
                server.listen(1)
                print(f'NDP_SE_HOST_READY port={server.getsockname()[1]} (relay listening; worker handshake pending)', flush=True)
                print("Waiting for terminal B; Ctrl+C cancels. Request timeout starts after connection.", flush=True)
                client = accept_client(server, process, connect_timeout)
                with client:
                    client.settimeout(timeout)
                    count = 0
                    while True:
                        request = frame(client.fileno(), timeout)
                        write_all(fds[0], request, timeout)
                        reply = frame(fds[1], timeout, process)
                        client.sendall(reply)
                        count += 1
                        if request[4:] == b'Q':
                            if reply != b'\x01\0\0\0\0':
                                raise RuntimeError('worker rejected close')
                            break
                if process.wait(timeout=timeout) != 0:
                    raise RuntimeError('worker process failed after close')
                if gem5:
                    print('NDP_SE_PIM_TRACE', verify_trace(directory / 'sim', cpu), flush=True)
                print(f'NDP_SE_SESSION_CLOSED requests={count}', flush=True)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for fd in fds:
            os.close(fd)
        for path in paths[:len(fds)]:
            path.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--gem5', type=Path)
    parser.add_argument('--port', type=int, default=45240)
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--timeout', type=float, default=240)
    parser.add_argument('--connect-timeout', type=float, default=0,
                        help='Seconds to wait for external client; 0 waits until connected or cancelled')
    parser.add_argument('--cpu', choices=['atomic', 'timing'], default='atomic')
    args = parser.parse_args()
    if args.timeout <= 0 or args.connect_timeout < 0:
        parser.error('timeout must be positive; connect-timeout must be nonnegative')
    serve(args.directory.resolve(), args.worker.resolve(), args.port, args.timeout,
          args.gem5.resolve() if args.gem5 else None, args.cpu, args.bind, args.connect_timeout)
