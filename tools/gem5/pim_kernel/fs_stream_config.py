"""Register FS stream rendezvous handlers before running the supplier board config."""
import os
from pathlib import Path
import runpy
import socket
import sys

import m5
from gem5.simulate.exit_handler import ExitHandler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fs_exchange import Exchange

directory = Path(m5.options.outdir)
case = os.environ['FS_CASE']
exchange = None


class Request(ExitHandler, hypercall_num=42):
    def _process(self, simulator):
        global exchange
        if exchange is None:
            mailbox, = directory.glob('readfile_*')
            with socket.create_server(('0.0.0.0', 45240)) as listener:
                host_port = os.environ.get('NDP_HOST_PORT', '45240')
                print(f'NDP_FS_HOST_READY port={host_port} (host loopback)', flush=True)
                client, _ = listener.accept()
            client.settimeout(600)
            exchange = Exchange(directory, mailbox, client)
        exchange.step()

    def _exit_simulation(self):
        return False


class SessionDrained(ExitHandler, hypercall_num=44):
    def _process(self, simulator):
        exchange.step(final=True)
        exchange.close()
        print('NDP_FS_SESSION_DRAINED: switching to Atomic for guest cleanup', flush=True)
        simulator.switch_processor()
        print('NDP_FS_CLEANUP_ATOMIC', flush=True)

    def _exit_simulation(self):
        return False


class Finished(ExitHandler, hypercall_num=43):
    def _process(self, simulator):
        ops = exchange.completed
        if case == 'partial':
            if exchange.partial_body != (b'G', 12, 13):
                raise ValueError(f'unexpected truncated frame: {exchange.partial_body}')
            print('NDP_FS_PARTIAL_GEMV received=12 expected=13', flush=True)
        if case in ('partial', 'lost'):
            expected = list('HCCCUU') + (['G'] if case == 'lost' else [])
            if not exchange.disconnected or ops != expected:
                raise ValueError(f'unexpected disconnect dispatch history: {ops}')
        elif case == 'gemv' and ops.count('G') != 2:
            raise ValueError('expected exactly two GEMVs')
        elif case == 'chain' and ops.count('G') != 4:
            raise ValueError('expected exactly four dependent GEMVs')
        elif exchange.disconnected or not ops or ops[-1] != 'Q':
            raise ValueError('missing acknowledged close')
        print(f'NDP_FS_SESSION_CLOSED case={case} requests={len(ops)} gemv={ops.count("G")}', flush=True)

    def _exit_simulation(self):
        return False


runpy.run_path(str(Path(__file__).resolve().parents[3] /
                   'gem5-ParallelPIM-main/configs/example/gem5_library/riscv-pim.py'),
               run_name='__m5_main__')
