"""Replay production commands on pinned CENT primitives, with explicit numeric modes."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

from codec import decode
from reference import bound32, f32, half_bytes, halves, matmul, packed_image, tree16

CENT_REVISION = '4c2aac6ce6aa3ddfe1944fc9c5d656b90d3fc7d2'
CENT_SHA256 = '9ac2e163c300caf503eea955ab9f3b051a38fd599c4f1b2c062fbc962458a39e'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def load_cent(source):
    source = Path(source)
    need(hashlib.sha256(source.read_bytes()).hexdigest() == CENT_SHA256,
         'CENT source differs from pinned aim_sim.py; do not silently validate a different model')
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    spec = importlib.util.spec_from_file_location('ndp_cent_upstream', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, torch


class Replay:
    def __init__(self, manifest, module, torch, mode, trace):
        self.m, self.torch, self.mode = manifest, torch, mode
        self.C, self.B, self.R = (manifest[k] for k in ('C', 'B', 'R'))
        self.P = self.C * self.B
        self.T = min(self.R, manifest['V']) // 2
        self.S = self.P * self.R
        self.mem = {1: bytearray(manifest['rows'] * self.S),
                    2: bytearray(manifest['gpr']), 3: bytearray(manifest['scratch'])}
        cls = module.PIM
        if mode == 'profile':
            class ProjectArithmetic(cls):
                def MAC(self, a, b, profile):
                    lanes = a * b
                    need(len(lanes) == 16, 'Expected 16-lane MAC')
                    while len(lanes) > 1:
                        lanes = lanes[0::2] + lanes[1::2]
                    return lanes[0]
            cls = ProjectArithmetic
        args = SimpleNamespace(DRAM_column=self.R // 2, DRAM_row=manifest['rows'],
                               burst_length=16, num_banks=self.B, num_channels=self.C,
                               threads=1, only_trace=False, op_trace=False, model_parallel=False,
                               FC_devices=1, reuse_size=1, trace_file=str(trace))
        self.device = cls(args)
        self.layouts = {}
        self.pending_pack = None
        self.mac_reference = {}
        self.products = {}
        self.counts = {'packed': 0, 'pim_jobs': 0, 'pnm_commands': 0, 'downloads': 0,
                       'native_different_downloads': 0}
        self.max_error = 0.0
        self.current_command = None

    def close(self):
        self.device.file.close()

    def channel(self, c):
        return self.device.pim_device['dimm_0'].dimm[f'channel_{c}']

    def channels(self, mask):
        need(0 < mask < (1 << self.C), 'Invalid channel mask')
        return [c for c in range(self.C) if mask & (1 << c)]

    def read(self, address, size):
        space, start = address
        need(space in self.mem and start >= 0 and size >= 0 and start % 2 == 0 and size % 2 == 0,
             'Replay requires aligned FP16 ranges')
        need(start + size <= len(self.mem[space]), 'Device range outside memory')
        return bytes(self.mem[space][start:start + size])

    def write(self, address, data):
        self.read(address, len(data))
        space, start = address
        self.mem[space][start:start + len(data)] = data
        if space == 1:
            offset = 0
            while offset < len(data):
                a = start + offset
                row, within = divmod(a, self.S)
                c, within = divmod(within, self.B * self.R)
                b, byte = divmod(within, self.R)
                size = min(self.R - byte, len(data) - offset)
                values = self.torch.tensor(halves(data[offset:offset + size]), dtype=self.torch.float32)
                self.device.store_to_DRAM_single_bank(0, c, b, row, byte // 2, size // 2, values, False)
                offset += size

    def audit_schedule(self, commands):
        mac = next(c for c in commands if c['op'] == 4)
        layout = next((v for a, v in self.layouts.items() if a[1] // self.S == mac['row']), None)
        need(layout is not None, 'MAC does not address a verified packed region')
        n, k = layout['rows'], layout['cols']
        G, H = (n + self.P - 1) // self.P, (k + self.T - 1) // self.T
        need(len(commands) == 1 + G * ((3 + self.C) * H + 3), 'Incorrect GEMV command count')
        mask = (1 << self.C) - 1
        bankmask = (1 << self.B) - 1
        zero, result = 2 * self.T, 2 * self.T + 2 * self.P
        expected = [{'op': 1, 'src': (1, 0), 'dst': (2, zero), 'size': 2 * self.P}]
        x = commands[3]['src']
        y = commands[(3 + self.C) * H + 3]['dst']
        need(x[0] == y[0] == 1, 'GEMV operands must be Shared')
        for g in range(G):
            expected.append({'op': 2, 'channels': mask, 'banks': bankmask, 'offset': zero})
            for t in range(H):
                q = min(self.T, k - t * self.T)
                expected.extend([
                    {'op': 1, 'src': (1, 0), 'dst': (2, 0), 'size': 2 * self.T},
                    {'op': 1, 'src': (1, x[1] + t * self.T * 2), 'dst': (2, 0), 'size': 2 * q},
                    *[{'op': 1, 'src': (2, 0), 'dst': (4, ch * self.m['V']), 'size': 2 * self.T}
                      for ch in range(self.C)],
                    {'op': 4, 'channels': mask, 'row': mac['row'] + g * H + t,
                     'column': 0, 'bursts': (q + 15) // 16}])
            expected.extend([
                {'op': 5, 'channels': mask, 'offset': result},
                {'op': 1, 'src': (2, result), 'dst': (1, y[1] + g * self.P * 2),
                 'size': 2 * min(self.P, n - g * self.P)}])
        need(commands == expected, 'GEMV address/order/GPR/tail schedule differs from specification')
        self.counts['pim_jobs'] += 1

    def pnm(self, c):
        inputs = [halves(self.read(a, elements * 2)) for a, elements in c['inputs']]
        address, elements = c['output']
        if c['kernel'] == 1:
            need(c['rows'] > 0 and c['width'] > 0 and c['inner'] > 0
                 and len(inputs) in (2, 3) and len(inputs[0]) == c['rows'] * c['inner']
                 and len(inputs[1]) == c['inner'] * c['width']
                 and elements == c['rows'] * c['width'], 'PNM matmul dimensions')
            if len(inputs) == 3:
                need(len(inputs[2]) == c['width'], 'PNM bias size')
            data = matmul(inputs[0], inputs[1], c['rows'], c['width'], c['inner'],
                          c['transpose'], c['factor'], inputs[2] if len(inputs) == 3 else None)
        elif c['kernel'] == 2:
            data = half_bytes([f32(a + inputs[1][i % len(inputs[1])]) for i, a in enumerate(inputs[0])])
        else:
            raise ValueError('PNM replay supports Matmul and Add only; no fallback')
        need(len(data) == elements * 2, 'PNM output size mismatch')
        self.write(address, data)
        self.counts['pnm_commands'] += 1

    def execute(self, commands):
        if any(c['op'] == 4 for c in commands):
            self.audit_schedule(commands)
        for index, c in enumerate(commands):
            self.current_command = (index, c)
            op = c['op']
            if op == 1 and c['dst'][0] == 4:
                channel, offset = divmod(c['dst'][1], self.m['V'])
                size = c['size']
                need(channel < self.C and c['src'][0] in (1, 2) and size > 0 and
                     c['src'][1] % 32 == offset % 32 == size % 32 == 0 and
                     offset + size <= self.m['V'], 'Invalid GB transfer')
                values = self.torch.tensor(halves(self.read(c['src'], size)))
                self.device.WR_GB(0, channel, self.C, offset // 2, size // 32, values, False)
            elif op in (1, 9):
                self.write(c['dst'], self.read(c['src'], c['size']))
            elif op == 10:
                self.write(c['dst'], bytes([c['value']]) * c['size'])
            elif op == 8:
                need(c['pack'] and c['tile'] == self.T and c['banks'] == self.P and c['row_bytes'] == self.R,
                     'Invalid relayout geometry or unsupported unpack')
                need((c['rows'], c['cols']) in [(self.m['N'], self.m['K']), (1, self.m['N'])],
                     'Relayout shape differs from fixture intent')
                need(c['dst'][0] == 1 and c['dst'][1] % self.S == 0, 'Packed base must be a stripe')
                dense = self.read(c['src'], c['rows'] * c['cols'] * 2)
                expected = packed_image(dense, c['rows'], c['cols'], self.P, self.T, self.R)
                self.pending_pack = (c['dst'], expected, c)
            elif op == 7:
                self.pnm(c)
            elif op == 2:
                need(c['banks'] == (1 << self.B) - 1, 'Partial-bank initialization not supported by replay')
                biases = halves(self.read((2, c['offset']), len(self.channels(c['channels'])) * self.B * 2))
                for i, channel in enumerate(self.channels(c['channels'])):
                    values = biases[i * self.B:(i + 1) * self.B]
                    self.device.WR_BIAS(0, channel, self.C, 0, self.torch.tensor(values), False)
                    for bank, value in enumerate(values):
                        self.mac_reference[channel, bank] = value
                        self.products[channel, bank] = [value]
            elif op == 4:
                need(c['bursts'] > 0 and (c['column'] + c['bursts']) * 32 <= min(self.R, self.m['V']),
                     'MAC burst exceeds row/GB')
                for channel in self.channels(c['channels']):
                    for bank in range(self.B):
                        for burst in range(c['column'], c['column'] + c['bursts']):
                            a = self.device.load_from_DRAM_single_bank(0, channel, bank, c['row'], burst * 16, 16, False).tolist()
                            b = self.channel(channel).GB[burst * 16:(burst + 1) * 16].tolist()
                            key = (channel, bank)
                            need(key in self.mac_reference, 'Uninitialized MAC')
                            self.mac_reference[key] = f32(self.mac_reference[key] + tree16(a, b))
                            self.products[key].extend(x * y for x, y in zip(a, b))
                    self.device.MAC_BK_GB(0, channel, self.C, c['row'], c['column'] * 16, 0,
                                          c['bursts'], False, 'breakdown_sa_weight')
            elif op == 5:
                values = []
                for channel in self.channels(c['channels']):
                    result = self.device.RD_MAC(0, channel, self.C, 0, False)
                    for bank, value in enumerate(result):
                        value = float(value)
                        key = (channel, bank)
                        error = abs(value - math.fsum(self.products[key]))
                        self.max_error = max(self.max_error, error)
                        need(error <= bound32(self.products[key]), 'FP32 accumulation outside independent bound')
                        if self.mode == 'profile':
                            need(value == self.mac_reference[key], 'CENT tree differs from scalar arithmetic oracle')
                        values.append(value)
                self.write((2, c['offset']), half_bytes(values))
            else:
                raise ValueError(f'Unsupported command {op}')


def replay(directory, source, mode='profile', loaded=None):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    need(manifest['version'] == 1 and mode in ('profile', 'native'), 'Unsupported fixture/mode')
    for key in ('C', 'B', 'R', 'V', 'rows', 'gpr', 'scratch', 'N', 'K', 'M'):
        need(type(manifest[key]) is int and 0 < manifest[key] <= 1000000, 'Invalid profile scalar')
    need(manifest['C'] <= 64 and manifest['B'] <= 64 and manifest['R'] % 32 == 0
         and manifest['V'] % 32 == 0 and manifest['operation'] in ('gemv', 'matmul', 'linear')
         and manifest['pattern'] in ('exact', 'random', 'ones', 'impulse'), 'Invalid profile')
    # Include inherited CENT Bank/Channel arrays, shadow bytes, Python fixture buffers and headroom.
    estimate = manifest['rows'] * (manifest['R'] // 2) * 4 * (1 + manifest['C'] * (manifest['B'] + 1))
    estimate += manifest['rows'] * manifest['C'] * manifest['B'] * manifest['R'] * 5
    need(estimate < 256 * 1024 * 1024, 'Estimated fixture/model storage exceeds 256 MiB budget')
    module, torch = loaded or load_cent(source)
    state = Replay(manifest, module, torch, mode, directory / f'{mode}-trace.txt')
    digest = hashlib.sha256((directory / 'manifest.json').read_bytes())
    try:
        for index, line in enumerate((directory / 'events.jsonl').read_text().splitlines()):
            e = json.loads(line)
            need(Path(e['file']).name == e['file'], 'Invalid fixture path')
            data = (directory / e['file']).read_bytes()
            digest.update(line.encode()); digest.update(data)
            try:
                state.current_command = None
                kind = e['kind']
                if kind == 'submit':
                    need(state.pending_pack is None, 'Missing packed snapshot')
                    state.execute(decode(data))
                else:
                    need(len(data) == e['bytes'], 'Fixture length mismatch')
                    address = (e['space'], e['offset'])
                    if kind == 'upload':
                        state.write(address, data)
                    elif kind == 'packed':
                        need(state.pending_pack is not None, 'Unexpected packed snapshot')
                        dest, expected, layout = state.pending_pack
                        need(address == dest and data == expected, 'Production packed bytes/padding differ from address oracle')
                        # Replay actual production bytes after independently verifying them.
                        state.write(dest, data)
                        state.layouts[dest] = layout
                        state.pending_pack = None
                        state.counts['packed'] += 1
                    elif kind == 'download':
                        actual = state.read(address, len(data))
                        if mode == 'profile' or manifest['pattern'] != 'random':
                            need(actual == data, 'Downloaded FP16 bits differ from production')
                        elif actual != data:
                            state.counts['native_different_downloads'] += 1
                        state.counts['downloads'] += 1
                    else:
                        raise ValueError('Unknown fixture event')
            except Exception as error:
                detail = {'event_index': index, 'event': e, 'command': state.current_command,
                          'mode': mode, 'case': manifest, 'error': str(error),
                          'gpr_prefix_hex': state.mem[2][:64].hex(),
                          'gb_channel0_prefix': state.channel(0).GB[:16].tolist(),
                          'expected_mac': {str(k): v for k, v in state.mac_reference.items()}}
                (directory / f'failure-{mode}.json').write_text(json.dumps(detail, indent=2) + '\n')
                raise ValueError(f'event {index} ({e["file"]}): {error}') from error
        need(state.pending_pack is None, 'Missing final packed snapshot')
        expected_pim = 6 if manifest['operation'] == 'gemv' else 0
        need(state.counts['pim_jobs'] == expected_pim, 'Missing PIM test jobs')
        return {**state.counts, 'mode': mode, 'case': manifest, 'fixture_sha256': digest.hexdigest(),
                'cent_commit': CENT_REVISION, 'cent_source_sha256': CENT_SHA256,
                'torch': torch.__version__, 'max_fp32_reference_error': state.max_error,
                'pnm_coverage': 'adapter scalar FMA/Add, not CENT RISC-V execution', 'status': 'pass'}
    finally:
        state.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('--cent-source', required=True, type=Path)
    parser.add_argument('--mode', choices=['native', 'profile'], default='profile')
    args = parser.parse_args()
    print(json.dumps(replay(args.fixture, args.cent_source, args.mode), indent=2))
