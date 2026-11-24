"""Write a deterministic v2 worker session and its expected binary replies."""
import argparse
import struct
import json
import re
from pathlib import Path

from decoder_fixture import decoder_requests
from decoder_state_fixture import decoder_state_requests


def lifecycle_requests(emit):
    """Reject invalid requests, then prove the same session can still compute."""
    def create(role, rows, cols, handle):
        emit(struct.pack('<ccII', b'C', role, rows, cols), struct.pack('<I', handle))

    def upload(handle, values):
        emit(struct.pack(f'<cI{len(values)}e', b'U', handle, *values))

    def compute(weight=1):
        emit(struct.pack('<cIII', b'G', 3, weight, 2))
        emit(struct.pack('<cI', b'D', 3), struct.pack('<2e', 6, 1))

    def reject(request):
        emit(request, status=1)

    reject(struct.pack('<ccII', b'C', b'W', 0, 3))
    reject(struct.pack('<cI', b'D', 999))
    create(b'W', 2, 3, 1)
    create(b'X', 1, 3, 2)
    create(b'Y', 1, 2, 3)
    reject(struct.pack('<cI', b'D', 3))
    reject(struct.pack('<cIII', b'G', 3, 1, 2))
    upload(1, [1, 2, 3, -1, 0, 2])
    upload(2, [1, 1, 1])
    compute()
    reject(struct.pack('<cIII', b'G', 2, 1, 2))  # Output/input alias.
    reject(struct.pack('<cI', b'U', 1))  # Truncated payload, complete frame.
    compute()  # Rejected upload must not modify resident weights.
    create(b'X', 1, 2, 4)
    upload(4, [1, 1])
    reject(struct.pack('<cIII', b'G', 3, 1, 4))  # Wrong input dimension.
    reject(struct.pack('<cI', b'D', 3))  # Failed execution invalidates output.
    compute()
    emit(struct.pack('<cI', b'R', 1))
    reject(struct.pack('<cI', b'D', 1))
    reject(struct.pack('<cI', b'R', 1))
    create(b'W', 2, 3, 5)
    upload(5, [1, 2, 3, -1, 0, 2])
    reject(struct.pack('<cIII', b'G', 3, 1, 2))  # Old ID stays invalid.
    compute(5)
    for handle in (5, 4, 3, 2):
        emit(struct.pack('<cI', b'R', handle))


def chain_requests(emit):
    """Chain GEMVs across banked, logical FP16 and mutable representations."""
    def half(value):
        return struct.unpack('<e', struct.pack('<e', value))[0]

    for index, hidden in enumerate((3, 257, 1025)):
        base = index * 9
        flat, w1, x, y, w2, z, copied, storage, direct = range(base+1, base+10)
        descriptors = [(flat, b'X', 1, hidden*3), (w1, b'W', hidden, 3),
                       (x, b'X', 1, 3), (y, b'Y', 1, hidden),
                       (w2, b'W', 2, hidden), (z, b'Y', 1, 2),
                       (copied, b'X', 1, hidden), (storage, b'B', 1, hidden),
                       (direct, b'Y', 1, 2)]
        for handle, role, rows, cols in descriptors:
            emit(struct.pack('<ccII', b'C', role, rows, cols), struct.pack('<I', handle))
        weights1 = [v for r in range(hidden)
                    for v in (1 + (r % 3)*2**-10, 2**-11, (r % 2)*2**-11)]
        weights2 = [((r*3+c) % 5-2)/8 for r in range(2) for c in range(hidden)]
        emit(struct.pack(f'<cI{len(weights1)}e', b'U', flat, *weights1))
        emit(struct.pack('<cII', b'T', w1, flat))  # Dense -> banked matrix.
        emit(struct.pack('<cII', b'F', flat, 0))
        emit(struct.pack('<cII', b'T', flat, w1))  # Banked -> dense.
        emit(struct.pack('<cII', b'T', w1, flat))  # Restore banked representation.
        emit(struct.pack('<cI', b'R', flat))  # Converted copy owns its data.
        emit(struct.pack(f'<cI{len(weights2)}e', b'U', w2, *weights2))
        for vector in ([1, 1, 1], [2, 0, -1]):
            intermediate = [half(sum(weights1[r*3+c]*vector[c] for c in range(3)))
                            for r in range(hidden)]
            result = [half(sum(weights2[r*hidden+c]*intermediate[c] for c in range(hidden)))
                      for r in range(2)]
            emit(struct.pack('<cI3e', b'U', x, *vector))
            emit(struct.pack('<cIII', b'G', y, w1, x))
            # Direct resident chaining observes the logical FP16 output boundary.
            emit(struct.pack('<cIII', b'G', direct, w2, y))
            emit(struct.pack('<cII', b'T', copied, y))
            emit(struct.pack('<cIIIII', b'B', storage, copied, 0, 0, hidden))
            emit(struct.pack('<cII', b'F', copied, 0))
            emit(struct.pack('<cIIIII', b'B', copied, storage, 0, 0, hidden))
            emit(struct.pack('<cIII', b'G', z, w2, copied))
            # Only final outputs cross the external-client boundary.
            for handle in (direct, z):
                emit(struct.pack('<cI', b'D', handle), struct.pack('<2e', *result))
        for handle in (direct, storage, copied, z, w2, y, x, w1):
            emit(struct.pack('<cI', b'R', handle))


def write_fixture(directory, suite="smoke", layout=None):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "requests.bin").open("wb") as requests, (directory / "expected.bin").open("wb") as replies:
        def emit(request, result=b"", status=0):
            requests.write(struct.pack("<I", len(request)) + request)
            replies.write(struct.pack("<IB", len(result) + 1, status) + result)

        emit(b"H\x02", b"LPDDR6/FP16/FP32")
        cases = [(2, 3)]
        if suite == 'boundary':
            constants = {name: int(value, 0) for name, value in re.findall(
                r'^#define PIM_LAYOUT_(\w+) (0x[0-9a-fA-F]+|[0-9]+)\s*$',
                layout.read_text(), re.MULTILINE)}
            if (constants['STD'], constants['NUM_FMT'], constants['ACC_FMT']) != (2, 1, 2):
                raise ValueError('Boundary suite requires LPDDR6 FP16/FP32')
            lanes = constants['CHANNELS'] * constants['UNITS'] * constants['BANKS_PER_UNIT']
            tile = constants['ROW_BYTES'] // constants['ELEM_BYTES']
            cases += [(lanes-1, 17), (lanes, 17), (lanes+1, 17),
                      (3, tile-1), (3, tile), (3, tile+1), (lanes+1, tile+1)]
        if suite == 'lifecycle':
            lifecycle_requests(emit)
        if suite == 'chain':
            chain_requests(emit)
            cases = [(3, 3), (257, 3), (1025, 3)]
        if suite == 'decoder':
            decoder_requests(emit)
            cases = []
        if suite == 'decoder_state':
            state_checks = decoder_state_requests(emit)
            cases = []
        for index, (rows, cols) in enumerate([] if suite in ('lifecycle', 'chain', 'decoder', 'decoder_state') else cases):
            w, x, y = (3*index + 1, 3*index + 2, 3*index + 3)
            for handle, role, m, n in [(w, b'W', rows, cols), (x, b'X', 1, cols), (y, b'Y', 1, rows)]:
                emit(struct.pack('<ccII', b'C', role, m, n), struct.pack('<I', handle))
            weights = ([1, 2, 3, -1, 0, 2] if suite == 'smoke' else
                       [((r*3 + c*5) % 17 - 8)/16 for r in range(rows) for c in range(cols)])
            emit(struct.pack(f'<cI{len(weights)}e', b'U', w, *weights))
            for repeat in range(2 if suite == 'smoke' else 3):
                vector = ([(1, 1, 1), (2, 0, -1)][repeat] if suite == 'smoke' else
                          [((c*7 + repeat) % 9 - 4)/8 for c in range(cols)])
                if repeat == 2:
                    # Overwrite an already materialized matrix to detect stale weights.
                    weights = [-value for value in weights]
                    emit(struct.pack(f'<cI{len(weights)}e', b'U', w, *weights))
                # Independent dense dot product; binary fractions sum exactly here.
                result = [sum(weights[r*cols+c] * vector[c] for c in range(cols))
                          for r in range(rows)]
                emit(struct.pack(f'<cI{cols}e', b'U', x, *vector))
                emit(struct.pack('<cIII', b'G', y, w, x))
                emit(struct.pack('<cI', b'D', y), struct.pack(f'<{rows}e', *result))
            for handle in (y, x, w):
                emit(struct.pack('<cI', b'R', handle))
        emit(b"Q")

    marker = ('NDP_PIM_SMOKE_PASS launches=2 outputs=[6,1],[-1,-4]' if suite == 'smoke'
              else f'NDP_PIM_BOUNDARY_PASS cases={len(cases)} launches={3*len(cases)}')
    if suite == 'lifecycle':
        marker = 'NDP_PIM_LIFECYCLE_PASS errors=11 launches=4'
    if suite == 'chain':
        marker = 'NDP_PIM_CHAIN_PASS cases=3 launches=18 intermediate_downloads=0'
    if suite == 'decoder':
        marker = 'NDP_PIM_DECODER_OPS_PASS checks=11 pim_gemv=1'
    if suite == 'decoder_state':
        marker = f'NDP_PIM_DECODER_STATE_PASS schedules=3 tokens=7 checks={state_checks}'
    (directory / 'cases.json').write_text(json.dumps(
        {'suite': suite, 'shapes': cases, 'marker': marker}, indent=2) + '\n')
    # Embed the same byte streams for SE/FS runs without guest file provisioning.
    with (directory / 'fixture.hpp').open('w') as header:
        header.write('// Generated deterministic worker protocol fixture.\n#pragma once\n')
        header.write('static const char *fixture_marker = ' + json.dumps(marker) + ';\n')
        for name, filename in [('requests', 'requests.bin'), ('expected', 'expected.bin')]:
            data = (directory / filename).read_bytes()
            header.write('static const unsigned char ' + name + '[] = {\n')
            for start in range(0, len(data), 16):
                header.write('  ' + ', '.join(str(x) for x in data[start:start + 16]) + ',\n')
            header.write('};\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument('--suite', choices=['smoke', 'boundary', 'lifecycle', 'chain', 'decoder', 'decoder_state'], default='smoke')
    parser.add_argument('--layout', type=Path)
    args = parser.parse_args()
    if args.suite == 'boundary' and args.layout is None:
        parser.error('--layout is required for boundary cases')
    write_fixture(args.directory, args.suite, args.layout)
