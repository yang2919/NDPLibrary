"""Check recorded downloads from logical host inputs without consulting command operands."""
import argparse
import json
from pathlib import Path

import numpy as np

from reference import matmul


def gemv(weights, vector):
    # Match the specified 16-lane reduction and round each burst addition to FP32.
    n, k = weights.shape
    products = weights.astype(np.float32) * vector.astype(np.float32)
    groups = np.pad(products, ((0, 0), (0, (-k) % 16))).reshape(n, -1, 16)
    while groups.shape[-1] > 1:
        groups = groups[:, :, ::2] + groups[:, :, 1::2]
    acc = np.zeros(n, dtype=np.float32)
    for burst in groups[:, :, 0].T:
        acc += burst
    return acc.astype('<f2')


def check_outputs(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'manifest.json').read_text())
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    inputs, downloads = [], []
    for i, event in enumerate(events):
        if event['kind'] not in ('upload', 'download'):
            continue
        if event['kind'] == 'upload' and i + 2 < len(events) and events[i+2]['kind'] == 'packed':
            # Recorder inserts a destination poison write just before each pack job.
            continue
        values = np.frombuffer((directory / event['file']).read_bytes(), dtype='<f2')
        (inputs if event['kind'] == 'upload' else downloads).append(values)
    n, k, m = (meta[x] for x in ('N', 'K', 'M'))
    op = meta['operation']
    gemv_case = op == 'gemv'
    required_inputs = 8 if gemv_case else 5 + (op == 'linear')
    if len(inputs) != required_inputs:
        raise ValueError('Logical input count differs from fixture schedule')
    guard, weights = inputs[:2]
    expected = [guard]
    for request in range(3):
        if gemv_case:
            x, projection = inputs[2+request*2:4+request*2]
            y = gemv(weights.reshape(n, k), x.reshape(k))
            doubled = (y.astype(np.float32) * 2).astype('<f2')
            expected.extend([y, doubled, gemv(projection.reshape(1, n), doubled)])
        else:
            linear = op == 'linear'
            x = inputs[2 + int(linear) + request]
            bias = inputs[2].tolist() if linear else None
            bits = matmul(x.tolist(), weights.tolist(), m, n, k, linear, 1.0, bias)
            expected.append(np.frombuffer(bits, dtype='<f2'))
    expected.append(guard)
    if len(downloads) != len(expected):
        raise ValueError('Missing or unexpected logical download')
    for index, (actual, reference) in enumerate(zip(downloads, expected)):
        if actual.tobytes() != reference.tobytes():
            raise ValueError(f'Logical output differs at download {index} in {directory}')
    return len(expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='A fixture directory or completed sweep')
    args = parser.parse_args()
    fixtures = ([args.directory] if (args.directory / 'manifest.json').exists()
                else sorted(file.parent for file in args.directory.glob('*/manifest.json')))
    if not fixtures:
        parser.error('No fixtures found')
    print(json.dumps({'fixtures': len(fixtures), 'logical_downloads': sum(map(check_outputs, fixtures))}))
