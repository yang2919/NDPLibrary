"""Run Qwen rank ROIs with Docker, then replay fixed-context decoder requests."""
import argparse
import csv
from collections import deque
import hashlib
import heapq
import json
import math
import os
import runpy
from pathlib import Path
import statistics
import subprocess

from qwen_stats import summarize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = ROOT / 'configs/qwen3-serving-16.json'


def cases(config):
    for model, spec in config['models'].items():
        for length, batch in config['workload']['sequence_length_including_current_and_resident_batch']:
            for pp, tp in config['pp_tp_candidates']:
                if spec['kv_heads'] % tp == 0:
                    assert pp * tp == config['device_count'] and spec['layers'] % pp == 0
                    yield model, length, batch, pp, tp


def run_roi(model, length, tp):
    env = dict(os.environ, STAGE='QWEN', MODEL=model, CONTEXT=str(length), TP=str(tp), CPU='timing')
    with subprocess.Popen([HERE / 'run.sh'], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          text=True, bufsize=1) as process:
        directory = None
        for line in process.stdout:
            print(line, end='', flush=True)
            if line.startswith('Results: '):
                directory = Path(line.removeprefix('Results: ').strip())
        if process.wait() or directory is None:
            raise RuntimeError(f'gem5 run failed: {model=} {length=} {tp=} {directory=}')
    return directory


def replay(config, case, roi, host_token_us, host_reduce_us, cent_vector=None):
    model, length, batch, pp, tp = case
    spec = config['models'][model]
    hidden = spec['hidden']
    fabric = config['fabric']
    lanes = fabric['endpoint_lanes']
    # CENT cxl_latency.vector_latency: 3 x 64-byte messages per 256-byte flit,
    # MULTICAST=True lane rate; each actual host-staged message is unicast.
    endpoint_bytes_us = lanes * 4 * 2**30 / 1e6
    uplink_bytes_us = fabric['shared_host_uplink_GBps_per_direction'] * 1000

    phases = roi['phases']
    def phase_us(name):
        return phases[name]['simTicks'] * 1e6 / phases[name]['simFreq']

    # Each tuple is a resource, duration in us and payload bytes. A stage
    # retains its token across barriers. TP groups have disjoint rank sets.
    layer = []
    def messages(target, direction, size, count):
        wire = 256 * math.ceil(size / 192)
        for _ in range(count):
            endpoint_us = (cent_vector(size // 2, lanes) * 1000 if cent_vector
                           else .180 + wire / endpoint_bytes_us)
            target.append((direction, endpoint_us, size))

    for phase in ('attention', 'ffn'):
        if tp > 1:
            messages(layer, 'write', hidden * 2, tp)
        if tp > 1 and phase == 'ffn' and model == 'moe':
            # Eight int32 expert IDs and eight float32 routing weights.
            messages(layer, 'write', spec['top_k'] * 8, tp - 1)
        layer.append(('compute', phase_us(phase), 0))
        if tp > 1:
            # Rank partials of the final projection. For fixed top-8 MoE,
            # this models a local weighted sum before the host reduction.
            messages(layer, 'read', hidden * 4, tp)
            layer.append(('host', host_reduce_us, 0))
    stage_ops = layer * (spec['layers'] // pp)
    boundary = []
    if tp == 1:
        messages(boundary, 'read', hidden * 2, 1)
        messages(boundary, 'write', hidden * 2, 1)
    # For TP>1, the final reduction and the next broadcast cover this boundary.
    ops = []
    for stage in range(pp):
        stage_work = []
        if stage == 0 and tp == 1:
            messages(stage_work, 'write', hidden * 2, 1)
        stage_work.extend(stage_ops)
        if stage + 1 < pp:
            stage_work.extend(boundary)
        else:
            if tp == 1:
                messages(stage_work, 'read', hidden * 2, 1)
            stage_work.append(('host', host_token_us, 0))
        ops.append(stage_work)
    queues = [deque() for _ in range(pp)]
    busy = [False] * pp
    stage_start = [0.] * pp
    occupied = [0.] * pp
    events = []
    available = dict(read=0., write=0., host=0.)
    warm = config['workload']['warmup_rounds']
    rounds = config['workload']['measurement_rounds']
    last = [None] * batch
    samples = []
    completions = 0
    window_start = None

    def admit(t, stage, req):
        if busy[stage]:
            queues[stage].append(req)
        else:
            busy[stage] = True
            stage_start[stage] = t
            heapq.heappush(events, (t, stage, req, 0))

    for req in range(batch):
        admit(0., 0, req)
    while events:
        t, stage, req, i = heapq.heappop(events)
        if i < len(ops[stage]):
            resource, duration, payload = ops[stage][i]
            if resource in ('read', 'write'):
                # Cut-through approximation: the endpoint and switch serialize
                # the same flits concurrently. The shared egress queues FIFO.
                wire = 256 * math.ceil(payload / 192)
                end = max(t, available[resource]) + wire / uplink_bytes_us
                available[resource] = end
                duration = max(duration, end - t + .180)
            elif resource in available:
                t = max(t, available[resource])
                available[resource] = t + duration
            heapq.heappush(events, (t + duration, stage, req, i + 1))
            continue
        if window_start is not None:
            occupied[stage] += t - max(window_start, stage_start[stage])
        busy[stage] = False
        if queues[stage]:
            admit(t, stage, queues[stage].popleft())
        if stage + 1 < pp:
            admit(t, stage + 1, req)
            continue
        completions += 1
        if completions > warm * batch:
            samples.append(t - last[req])
        if completions == warm * batch:
            window_start = t
        last[req] = t
        if completions == (warm + rounds) * batch:
            break
        admit(t, 0, req)
    for stage in range(pp):
        if busy[stage]:
            occupied[stage] += t - max(window_start, stage_start[stage])
    samples.sort()
    return {
        'estimated_tpot_mean_ms': statistics.mean(samples) / 1000,
        'estimated_tpot_p50_ms': statistics.median(samples) / 1000,
        'estimated_tpot_p95_ms': samples[math.ceil(.95 * len(samples)) - 1] / 1000,
        'estimated_tokens_per_second': len(samples) * 1e6 / (t - window_start),
        'intervals': len(samples),
        'estimated_stage_occupancy_fraction': json.dumps([round(x / (t - window_start), 6) for x in occupied]),
        'tp_reductions_per_token': (2 * spec['layers'] if tp > 1 else 0),
        'logical_kv_bytes_per_rank': batch * length * (spec['layers'] // pp) * (spec['kv_heads'] // tp) * spec['head_dim'] * 4,
        'cxl_payload_bytes_per_token': sum(b for stage in ops for _, _, b in stage),
        'rank_attention_ticks': phases['attention']['simTicks'],
        'rank_ffn_ticks': phases['ffn']['simTicks'],
        'rank_cpu_cycles': sum(p['cpuCycles'] for p in phases.values()),
        'rank_abmac_requests': sum(p['pim_requests']['AbMac'] for p in phases.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'run', 'report'])
    parser.add_argument('--output', type=Path, default=ROOT / 'build/extended-gem5/qwen-serving')
    parser.add_argument('--cent-root', type=Path, help='use pinned CENT vector_latency directly')
    parser.add_argument('--model', choices=['dense', 'moe'])
    parser.add_argument('--context', type=int)
    parser.add_argument('--tp', type=int)
    parser.add_argument('--host-token-us', type=float, default=0, help='host embedding and output time per token')
    parser.add_argument('--host-reduce-us', type=float, default=0, help='host compute time per TP reduction')
    args = parser.parse_args()
    if any(not math.isfinite(v) or v < 0 for v in (args.host_token_us, args.host_reduce_us)):
        parser.error('host costs must be finite and nonnegative')
    config = json.loads(CONFIG.read_text())
    selected = [c for c in cases(config) if all(v is None or v == c[i] for v, i in
                [(args.model, 0), (args.context, 1), (args.tp, 4)])]
    if not selected:
        parser.error('no valid profile matches the filters')
    if args.action == 'list':
        print('model context batch pp tp')
        for case in selected:
            print(*case)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for path in [CONFIG, HERE / 'qwen_roi.cpp', HERE / 'run-container.sh', HERE / 'qwen_stats.py']:
        digest.update(path.read_bytes())
    signature = digest.hexdigest()
    cent_vector = None
    if args.cent_root:
        source = args.cent_root / 'cent_simulation/cxl_latency.py'
        expected = '800388e1dc49c06a2668fd9e615ef64da57e86f771c28f9b9146691d27e6deb2'
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            parser.error('CENT source differs from the pinned revision')
        cent_vector = runpy.run_path(str(source))['vector_latency']
    rows = []
    for case in selected:
        model, length, batch, pp, tp = case
        ref = args.output / f'{model}-s{length}-tp{tp}.json'
        if ref.exists():
            record = json.loads(ref.read_text())
            if record['signature'] != signature:
                raise ValueError(f'ROI configuration changed; use a fresh --output: {ref}')
            directory = Path(record['run'])
        elif args.action == 'run':
            directory = run_roi(model, length, tp)
        else:
            raise FileNotFoundError(f'missing completed ROI: {ref}')
        roi = summarize(directory, model, length, tp)
        if not ref.exists():
            tmp = ref.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(signature=signature, run=str(directory))) + '\n')
            tmp.replace(ref)
        rows.append(dict(model=model, context=length, batch=batch, pp=pp, tp=tp, run=str(directory),
                         host_token_us=args.host_token_us, host_reduce_us=args.host_reduce_us,
                         **replay(config, case, roi, args.host_token_us, args.host_reduce_us, cent_vector)))
    # Filters get separate tables so a smoke report cannot overwrite the sweep.
    tag = '-'.join(str(v) for v in (args.model, args.context, args.tp) if v is not None)
    output = args.output / ('serving' + ('-' + tag if tag else '') + '.csv')
    with output.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f'NDP_QWEN_SERVING_REPORT cases={len(rows)} {output}')


if __name__ == '__main__':
    main()
