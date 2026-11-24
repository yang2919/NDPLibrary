"""Run bounded, reproducible production-export/CENT-replay dimension sweeps."""
import argparse
import gc
import hashlib
import json
import subprocess
import signal
import sys
import time
from pathlib import Path

from replay import load_cent, replay
from check_outputs import check_outputs

PROFILES = {
    'tiny': (1, 1, 64, 64),
    'asymmetric': (2, 3, 96, 64),
    'gb-limited': (2, 2, 128, 64),
    'row-limited': (2, 2, 64, 128),
    'wide': (2, 16, 2048, 2048),
}


def cases(suite, selected):
    if suite == 'large':
        for n, k in [(257, 4097), (129, 11009)]:
            yield 'wide', n, k, 42, 'random', 'gemv', 1
        return
    for name in selected:
        c, b, r, v = PROFILES[name]
        p, t = c * b, min(r, v) // 2
        if suite == 'smoke':
            dimensions = [(1, 1), (p + 1, t + 3)]
        else:
            ns = sorted({n for n in [1, p-1, p, p+1, 2*p-1, 2*p, 2*p+1, 3*p+5] if n > 0})
            ks = sorted({1, 15, 16, 17, 31, 32, 33, t-1, t, t+1, 2*t-1, 2*t, 2*t+1, 3*t+7})
            dimensions = [(n, k) for n in ns for k in ks]
        for n, k in dimensions:
            yield name, n, k, 0, 'exact', 'gemv', 1
        for seed in (0, 1, 7, 42):
            yield name, p+1, t+7, seed, 'random', 'gemv', 1
        for pattern in ('ones', 'impulse'):
            yield name, p+1, 17, 16, pattern, 'gemv', 1
    # Independent PNM functional/FMA coverage is labelled separately from CENT AiM execution.
    dims = [(3, 17, 5), (1, 1, 1)] if suite == 'smoke' else [
        (m, k, n) for m in (1, 2, 3, 7, 17) for k in (1, 15, 16, 17, 31, 33, 65)
        for n in (1, 3, 15, 16, 17, 33)]
    for m, k, n in dims:
        for op in ('matmul', 'linear'):
            yield 'asymmetric', n, k, 7, 'random', op, m
    if suite == 'extended':
        yield from cases('large', ['wide'])


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output / 'report.jsonl'
    if report.exists():
        raise ValueError('Report already exists; use a fresh output directory')
    loaded = load_cent(args.cent_source)
    start = time.monotonic()
    count = 0
    with report.open('w') as out:
        for index, case in enumerate(cases(args.suite, args.profiles.split(','))):
            if args.limit and index >= args.limit:
                break
            name, n, k, seed, pattern, op, m = case
            fixture = args.output / f'{index:04d}-{name}-{op}-{n}x{k}x{m}-{pattern}-{seed}'
            cmd = [str(args.exporter.resolve()), str(fixture), *map(str, PROFILES[name]),
                   str(n), str(k), str(seed), pattern, op, str(m)]
            def expired(signum, frame):
                raise TimeoutError('Case exceeded 60-second wall-clock limit')
            previous = signal.signal(signal.SIGALRM, expired)
            signal.setitimer(signal.ITIMER_REAL, 60)
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
                logical_downloads = check_outputs(fixture)
                for mode in ('native', 'profile'):
                    result = replay(fixture, args.cent_source, mode, loaded)
                    result.update(fixture=str(fixture), profile_name=name, logical_downloads=logical_downloads)
                    out.write(json.dumps(result) + '\n'); out.flush()
            except Exception as error:
                out.write(json.dumps({'status': 'fail', 'fixture': str(fixture), 'error': str(error)}) + '\n')
                raise
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
            count += 1
            gc.collect()
            if count % 10 == 0:
                print(f'{count} cases passed in both numeric modes', flush=True)
    summary = {'status': 'pass', 'cases': count, 'replays': 2*count, 'suite': args.suite,
               'profiles': args.profiles, 'limit': args.limit,
               'elapsed_seconds': round(time.monotonic()-start, 2),
               'python': sys.version, 'exporter_sha256': hashlib.sha256(args.exporter.read_bytes()).hexdigest()}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exporter', type=Path, default=Path('build/ndp_cent_export'))
    parser.add_argument('--cent-source', required=True, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', choices=['smoke', 'boundary', 'large', 'extended'], default='smoke')
    parser.add_argument('--profiles', default=','.join(PROFILES))
    parser.add_argument('--limit', type=int, default=0, help='Explicit partial sweep; 0 runs all cases')
    args = parser.parse_args()
    if any(x not in PROFILES for x in args.profiles.split(',')) or args.limit < 0:
        parser.error('Invalid profiles/limit')
    run(args)
