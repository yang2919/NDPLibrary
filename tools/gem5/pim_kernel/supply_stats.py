"""Check the three measured GEMVs and report supply-path timings from SE output."""
import argparse
from collections import defaultdict, deque
import csv
import json
from pathlib import Path
import re
import statistics


def channel_columns(sim, tick):
    path = sim / f'board.memory.pim_memory.ramulator2.ramulator_stats.{tick}.yaml'
    counts = {}
    for section in re.split(r'(?m)^\s*id: Channel ', path.read_text())[1:]:
        channel = int(section.splitlines()[0])
        counts[channel] = int(re.search(r'pim_col_cmds_AbMac: (\d+)', section)[1])
    if set(counts) != set(range(16)):
        raise ValueError(f'incomplete channel snapshot at {tick}')
    return counts


def trace_report(run, windows):
    # Join ingress to the first core attempt in FIFO order for each address.
    # The TimingSimpleCPU is single-threaded; retries keep the original attempt.
    attempt = defaultdict(deque)
    jobs = {}
    for line in (run / 'sim/pim.trace').open():
        if 'AbMac' not in line:
            continue
        tick = int(line.split(':', 1)[0])
        if not any(lo <= tick <= hi for lo, hi, _ in windows):
            continue
        if m := re.search(r'cpu attempt AbMac at (0x[0-9a-fA-F]+)', line):
            attempt[int(m[1], 16)].append([tick, 0])
        elif m := re.search(r'cpu retry AbMac at (0x[0-9a-fA-F]+)', line):
            attempt[int(m[1], 16)][0][1] += 1
        elif m := re.search(r'accept AbMac at (\S+) size \d+ \(seq (\d+)\)', line):
            start, retries = attempt[int(m[1], 16)].popleft()
            jobs[int(m[2])] = dict(attempt=start, retries=retries, accept=tick)
        elif m := re.search(r'dispatch AbMac \(seq (\d+)\) to channel (\d+), (\d+) columns', line):
            jobs[int(m[1])].update(dispatch=tick, channel=int(m[2]), columns=int(m[3]))
        elif m := re.search(r'last command AbMac \(seq (\d+)\)', line):
            jobs[int(m[1])]['last'] = tick
        elif m := re.search(r'done AbMac \(seq (\d+)\)', line):
            jobs[int(m[1])]['done'] = tick
    if any(attempt.values()):
        raise ValueError('unmatched core MAC attempts')
    summary = []
    with (run / 'mac-events.csv').open('w', newline='') as out:
        fields = 'trial seq channel columns retries attempt accept dispatch last done'.split()
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for trial, (lo, hi, expected) in enumerate(windows, 1):
            selected = [v for v in jobs.values() if lo <= v['accept'] <= hi]
            if len(selected) != expected:
                raise ValueError(f'trial {trial}: MAC trace count differs from stats')
            intervals = defaultdict(list)
            for seq, v in jobs.items():
                if not lo <= v['accept'] <= hi:
                    continue
                for a, b in [('attempt', 'accept'), ('accept', 'dispatch'),
                             ('dispatch', 'last'), ('last', 'done')]:
                    delta = v[b] - v[a]
                    if delta < 0:
                        raise ValueError(f'out of order: seq {seq}')
                    intervals[f'{a}_to_{b}'].append(delta)
                writer.writerow(dict(trial=trial, seq=seq, **v))
            attempts = sorted(v['attempt'] for v in selected)
            spans = sorted((v['accept'], v['done']) for v in selected)
            covered = 0
            previous_end = lo
            for begin, end in spans:
                covered += max(0, end - max(begin, previous_end))
                previous_end = max(previous_end, end)
            means = {k: statistics.mean(values) for k, values in intervals.items()}
            summary.append(dict(mean_path_ticks=means,
                                mean_core_attempt_gap_ticks=(attempts[-1]-attempts[0])/(len(attempts)-1),
                                no_accepted_mac_fraction=1-covered/(hi-lo),
                                core_mac_retries=sum(v['retries'] for v in selected)))
    return summary


def summarize(run, shape):
    rows, cols = (256, 2048) if shape == 'small' else (25600, 5120)
    sim = run / 'sim'
    if f'NDP_SUPPLY_OUTPUT_PASS {shape} trials=3' not in (sim / 'program.out').read_text().splitlines():
        raise ValueError('missing guest output check')
    if 'exiting with last active thread context (code 0,' not in (run / 'simulation.log').read_text():
        raise ValueError('missing successful SE exit')
    blocks = re.findall(r'Begin Simulation Statistics -+\n(.*?)\n-+ End Simulation Statistics',
                        (sim / 'stats.txt').read_text(), re.S)
    if len(blocks) != 10:
        raise ValueError(f'expected nine ROI dumps and teardown, found {len(blocks)}')
    trials, windows = [], []
    for i in range(0, 9, 3):
        phases = {}
        for phase, block in zip(('input_copy', 'gemv', 'output_copy'), blocks[i:i+3]):
            s = dict(re.findall(r'^([^\s#]+)\s+([\d.eE+-]+)\s', block, re.M))
            ticks, end, freq = (int(s[k]) for k in ('simTicks', 'finalTick', 'simFreq'))
            if ticks <= 0 or freq <= 0:
                raise ValueError('empty ROI')
            phases[phase] = dict(ticks=ticks, final_tick=end, sim_freq=freq, ms=1000*ticks/freq,
                                 cpu_instructions=int(s['simInsts']))
            if phase == 'gemv':
                p = phases[phase]
                start = phases['input_copy']['final_tick']
                if end-start != ticks:
                    raise ValueError('unexpected GEMV start tick')
                before, after = channel_columns(sim, start), channel_columns(sim, end)
                p['mac_columns_by_channel'] = [after[ch]-before[ch] for ch in range(16)]
                if p['mac_columns_by_channel'] != [rows*cols//4096]*16:
                    raise ValueError('unexpected per-channel work')
                prefix = 'board.memory.pim_memory.ramulator2.pim.'
                p['requests'] = {k.split('::')[-1]: int(v) for k, v in s.items()
                                 if k.startswith(prefix+'reqs::')}
                n = p['requests']['AbMac']
                if n <= 0:
                    raise ValueError('missing MAC requests')
                windows.append((start, end, n))
                p['mean_outstanding_mac'] = int(s[prefix+'totLat::AbMac']) / ticks
                p['controller_retries'] = int(s.get(prefix+'numRetry', 0))
                p['useful_gmac_per_second'] = rows*cols*freq/ticks/1e9
        trials.append(phases)
    report = dict(shape=[rows, cols], warmup=1, trials=trials,
                  median_gemv_ms=statistics.median(p['gemv']['ms'] for p in trials))
    if shape == 'small':
        report['mac_trace_by_trial'] = trace_report(run, windows)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('shape', choices=('small', 'gate'))
    args = parser.parse_args()
    result = summarize(args.run, args.shape)
    (args.run / 'supply.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"NDP_SUPPLY_A_PASS case={args.shape} trials=3 median_gemv_ms={result['median_gemv_ms']:.6f}")
