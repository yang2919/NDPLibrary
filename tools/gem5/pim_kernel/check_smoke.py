"""Require guest numeric success and completed PIM controller operations."""
import argparse
import json
from pathlib import Path


MARKER = 'NDP_PIM_SMOKE_PASS launches=2 outputs=[6,1],[-1,-4]'


def verify(directory, mode, cpu="timing", marker=MARKER):
    output = directory / ('program.out' if mode == 'SE' else 'board.platform.terminal')
    lines = output.read_text(errors='replace').replace('\r', '').splitlines()
    if marker not in lines:
        raise ValueError(f'Missing exact numerical pass marker in {output}')
    if mode == 'FS' and 'PIM-FS exit 0' not in lines:
        raise ValueError(f'Missing successful guest script exit in {output}')
    return verify_trace(directory, cpu)


def verify_trace(directory, cpu):
    counts = {op: 0 for op in ('WrGb', 'AbMac', 'RdReg')}
    with (directory / 'pim.trace').open(errors='replace') as trace:
        for line in trace:
            for op in counts:
                if (f'done {op} (seq ' in line if cpu == 'timing' else f'atomic {op} at ' in line):
                    counts[op] += 1
    if any(value == 0 for value in counts.values()):
        raise ValueError(f'Missing completed controller operations: {counts}')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--mode', choices=['SE', 'FS'], required=True)
    parser.add_argument('--cpu', choices=['timing', 'atomic'], default='timing')
    parser.add_argument('--cases', type=Path)
    args = parser.parse_args()
    marker = json.loads(args.cases.read_text())['marker'] if args.cases else MARKER
    counts = verify(args.directory, args.mode, args.cpu, marker)
    print(f'NDP_PIM_{args.mode}_PASS cpu={args.cpu}: exact replies and PIM trace')
    print(marker)
    print('Completed controller requests:' if args.cpu == 'timing' else 'Atomic PIM accesses:', counts)


if __name__ == '__main__':
    main()
