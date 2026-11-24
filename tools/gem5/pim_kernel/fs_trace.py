"""Validate timing-controller completions for the selected FS client scenario."""
import sys
from pathlib import Path
from check_smoke import verify_trace

directory, case = Path(sys.argv[1]), sys.argv[2]
if case == 'partial':
    with (directory / 'pim.trace').open() as trace:
        if any('done AbMac (seq ' in line for line in trace):
            raise ValueError('truncated request reached the MAC controller')
else:
    print('Completed controller requests:', verify_trace(directory, 'timing'))
print(f'NDP_FS_EXTERNAL_PASS case={case}')
