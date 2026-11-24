"""Record the exact libpim profile, sources and worker artifacts for a run."""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pim-root', type=Path, required=True)
    parser.add_argument('--layout', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--compiler', required=True)
    parser.add_argument('--kind', choices=['native-reference', 'riscv'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.pim_root.resolve()
    layout = args.layout.resolve()
    constants = dict((name, int(value, 0)) for name, value in re.findall(
        r'^#define PIM_LAYOUT_(\w+) (0x[0-9a-fA-F]+|[0-9]+)\s*$',
        layout.read_text(), re.MULTILINE))
    required = {'STD': 2, 'NUM_FMT': 1, 'ACC_FMT': 2}
    if any(constants.get(name) != value for name, value in required.items()):
        raise ValueError('Expected a generated LPDDR6 FP16/FP32 profile')
    ndp = Path(__file__).resolve().parents[3]
    sources = {}
    for directory in ['include/gem5/pim', 'libpim/src',
                      'ext/ramulator2/python/ramulator/dram']:
        for path in sorted((root / directory).rglob('*')):
            if path.is_file() and path.suffix in {'.c', '.h', '.py'}:
                sources[str(path.relative_to(root))] = digest(path)
    for name in ['libpim/SConstruct', 'libpim/src/SConscript', 'libpim/gen_pim_layout.py']:
        sources[name] = digest(root / name)
    ndp_sources = {name: digest(ndp / name) for name in [
        'tools/gem5/pim_kernel/Makefile', 'tools/gem5/pim_kernel/worker.cpp',
        'tools/gem5/pim_kernel/decoder_fixture.py',
        'tools/gem5/pim_kernel/decoder_state_fixture.py',
        'tools/gem5/pim_kernel/smoke.cpp', 'tools/gem5/pim_kernel/target_fixture.py',
        'include/ndp/l1/gem5_pim/dense_kernel.hpp', 'include/ndp/common/fp16.hpp',
        'src/common/fp16.cpp']}
    artifacts = {}
    for name, path in [('layout', layout), ('archive', args.archive), ('worker', args.worker)]:
        artifacts[name] = {'path': str(path.resolve()), 'sha256': digest(path)}
    compiler = subprocess.check_output(
        shlex.split(args.compiler) + ['--version'], text=True).strip()
    result = {
        'schema_version': 1, 'kind': args.kind, 'pim_root': str(root),
        'compiler': compiler, 'profile': constants, 'artifacts': artifacts,
        'libpim_source_hashes': sources, 'ndp_source_hashes': ndp_sources,
        'scope': 'Worker build identity; not a full gem5 source or simulation manifest.',
        'simulation': None,
        'required_run_fields': ['gem5_source_revision_or_archive_hash', 'gem5_binary_hash',
                                'config', 'cpu', 'vlen_bits', 'cache', 'memory_mode',
                                'posted', 'ab_mode', 'arb_policy', 'kernel_hash'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
