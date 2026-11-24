"""Check a static RISC-V worker and save ELF/disassembly evidence without running it."""
import argparse
import json
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('elf', type=Path)
    parser.add_argument('--prefix', default='riscv64-linux-gnu-')
    args = parser.parse_args()
    elf = args.elf
    header = elf.read_bytes()[:20]
    if header[:6] != b'\x7fELF\x02\x01' or int.from_bytes(header[18:20], 'little') != 243:
        raise ValueError('Expected ELF64 little-endian RISC-V')
    evidence = {}
    for name, command in {
        'readelf': ['readelf', '-h', '-l', '-A'],
        'symbols': ['nm'], 'disassembly': ['objdump', '-d'],
    }.items():
        evidence[name] = subprocess.check_output(
            [args.prefix + command[0], *command[1:], str(elf)], text=True)
        elf.with_suffix('.' + name + '.txt').write_text(evidence[name])
    if re.search(r'^\s*INTERP\s', evidence['readelf'], re.MULTILINE):
        raise ValueError('Dynamic interpreter found; static Linux ELF required')
    if re.search(r'\bpim_ref_(scalar|vector|fence|flush)\b', evidence['symbols']):
        raise ValueError('Native reference functions linked into target ELF')
    if not re.search(r'\bpim_gemv\b', evidence['symbols']):
        raise ValueError('Missing libpim GEMV symbol')
    # Count encoded custom instructions in executable disassembly, not raw data.
    words = re.findall(r'^\s*[0-9a-f]+:\s+([0-9a-f]{8})\s',
                       evidence['disassembly'], re.MULTILINE)
    counts = {hex(op): sum((int(word, 16) & 0x7f) == op for word in words)
              for op in (0x0b, 0x2b, 0x5b)}
    if not counts['0xb'] or not counts['0x5b']:
        raise ValueError('Expected PIM vector read/write custom opcode sites')
    summary = {'elf': str(elf), 'static_riscv64': True,
               'reference_backend_symbols': False, 'custom_opcode_sites': counts,
               'note': 'Static instruction sites, not executed instruction counts. '
                       'ISA attributes still require comparison with the target CPU.'}
    elf.with_suffix('.inspection.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('NDP_RISCV_ELF_PASS', elf)


if __name__ == '__main__':
    main()
