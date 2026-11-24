#!/usr/bin/env python3
"""Compare pim_ref.py's port of pim_addr.h with the C header.

Reads geom_c's output (one sample per line: the geometry, an offset, its
decoded coordinates and byte, and their re-encoding) and decodes and
encodes every offset with ``pim_ref.Geometry``. Prints the count and exits
with 1 on any difference. Usage: geom_py.py <geom_c output>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pim_ref as R  # noqa: E402


def main():
    n = bad = 0
    for line in open(sys.argv[1]):
        v = [int(x) for x in line.split()]
        g = R.Geometry(*v[:10])
        off = v[10]
        want = v[11:18]
        c, byte = g.decode(off)
        got = list(c.key()[:2]) + [c.sid, c.bg, c.bank, c.row, c.col]
        n += 1
        if got != want or byte != v[18] or g.encode(c) != v[19]:
            bad += 1
            if bad <= 10:
                print(f"MISMATCH {line.strip()}: got {got} {byte}")
    print(f"geom: {n} samples, {bad} mismatches")
    return 1 if bad or not n else 0


if __name__ == "__main__":
    sys.exit(main())
