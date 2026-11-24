#!/usr/bin/env python3
"""Compare each gem5 run of an LLM example with its host reference build.

    check_digest.py <example dir>

For every run in configs/runs.csv that has results/runs/<name>/program.out,
runs bin/host-<memory>/<program> with the same arguments (the PIM library's
host reference backend and the plain C kernels) and compares the "digest"
lines: per sequence, the sum and sum of squares of the layer output, two
output elements and one attention output. The RVV kernels and the C ones
differ only in summation order, so the values agree to about 1e-5 of the
output's size. Programs that route tokens to experts (the MoE example)
also print "route" lines, the experts each sequence chose; they must be
equal, since a different choice changes the output by more than rounding
(the program prints the margin of each choice). Writes results/check.md
and exits with 1 on any mismatch.
"""

import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

TOL = 1e-2  # relative to the output vector's root-mean-square size


def digests(text: str):
    out = {}
    for m in re.finditer(
        r"^digest (\d+) sum (\S+) sumsq (\S+) x0 (\S+) x1 (\S+) attn0 (\S+)",
        text,
        re.M,
    ):
        out[int(m.group(1))] = [float(v) for v in m.groups()[1:]]
    return out


def routes(text: str):
    return re.findall(r"^route ([\d ]+) margin", text, re.M)


def main() -> int:
    ex = Path(sys.argv[1]).resolve()
    cfg = json.loads((ex / "configs" / "bench.json").read_text())
    with open(ex / "configs" / "runs.csv", newline="") as f:
        runs = list(
            csv.DictReader(line for line in f if not line.startswith("#"))
        )
    rows = []
    failed = False
    for r in runs:
        out = ex / "results" / "runs" / r["name"] / "program.out"
        if not out.exists():
            continue
        g = digests(out.read_text())
        host = ex / "bin" / f"host-{r['memory']}" / cfg["program"]
        htext = subprocess.run(
            [str(host)] + r["args"].split(),
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        h = digests(htext)
        gr, hr = routes(out.read_text()), routes(htext)
        hidden = int(re.search(r"hidden (\d+)", out.read_text()).group(1))
        worst, ok = 0.0, bool(g) and g.keys() == h.keys()
        for b in g.keys() & h.keys():
            sum_, sq, x0, x1, a0 = (abs(x - y) for x, y in zip(g[b], h[b]))
            rms = math.sqrt(max(h[b][1], 1e-30) / hidden)
            worst = max(
                worst,
                sum_ / math.sqrt(max(h[b][1], 1e-30)),  # sum of n values
                sq / max(h[b][1], 1e-30),
                x0 / rms,
                x1 / rms,
                a0 / max(abs(h[b][4]), 1e-3),
            )
        ok = ok and worst <= TOL and gr == hr
        failed |= not ok
        rows.append(
            [r["name"], r["args"], str(len(g)), f"{worst:.2e}"]
            + (["equal" if gr == hr else "DIFFER"] if gr or hr else [None])
            + ["pass" if ok else "FAIL"]
        )
    with_routes = any(x[4] is not None for x in rows)
    head = ["run", "args", "sequences", "largest difference / size"]
    head += ["routes"] if with_routes else []
    head += ["check"]
    lines = [
        "# Host reference check",
        "",
        "Layer output digests of each gem5 run against the same program on "
        "the host reference backend (`check_digest.py`); tolerance "
        f"{TOL:g} of the output's root-mean-square size."
        + (
            " Routes: the experts each sequence chose, gem5 against host."
            if with_routes
            else ""
        ),
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for x in rows:
        cells = [c for c in x if c is not None]
        lines.append("| " + " | ".join(cells) + " |")
    (ex / "results" / "check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
