#!/usr/bin/env python3
"""Build, run and collect one PIM example (workload/microbench, workload/llm).

    bench.py build   <dir> [--host]      libpim + one binary per layout
    bench.py run     <dir> [-j N] [--only NAME ...]
    bench.py collect <dir>               results/summary.csv, summary.md

An example directory holds:
    src/                    the program; regions marked with common/bench.h
    configs/bench.json      what to build and how to run it (below)
    configs/runs.csv        one run per line
    scripts/                build.sh, run.sh, collect.sh: wrappers of this file
    results/runs/<name>/    one gem5 output directory per run (kept small)
    results/summary.{csv,md}
Binaries go to bin/<layout>/ and, with --host, bin/host-<layout>/: the same
program on the host reference backend (PIM_BACKEND_REF, libpim/src/ref).

configs/bench.json:
    program     binary name
    sources     source files, relative to the example directory
    includes    extra include directories, relative to workload/
    cflags      extra compiler flags (optional)
    config      gem5 configuration, relative to the tree root (default
                configs/example/gem5_library/riscv-pim.py)
    speedups    [[slower region, faster region], ...] for summary.md

configs/runs.csv: columns name, memory, args, optional policy (empty =
RoRaBaChCo; another policy builds a generated layout header), and any other
column, passed to the gem5 configuration as --<column with - for _>.

Environment: GEM5 (default <tree>/build/RISCV_RAMULATOR2/gem5.opt),
CROSS_COMPILE (default riscv64-linux-gnu-).
"""

import argparse
import concurrent.futures
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORKLOAD = Path(__file__).resolve().parents[1]
ROOT = WORKLOAD.parent
PIM_LIB = ROOT / "libpim"
CROSS = os.environ.get("CROSS_COMPILE", "riscv64-linux-gnu-")
GEM5 = Path(
    os.environ.get("GEM5", ROOT / "build" / "RISCV_RAMULATOR2" / "gem5.opt")
)
DEFAULT_POLICY = "RoRaBaChCo"
FIXED = {"name", "memory", "args", "policy", "layout"}
KEEP = {"program.out", "stats.txt", "cmd.txt", "rc", "hosttime", "gem5.log"}


def load(bench: Path):
    cfg = json.loads((bench / "configs" / "bench.json").read_text())
    with open(bench / "configs" / "runs.csv", newline="") as f:
        runs = list(
            csv.DictReader(line for line in f if not line.startswith("#"))
        )
    for r in runs:
        r["policy"] = r.get("policy") or DEFAULT_POLICY
        r["layout"] = (
            r["memory"]
            if r["policy"] == DEFAULT_POLICY
            else f"{r['memory']}-{r['policy']}"
        )
    return cfg, runs


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def build(bench: Path, args) -> None:
    cfg, runs = load(bench)
    jobs = ["-j", str(os.cpu_count() or 8)]
    sh(
        ["scons", "-C", PIM_LIB, f"riscv.CROSS_COMPILE={CROSS}", "build/riscv"]
        + jobs
    )
    if args.host:
        sh(["scons", "-C", PIM_LIB, "build/native"] + jobs)
    common = [
        "-O2",
        "-std=gnu11",
        "-ffp-contract=off",
        "-pthread",
        f"-I{ROOT / 'include'}",
        f"-I{WORKLOAD / 'common'}",
    ] + [f"-I{WORKLOAD / i}" for i in cfg.get("includes", [])]
    common += cfg.get("cflags", [])
    srcs = [bench / s for s in cfg["sources"]]
    for layout in sorted({r["layout"] for r in runs}):
        r = next(x for x in runs if x["layout"] == layout)
        out = bench / "bin" / layout
        out.mkdir(parents=True, exist_ok=True)
        if r["policy"] == DEFAULT_POLICY:
            lay, lib = [f"-DPIM_STD={r['memory']}"], r["memory"]
        else:
            hdr = out / "layout.h"
            sh(
                [sys.executable, PIM_LIB / "gen_pim_layout.py"]
                + ["--standard", r["memory"], "--policy", r["policy"]]
                + ["-o", hdr]
            )
            # libpim_custom.a holds one generated layout at a time.
            sh(
                ["scons", "-C", PIM_LIB, f"riscv.CROSS_COMPILE={CROSS}"]
                + [f"PIM_LAYOUT={hdr}", "build/riscv"]
            )
            lay, lib = [f'-DPIM_LAYOUT_HEADER="{hdr}"'], "custom"
        sh(
            [f"{CROSS}gcc", "-static", "-march=rv64gcv_zicbom", "-mabi=lp64d"]
            + common
            + lay
            + srcs
            + [f"-L{PIM_LIB / 'build' / 'riscv' / 'out'}", f"-lpim_{lib}"]
            + ["-lm", "-o", out / cfg["program"]]
        )
        if args.host and r["policy"] == DEFAULT_POLICY:
            hout = bench / "bin" / f"host-{layout}"
            hout.mkdir(parents=True, exist_ok=True)
            sh(
                ["gcc", "-D_DEFAULT_SOURCE", "-DPIM_BACKEND_REF"]
                + common
                + lay
                + srcs
                + [f"-L{PIM_LIB / 'build' / 'native'}", f"-lpim_ref_{lib}"]
                + ["-lm", "-o", hout / cfg["program"]]
            )


def run_one(bench: Path, cfg, r) -> str:
    out = bench / "results" / "runs" / r["name"]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    config = ROOT / cfg.get(
        "config", "configs/example/gem5_library/riscv-pim.py"
    )
    cmd = [GEM5, "-d", out, config, "--memory", r["memory"]]
    if r["policy"] != DEFAULT_POLICY:
        cmd += ["--policy", r["policy"]]
        cmd += ["--layout", bench / "bin" / r["layout"] / "layout.h"]
    for k, v in r.items():
        if k == "extra":  # free-form options, for example --no-posted
            cmd += v.split()
        elif k not in FIXED and v:
            cmd += [f"--{k.replace('_', '-')}", v]
    cmd += ["--binary", bench / "bin" / r["layout"] / cfg["program"]]
    if r["args"].strip():
        cmd += ["--args"] + r["args"].split()
    (out / "cmd.txt").write_text(" ".join(str(c) for c in cmd) + "\n")
    t0 = time.time()
    with open(out / "gem5.log", "w") as log:
        rc = subprocess.run(
            [str(c) for c in cmd], cwd=ROOT, stdout=log, stderr=log
        ).returncode
    (out / "rc").write_text(f"{rc}\n")
    (out / "hosttime").write_text(f"{time.time() - t0:.0f}\n")
    for p in out.iterdir():  # keep the results small
        if p.name not in KEEP:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    return f"{r['name']}: rc={rc}"


def run(bench: Path, args) -> None:
    cfg, runs = load(bench)
    if args.only:
        runs = [r for r in runs if r["name"] in args.only]
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
        for line in ex.map(lambda r: run_one(bench, cfg, r), runs):
            print(line, flush=True)


STAT = re.compile(r"^(\S+)\s+([-\d.e+]+)", re.M)
R2 = r"board\.memory\.pim_memory\.ramulator2\."


def total(block: str, pat: str) -> float:
    vals = [
        float(m.group(2))
        for m in STAT.finditer(block)
        if re.fullmatch(pat, m.group(1))
    ]
    return sum(vals) if vals else 0.0


def collect(bench: Path, args) -> None:
    cfg, runs = load(bench)
    rows = []
    for r in runs:
        out = bench / "results" / "runs" / r["name"]
        if not (out / "rc").exists():  # not run, or still running
            continue
        prog = (out / "program.out").read_text()
        names = re.findall(r"^region \d+ (\S+)$", prog, re.M)
        blocks = (
            (out / "stats.txt")
            .read_text()
            .split("---------- Begin Simulation Statistics ----------")[1:]
        )
        # A program reports wrong results as "...: <count>" lines.
        bad = re.findall(r"(?:outside \S+ tolerance|wrong): (\d+)", prog)
        rc = (out / "rc").read_text().strip()
        ok = rc == "0" and bad and all(b == "0" for b in bad)
        # The regions' dumps are the first blocks: nothing dumps before them
        # (a work-begin hypercall only resets).
        blocks = blocks[: len(names)]
        cum = {"acts": 0.0, "ab_acts": 0.0}
        for name, b in zip(names, blocks):
            # Ramulator 2.1 counters are cumulative: take differences.
            now = {
                "acts": total(
                    b, R2 + r"channel\d+\.controller_plugin\.pim_acts"
                ),
                "ab_acts": total(
                    b, R2 + r"channel\d+\.controller_plugin\.pim_ab_acts"
                ),
            }
            row = {"run": r["name"]}
            row.update(
                {k: v for k, v in r.items() if k not in ("name", "layout")}
            )
            row.update(
                region=name,
                time_us=round(total(b, r"simTicks") / 1e6, 3),
                host_insts=int(total(b, r"simInsts")),
                pim_reqs=int(total(b, R2 + r"pim\.reqs::total")),
                pim_acts=int(now["acts"] - cum["acts"]),
                pim_ab_acts=int(now["ab_acts"] - cum["ab_acts"]),
                checked="pass" if ok else f"FAIL (rc {rc})",
            )
            rows.append(row)
            cum = now
    if not rows:
        print("no results")
        return
    res = bench / "results"
    with open(res / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    regions = list(dict.fromkeys(x["region"] for x in rows))
    sp = cfg.get("speedups", [])
    params = [k for k in runs[0] if k not in ("name", "layout")]
    head = (
        ["run"]
        + params
        + [f"{g} (µs)" for g in regions]
        + [f"{s} / {f}" for s, f in sp]
        + ["check"]
    )
    lines = [
        f"# {cfg['program']} results",
        "",
        "Simulated time per region in µs; speedups are time ratios.",
        "Generated by `bench.py collect` from `results/runs/`.",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for r in runs:
        mine = {x["region"]: x for x in rows if x["run"] == r["name"]}
        if not mine:
            continue
        t = {k: v["time_us"] for k, v in mine.items()}
        cells = [r["name"]] + [r[k] for k in params]
        cells += [f"{t[g]:.1f}" if g in t else "" for g in regions]
        cells += [
            f"{t[s] / t[f]:.2f}×" if s in t and f in t and t[f] else ""
            for s, f in sp
        ]
        cells.append(next(iter(mine.values()))["checked"])
        lines.append("| " + " | ".join(cells) + " |")
    (res / "summary.md").write_text("\n".join(lines) + "\n")
    print((res / "summary.md").read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("build", "run", "collect"))
    ap.add_argument("bench", type=Path)
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--host", action="store_true", help="also host builds")
    a = ap.parse_args()
    {"build": build, "run": run, "collect": collect}[a.cmd](
        a.bench.resolve(), a
    )


if __name__ == "__main__":
    main()
