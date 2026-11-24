#!/usr/bin/env python3
"""SE runs of the PIM library's programs on a PimSystemMemory.

Laid out like a TestLib suite, but not one (TestLib builds only
``build/<ISA>``, and Ramulator 2.1 exists only in
``build/RISCV_RAMULATOR2``):

1. Fixtures: the PIM library and each program, for RISC-V and for the host
   reference backend, per layout: the default headers
   ``include/gem5/pim/pim_layout_<std>.h``, or headers generated with
   ``libpim/gen_pim_layout.py`` for the other policies. For every run,
   the host build of the same layout, given the run's VLENB, predicts
   gem5's counts (``PIM_REF_COUNTS``, ``libpim/src/ref/model.c``) and
   its output must equal the committed reference (the results do not
   depend on the policy).
2. Runs: ``build/RISCV_RAMULATOR2/gem5.opt`` with
   ``tests/gem5/pim/configs/pim_lib_se.py``, in parallel.
3. Verifiers: the program's output equals
   ``tests/gem5/pim/ref/<program>/<std>.out`` (which includes the count of
   results outside bf16 tolerance, 0); gem5 exits with 0; ``pim.reqs::<Op>``
   and every ``channel<i>`` ``pim_col_cmds_<Class>`` equal the prediction; on
   the Atomic CPU every ``pim_col_cmds_<Class>`` is 0 instead, as atomic mode
   executes PIM requests without Ramulator 2.1 commands, and
   ``pim.atomicReqs::<Op>`` equals the predicted requests; ``pim.swErrors`` is
   0; the operating-system memory has no PIM statistic and gem5 printed no
   fatal or panic.

Also the prefetcher ablation: both programs on the LPDDR6 default, Timing and
O3, in three settings of the stdlib caches' stride prefetchers (``pim_lib_se.py
--prefetcher``): ``pf-on`` (the PIM memory excluded, the default), ``pf-none``
(no prefetcher) and ``pf-excluded-off`` (prefetchers without the exclusion).
``pf-on`` and ``pf-none`` must pass the verifiers above; ``pf-none`` has no
prefetcher statistic. By the ``HWPrefetch`` traces, ``pf-on`` generates no
prefetch for the PIM memory and ``pf-excluded-off`` generates some, so the
exclusion is what keeps prefetches out. (An excluded range also stops training,
so with the whole PIM memory excluded no prefetch for it is generated and
``pfExcluded`` stays 0.) ``pf-excluded-off`` must either pass the verifiers or
stop on the dirty-read diagnostic; which one happens is reported. Host-visible
cycles and ``pfUseful`` of each run go to ``<work>/ablation.txt``.

Also ``pim-reuse`` on Timing and O3: an element-wise output written again by
PIM after the host read it; its results must be right and no PIM write may meet
a valid copy in L1 or L2 (``pimValidBlk``).

Prints a summary with each run's host time and exits with 1 if anything
failed. From the tree root::

    tests/gem5/pim/run_lib_se.py -j 26
    tests/gem5/pim/run_lib_se.py --program pim-gemv \\
        --run lpddr6-RoRaBaChCo-timing
"""

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PIM_DIR = ROOT / "libpim"
REF_DIR = ROOT / "tests" / "gem5" / "pim" / "ref"
CONFIG = ROOT / "tests" / "gem5" / "pim" / "configs" / "pim_lib_se.py"
PROGRAMS = ("pim-gemv", "pim-eltwise")
STANDARDS = ("ddr5", "lpddr6", "gddr7", "hbm4")
CROSS = "riscv64-linux-gnu-"

# (name, standard, policy, interleave, CPU) per program.
RUNS = (
    [
        (f"{std}-RoRaBaChCo-timing", std, "RoRaBaChCo", None, "TIMING")
        for std in STANDARDS
    ]
    + [
        (f"lpddr6-{p}-timing", "lpddr6", p, None, "TIMING")
        for p in (
            "RoRaBaCoCh",
            "RoCoRaBaCh",
            "RoBaRaCoCh",
            "ChRaBaRoCo",
            "MOP4CLXOR",
        )
    ]
    + [
        (
            "lpddr6-RoBaRaCoCh-i256-timing",
            "lpddr6",
            "RoBaRaCoCh",
            256,
            "TIMING",
        )
    ]
    + [
        (f"lpddr6-RoRaBaChCo-{cpu.lower()}", "lpddr6", "RoRaBaChCo", None, cpu)
        for cpu in ("ATOMIC", "MINOR", "O3")
    ]
)


# The prefetcher ablation: (name, standard, policy, interleave, CPU,
# prefetcher setting).
PF_SETTINGS = ("on", "none", "excluded-off")
PF_RUNS = tuple(
    (f"lpddr6-pf-{pf}-{cpu.lower()}", "lpddr6", "RoRaBaChCo", None, cpu, pf)
    for cpu in ("TIMING", "O3")
    for pf in PF_SETTINGS
)
DIRTY_READ_FATAL = "flush the line with cbo.flush"


def pf_checks(run, outdir, rc, fails):
    """The ablation's checks; returns (failures, report line)."""
    pf = run[5]
    log_text = (outdir / "gem5.log").read_text(errors="replace")
    st = read_stats(outdir / "stats.txt") if rc == 0 else {}

    def total(suffix):
        return sum(v for k, v in st.items() if k.endswith(suffix))

    cycles = total("processor.cores.core.numCycles")
    report = (
        f"cycles {cycles:.0f} pfIssued {total('prefetcher.pfIssued'):.0f} "
        f"pfUseful {total('prefetcher.pfUseful'):.0f} pfExcluded "
        f"{total('pfExcluded'):.0f}"
    )
    if pf == "none":
        pfs = [k for k in st if ".prefetcher." in k]
        if pfs:
            fails = fails + [f"prefetcher statistic {pfs[0]}"]
        return fails, report
    hdr = ROOT / "include" / "gem5" / "pim" / "pim_layout_lpddr6.h"
    lay = layout_values(hdr)
    lo, hi = lay["WINDOW_BASE_SE"], lay["WINDOW_BASE_SE"] + lay["WINDOW_SIZE"]
    gen = 0
    trace = outdir / "trace.out"
    for line in trace.open() if trace.exists() else []:
        m = re.search(r"Generating prefetch for (0x[0-9a-f]+)", line)
        if m and lo <= int(m.group(1), 16) < hi:
            gen += 1
    if trace.exists():
        subprocess.run(["gzip", "-f", str(trace)])
    report += f"; prefetches generated for the PIM memory {gen}"
    if pf == "on":
        if gen:
            fails = fails + [f"{gen} prefetches for the PIM memory"]
        return fails, report
    if rc != 0 and DIRTY_READ_FATAL in log_text:
        return [], "stopped on the dirty-read diagnostic; " + report
    if not gen:
        fails = fails + [
            "no prefetch for the PIM memory without the exclusion"
        ]
    return fails, ("correct; " if not fails else "") + report


# pim-reuse writes an element-wise output again after the host read it; the
# library must flush it first, so no PIM write may meet a valid copy
# (pimValidBlk) in L1 or L2.
REUSE = "tests/test-progs/pim-reuse"
REUSE_RUNS = (("reuse-timing", "TIMING"), ("reuse-o3", "O3"))


def reuse_one(job):
    """One pim-reuse run on the LPDDR6 default layout and its checks."""
    gem5, name, cpu, outdir = job
    outdir.mkdir(parents=True, exist_ok=True)
    hdr = ROOT / "include" / "gem5" / "pim" / "pim_layout_lpddr6.h"
    cmd = [gem5, "-d", outdir, CONFIG, "--std", "LPDDR6", "--cpu", cpu]
    cmd += ["--layout-file", hdr, "--binary"]
    cmd += [ROOT / REUSE / "bin" / "riscv" / "lpddr6" / "pim-reuse"]
    t0 = time.time()
    with open(outdir / "gem5.log", "w") as f:
        rc = subprocess.call(
            [str(c) for c in cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT
        )
    dt = time.time() - t0
    fails = []
    out = (outdir / "program.out").read_text() if rc == 0 else ""
    if rc != 0:
        fails.append(f"gem5 exited with {rc}")
    elif len(re.findall(r": 0 of \d+ wrong", out)) != 2:
        fails.append(f"wrong results: {out.strip()}")
    stats = (outdir / "stats.txt").read_text()
    valid = re.findall(r"^(\S*(?:l1d|l2)\S*pimValidBlk)\s+(\d+)", stats, re.M)
    if any(int(v) for _, v in valid):
        fails.append(f"PIM writes met valid copies: {valid}")
    return "pim-reuse", name, dt, fails


def log(msg):
    print(msg, flush=True)


def sh(cmd, cwd=ROOT, env=None, out=None):
    """Run a command; raise with its output on failure."""
    r = subprocess.run(
        [str(c) for c in cmd],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if out:
        Path(out).write_text(r.stdout)
    if r.returncode:
        raise RuntimeError(
            f"{' '.join(map(str, cmd))} failed ({r.returncode}):\n"
            + r.stdout[-3000:]
        )
    return r.stdout


def layout_name(std, policy, interleave):
    if policy == "RoRaBaChCo" and interleave is None:
        return std
    return f"{std}-{policy}" + (f"-i{interleave}" if interleave else "")


def layout_header(name, std, policy, interleave, work):
    """The default header, or one generated into the work directory."""
    if name == std:
        return ROOT / "include" / "gem5" / "pim" / f"pim_layout_{std}.h"
    hdr = work / "layouts" / f"{name}.h"
    hdr.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["python3", "-B", PIM_DIR / "gen_pim_layout.py", "--standard", std]
    cmd += ["--policy", policy, "-o", hdr]
    if interleave:
        cmd += ["--interleave", str(interleave)]
    sh(cmd)
    return hdr


def layout_values(hdr):
    values = {}
    for line in Path(hdr).read_text().splitlines():
        m = re.match(r"^#define PIM_LAYOUT_(\w+) (\S+)$", line)
        if m:
            values[m.group(1)] = int(m.group(2), 0)
    return values


def build_programs(name, std, hdr, work):
    """Build both programs for RISC-V and the host; one layout at a time,
    as a generated layout reuses libpim_custom.a."""
    bins = {}
    for prog in PROGRAMS:
        src = ROOT / "tests" / "test-progs" / prog / "src"
        for target in ("riscv", "host"):
            make = ["make", "-C", src, f"TARGET={target}"]
            make += [f"CROSS_COMPILE={CROSS}", f"LAYOUT_NAME={name}"]
            if name == std:
                make.append(f"PIM_STD={std}")
            else:
                make.append(f"PIM_LAYOUT={hdr}")
                # The custom archive is rebuilt for each generated layout.
                (src.parent / "bin" / target / name / prog).unlink(
                    missing_ok=True
                )
            sh(make, out=work / "build" / f"{prog}-{target}-{name}.log")
            bins[(prog, target)] = (
                src.parent / "bin" / target / name / prog
            ).resolve()
    return bins


def predict(host_bin, args, vlenb, work, tag):
    """The host reference run: output and gem5's predicted counts."""
    counts = work / "predict" / f"{tag}.counts"
    counts.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PIM_REF_VLENB=str(vlenb), PIM_REF_COUNTS=counts)
    env = {k: str(v) for k, v in env.items()}
    r = subprocess.run(
        [str(host_bin)] + args, env=env, stdout=subprocess.PIPE, text=True
    )
    pred = {"reqs": {}, "cols": {}}
    for line in counts.read_text().splitlines():
        f = line.split()
        if f[0] == "reqs":
            pred["reqs"][f[1]] = int(f[2])
        elif f[0] == "cols":
            pred["cols"][(int(f[1]), f[2])] = int(f[3])
    return r.returncode, r.stdout, pred


def read_stats(path):
    """The first dump of stats.txt as {name: value}."""
    stats = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("---------- End"):
            break
        f = line.split()
        if len(f) >= 2 and not line.startswith("-"):
            try:
                stats[f[0]] = float(f[1])
            except ValueError:
                pass
    return stats


def verify(run, outdir, ref_text, pred, rc):
    """The verifiers; returns a list of failure messages."""
    fails = []
    if rc != 0:
        fails.append(f"gem5 exited with {rc}")
    log_text = (outdir / "gem5.log").read_text(errors="replace")
    for word in ("fatal:", "panic:"):
        if word in log_text:
            line = next(l for l in log_text.splitlines() if word in l)
            fails.append(f"gem5 printed {line.strip()}")
    out = outdir / "program.out"
    got = out.read_text() if out.exists() else ""
    if got != ref_text:
        a, b = got.splitlines(), ref_text.splitlines()
        i = next(
            (k for k in range(min(len(a), len(b))) if a[k] != b[k]),
            min(len(a), len(b)),
        )
        fails.append(
            f"output differs from the reference at line {i + 1}: "
            f"{a[i] if i < len(a) else '<end>'!r} vs "
            f"{b[i] if i < len(b) else '<end>'!r}"
        )
    stats_path = outdir / "stats.txt"
    if not stats_path.exists():
        return fails + ["no stats.txt"]
    st = read_stats(stats_path)
    base = "board.memory.pim_memory.ramulator2"
    # Atomic mode executes PIM requests functionally with the
    # unloaded latency and issues no Ramulator 2.1 command.
    atomic = run[4] == "ATOMIC"
    for op, n in pred["reqs"].items():
        g = st.get(f"{base}.pim.reqs::{op}", 0)
        if g != n:
            fails.append(f"pim.reqs::{op} {g:g}, predicted {n}")
        if atomic:
            g = st.get(f"{base}.pim.atomicReqs::{op}", 0)
            if g != n:
                fails.append(f"pim.atomicReqs::{op} {g:g}, predicted {n}")
    for (ch, cls), n in pred["cols"].items():
        key = f"{base}.channel{ch}.controller_plugin.pim_col_cmds_{cls}"
        want = 0 if atomic else n
        if key not in st:
            fails.append(f"no statistic {key}")
        elif st[key] != want:
            fails.append(
                f"channel{ch} pim_col_cmds_{cls} {st[key]:g}, "
                f"expected {want}"
            )
    if st.get(f"{base}.pim.swErrors", 0):
        fails.append(f"pim.swErrors {st[base + '.pim.swErrors']:g}")
    if not any(k.startswith(base + ".pim.reqs::") for k in st):
        fails.append("no PIM request reached the PIM memory")
    os_pim = [
        k
        for k in st
        if k.startswith("board.memory.os_memory.") and "pim" in k.lower()
    ]
    if os_pim:
        fails.append(f"PIM statistic at the OS memory: {os_pim[0]}")
    return fails


def run_one(job):
    """One gem5 run and its verification."""
    prog, run, gem5, args, hdr, binary, ref_text, pred, outdir = job
    name, std, policy, interleave, cpu = run[:5]
    pf = run[5] if len(run) > 5 else None
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [gem5]
    if pf in ("on", "excluded-off"):
        cmd += ["--debug-flags=HWPrefetch", "--debug-file=trace.out"]
    cmd += ["-d", outdir, CONFIG, "--std", std.upper()]
    if pf:
        cmd += ["--prefetcher", pf]
    cmd += ["--policy", policy, "--layout-file", hdr, "--cpu", cpu]
    if interleave:
        cmd += ["--interleave", str(interleave)]
    cmd += ["--binary", binary, "--args"] + args
    t0 = time.time()
    with open(outdir / "gem5.log", "w") as f:
        rc = subprocess.call(
            [str(c) for c in cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT
        )
    dt = time.time() - t0
    fails = verify(run, outdir, ref_text, pred, rc)
    if pf:
        fails, report = pf_checks(run, outdir, rc, fails)
        (outdir / "ablation.txt").write_text(report + "\n")
    return prog, name, dt, fails


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--program", action="append", choices=PROGRAMS)
    ap.add_argument("--run", action="append", help="run name (repeatable)")
    ap.add_argument("-j", type=int, default=os.cpu_count())
    ap.add_argument(
        "--gem5", default=ROOT / "build" / "RISCV_RAMULATOR2" / "gem5.opt"
    )
    ap.add_argument("--work", default=ROOT / "m5out" / "pim-lib-se")
    ap.add_argument(
        "--build-only", action="store_true", help="fixtures and predictions"
    )
    a = ap.parse_args()
    programs = a.program or list(PROGRAMS)
    runs = [
        r for r in list(RUNS) + list(PF_RUNS) if not a.run or r[0] in a.run
    ]
    work = Path(a.work).resolve()
    (work / "build").mkdir(parents=True, exist_ok=True)

    # 1. Fixtures: the default archives, then each layout's programs.
    log("== fixtures")
    targets = [f"build/riscv/out/libpim_{s}.a" for s in STANDARDS]
    targets += [f"build/native/libpim_ref_{s}.a" for s in STANDARDS]
    sh(
        [
            "scons",
            "-C",
            PIM_DIR,
            "-j",
            str(a.j),
            f"riscv.CROSS_COMPILE={CROSS}",
        ]
        + targets,
        out=work / "build" / "libpim.log",
    )
    layouts = {}
    for run in runs:
        name = layout_name(run[1], run[2], run[3])
        if name not in layouts:
            hdr = layout_header(name, run[1], run[2], run[3], work)
            # Programs of a generated layout are copied before the next
            # layout reuses the custom archive and output directory.
            bins = build_programs(name, run[1], hdr, work)
            keep = work / "bin" / name
            keep.mkdir(parents=True, exist_ok=True)
            for (prog, target), path in bins.items():
                dst = keep / f"{prog}.{target}"
                dst.write_bytes(path.read_bytes())
                dst.chmod(0o755)
                bins[(prog, target)] = dst
            layouts[name] = (hdr, bins)
            log(f"   built {name}")

    # Predictions from the host build of each run's layout.
    jobs, failed = [], []
    for prog in programs:
        for run in runs:
            name, std = run[0], run[1]
            lname = layout_name(std, run[2], run[3])
            hdr, bins = layouts[lname]
            args = (REF_DIR / prog / f"{std}.args").read_text().split()
            ref_text = (REF_DIR / prog / f"{std}.out").read_text()
            vlenb = layout_values(hdr)["COL_BYTES"]  # VLEN 8 x column bytes
            rc, host_out, pred = predict(
                bins[(prog, "host")], args, vlenb, work, f"{prog}-{name}"
            )
            if rc != 0 or host_out != ref_text:
                failed.append(
                    (
                        prog,
                        name,
                        0.0,
                        [
                            f"the host build of {lname} does not "
                            f"reproduce the reference (exit {rc})"
                        ],
                    )
                )
                continue
            outdir = work / "runs" / prog / name
            jobs.append(
                (
                    prog,
                    run,
                    a.gem5,
                    args,
                    hdr,
                    (ROOT / bins[(prog, "riscv")]).relative_to(ROOT),
                    ref_text,
                    pred,
                    outdir,
                )
            )
    log(f"== {len(jobs)} runs predicted")
    reuse = [r for r in REUSE_RUNS if not a.run or r[0] in a.run]
    if reuse:
        sh(["make", "-C", f"{REUSE}/src", "PIM_STD=lpddr6"])
    if a.build_only:
        return 1 if failed else 0

    # 2. Runs and 3. verifiers.
    results = list(failed)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.j) as pool:
        reuse_jobs = [
            (a.gem5, name, cpu, work / "runs" / "pim-reuse" / name)
            for name, cpu in reuse
        ]
        both = [(run_one, j) for j in jobs] + [
            (reuse_one, j) for j in reuse_jobs
        ]
        for res in pool.map(lambda fj: fj[0](fj[1]), both):
            prog, name, dt, fails = res
            log(
                f"   {'PASS' if not fails else 'FAIL'} {prog} {name} "
                f"({dt:.0f} s)"
            )
            for msg in fails:
                log(f"        {msg}")
            results.append(res)

    log("== summary")
    for prog, name, dt, fails in sorted(results):
        log(
            f"{'PASS' if not fails else 'FAIL'} {prog:12} {name:32} "
            f"{dt:7.0f} s"
        )
    lines = []
    for prog in programs:
        for run in PF_RUNS:
            f = work / "runs" / prog / run[0] / "ablation.txt"
            if f.exists():
                lines.append(f"{prog:12} {run[0]:28} {f.read_text().strip()}")
    if lines:
        (work / "ablation.txt").write_text("\n".join(lines) + "\n")
        log("== prefetcher ablation")
        for line in lines:
            log(line)
    nfail = sum(1 for r in results if r[3])
    log(f"{len(results) - nfail} passed, {nfail} failed")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
