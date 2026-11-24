#!/usr/bin/env python3
"""The processing-in-memory (PIM) Ramulator 2.1 test run list.

Every acceptance run of the PIM memory from one driver, with a PASS or
FAIL verdict per case. Groups (``--groups``, default all but ``fs-boot``,
``fs-way1``, ``fs-way1-whole``, ``fs-protocol``, ``fs-way2``, ``sv48``
and ``fs-ddr5``):

- ``cpuless``: the CPU-less runs on ``PimTester``
  (``configs/pim_r2_cpuless.py``): C1-C9, C12-C26, C29, normal traffic;
- ``refresh``: C10 (Chopim, host traffic, refresh every 3000 clocks) and
  C11 (a long run over many refresh intervals), also on ``gem5.debug``
  when it exists, and each with ``--close-rows`` (names ``...-close``);
- ``c28``: C28, the per-bank ACTs of a Baseline all-bank job against the
  T12 sequence of ``ext/ramulator2/tests/device_timings/pim_cases.py``
  (``t12_acts.py``, run in an offline scratch build of
  ``ext/ramulator2`` with its Python bindings);
- the checks (not groups) ``c26-model``, in ``cpuless``, and
  ``c28-model-<std>``, in ``c28``: C26's timing-mode latencies
  and C28's ACT clocks against the analytic model of
  ``libpim/timing/pim_timing.py`` on the committed tables
  ``timing_<std>.json`` (U plus the issue term; A(k));
- ``path``: the request path on an LPDDR6 ``PimMemory``
  (``configs/pim_path_se.py --memory lpddr6``): ``pim-path``'s PIM
  kernels on a matrix of CPU models and vector lengths, its protocol
  cases P1-P7 (cache maintenance and dirty copies around PIM accesses,
  prefetchers) and its speculation cases s1-s3;
- ``config``: V12-V15, C30 and C31, the layout header generator's
  checks, and Way 2's device tree on the four default memories;
- ``v16``: V16 in SE (``configs/example/gem5_library/riscv-pim.py``);
- ``decode``: the decoder of the PIM instructions on the native
  ``build/RISCV/gem5.opt``: the static check of the generated decoder
  (``check_generated.py``), the decode matrix (four modes on nine CPU
  and VLEN configurations, ``configs/pim_decode_se.py`` and
  ``check_exec_trace.py``) and the 18 negative runs (16 reserved
  encodings, 2 guards);
- ``fsbm``: the cases F1-F10 of ``pim-fs-bm`` in FS bare metal
  (``tests/test-progs/pim-fs-bm``, ``check_fs_bm.py``) on Atomic, Timing,
  Minor and O3, and F7's PMA runs on Timing and O3
  (``configs/pim_fs_bm_pma.py``), on the native build; with them
  ``pim-fs-bm-nopim`` on every model, and the Sv48 cases F1-F18
  (``pim-fs-bm-sv48`` with ``max_paging_mode`` "sv48", ``check_fs_bm.py
  --sv48``) on every model plus one Timing run without the parameter
  (only F14 fails);
- ``cache``: the CPU-less cache cases P2-P6, the D-3 write-buffer case
  and its 0-60 ns sweep, the non-coherent guard
  (``configs/pim_cache_cpuless.py``), an end-to-end check on a
  ``PimMemory`` (a PIM operation on a block whose dirty copy is cached
  sees the written data), and two PIM writes of one tick behind a
  pending writeback, which must reach the memory in order;
- ``se-misc``: the ``path`` group's matrix and protocol cases on the
  native ``build/RISCV/gem5.opt`` (names ``native-...``), the
  ``/dev/mem`` unmap check (``pim-devmem unmap``) on Atomic and Timing;
  ``pim-decode`` on Ruby (``build/ALL/gem5.opt``, Timing and Atomic),
  which must stop at the Ruby guard; and ``pim-order``, two PIM writes
  per iteration behind a cached line, on O3, Timing and Minor;
- ``r2suites``: Ramulator 2.1's own pytest suites
  (``device_timings``, ``controller_scheduling``, ``unit_tests``) and K6
  (``k6_trace.py``: the stock standards' command traces, compared by hash
  with the trace taken before the PIM changes to Ramulator 2.1) in the
  offline scratch build of ``ext/ramulator2`` that ``c28`` also uses;
- ``xcheck``: ``pim_ref.py --dump`` against ``libpim``'s
  ``test_model --dump`` on the shared vectors; gem5's functional model
  against the C reference model (``xmodel/``: the vectors and 300 random
  programs) and against ``pim_ref.py``; ``pim_ref.py``'s address port
  against ``pim_addr.h`` (80000 samples); ``ref_decode.py``'s vectors
  against Ramulator 2.1's ported mappers;
- ``r6``: the default builds stay free of Ramulator 2.1: on
  ``build/RISCV`` and ``build/ALL``, no ``ramulator2`` object, no
  Ramulator 2.1, ``PimRamulator2``, ``PimTester`` or ``PimSequencer``
  symbol, no ``Ramulator2VectorPorts`` in ALL, and ``size`` text within
  the relink noise of the last accepted build (``--size-ref``; written by
  ``--accept-sizes``);
- ``fs-boot`` (not run by default): V16's full-system boot on Timing to
  a booted kernel (about 70 minutes on ``gem5.fast``);
- ``fs-way1`` (not run by default): the Way 1 request-path run in FS
  Linux, stock kernel: ``pim-path`` mode ``all`` on the LPDDR6 default
  as root, on Atomic, and on Timing, Minor and O3 after an Atomic
  ``no_systemd`` boot and a switch; checked by ``check_fs_linux.py`` (on
  ``gem5.opt``, which has the debug trace; the four in parallel). The
  program maps only its bank buffer and the register rows (``regions``),
  because Linux fills and clears one page-table entry per mapped page;
- ``fs-way1-whole`` (not run by default, for the ``release`` profile):
  the same run on Atomic with the whole PIM memory mapped (about an hour:
  Linux fills and clears 8.4 M page-table entries);
- ``fs-protocol`` (not run by default): the full flush protocol under
  Linux, on the rebuilt kernel and image of ``util/pim-kernel`` (in
  ``$PIM_KERNEL_WORK``, default ``~/pim-kernel``), LPDDR6 default, two
  cores. One Atomic ``no_systemd`` boot, checkpointed when
  ``after_boot.sh`` starts (``build_logs/p6/ckpt/lpddr6-pimkernel-2c``;
  re-taken when the kernel, the image, the arguments or ``CKPT_EPOCH``
  change, and a case fails when its device tree or its CPUs' ISA settings
  differ from the checkpoint's), is restored by every case: the FS
  checkpoint case (PIM state written on Atomic, checkpointed after the
  PIM work, read back on Timing and O3); the library programs
  ``pim-gemv`` and ``pim-eltwise`` built with ``FS_MARKERS=1`` on Timing
  and O3 (the switch at the program's first marker, after ``pim_init``
  has mapped the PIM memory on Atomic): output equal to
  ``tests/gem5/pim/ref``, requests and commands as the host build
  predicts, and normal accesses to the PIM memory cached; and
  ``pim-path``'s flush-protocol modes P1-P6 (and P5 with the dirty-read
  warning) with the ``path`` group's criteria, on Timing and O3. At most
  ``$PIM_FS_PARALLEL`` (4) restores run at once: each fills 64 GiB of
  host memory;
- ``fs-way2`` (not run by default): the library under Linux on Way 2:
  the same kernel and board with ``--way 2`` (the PIM memory NUMA node
  1, its own boot checkpoint
  ``build_logs/p6/ckpt/lpddr6-way2-pimkernel-2c``), ``pim-gemv`` and
  ``pim-eltwise`` with ``PIM_WAY=2`` on Timing and O3 after 64 huge pages
  are reserved on node 1: the checks of ``fs-protocol``'s library runs,
  the node read from the device tree equal to the board's, every huge
  page on the PIM node below its register rows and 2 MiB-aligned; and
  with ``PIM_NODE=0`` the library's pagemap check refuses the OS node's
  page;
- ``sv48`` (not run by default): Linux in Sv48 (``riscv-pim.py
  --paging-mode sv48``, LPDDR6, Atomic ``no_systemd`` boot, switch to
  Timing): ``fs-way1``'s request-path run (``regions``) with
  ``check_fs_linux.py``, then every hart's ``/proc/cpuinfo`` must say
  ``mmu : sv48``;
- ``fs-ddr5`` (not run by default): the 128 GiB DDR5 system in FS
  (``--memory ddr5 --channels 16``, its header
  ``build_logs/p6/ckpt/pim_layout_ddr5x16.h``, the rebuilt kernel, one
  core): Linux maps the whole OS memory up to 0x207FFFFFFF under Sv48
  (the boot checkpoint's console) and up to 0x1F7FFFFFFF under Sv39 (a
  boot to the booted kernel); from the boot checkpoint, the Way 1
  request-path run (``regions``) and ``pim-gemv`` with the DDR5
  arguments on Timing, one at a time (a restore fills 256 GiB of host
  memory).

A case that cannot run here (a missing build, input or scratch build) is
reported as SKIP with the reason, never left out. Run from anywhere,
after ``scons build/RISCV_RAMULATOR2/gem5.opt`` (and ``gem5.fast`` for
``fs-boot``) and ``build/RISCV/gem5.opt`` (the native cases)::

    tests/gem5/pim/run_p4r.py --out <dir> [-j N] [--groups a,b] [--only RE]
        [--native BINARY] [--checks-only]

The summary is ``<dir>/summary.txt``; the exit status is 1 when a case
fails, 0 otherwise.
"""

import argparse
import concurrent.futures
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import pim_ref as R  # noqa: E402

TIMING = ROOT / "libpim/timing"


def timing_model():
    """libpim/timing/pim_timing.py, imported once."""
    if str(TIMING) not in sys.path:
        sys.path.insert(0, str(TIMING))
    import pim_timing

    return pim_timing


# The scalar SB operations carry their column count in the descriptor
# (pim_ref.py writes it as the data of the script line).
SCALAR_SB = ("SbMac", "SbEwadd", "SbEwmul", "SbEwaddGbuf", "SbEwmulGbuf")


def c26_requests(pt, table, script):
    """(operation, model class, columns) of every PIM line of a C26
    script, in order."""
    cb = table["geometry"]["col_bytes"]
    out = []
    for line in open(script):
        f = line.split()
        if len(f) < 4 or not f[1].startswith("pim:"):
            continue
        op, size = f[1][4:], int(f[3])
        if op in SCALAR_SB:
            n = int.from_bytes(bytes.fromhex(f[4]), "little") & 0xFFFF
        else:
            n = size // cb
        out.append((op, pt.OP_CLASS[op], n))
    return out


R2 = ROOT / "build/RISCV_RAMULATOR2"
OPT, FAST, DEBUG = R2 / "gem5.opt", R2 / "gem5.fast", R2 / "gem5.debug"
# The default build without Ramulator 2.1; ``--native`` may name another
# one (TestLib's suites pass build/ALL/gem5.opt).
# The r6 build checks always use RISCV_OPT.
RISCV_OPT = ROOT / "build/RISCV/gem5.opt"
NATIVE = RISCV_OPT
ALL_OPT = ROOT / "build/ALL/gem5.opt"
CPULESS = "tests/gem5/pim/configs/pim_r2_cpuless.py"
REF = "tests/gem5/pim/pim_ref.py"
CTRL = "tests/gem5/pim/check_pim_ctrl.py"
TRACE = "tests/gem5/pim/check_pim_trace.py"
SE_PIM = "tests/gem5/pim/configs/pim_se.py"
FS_DTS = "tests/gem5/pim/configs/pim_fs_dts.py"
EXAMPLE = "configs/example/gem5_library/riscv-pim.py"
STDS = ("LPDDR6", "DDR5", "GDDR7", "HBM4")
FORMATS = (
    ("bf16", "fp32"),
    ("fp32", "fp32"),
    ("int16", "int32"),
    ("int8", "int32"),
    ("fp16", "fp32"),
)
ERROR_CASES = (
    "descriptor",
    "ncols",
    "alignment",
    "odd-bank-sb",
    "odd-bank-ab",
    "window-bank",
    "reg-outside",
)
GROUPS = (
    "cpuless",
    "refresh",
    "c28",
    "path",
    "config",
    "v16",
    "decode",
    "fsbm",
    "cache",
    "se-misc",
    "r2suites",
    "xcheck",
    "r6",
    "fs-boot",
    "fs-way1",
    "fs-way1-whole",
    "fs-protocol",
    "fs-way2",
    "sv48",
    "fs-ddr5",
)
M = "system.memory.ramulator2"
# Groups whose cases run only on the native build (se-misc: with --only).
NATIVE_GROUPS = {"decode", "fsbm", "se-misc"}


class Suite:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out).resolve()
        self.runs = self.out / "runs"
        self.gen = self.out / "gen"
        for d in (self.runs, self.gen):
            d.mkdir(parents=True, exist_ok=True)
        self.groups = set(args.groups.split(","))
        self.only = re.compile(args.only) if args.only else None
        self.tasks = []
        self.checks = []
        self.geo = {}
        self.lock = threading.Lock()

    # Registration.
    def want(self, name):
        return not self.only or self.only.search(name)

    def task(self, name, fn):
        self.tasks.append((name, fn))

    def check(self, name, fn):
        if self.want(name):
            self.checks.append((name, fn))

    # Running.
    def sh(self, cmd, log, cwd=ROOT, env=None):
        with open(log, "w") as f:
            return subprocess.run(
                [str(c) for c in cmd],
                cwd=cwd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            ).returncode

    def g5(self, name, argv, binary=OPT, flags=None, redirect=False):
        """Run gem5 into runs/<name>; the exit status goes to <name>.rc."""
        d = self.runs / name
        cmd = [binary]
        if redirect:
            cmd.append("-re")
        if flags:
            cmd += [f"--debug-flags={flags}", "--debug-file=trace.out"]
        cmd += ["-d", d] + list(argv)
        rc = self.sh(cmd, self.runs / f"{name}.log")
        (self.runs / f"{name}.rc").write_text(f"{rc}\n")
        return rc

    def cpuless(self, name, script, *args, binary=OPT, flags=None):
        return self.g5(
            name,
            [CPULESS, "--memory", "pim", "--script", script] + list(args),
            binary=binary,
            flags=flags,
        )

    # Results.
    def rc(self, name):
        p = self.runs / f"{name}.rc"
        return int(p.read_text()) if p.exists() else None

    def text(self, name):
        """The run's log, simout and simerr."""
        out = ""
        for p in (
            self.runs / f"{name}.log",
            self.runs / name / "simout.txt",
            self.runs / name / "simerr.txt",
        ):
            if p.exists():
                out += p.read_text(errors="replace")
        return out

    def ctrl(self, name, *conds, csv_name=None, lines=None, positive=False):
        """check_pim_ctrl.py on runs/<name>: (ok, its last line)."""
        cmd = [sys.executable, CTRL, self.runs / name]
        if csv_name:
            cmd += ["--csv", csv_name]
        if lines:
            cmd += ["--lines", lines]
        if positive:
            cmd.append("--positive-latency")
        for c in conds:
            cmd += ["--stat", c]
        p = subprocess.run(
            [str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True
        )
        last = (p.stdout.strip().splitlines() or ["(no output)"])[-1]
        return p.returncode == 0, last

    def trace(self, name, *args):
        p = subprocess.run(
            [sys.executable, TRACE, str(self.runs / name)]
            + [str(a) for a in args],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        last = (p.stdout.strip().splitlines() or ["(no output)"])[-1]
        return p.returncode == 0, last

    def stats(self, name):
        st = {}
        p = self.runs / name / "stats.txt"
        if not p.exists():
            return st
        for line in p.open():
            if line.startswith("---------- End"):
                break
            f = line.split()
            if len(f) >= 2 and not line.startswith("-"):
                st[f[0]] = f[1]
        return st

    # Generation.
    def geometry(self, key, *args):
        """The PIM memory's geometry for these CPU-less options."""
        if key not in self.geo:
            d = self.gen / f"geo-{key}"
            out = subprocess.run(
                [str(OPT), "-d", str(d), CPULESS, "--memory", "pim"]
                + list(args)
                + ["--print-geometry"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.geo[key] = out.splitlines()[-1] if out else "{}"
        return self.geo[key]

    def script(self, gen, path, key, *args):
        """A pim_ref.py script for geometry <key>; returns its path."""
        p = self.gen / path
        subprocess.run(
            [sys.executable, REF, gen, str(p), "--geometry", self.geo[key]]
            + list(args),
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        return p

    def g(self, key):
        return R.Geometry(**json.loads(self.geo[key]))


def ok_rc(s, name, want=0):
    rc = s.rc(name)
    return rc == want, f"rc={rc}"


def nonzero_with(s, name, text):
    """The run stopped (exit status not 0) and its output has <text>."""
    rc = s.rc(name)
    m = re.search(re.escape(text) + r".{0,80}", s.text(name))
    return (
        rc not in (0, None) and m is not None,
        f"rc={rc}: {m.group(0) if m else 'message not found'}",
    )


def pim_lines(path):
    return sum(1 for line in open(path) if re.match(r"^\d+ pim:", line))


# ---------------------------------------------------------------------
# cpuless: the CPU-less runs on PimTester.
# ---------------------------------------------------------------------

CPULESS_GEO = {
    "LPDDR6": ("--std", "LPDDR6", "--channels", "2"),
    "DDR5": ("--std", "DDR5", "--channels", "2"),
    "GDDR7": ("--std", "GDDR7", "--channels", "2"),
    "HBM4": ("--std", "HBM4", "--channels", "2"),
    "lp-RoRaBaCoCh": (
        "--std LPDDR6 --channels 2 --policy RoRaBaCoCh --interleave 64"
    ).split(),
    "lp-RoCoRaBaCh": (
        "--std LPDDR6 --channels 2 --policy RoCoRaBaCh --interleave 256"
    ).split(),
    "lp-RoBaRaCoCh": (
        "--std LPDDR6 --channels 2 --policy RoBaRaCoCh --interleave 64"
    ).split(),
    "lp-ChRaBaRoCo": "--std LPDDR6 --channels 2 --policy ChRaBaRoCo".split(),
    "lp1-ChRaBaRoCo": "--std LPDDR6 --channels 1 --policy ChRaBaRoCo".split(),
    "lp-MOP4CLXOR": (
        "--std LPDDR6 --channels 2 --policy MOP4CLXOR --interleave 64"
    ).split(),
    "hbm-RoCoRaBaCh": (
        "--std HBM4 --channels 2 --policy RoCoRaBaCh --interleave 128"
    ).split(),
    "lp1": "--std LPDDR6 --channels 1".split(),
    "lp1-MOP4CLXOR": "--std LPDDR6 --channels 1 --policy MOP4CLXOR".split(),
    # C16's DDR5 case needs two units per channel: two ranks (the default
    # DDR5 has one; --ranks sets the count).
    "DDR5-2r": "--std DDR5 --channels 2 --ranks 2".split(),
}
LP = CPULESS_GEO["LPDDR6"]
# C16: units 0 and 1 of channel 0.
C16_GEO = {"DDR5": "DDR5-2r", "HBM4": "HBM4"}


def prepare_cpuless(s):
    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        list(
            ex.map(
                lambda kv: s.geometry(*[kv[0]] + list(kv[1])),
                CPULESS_GEO.items(),
            )
        )
    jobs = []
    for k in CPULESS_GEO:
        if k not in ("lp1", "lp1-MOP4CLXOR", "DDR5-2r"):
            jobs.append(("c1", f"c1-{k}.txt", k))
    for n, a in FORMATS:
        for k in ("LPDDR6", "DDR5"):
            jobs.append(
                (
                    "c1",
                    f"c20-{k}-{n}-{a}.txt",
                    k,
                    "--num-fmt",
                    n,
                    "--acc-fmt",
                    a,
                )
            )
    jobs += [
        ("c1", "c1-LPDDR6-off.txt", "LPDDR6", "--no-values"),
        ("c14", "c14-on.txt", "LPDDR6"),
        ("c14", "c14-off.txt", "LPDDR6", "--no-values"),
        ("errors", "c15-all.txt", "LPDDR6"),
    ]
    jobs += [
        ("errors", f"c15-{c}.txt", "LPDDR6", "--case", c) for c in ERROR_CASES
    ]
    for c in ("c3", "c4", "c5", "c6", "c7", "c8", "c9", "c18", "c23-order"):
        jobs.append((c, f"{c}.txt", "LPDDR6"))
    jobs += [("c8", f"c8-{k}.txt", k) for k in ("DDR5", "GDDR7", "HBM4")]
    jobs += [("c16", f"c16-{k}.txt", C16_GEO[k]) for k in C16_GEO]
    jobs += [
        ("host", "host-u0.txt", "LPDDR6", "--count", "1500", "--unit", "0"),
        ("host", "host-u1.txt", "LPDDR6", "--count", "600", "--unit", "1"),
        ("c24-pim", "c24-pim.txt", "LPDDR6"),
    ]
    for k in STDS:
        jobs += [
            ("c13", f"c13-{k}.txt", k),
            ("c25", f"c25-{k}.txt", k),
            ("c26", f"c26-{k}.txt", k),
        ]
    # C26 on two DDR5 ranks.
    jobs.append(("c26", "c26-DDR5-2r.txt", "DDR5-2r"))
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        list(ex.map(lambda j: s.script(*j), jobs))
    subprocess.run(
        [
            sys.executable,
            REF,
            "rdwr",
            str(s.gen / "rdwr.txt"),
            "--size",
            "0x20000000",
            "--seed",
            "3",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    g = s.g("LPDDR6")
    # C15 with two registers (EW refused) and across banks; C21 (normal
    # requests at a register row); C22 (PIM at a plain memory).
    p = R.PimScript(g, num_regs=2)
    p.pim(R.SB_EWADD, p.addr(0, 0, 1, 0), desc=1)
    p.pim(R.RD_REG, p.addr(0, 0, p.acc_row(), 0), g.col_bytes)
    p.drain()
    p.write(str(s.gen / "c15-regs.txt"))
    for k in ("lp1", "lp1-MOP4CLXOR"):
        p = R.PimScript(s.g(k))
        R.crossing_cases(p)
        p.write(str(s.gen / f"c15-cross-{k}.txt"))
        (s.gen / f"c15-cross-{k}.errors").write_text(f"{p.m.errors}\n")
    p = R.PimScript(g)
    a = p.addr(0, 3, p.reg_row(), 2)
    p.s.add(0, "rd", a, 64, None, bytes(64))
    p.s.add(0, "wr", a, 64, bytes(range(64)))
    p.drain()
    p.write(str(s.gen / "c21.txt"))
    sc = R.Script()
    sc.add(0, "pim:RdGb", 0, 32, None, bytes(32))
    sc.write(str(s.gen / "c22.txt"))
    # C23: host reads of row 1 held behind the AB MACs.
    p = R.c23_pim_script(g)
    p.write(str(s.gen / "c23-pim.txt"))
    h = R.PimScript(g)
    h.m = p.m
    n = min(64, h.contig())
    import random

    rng = random.Random(15)
    for i in range(60):
        h.rd(
            h.addr(
                0,
                rng.randrange(h.banks()),
                1,
                rng.randrange(0, g.cols, n // g.col_bytes),
            ),
            n,
            500 if i == 0 else 0,
        )
    h.drain()
    h.write(str(s.gen / "c23-host.txt"))
    e = R.Script()
    e.add(1000, "drain")
    e.write(str(s.gen / "empty.txt"))


# r3-swap: uncacheable atomic swaps (AMOs) at a row, then
# pim.sb.mov.gbufbk of that row, then more swaps, and a checkpoint at the
# end (on gem5.debug).
SWAP_SCRIPT = (
    "0 wr 0x1000 8 1111111111111111\n"
    "100000 swap 0x1000 8 2222222222222222 1111111111111111\n"
    "100000 pim:SbMovGbufbk 0x1000 32\n"
    "100000 swap 0x1000 8 3333333333333333 0000000000000000\n"
    "100000 swap 0x1000 8 4444444444444444 3333333333333333\n"
    "100000 rd 0x1000 8 - 4444444444444444\n"
)


def tasks_cpuless(s):
    tasks_swap(s)
    tasks_aespa_pair(s)
    G = s.gen
    t = s.task
    geo = CPULESS_GEO

    def at(name, script, *args, **kw):
        if s.want(name):
            t(name, lambda: s.cpuless(name, script, *args, **kw))

    # Atomic.
    for k in (
        "LPDDR6",
        "DDR5",
        "GDDR7",
        "HBM4",
        "lp-RoRaBaCoCh",
        "lp-RoCoRaBaCh",
        "lp-RoBaRaCoCh",
        "lp-ChRaBaRoCo",
        "lp1-ChRaBaRoCo",
        "lp-MOP4CLXOR",
        "hbm-RoCoRaBaCh",
    ):
        at(f"c1-{k}-atomic", G / f"c1-{k}.txt", *geo[k], "--mode", "atomic")
    at(
        "c1-LPDDR6-off-atomic",
        G / "c1-LPDDR6-off.txt",
        *LP,
        "--mode",
        "atomic",
        "--no-values",
    )
    at(
        "c19",
        G / "c1-LPDDR6.txt",
        *LP,
        "--mode",
        "atomic_noncaching",
        "--use-backdoor",
    )
    for n, a in FORMATS:
        for k in ("LPDDR6", "DDR5"):
            at(
                f"c20-{k}-{n}-{a}",
                G / f"c20-{k}-{n}-{a}.txt",
                *geo[k],
                "--mode",
                "atomic",
                "--num-fmt",
                n,
                "--acc-fmt",
                a,
            )
    at(
        "c20-invalid",
        G / "c1-LPDDR6.txt",
        *LP,
        "--mode",
        "atomic",
        "--num-fmt",
        "fp32",
        "--acc-fmt",
        "bf16",
    )
    at("c14-on", G / "c14-on.txt", *LP, "--mode", "atomic")
    at("c14-off", G / "c14-off.txt", *LP, "--mode", "atomic", "--no-values")
    at(
        "c15-all",
        G / "c15-all.txt",
        *LP,
        "--mode",
        "atomic",
        "--sw-errors-counted",
    )
    for c in ERROR_CASES:
        at(f"c15-{c}", G / f"c15-{c}.txt", *LP, "--mode", "atomic")
    at(
        "c15-regs",
        G / "c15-regs.txt",
        *LP,
        "--mode",
        "atomic",
        "--num-regs",
        "2",
        "--sw-errors-counted",
    )
    at(
        "c15-regs-fatal",
        G / "c15-regs.txt",
        *LP,
        "--mode",
        "atomic",
        "--num-regs",
        "2",
    )
    for k in ("lp1", "lp1-MOP4CLXOR"):
        at(
            f"c15-cross-{k}",
            G / f"c15-cross-{k}.txt",
            *geo[k],
            "--mode",
            "atomic",
            "--sw-errors-counted",
        )
    at("c21", G / "c21.txt", *LP, "--mode", "atomic", "--sw-errors-counted")
    at("c21-fatal", G / "c21.txt", *LP, "--mode", "atomic")
    at("c21-timing", G / "c21.txt", *LP)
    for name, mode in (("c22", ["--mode", "atomic"]), ("c22-timing", [])):
        if s.want(name):
            t(
                name,
                lambda name=name, mode=mode: s.g5(
                    name,
                    [
                        CPULESS,
                        "--memory",
                        "ramulator2",
                        "--std",
                        "LPDDR6",
                        "--channels",
                        "2",
                        "--script",
                        G / "c22.txt",
                    ]
                    + mode,
                ),
            )
    for m in ("timing", "atomic"):
        at(f"rdwr-{m}", G / "rdwr.txt", *LP, "--mode", m)
        for k in ("DDR4", "LPDDR6", "HBM4"):
            name = f"rdwr-plain-{k}-{m}"
            if s.want(name):
                t(
                    name,
                    lambda name=name, k=k, m=m: s.g5(
                        name,
                        [
                            CPULESS,
                            "--std",
                            k,
                            "--channels",
                            "2",
                            "--mode",
                            m,
                            "--script",
                            G / "rdwr.txt",
                        ],
                    ),
                )
    # Timing.
    for k in (
        "LPDDR6",
        "DDR5",
        "GDDR7",
        "HBM4",
        "lp-RoRaBaCoCh",
        "lp-MOP4CLXOR",
    ):
        at(f"c1-{k}", G / f"c1-{k}.txt", *geo[k])
    for m in ("ideal", "aim", "aespa"):
        at(f"c1-lp-{m}", G / "c1-LPDDR6.txt", *LP, "--ab-mode", m)
    at("c7-posted", G / "c7.txt", *LP)
    at("c7-nonposted", G / "c7.txt", *LP, "--no-posted")
    at("c12-off", G / "c1-LPDDR6-off.txt", *LP, "--no-values")
    at("c3", G / "c3.txt", *LP, "--trace", "--host-script", G / "host-u0.txt")
    at("c4", G / "c4.txt", *LP)
    at("c5", G / "c5.txt", *LP, "--trace")
    at("c6", G / "c6.txt", *LP, "--host-script", G / "host-u0.txt")
    for m in ("ideal", "aim", "baseline", "aespa"):
        at(
            f"c8-LPDDR6-{m}",
            G / "c8.txt",
            *LP,
            "--ab-mode",
            m,
            "--close-rows",
            "--refresh",
            "NoRefresh",
        )
        for k in ("DDR5", "GDDR7", "HBM4"):
            at(
                f"c8-{k}-{m}",
                G / f"c8-{k}.txt",
                *geo[k],
                "--ab-mode",
                m,
                "--close-rows",
                "--refresh",
                "NoRefresh",
            )
    for p in ("frfcfs", "chopim", "asyncdimm"):
        at(
            f"c9-{p}",
            G / "c9.txt",
            *LP,
            "--arb-policy",
            p,
            "--host-script",
            G / "host-u0.txt",
            "--trace",
        )
    at("c9-cosm", G / "c9.txt", *LP, "--arb-policy", "cosm")
    at(
        "c29",
        G / "c9.txt",
        *LP,
        "--arb-policy",
        "asyncdimm",
        "--row-policy",
        "ClosedCAP",
        "--host-script",
        G / "host-u0.txt",
        "--trace",
    )
    for k in C16_GEO:
        at(f"c16-{k}", G / f"c16-{k}.txt", *geo[C16_GEO[k]], "--no-posted")
        at(
            f"c16-{k}-full",
            G / f"c16-{k}.txt",
            *geo[C16_GEO[k]],
            "--no-posted",
            "--pim-queue-size",
            "4",
        )
    at("c18", G / "c18.txt", *LP)
    at(
        "c23",
        G / "c23-pim.txt",
        *LP,
        "--read-buffer",
        "4",
        "--write-buffer",
        "4",
        "--host-script",
        G / "c23-host.txt",
    )
    at("c23-order", G / "c23-order.txt", *LP, "--trace", flags="PIM")
    at("c24-with", G / "c24-pim.txt", *LP, "--host-script", G / "host-u1.txt")
    at("c24-without", G / "empty.txt", *LP, "--host-script", G / "host-u1.txt")
    # C13 and C25: save, then restore (timing and atomic).
    for k in STDS:
        name = f"c13-{k}"
        if s.want(name):
            cpt = s.runs / f"c13-cpt-{k}"

            def c13(k=k, cpt=cpt):
                a = list(geo[k])
                s.cpuless(
                    f"c13a-{k}",
                    G / f"c13-{k}.txt",
                    *a,
                    "--checkpoint-at-end",
                    "--checkpoint-dir",
                    cpt,
                )
                s.cpuless(
                    f"c13b-{k}",
                    G / f"c13-{k}.txt.restore",
                    *a,
                    "--restore",
                    cpt,
                )
                s.cpuless(
                    f"c13c-{k}",
                    G / f"c13-{k}.txt.restore",
                    *a,
                    "--mode",
                    "atomic",
                    "--restore",
                    cpt,
                )
                if k == "LPDDR6":
                    # Without the PIM state: zeros and a warning.
                    nost = s.runs / "c13-cpt-nostate"
                    shutil.rmtree(nost, ignore_errors=True)
                    shutil.copytree(cpt, nost)
                    f = nost / "m5.cpt"
                    f.write_text(
                        "".join(
                            x
                            for x in f.read_text().splitlines(True)
                            if not x.startswith("pim_")
                        )
                    )
                    s.cpuless(
                        "c13-nostate",
                        G / f"c13-{k}.txt.restore",
                        *a,
                        "--restore",
                        nost,
                    )
                    # Another operand format must not restore.
                    s.cpuless(
                        "c13-fmt",
                        G / f"c13-{k}.txt.restore",
                        *a,
                        "--num-fmt",
                        "fp16",
                        "--restore",
                        cpt,
                    )

            t(name, c13)
        name = f"c25-{k}"
        if s.want(name):
            cpt = s.runs / f"c25-cpt-{k}"
            # DDR5's 8 KiB rows keep the crossbar busy longer: its reads
            # reach the memory later, so its checkpoint comes later.
            when = "400000" if k == "DDR5" else "200000"

            def c25(k=k, cpt=cpt, when=when):
                a = list(geo[k])
                s.cpuless(
                    f"c25a-{k}",
                    G / f"c25-{k}.txt",
                    *a,
                    "--checkpoint-at",
                    when,
                    "--checkpoint-dir",
                    cpt,
                    flags="Drain",
                )
                s.cpuless(
                    f"c25b-{k}",
                    G / f"c25-{k}.txt.restore",
                    *a,
                    "--restore",
                    cpt,
                )

            t(name, c25)
    # C26 on LPDDR6 under ClosedCAP without --close-rows; the
    # sequencer must close rows as with --close-rows.
    cc = list(geo["LPDDR6"]) + [
        "--row-policy",
        "ClosedCAP",
        "--refresh",
        "NoRefresh",
        "--no-posted",
    ]
    at(
        "c26cc-LPDDR6-atomic",
        G / "c26-LPDDR6.txt",
        *cc,
        "--mode",
        "atomic",
        flags="PIM",
    )
    for m in ("ideal", "aim", "baseline", "aespa"):
        at(
            f"c26cc-LPDDR6-{m}",
            G / "c26-LPDDR6.txt",
            *cc,
            "--ab-mode",
            m,
            flags="PIM",
        )
    # C26 and C2.
    for k in STDS:
        a = list(geo[k]) + [
            "--close-rows",
            "--refresh",
            "NoRefresh",
            "--no-posted",
        ]
        at(
            f"c26-{k}-atomic",
            G / f"c26-{k}.txt",
            *a,
            "--mode",
            "atomic",
            flags="PIM",
        )
        for m in ("ideal", "aim", "baseline", "aespa"):
            at(
                f"c26-{k}-{m}",
                G / f"c26-{k}.txt",
                *a,
                "--ab-mode",
                m,
                flags="PIM",
            )
    # The same requests on two DDR5 ranks; isolated
    # requests never switch ranks, so the one-rank table applies.
    for m in ("ideal", "aim", "baseline", "aespa"):
        at(
            f"c26-DDR5-2r-{m}",
            G / "c26-DDR5-2r.txt",
            *geo["DDR5-2r"],
            "--close-rows",
            "--refresh",
            "NoRefresh",
            "--no-posted",
            "--ab-mode",
            m,
            flags="PIM",
        )


# r5-pair: two AB MAC jobs on one row; under
# AESPA the second finds its row open and must not wait for a refresh.
AESPA_PAIR = "0 pim:AbMac 0x0 64\n0 pim:AbMac 0x40 64\n"


def tasks_aespa_pair(s):
    script = s.gen / "r5-aespa-pair.txt"
    script.write_text(AESPA_PAIR)
    for m in ("aespa", "baseline"):
        name = f"r5-pair-{m}"
        if s.want(name) or s.want("r5-pair"):
            s.task(
                name,
                lambda name=name, m=m: s.cpuless(
                    name, script, *LP, "--ab-mode", m, "--no-posted"
                ),
            )


def check_aespa_pair(s):
    """AESPA may take longer than Baseline only by its per-bank columns:
    banks x columns x the SB read spacing of timing_lpddr6.json."""
    t = timing_model().load("LPDDR6")

    def end(name):
        rows = list(csv.DictReader(open(s.runs / name / "pim_tester.csv")))
        return max(int(r["resp_tick"]) for r in rows), len(rows)

    (a, na), (b, nb) = end("r5-pair-aespa"), end("r5-pair-baseline")
    g, d = t["geometry"], t["derived"]
    slack = g["banks_per_unit"] * 4 * d["sb_rd_spacing"] * t["clock"]["tCK_ps"]
    return (
        s.rc("r5-pair-aespa") == 0
        and s.rc("r5-pair-baseline") == 0
        and na == nb == 2
        and a <= b + slack,
        f"last response: AESPA {a} ps, Baseline {b} ps (allowed "
        f"difference {slack} ps)",
    )


def tasks_swap(s):
    if not DEBUG.exists() or not s.want("r3-swap"):
        return
    script = s.gen / "r3-swap.txt"
    script.write_text(SWAP_SCRIPT)
    s.task(
        "r3-swap",
        lambda: s.cpuless(
            "r3-swap",
            script,
            *LP,
            "--checkpoint-at-end",
            "--checkpoint-dir",
            s.runs / "r3-swap-cpt",
            binary=DEBUG,
        ),
    )


def checks_cpuless(s):
    G = s.gen
    c = s.check
    c("r5-pair", lambda: check_aespa_pair(s))
    if DEBUG.exists():
        c(
            "r3-swap",
            lambda: (
                s.rc("r3-swap") == 0
                and "panic" not in s.text("r3-swap")
                and (s.runs / "r3-swap-cpt" / "m5.cpt").exists(),
                f"rc={s.rc('r3-swap')} "
                + (
                    re.search(r"panic: .{0,100}", s.text("r3-swap")).group(0)
                    if "panic" in s.text("r3-swap")
                    else "no panic, swaps as expected, checkpoint written"
                ),
            ),
        )
    else:
        c("r3-swap", lambda: (None, f"SKIP: {DEBUG} not found"))

    def ctrl_ok(name, *conds, **kw):
        def fn():
            rc = s.rc(name)
            ok, line = s.ctrl(name, *conds, **kw)
            return rc == 0 and ok, f"rc={rc} {line}"

        return fn

    for k in (
        "LPDDR6",
        "DDR5",
        "GDDR7",
        "HBM4",
        "lp-RoRaBaCoCh",
        "lp-RoCoRaBaCh",
        "lp-RoBaRaCoCh",
        "lp-ChRaBaRoCo",
        "lp1-ChRaBaRoCo",
        "lp-MOP4CLXOR",
        "hbm-RoCoRaBaCh",
    ):
        n = pim_lines(G / f"c1-{k}.txt")
        c(
            f"c1-{k}-atomic",
            ctrl_ok(
                f"c1-{k}-atomic",
                "system.tester.mismatches=0",
                f"{M}.pim.reqs::total={n}",
                f"{M}.pim.atomicReqs::total={n}",
                positive=True,
            ),
        )
    c(
        "c1-LPDDR6-off-atomic",
        ctrl_ok(
            "c1-LPDDR6-off-atomic",
            "system.tester.mismatches=0",
            f"{M}.pim.reqs::total={pim_lines(G / 'c1-LPDDR6-off.txt')}",
        ),
    )
    c(
        "c19",
        ctrl_ok(
            "c19",
            "system.tester.mismatches=0",
            "system.tester.backdoors=0",
            positive=True,
        ),
    )
    for n, a in FORMATS:
        for k in ("LPDDR6", "DDR5"):
            c(
                f"c20-{k}-{n}-{a}",
                ctrl_ok(f"c20-{k}-{n}-{a}", "system.tester.mismatches=0"),
            )
    c("c20-invalid", lambda: nonzero_with(s, "c20-invalid", "fatal:"))
    for n in ("c14-on", "c14-off"):
        c(
            n,
            ctrl_ok(
                n,
                "system.tester.mismatches=0",
                f"{M}.pim.regWindowFunctional>0",
            ),
        )
    c(
        "c15-all",
        ctrl_ok(
            "c15-all",
            "system.tester.mismatches=0",
            f"{M}.pim.swErrors=7",
            f"{M}.pim.reqs::total=2",
        ),
    )
    for e in ERROR_CASES:
        c(
            f"c15-{e}",
            lambda e=e: nonzero_with(s, f"c15-{e}", "PIM software error:"),
        )
    c(
        "c15-regs",
        ctrl_ok(
            "c15-regs", "system.tester.mismatches=0", f"{M}.pim.swErrors=1"
        ),
    )
    c(
        "c15-regs-fatal",
        lambda: nonzero_with(s, "c15-regs-fatal", "PIM software error:"),
    )
    for k in ("lp1", "lp1-MOP4CLXOR"):
        e = int((G / f"c15-cross-{k}.errors").read_text())
        c(
            f"c15-cross-{k}",
            ctrl_ok(
                f"c15-cross-{k}",
                "system.tester.mismatches=0",
                f"{M}.pim.swErrors={e}",
            ),
        )
    c(
        "c21",
        ctrl_ok("c21", "system.tester.mismatches=0", f"{M}.pim.swErrors=2"),
    )
    for n in ("c21-fatal", "c21-timing"):
        c(n, lambda n=n: nonzero_with(s, n, "PIM software error:"))
    for n in ("c22", "c22-timing"):
        c(n, lambda n=n: nonzero_with(s, n, "fatal:"))
    for m in ("timing", "atomic"):
        c(
            f"rdwr-{m}",
            ctrl_ok(f"rdwr-{m}", "system.tester.mismatches=0", lines="753"),
        )
        for k in ("DDR4", "LPDDR6", "HBM4"):
            n = f"rdwr-plain-{k}-{m}"
            c(n, ctrl_ok(n, "system.tester.mismatches=0", lines="753"))
    for k in (
        "LPDDR6",
        "DDR5",
        "GDDR7",
        "HBM4",
        "lp-RoRaBaCoCh",
        "lp-MOP4CLXOR",
    ):
        c(
            f"c1-{k}",
            ctrl_ok(
                f"c1-{k}",
                "system.tester.mismatches=0",
                f"{M}.pim.reqs::total={pim_lines(G / f'c1-{k}.txt')}",
            ),
        )
    for m in ("ideal", "aim", "aespa"):
        c(f"c1-lp-{m}", ctrl_ok(f"c1-lp-{m}", "system.tester.mismatches=0"))
    c("c17-DDR5", ctrl_ok("c1-DDR5", f"{M}.pim.partialCols>0"))

    def with_trace(name, conds, *targs):
        def fn():
            rc = s.rc(name)
            ok1, l1 = s.ctrl(name, *conds)
            ok2, l2 = s.trace(name, *targs)
            return rc == 0 and ok1 and ok2, f"rc={rc} {l1} | {l2}"

        return fn

    c(
        "c3",
        with_trace(
            "c3",
            [
                "system.tester.mismatches=0",
                "system.host.mismatches=0",
                f"{M}.pim.hazardWaitTicks::SbMac>0",
            ],
            "after-host-write",
            0,
        ),
    )
    c(
        "c4",
        ctrl_ok(
            "c4",
            "system.tester.mismatches=0",
            f"{M}.pim.heldPkts=2",
            f"{M}.pim.heldTicks>0",
            f"{M}.channel0.num_read_reqs_forwarded=0",
        ),
    )
    c(
        "c5",
        with_trace(
            "c5",
            [
                "system.tester.mismatches=0",
                f"{M}.pim.heldPkts>0",
                f"{M}.channel0.num_write_reqs_coalesced=0",
            ],
            "count",
            0,
            1,
            "WR*",
            "=",
            4,
        ),
    )
    c(
        "c6",
        ctrl_ok(
            "c6",
            "system.tester.mismatches=0",
            "system.host.mismatches=0",
            f"{M}.pim.writeDrainTicks>0",
        ),
    )

    def c7():
        rcs = (s.rc("c7-posted"), s.rc("c7-nonposted"))
        ok1, _ = s.ctrl("c7-posted", "system.tester.mismatches=0")
        ok2, _ = s.ctrl("c7-nonposted", "system.tester.mismatches=0")

        def lat(d):
            out = {}
            for r in csv.DictReader(open(s.runs / d / "pim_tester.csv")):
                if r["kind"].startswith("pim:") and r["kind"] not in (
                    "pim:RdGb",
                    "pim:RdReg",
                ):
                    out[r["line"]] = int(r["resp_tick"]) - int(r["send_tick"])
            return out

        size = {
            r["line"]: int(r["size"])
            for r in csv.DictReader(
                open(s.runs / "c7-posted" / "pim_tester.csv")
            )
        }
        p, n = lat("c7-posted"), lat("c7-nonposted")
        # Posted: the static frontend latency (10 ns) plus the crossbar's
        # header and payload delays (1 GHz, 64 bytes wide).
        bound = all(
            10000 <= p[k] <= 10000 + 1001 + 1000 * -(-size[k] // 64) for k in p
        )
        larger = all(n[k] > p[k] for k in p)
        return (
            rcs == (0, 0) and ok1 and ok2 and bound and larger and p,
            f"rc={rcs} posted within 10 ns + crossbar {bound}, "
            f"non-posted larger {larger} ({len(p)} store-like)",
        )

    c("c7", c7)
    for k in STDS:
        g = json.loads(s.geo[k])
        b = g["sids"] * g["bgs"] * g["banks"]
        P = f"{M}.channel0.controller_plugin"
        c(
            f"c8-{k}-ideal",
            ctrl_ok(
                f"c8-{k}-ideal",
                f"{P}.pim_ab_acts=1",
                f"{P}.pim_ab_pres=1",
                f"{P}.pim_acts=0",
                f"{P}.pim_pres=0",
            ),
        )
        c(
            f"c8-{k}-aim",
            ctrl_ok(
                f"c8-{k}-aim",
                f"{P}.pim_ab_acts=1",
                f"{P}.pim_pres={b}",
                f"{P}.pim_ab_pres=0",
            ),
        )
        for m in ("baseline", "aespa"):
            c(
                f"c8-{k}-{m}",
                ctrl_ok(
                    f"c8-{k}-{m}",
                    f"{P}.pim_acts={b}",
                    f"{P}.pim_pres={b}",
                    f"{P}.pim_ab_acts=0",
                ),
            )
    P = f"{M}.channel0.controller_plugin*"
    c(
        "c9-frfcfs",
        ctrl_ok(
            "c9-frfcfs",
            "system.tester.mismatches=0",
            "system.host.mismatches=0",
            f"{P}.pim_interrupts=0",
        ),
    )
    c(
        "c9-chopim",
        ctrl_ok(
            "c9-chopim",
            "system.tester.mismatches=0",
            "system.host.mismatches=0",
            f"{P}.pim_interrupts>0",
        ),
    )

    def switches(name):
        def fn():
            d = s.runs / name

            def last(f):
                return max(
                    int(r["send_tick"])
                    for r in csv.DictReader(open(d / f))
                    if r["kind"] != "drain"
                )

            until = min(last("host.csv"), last("pim_tester.csv")) // 375
            rc = s.rc(name)
            ok1, l1 = s.ctrl(
                name, "system.tester.mismatches=0", "system.host.mismatches=0"
            )
            ok2, l2 = s.trace(name, "switches", 0, until)
            return rc == 0 and ok1 and ok2, f"rc={rc} {l1} | {l2}"

        return fn

    c("c9-asyncdimm", switches("c9-asyncdimm"))
    c("c29", switches("c29"))
    c("c9-cosm", lambda: nonzero_with(s, "c9-cosm", "pending its definition"))

    def c12():
        rc = s.rc("c12-off")
        ok1, l1 = s.ctrl("c12-off", "system.tester.mismatches=0")
        p = subprocess.run(
            [
                sys.executable,
                "util/ramulator2_stats_diff.py",
                str(s.runs / "c1-LPDDR6"),
                str(s.runs / "c12-off"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        last = (p.stdout.strip().splitlines() or [""])[-1]
        return (
            rc == 0 and ok1 and p.returncode == 0,
            f"rc={rc} {l1} | values on and off equal: {last}",
        )

    c("c12", c12)
    for k in C16_GEO:

        def c16(k=k):
            rc = s.rc(f"c16-{k}")
            ok1, l1 = s.ctrl(f"c16-{k}", "system.tester.mismatches=0")
            g = R.Geometry(**json.loads(s.geo[C16_GEO[k]]))
            done = {0: [], 1: []}
            for x in csv.DictReader(
                open(s.runs / f"c16-{k}" / "pim_tester.csv")
            ):
                if x["kind"] == "pim:SbMac":
                    co, _ = g.decode(int(x["addr"], 16))
                    done[co.ch * g.units + co.unit].append(int(x["resp_tick"]))
            inorder = all(
                a < b for u in done for a, b in zip(done[u], done[u][1:])
            )
            overlap = any(
                t1 < t0 for i, t0 in enumerate(done[0]) for t1 in done[1][:i]
            )
            return (
                rc == 0 and ok1 and inorder and overlap,
                f"rc={rc} {l1} | in order per unit {inorder}, "
                f"overlap across units {overlap}",
            )

        c(f"c16-{k}", c16)
        c(
            f"c16-{k}-full",
            ctrl_ok(
                f"c16-{k}-full",
                "system.tester.mismatches=0",
                f"{M}.pim.numRetry>0",
            ),
        )
    c("c18", ctrl_ok("c18", "system.tester.mismatches=0"))
    c(
        "c23",
        ctrl_ok(
            "c23",
            "system.tester.mismatches=0",
            "system.host.mismatches=0",
            f"{M}.pim.holdFull>0",
            f"{M}.pim.heldPkts>0",
        ),
    )

    def c23_order():
        d = s.runs / "c23-order"
        rc = s.rc("c23-order")
        ok1, l1 = s.ctrl("c23-order", "system.tester.mismatches=0")
        disp = [
            int(m.group(1))
            for m in re.finditer(
                r"^\s*(\d+): \S+: dispatch SbMac",
                (d / "trace.out").read_text(),
                re.M,
            )
        ]
        rd = [
            int(r[0])
            for r in csv.reader(open(d / "cmd_trace.csv.ch0"))
            if r[0] != "clock" and r[-2] == "0" and r[1].startswith("RD")
        ]
        lastrd = max(rd) * 375  # LPDDR6 clock period, ps
        # The ordering rule (no older conflicting normal request
        # outstanding) releases the PIM request at the reads' RD
        # commands, not at their data (one-cycle trace offset allowed).
        ok = bool(disp) and -375 <= disp[0] - lastrd <= 4 * 375
        return (
            rc == 0 and ok1 and ok,
            f"rc={rc} {l1} | SbMac dispatched "
            f"{disp[0] - lastrd if disp else None} ps after the older "
            f"reads' last RD",
        )

    c("c23-order", c23_order)

    def c24():
        rcs = (s.rc("c24-with"), s.rc("c24-without"))
        ok1, l1 = s.ctrl(
            "c24-with", "system.host.mismatches=0", csv_name="host.csv"
        )

        def lat(d):
            return [
                int(r["resp_tick"]) - int(r["send_tick"])
                for r in csv.DictReader(open(s.runs / d / "host.csv"))
            ]

        a, b = lat("c24-with"), lat("c24-without")
        return (
            rcs == (0, 0) and ok1 and a == b and a,
            f"rc={rcs} channel-1 host latencies identical with and "
            f"without channel-0 PIM: {a == b} ({len(a)} lines)",
        )

    c("c24", c24)
    for k in STDS:

        def c13(k=k):
            log = s.text(f"c13a-{k}")
            m = re.search(r"the drain took (\d+)", log)
            took = int(m.group(1)) if m else 0
            r = [s.rc(f"c13{x}-{k}") for x in "abc"]
            ok1, l1 = s.ctrl(f"c13b-{k}", "system.tester.mismatches=0")
            ok2, l2 = s.ctrl(f"c13c-{k}", "system.tester.mismatches=0")
            return (
                r == [0, 0, 0] and took > 0 and ok1 and ok2,
                f"rc={r} drain {took} ticks | restore {l1} | " f"atomic {l2}",
            )

        c(f"c13-{k}", c13)

        def c25(k=k):
            tr = (s.runs / f"c25a-{k}" / "trace.out").read_text().splitlines()
            lines = [x for x in tr if "PIM drain" in x]
            num = re.compile(
                r"(\d+) (?:PIM requests|held|released|busy|outstanding)"
            )
            first = [int(x) for x in num.findall(lines[0])] if lines else []
            last = [int(x) for x in num.findall(lines[-1])] if lines else []
            good = (
                bool(lines)
                and "draining" in lines[0]
                and first[0] > 0
                and first[1] > 0
                and "drained" in lines[-1]
                and not any(last)
            )
            r = [s.rc(f"c25{x}-{k}") for x in "ab"]
            ok1, _ = s.ctrl(f"c25a-{k}", "system.tester.mismatches=0")
            ok2, _ = s.ctrl(f"c25b-{k}", "system.tester.mismatches=0")
            return (
                r == [0, 0] and good and ok1 and ok2,
                f"rc={r} at the request {first}, at the end {last}",
            )

        c(f"c25-{k}", c25)

    def nostate():
        t = s.text("c13-nostate")
        return (
            "holds no PIM state" in t,
            f"rc={s.rc('c13-nostate')} (mismatches expected), "
            f"warning {'present' if 'holds no PIM state' in t else 'missing'}",
        )

    c("c13-nostate", nostate)

    def c26():
        bad = total = 0
        failed = []
        for k in STDS:
            for m in ("ideal", "aim", "baseline", "aespa"):
                name = f"c26-{k}-{m}"
                log = s.text(name)
                ps = int(
                    re.search(r"DRAM clock period (\d+) ps", log).group(1)
                )
                if "PimTester finished (code 0)" not in log:
                    failed.append(name)
                for x in re.finditer(
                    r"done (\S+) \(seq \d+\) after \d+ ticks, clocks "
                    r"(\d+) to (\d+), U (\d+) ticks",
                    (s.runs / name / "trace.out").read_text(),
                ):
                    total += 1
                    bad += (int(x.group(3)) - int(x.group(2))) * ps != int(
                        x.group(4)
                    )
        return (
            bad == 0 and total > 0 and not failed,
            f"{total} requests, {bad} with U != the timing-mode "
            f"latency, failed runs {failed or 'none'}",
        )

    c("c26", c26)

    def c2():
        n = b = 0
        failed = []
        for k in STDS:
            at = [
                int(x.group(1))
                for x in re.finditer(
                    r"atomic \S+ at \S+ size \d+: U (\d+) ticks",
                    (s.runs / f"c26-{k}-atomic" / "trace.out").read_text(),
                )
            ]
            ps = int(
                re.search(
                    r"DRAM clock period (\d+) ps", s.text(f"c26-{k}-baseline")
                ).group(1)
            )
            tm = [
                ((int(x.group(2)) - int(x.group(1))) * ps, int(x.group(1)))
                for x in re.finditer(
                    r"done \S+ \(seq \d+\) after \d+ ticks, clocks (\d+) "
                    r"to (\d+)",
                    (s.runs / f"c26-{k}-baseline" / "trace.out").read_text(),
                )
            ]
            if "PimTester finished (code 0)" not in s.text(
                f"c26-{k}-atomic"
            ) or len(at) != len(tm):
                failed.append(k)
            for u, (tt, a) in zip(at, tm):
                # HBM4: only requests whose first push fell on a rising
                # edge, as U assumes.
                if k == "HBM4" and (a + 2) % 2 != 1:
                    continue
                n += 1
                b += u != tt
        return (
            b == 0 and n > 0 and not failed,
            f"{n} atomic requests compared, {b} differ from timing, "
            f"failed {failed or 'none'}",
        )

    c("c2", c2)

    def c26_model(runs=tuple((k, k) for k in STDS)):
        # Every request's acceptance-to-Done clocks equal the
        # model's U on timing_<std>.json plus the issue term, counted once.
        # runs: (run key, standard of the table).
        pt = timing_model()
        bad = total = 0
        first = []
        for k, std in runs:
            t = pt.load(std)
            reqs = c26_requests(pt, t, G / f"c26-{k}.txt")
            mul = t["clock"]["tick_multiplier"]
            issue = t["derived"]["issue_offset"]
            for m in pt.MODES:
                done = re.findall(
                    r"done (\S+) \(seq \d+\) after \d+ ticks, clocks "
                    r"(\d+) to (\d+)",
                    (s.runs / f"c26-{k}-{m}" / "trace.out").read_text(),
                )
                if len(done) != len(reqs):
                    return False, (
                        f"c26-{k}-{m}: {len(done)} requests done, "
                        f"{len(reqs)} in the script"
                    )
                for (op, cls, n), (op2, a, e) in zip(reqs, done):
                    mode = m if cls in pt.AB_CLASSES else None
                    want = (pt.unloaded(t, cls, mode, n) + issue) * mul
                    total += 1
                    if op != op2 or int(e) - int(a) != want:
                        bad += 1
                        if len(first) < 3:
                            first.append(
                                f"{k}-{m} {op} n={n}: "
                                f"{int(e) - int(a)} != {want}"
                            )
        return (
            bad == 0 and total > 0,
            f"{total} requests against timing_<std>.json, {bad} differ"
            + (f" ({'; '.join(first)})" if first else ""),
        )

    c("c26-model", c26_model)
    # Two DDR5 ranks against the DDR5 table.
    c("c26-model-DDR5-2r", lambda: c26_model((("DDR5-2r", "DDR5"),)))
    c(
        "c13-fmt",
        lambda: nonzero_with(
            s,
            "c13-fmt",
            "the checkpoint's PIM state holds bf16 operands and fp32 "
            "accumulators, this memory uses fp16",
        ),
    )

    def c26cc():
        # The same requests and latencies as C26's --close-rows runs.
        def lat(name):
            p = s.runs / name / "trace.out"
            text = p.read_text() if p.exists() else ""
            return re.findall(
                r"(?:done (\S+) \(seq \d+\) after (\d+) ticks|"
                r"atomic (\S+) at \S+ size \d+: U (\d+) ticks)",
                text,
            )

        bad = []
        n = 0
        for m in ("atomic", "ideal", "aim", "baseline", "aespa"):
            a, b = lat(f"c26cc-LPDDR6-{m}"), lat(f"c26-LPDDR6-{m}")
            n += len(a)
            if not a or a != b:
                bad.append(m)
        return (
            not bad,
            f"{n} requests under ClosedCAP equal the --close-rows runs "
            f"(differ: {bad or 'none'})",
        )

    c("c26-closedcap", c26cc)


# ---------------------------------------------------------------------
# refresh: C10 and C11.
# ---------------------------------------------------------------------

LPDDR6_NREFI = 3906000 // 375  # lpddr6.py _resolve_nREFI at tCK 375 ps
C10_NREFI = 3000  # about 3.5 x the default rate; 1000 stalls stock LPDDR6


def prepare_refresh(s):
    s.geometry("LPDDR6", *LP)
    s.script("c9", "c9.txt", "LPDDR6")
    s.script("host", "host-u0.txt", "LPDDR6", "--count", "1500", "--unit", "0")
    R.c9_script(s.g("LPDDR6"), jobs=200, seed=21).write(str(s.gen / "c11.txt"))


def refresh_binaries(s):
    """gem5.opt, and gem5.debug if it existed when the run began."""
    if not hasattr(s, "debug_ok"):
        s.debug_ok = DEBUG.exists()
    return [("", OPT)] + ([("-debug", DEBUG)] if s.debug_ok else [])


def refresh_variants(s):
    """(suffix, binary, extra options): each binary, and the same
    runs with close_rows, so that refresh closes rows of closing jobs."""
    return [
        (sfx + close, b, extra)
        for sfx, b in refresh_binaries(s)
        for close, extra in (("", []), ("-close", ["--close-rows"]))
    ]


def tasks_refresh(s):
    G = s.gen
    for sfx, b, extra in refresh_variants(s):
        name = f"c10{sfx}"
        if s.want(name):
            s.task(
                name,
                lambda name=name, b=b: s.cpuless(
                    name,
                    G / "c9.txt",
                    *LP,
                    "--arb-policy",
                    "chopim",
                    "--host-script",
                    G / "host-u0.txt",
                    "--override",
                    f"nREFI={C10_NREFI}",
                    *extra,
                    binary=b,
                ),
            )
        for m in ("baseline", "aespa"):
            name = f"c11-{m}{sfx}"
            if s.want(name):
                s.task(
                    name,
                    lambda name=name, b=b, m=m, extra=extra: s.cpuless(
                        name,
                        G / "c11.txt",
                        *LP,
                        "--ab-mode",
                        m,
                        *extra,
                        binary=b,
                    ),
                )


def refresh_law(s, name, nrefi, chans=2):
    st = s.stats(name)
    got = []
    for ch in range(chans):
        cyc = int(float(st.get(f"{M}.channel{ch}.cycles", 0)))
        ref = int(float(st.get(f"{M}.channel{ch}.num_maintenance_reqs", -1)))
        got.append((cyc, ref, cyc // nrefi))
    return all(r == want for _, r, want in got), got


def checks_refresh(s):
    for sfx, _, _ in refresh_variants(s):

        def c10(sfx=sfx):
            n = f"c10{sfx}"
            rc = s.rc(n)
            ok1, l1 = s.ctrl(
                n,
                "system.tester.mismatches=0",
                "system.host.mismatches=0",
                f"{M}.channel0.controller_plugin.pim_reopens>0",
                f"{M}.channel0.controller_plugin.pim_interrupts>0",
            )
            law, got = refresh_law(s, n, C10_NREFI)
            return (
                rc == 0 and ok1 and law,
                f"rc={rc} {l1} | refreshes per channel (cycles, "
                f"issued, cycles // nREFI {C10_NREFI}): {got}",
            )

        s.check(f"c10{sfx}", c10)
        for m in ("baseline", "aespa"):

            def c11(m=m, sfx=sfx):
                n = f"c11-{m}{sfx}"
                rc = s.rc(n)
                ok1, l1 = s.ctrl(n, "system.tester.mismatches=0")
                law, got = refresh_law(s, n, LPDDR6_NREFI)
                # The close_rows variants finish sooner (Baseline: about
                # 20 intervals instead of 40).
                least = 10 if sfx.endswith("-close") else 40
                many = all(r >= least for _, r, _ in got)
                return (
                    rc == 0 and ok1 and law and many,
                    f"rc={rc} {l1} | refreshes (cycles, issued, "
                    f"cycles // nREFI {LPDDR6_NREFI}): {got}",
                )

            s.check(f"c11-{m}{sfx}", c11)
    if not s.debug_ok:
        for n in (
            f"{c}-debug{close}"
            for c in ("c10", "c11-baseline", "c11-aespa")
            for close in ("", "-close")
        ):
            s.check(n, lambda: (None, f"SKIP: {DEBUG} not found"))


# ---------------------------------------------------------------------
# c28: Baseline ACTs against Ramulator 2.1's T12 sequence
# (ext/ramulator2/tests/device_timings/pim_cases.py).
# ---------------------------------------------------------------------


def prepare_c28(s):
    t12 = s.gen / "t12.json"
    d = r2_scratch(s)
    if d is not None and not t12.exists():
        env = dict(os.environ, PYTHONPATH="python")
        with open(t12, "w") as f:
            subprocess.run(
                [str(r2_python()), str(HERE / "t12_acts.py")],
                cwd=d,
                env=env,
                stdout=f,
                check=False,
            )
    for k in STDS:
        s.geometry(k, *CPULESS_GEO[k])
        g = s.g(k)
        p = R.PimScript(g)
        # One AB MAC after 1 us idle: no earlier command constrains it.
        p.pim(R.AB_MAC, p.addr(0, 0, 1, 0), g.col_bytes, delay=1000000)
        p.drain()
        p.write(str(s.gen / f"c28-{k}.txt"))


def tasks_c28(s):
    for k in STDS:
        name = f"c28-{k}"
        if s.want(name):
            s.task(
                name,
                lambda name=name, k=k: s.cpuless(
                    name,
                    s.gen / f"c28-{k}.txt",
                    *CPULESS_GEO[k],
                    "--ab-mode",
                    "baseline",
                    "--close-rows",
                    "--refresh",
                    "NoRefresh",
                    "--no-posted",
                    "--trace",
                ),
            )


def checks_c28(s):
    for k in STDS:

        def c28(k=k):
            if r2_scratch(s) is None:
                return r2_missing(s)
            t12 = s.gen / "t12.json"
            if not t12.exists() or not t12.read_text().strip():
                return False, "no T12 sequence (t12_acts.py failed)"
            want = json.loads(t12.read_text())[k]["acts"]
            rows = [
                r
                for r in csv.reader(
                    open(s.runs / f"c28-{k}" / "cmd_trace.csv.ch0")
                )
                if r[0] != "clock" and r[-2] == "2" and r[1].startswith("ACT")
            ]
            got = [(r[1], int(r[0]) - int(rows[0][0])) for r in rows]
            same = got == [tuple(x) for x in want]
            return (
                s.rc(f"c28-{k}") == 0 and same,
                f"{len(got)} ACTs, last at {got[-1] if got else None}; "
                f"T12 last {want[-1]}; equal {same}",
            )

        s.check(f"c28-{k}", c28)

        def c28_model(k=k):
            # The last phase of every per-bank ACT, in clocks after the
            # job's first command, equals the model's A(k) (banks opened
            # in flat order).
            pt = timing_model()
            t = pt.load(k)
            last = (
                "ACTsb2" if t["command_cycles"]["act_phases"] == 2 else "ACT"
            )
            rows = [
                r
                for r in csv.reader(
                    open(s.runs / f"c28-{k}" / "cmd_trace.csv.ch0")
                )
                if r[0] != "clock" and r[-2] == "2" and r[1].startswith("ACT")
            ]
            mul = t["clock"]["tick_multiplier"]
            got = [
                pt._num((int(r[0]) - int(rows[0][0])) / mul)
                for r in rows
                if r[1] == last
            ]
            want = [a for _, a in pt.act_sequence(t)]
            return (
                s.rc(f"c28-{k}") == 0 and got == want,
                f"{len(got)} ACTs, last at {got[-1] if got else None} "
                f"clocks; model A(k) last {want[-1]}; equal {got == want}",
            )

        s.check(f"c28-model-{k}", c28_model)


# ---------------------------------------------------------------------
# path: the request path (pim-path) on an LPDDR6 PimMemory.
# ---------------------------------------------------------------------

PATH_CFG = ["tests/gem5/pim/configs/pim_path_se.py", "--memory", "lpddr6"]
PATH_FLAGS = "PIM,Cache,MemCtrl,ExecEnable,ExecMicro,ExecEffAddr"
PATH_MATRIX = (
    ("atomic", 256, False),
    ("atomic", 256, True),
    ("timing", 256, False),
    ("timing", 512, False),
    ("minor", 256, False),
    ("minor", 512, False),
    ("minor", 1024, False),
    ("o3", 256, False),
    ("o3", 512, False),
    ("o3", 1024, False),
)


# Speculation cases: s1 no forwarding to a PIM load, s2 no
# request from a wrong-path PIM instruction, s3 PIM loads in program order.
SPEC_RUNS = tuple(
    (f"spec-{mode}-{cpu}-{vlen}", mode, cpu, vlen)
    for mode in ("s1", "s2", "s3")
    for cpu in ("minor", "o3")
    for vlen in (256, 1024)
)


def path_name(cpu, vlen, caches):
    return f"path-{cpu}{'-caches' if caches else ''}-{vlen}"


def tasks_path(s, cfg=PATH_CFG):
    for cpu, vlen, caches in PATH_MATRIX:
        name = path_name(cpu, vlen, caches)
        if s.want(name):
            a = cfg + ["--cpu", cpu, "--vlen", vlen, "--mode", "all"]
            if caches:
                a.append("--caches")
            s.task(
                name, lambda name=name, a=a: s.g5(name, a, flags=PATH_FLAGS)
            )
    proto = []
    for cpu in ("timing", "o3"):
        proto += [
            (f"p1-{cpu}", ["--cpu", cpu, "--mode", "p1"]),
            (f"p2-{cpu}", ["--cpu", cpu, "--mode", "p2"]),
        ]
        proto += [
            (f"p{i}-{cpu}", ["--cpu", cpu, "--cores", "2", "--mode", f"p{i}"])
            for i in (3, 4, 5, 6)
        ]
        proto.append(
            (
                f"p5w-{cpu}",
                [
                    "--cpu",
                    cpu,
                    "--cores",
                    "2",
                    "--mode",
                    "p5",
                    "--dirty-read-warn",
                ],
            )
        )
    proto += [
        ("p2-atomic", ["--cpu", "atomic", "--caches", "--mode", "p2"]),
        ("p7-excl", ["--cpu", "timing", "--mode", "p7"]),
        ("p7-noexcl", ["--cpu", "timing", "--mode", "p7", "--no-exclude"]),
    ]
    for name, a in proto:
        if s.want(name):
            s.task(
                name,
                lambda name=name, a=a: s.g5(
                    name,
                    cfg + a,
                    flags="Cache,PIM,HWPrefetch,CoherentXBar",
                ),
            )
    if isinstance(s, NativePath):
        return
    # The speculation cases on the PIM memory.
    for name, mode, cpu, vlen in SPEC_RUNS:
        if s.want(name):
            a = cfg + ["--cpu", cpu, "--vlen", vlen, "--mode", mode]
            s.task(
                name,
                lambda name=name, a=a: s.g5(
                    name, a, flags="Cache,PIM,CoherentXBar"
                ),
            )


def checks_path(s):
    for cpu, vlen, caches in PATH_MATRIX:
        name = path_name(cpu, vlen, caches)

        def fn(name=name, cpu=cpu, vlen=vlen, caches=caches):
            cmd = [
                sys.executable,
                "tests/gem5/pim/check_request_path.py",
                "--dir",
                str(s.runs / name),
                "--log",
                str(s.runs / f"{name}.log"),
                "--mode",
                "all",
                "--vlen",
                str(vlen),
                "--cpu",
                cpu,
            ] + (["--caches"] if caches else [])
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            (s.runs / f"{name}.check").write_text(p.stdout + p.stderr)
            done = "PIM-PATH done" in s.text(name)
            last = (p.stdout.strip().splitlines() or [""])[-1]
            return (
                s.rc(name) == 0 and done and p.returncode == 0,
                f"rc={s.rc(name)} done={done} {last}",
            )

        s.check(name, fn)

    if not isinstance(s, NativePath):
        for name, mode, cpu, vlen in SPEC_RUNS:

            def spec(name=name, mode=mode):
                cmd = [
                    sys.executable,
                    "tests/gem5/pim/check_request_path.py",
                    "--dir",
                    str(s.runs / name),
                    "--log",
                    str(s.runs / f"{name}.log"),
                    "--mode",
                    mode,
                ]
                p = subprocess.run(
                    cmd, cwd=ROOT, capture_output=True, text=True
                )
                (s.runs / f"{name}.check").write_text(p.stdout + p.stderr)
                fails = re.findall(r"^FAIL (.*)$", p.stdout, re.M)
                return (
                    s.rc(name) == 0 and p.returncode == 0,
                    f"rc={s.rc(name)} "
                    + ("; ".join(fails) if fails else "0 failures"),
                )

            s.check(name, spec)

    def done(name):
        return "PIM-PATH done" in s.text(name)

    def st(name, pattern):
        return {
            k: float(v)
            for k, v in s.stats(name).items()
            if re.search(pattern, k)
        }

    def trace_lines(name):
        p = s.runs / name / "trace.out"
        return p.read_text(errors="replace") if p.exists() else ""

    def window(name):
        m = re.search(
            r"PIM window (0x[0-9a-f]+) size (0x[0-9a-f]+)", s.text(name)
        )
        return (int(m.group(1), 16), int(m.group(2), 16)) if m else (0, 0)

    def inwin(name, cmd):
        base, size = window(name)
        n = 0
        for m in re.finditer(
            r"\b" + cmd + r" \[([0-9a-f]+):", trace_lines(name)
        ):
            n += base <= int(m.group(1), 16) < base + size
        return n

    for cpu in ("timing", "o3"):

        def p1(cpu=cpu):
            n = f"p1-{cpu}"
            a, b = inwin(n, "CleanSharedReq"), inwin(n, "CleanInvalidReq")
            # cbo.inval with senvcfg.CBIE = 01 keeps the dirty line.
            kept = "PIM-PATH inval kept 8/8 zero 8/8" in s.text(n)
            return (
                s.rc(n) == 0 and done(n) and a > 0 and b > 0 and kept,
                f"rc={s.rc(n)} CleanSharedReq {a} CleanInvalidReq {b} "
                f"inval kept and zero: {kept}",
            )

        s.check(f"p1-{cpu}", p1)

        def p2(cpu=cpu, n=None):
            n = n or f"p2-{cpu}"
            # The memory bus's requests of the two kinds, in order.
            order = []
            for line in trace_lines(n).splitlines():
                if (
                    re.search(r"membus: recv(Timing|Atomic)", line)
                    and "SF" in line
                    and ("WritebackDirty" in line or "PimSbMacReq" in line)
                ):
                    m = re.search(r"packet ([A-Za-z]+)", line)
                    if m and (not order or order[-1] != m.group(1)):
                        order.append(m.group(1))
            dirty = st(n, r"l1d.*pimDirtyBlk$")
            ok = order[:2] == ["WritebackDirty", "PimSbMacReq"] and any(
                v == 1 for v in dirty.values()
            )
            return (
                s.rc(n) == 0 and ok,
                f"rc={s.rc(n)} order {order[:2]} "
                f"pimDirtyBlk {list(dirty.values())}",
            )

        s.check(f"p2-{cpu}", p2)

        def p3(cpu=cpu):
            n = f"p3-{cpu}"
            v = st(n, r"pimSnoopValidBlk")
            return (
                s.rc(n) == 0 and done(n) and not any(v.values()),
                f"rc={s.rc(n)} pimSnoopValidBlk {v or 'none'}",
            )

        s.check(f"p3-{cpu}", p3)
        s.check(
            f"p4-{cpu}",
            lambda cpu=cpu: nonzero_with(
                s, f"p4-{cpu}", "PIM SbMac request for"
            ),
        )
        s.check(
            f"p5-{cpu}",
            lambda cpu=cpu: nonzero_with(
                s, f"p5-{cpu}", "PIM RdGb request for"
            ),
        )

        def p5w(cpu=cpu):
            n = f"p5w-{cpu}"
            w = s.text(n).count("warn: PIM RdGb")
            v = st(n, r"pimDirtyPeerReads$")
            return (
                s.rc(n) == 0 and done(n) and w == 1 and sum(v.values()) == 1,
                f"rc={s.rc(n)} warnings {w} pimDirtyPeerReads "
                f"{sum(v.values())}",
            )

        s.check(f"p5w-{cpu}", p5w)

        def p6(cpu=cpu):
            n = f"p6-{cpu}"
            v = st(n, r"l1d.*pimSnoopValidBlk$")
            return (
                s.rc(n) == 0 and done(n) and any(x == 1 for x in v.values()),
                f"rc={s.rc(n)} pimSnoopValidBlk {list(v.values())}",
            )

        s.check(f"p6-{cpu}", p6)
    s.check("p2-atomic", lambda: p2(n="p2-atomic"))

    def p7():
        a, b = inwin("p7-excl", "HardPFReq"), inwin("p7-noexcl", "HardPFReq")
        rcs = (s.rc("p7-excl"), s.rc("p7-noexcl"))
        return (
            rcs == (0, 0) and a == 0 and b > 0,
            f"rc={rcs} HardPFReq in the PIM memory: {a} with the "
            f"excluded range, {b} without",
        )

    s.check("p7", p7)


# ---------------------------------------------------------------------
# config: V12-V15, C30, C31, layout headers.
# ---------------------------------------------------------------------

HELLO_SRC = "tests/test-progs/hello/src/hello.c"
DEVMEM = "tests/test-progs/pim-devmem/bin/riscv/linux/pim-devmem"


def prepare_config(s):
    s.sh(
        ["make", "-C", "tests/test-progs/pim-devmem/src"],
        s.gen / "make-devmem.log",
    )
    s.sh(
        [
            "riscv64-linux-gnu-gcc",
            "-static",
            "-O2",
            "-o",
            s.gen / "hello",
            HELLO_SRC,
        ],
        s.gen / "hello.log",
    )
    src = (ROOT / "include/gem5/pim/pim_layout_lpddr6.h").read_text()
    lines = src.splitlines()
    rx = re.compile(r"^#define PIM_LAYOUT_(\w+) (0x[0-9a-f]+|\d+)$")
    names = []
    hdr = s.gen / "hdr"
    hdr.mkdir(exist_ok=True)
    for i, line in enumerate(lines):
        m = rx.match(line)
        if not m:
            continue
        names.append(m.group(1))
        v = lines[:]
        v[i] = f"#define PIM_LAYOUT_{m.group(1)} {int(m.group(2), 0) + 1}"
        (hdr / f"wrong-{m.group(1)}.h").write_text("\n".join(v) + "\n")
    (hdr / "unknown.h").write_text(
        "\n".join(lines + ["#define PIM_LAYOUT_FOO 1"]) + "\n"
    )
    (hdr / "missing.h").write_text(
        "\n".join(x for x in lines if "PIM_LAYOUT_ACC_BYTES" not in x) + "\n"
    )
    s.v15_names = names
    for p in POLICY_RUNS:
        pol, intlv = p
        a = ["--standard", "lpddr6", "--policy", pol]
        if intlv:
            a += ["--interleave", str(intlv)]
        subprocess.run(
            [sys.executable, "libpim/gen_pim_layout.py"]
            + a
            + ["-o", str(s.gen / f"layout-{pol}-{intlv}.h")],
            cwd=ROOT,
            check=True,
        )


POLICY_RUNS = [
    (p, i)
    for p in (
        "RoRaBaChCo",
        "RoRaBaCoCh",
        "RoCoRaBaCh",
        "RoBaRaCoCh",
        "ChRaBaRoCo",
        "MOP4CLXOR",
    )
    for i in (None, 256)
    if not (i and p in ("RoRaBaChCo", "ChRaBaRoCo"))
]


def tasks_config(s):
    G = s.gen
    hello = G / "hello"
    t = s.task

    def g5(name, argv, **kw):
        if s.want(name):
            t(name, lambda: s.g5(name, argv, **kw))

    # V12: the PIM memory is never in the SE frame pool; the driver maps it
    # whole. The pattern stays below the register rows (LPDDR6 x 16: rows
    # 65534 and 65535 of every bank, the top 1 MiB of the 32 GiB window).
    dm = [
        "--binary",
        DEVMEM,
        "--args",
        "0x800000000",
        "0x800000000",
        "0x800000000",
        "128",
        "0x7fff00000",
    ]
    g5("v12-map", [SE_PIM] + dm, redirect=True)
    g5("v12-oom", [SE_PIM, "--os-size", "64MiB"] + dm, redirect=True)
    # V13 and V14 (FS).
    for m in ("r2", "pim", "pim-lpddr6", "pim-ddr5", "pim-gddr7", "pim-hbm4"):
        g5(f"v13-{m}", [FS_DTS, "--memory", m])
    # Way 2, the PIM memory as NUMA node 1.
    for k in ("lpddr6", "ddr5", "gddr7", "hbm4"):
        g5(f"v13-way2-{k}", [FS_DTS, "--memory", f"pim-{k}", "--way", "2"])
    g5(
        "v13-way2-refused",
        [FS_DTS, "--memory", "pim", "--way", "2", "--policy", "ChRaBaRoCo"],
        redirect=True,
    )
    g5("v14-se-true", [SE_PIM, "--binary", hello], redirect=True)
    g5(
        "v14-se-false",
        [SE_PIM, "--binary", hello, "--dirty-read-warn"],
        redirect=True,
    )
    g5("v14-fs-true", [FS_DTS, "--memory", "pim", "--instantiate"])
    g5(
        "v14-fs-false",
        [FS_DTS, "--memory", "pim", "--instantiate", "--dirty-read-warn"],
    )
    # V15.
    for k in ("lpddr6", "ddr5", "gddr7", "hbm4"):
        h = f"include/gem5/pim/pim_layout_{k}.h"
        g5(
            f"v15-se-default-{k}",
            [
                SE_PIM,
                "--binary",
                hello,
                "--std",
                k.upper(),
                "--layout-file",
                h,
            ],
            redirect=True,
        )
        g5(
            f"v15-fs-default-{k}",
            [FS_DTS, "--memory", f"pim-{k}", "--layout-file", h, "--simulate"],
            redirect=True,
        )
    for n in s.v15_names:
        g5(
            f"v15-se-wrong-{n}",
            [
                SE_PIM,
                "--binary",
                hello,
                "--layout-file",
                G / "hdr" / f"wrong-{n}.h",
            ],
            redirect=True,
        )
    for v in ("unknown", "missing"):
        g5(
            f"v15-se-{v}",
            [SE_PIM, "--binary", hello, "--layout-file", G / "hdr" / f"{v}.h"],
            redirect=True,
        )
    for n in ("WINDOW_BASE_FS", "WINDOW_BASE_SE"):
        g5(
            f"v15-fs-wrong-{n}",
            [
                FS_DTS,
                "--memory",
                "pim",
                "--layout-file",
                G / "hdr" / f"wrong-{n}.h",
                "--simulate",
            ],
            redirect=True,
        )
    # C30, C31.
    g5("c30-c31", ["tests/gem5/pim/configs/pim_policy_checks.py"])
    for pol, intlv in POLICY_RUNS:
        a = [
            SE_PIM,
            "--binary",
            hello,
            "--policy",
            pol,
            "--layout-file",
            G / f"layout-{pol}-{intlv}.h",
        ]
        if intlv:
            a += ["--interleave", str(intlv)]
        g5(f"c31-{pol}-{intlv or 'default'}", a, redirect=True)


def checks_config(s):
    def v12():
        t = s.text("v12-map")
        npass = len(re.findall(r"^PASS", t, re.M))
        fails = re.search(r"devmem: (\d+) failures", t)
        oom = "Out of memory" in s.text("v12-oom")
        ok = (
            s.rc("v12-map") == 0
            and fails
            and fails.group(1) == "0"
            and npass >= 14
            and oom
        )
        return ok, (
            f"map rc={s.rc('v12-map')} {npass} PASS, "
            f"{fails.group(0) if fails else 'no summary'}; 128 MiB "
            f"malloc on a 64 MiB OS memory runs out of memory: {oom}"
        )

    s.check("v12", v12)

    def nodes(name):
        p = s.runs / name / "device.dts"
        t = p.read_text() if p.exists() else ""
        return re.findall(r"memory@([0-9a-f]+)", t), t

    def v13():
        out, ok = [], True
        for m in ("pim", "pim-lpddr6", "pim-ddr5", "pim-gddr7", "pim-hbm4"):
            n, _ = nodes(f"v13-{m}")
            ok &= n == ["80000000"]
            out.append(f"{m} {n}")
        _, r2 = nodes("v13-r2")
        _, pim = nodes("v13-pim")
        same = r2 == pim and bool(r2)
        return (
            ok and same,
            "; ".join(out)
            + f"; pim device tree byte-identical to the plain memory's: {same}",
        )

    s.check("v13", v13)

    def v13_way2():
        """Way 2's device tree: the OS memory on
        node 0, the PIM memory from WINDOW_BASE_FS up to its register rows
        (REG_ROW0 << ROW_SHIFT) on node 1, every CPU on node 0; a policy
        without the row on top refused."""
        node_re = re.compile(
            r"memory@[0-9a-f]+ \{\s*device_type = \"memory\";\s*"
            r"reg = <([^>]*)>;\s*numa-node-id = <(0x[0-9a-f]+)>"
        )
        cpu_re = re.compile(
            r"cpu@\d+ \{\s*device_type = \"cpu\";\s*"
            r"numa-node-id = <(0x[0-9a-f]+)>"
        )
        out, ok = [], True
        for k in ("lpddr6", "ddr5", "gddr7", "hbm4"):
            _, t = nodes(f"v13-way2-{k}")
            h = {
                m.group(1): int(m.group(2), 0)
                for m in re.finditer(
                    r"#define PIM_LAYOUT_(\w+) (0x[0-9a-fA-F]+|\d+)",
                    (ROOT / f"include/gem5/pim/pim_layout_{k}.h").read_text(),
                )
            }
            got = []
            for reg, nid in node_re.findall(t):
                a, b, c, d = (int(x, 16) for x in reg.split())
                got.append((a << 32 | b, c << 32 | d, int(nid, 16)))
            want1 = (h["WINDOW_BASE_FS"], h["REG_ROW0"] << h["ROW_SHIFT"], 1)
            cpus = cpu_re.findall(t)
            good = (
                len(got) == 2
                and got[0][0] == 0x80000000
                and got[0][2] == 0
                and got[1] == want1
                and bool(cpus)
                and all(int(c, 16) == 0 for c in cpus)
            )
            ok &= good
            out.append(
                f"{k} "
                + ", ".join(f"{a:#x}+{n:#x} node {i}" for a, n, i in got)
                + f" (CPUs {len(cpus)} on node 0)"
            )
        refused = "Way 2 needs a policy with the row on top" in s.text(
            "v13-way2-refused"
        )
        return ok and refused, (
            "; ".join(out) + f"; ChRaBaRoCo refused: {refused}"
        )

    s.check("v13-way2", v13_way2)

    def v14(name, want):
        import configparser

        cfg = configparser.ConfigParser(interpolation=None, strict=False)
        cfg.optionxform = str
        cfg.read(s.runs / name / "config.ini")
        win = None
        for sec in cfg.sections():
            if sec.endswith("pim_memory.ramulator2"):
                win = cfg[sec]["range"]
        caches = [
            x for x in cfg.sections() if "pim_dirty_read_fatal" in cfg[x]
        ]
        pfs = [
            x
            for x in cfg.sections()
            if cfg[x].get("type", "").endswith("Prefetcher")
        ]
        bad = [x for x in caches if cfg[x]["pim_dirty_read_fatal"] != want]
        bad += [
            x
            for x in pfs
            if win not in cfg[x].get("excluded_ranges", "").split()
        ]
        tagged = [
            x
            for x in cfg.sections()
            if cfg[x].get("type", "") in ("Cache", "NoncoherentCache")
        ]
        ok = bool(win) and caches and len(caches) == len(tagged) and not bad
        return ok, (
            f"window {win}; {len(caches)} caches with the knob, "
            f"{len(pfs)} prefetchers; bad: {bad or 'none'}"
        )

    for n, w in (
        ("v14-se-true", "true"),
        ("v14-se-false", "false"),
        ("v14-fs-true", "true"),
        ("v14-fs-false", "false"),
    ):
        s.check(n, lambda n=n, w=w: v14(n, w))

    def v15(name, want):
        t = s.text(name)
        if "the PIM layout header matches" in t:
            got = "PASS"
        else:
            m = re.search(r"PIM layout mismatch: ([A-Z_0-9]+)", t)
            got = m.group(1) if m else None
        return got == want, f"got {got}, expected {want}"

    for k in ("lpddr6", "ddr5", "gddr7", "hbm4"):

        def se(k=k):
            ok, d = v15(f"v15-se-default-{k}", "PASS")
            h = "Hello world!" in s.text(f"v15-se-default-{k}")
            return ok and h, f"{d}; Hello world! {h}"

        s.check(f"v15-se-default-{k}", se)
        s.check(
            f"v15-fs-default-{k}",
            lambda k=k: v15(f"v15-fs-default-{k}", "PASS"),
        )
    for n in s.v15_names:
        # In SE the FS base is not compared.
        want = "PASS" if n == "WINDOW_BASE_FS" else n
        s.check(
            f"v15-se-wrong-{n}",
            lambda n=n, w=want: v15(f"v15-se-wrong-{n}", w),
        )
    s.check("v15-se-unknown", lambda: v15("v15-se-unknown", "FOO"))
    s.check("v15-se-missing", lambda: v15("v15-se-missing", "ACC_BYTES"))
    s.check(
        "v15-fs-wrong-WINDOW_BASE_FS",
        lambda: v15("v15-fs-wrong-WINDOW_BASE_FS", "WINDOW_BASE_FS"),
    )
    s.check(
        "v15-fs-wrong-WINDOW_BASE_SE",
        lambda: v15("v15-fs-wrong-WINDOW_BASE_SE", "PASS"),
    )

    def c30_c31():
        t = s.text("c30-c31")
        m = re.search(r"pim_policy_checks: (\d+) failures", t)
        n = len(re.findall(r"^PASS: ", t, re.M))
        return (
            s.rc("c30-c31") == 0 and m and m.group(1) == "0",
            f"{n} checks, {m.group(0) if m else 'no summary'}",
        )

    s.check("c30-c31", c30_c31)
    for pol, intlv in POLICY_RUNS:
        n = f"c31-{pol}-{intlv or 'default'}"

        def run(n=n):
            t = s.text(n)
            ok = (
                s.rc(n) == 0
                and "the PIM layout header matches" in t
                and "Hello world!" in t
            )
            return ok, f"rc={s.rc(n)} header matches and Hello world!: {ok}"

        s.check(n, run)

    def headers():
        p = subprocess.run(
            [sys.executable, "libpim/gen_pim_layout.py", "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        q = subprocess.run(
            [sys.executable, "tests/test_gen_layout.py"],
            cwd=ROOT / "libpim",
            capture_output=True,
            text=True,
        )
        tail = (q.stdout + q.stderr).strip().splitlines()[-1:]
        return (
            p.returncode == 0 and q.returncode == 0,
            f"gen_pim_layout --check rc={p.returncode}; generator test "
            f"rc={q.returncode} {tail}",
        )

    s.check("layout-headers", headers)


# ---------------------------------------------------------------------
# v16: riscv-pim.py in SE.
# ---------------------------------------------------------------------

GEMV = "tests/test-progs/pim-gemv"
GEMV_REF = "tests/gem5/pim/ref/pim-gemv"


def prepare_v16(s):
    s.sh(
        [
            "scons",
            "-C",
            "libpim",
            "riscv.CROSS_COMPILE=riscv64-linux-gnu-",
            "build/riscv",
            "build/native",
            f"-j{min(s.args.jobs, 32)}",
        ],
        s.gen / "libpim.log",
    )
    for k in ("lpddr6", "ddr5", "gddr7", "hbm4"):
        s.sh(
            ["make", "-C", f"{GEMV}/src", f"PIM_STD={k}"],
            s.gen / f"make-gemv-{k}.log",
        )
    s.sh(
        ["make", "-C", f"{GEMV}/src", "TARGET=host", "PIM_STD=lpddr6"],
        s.gen / "make-gemv-host.log",
    )
    s.sh(
        ["make", "-C", f"{ELTWISE}/src", "PIM_STD=lpddr6"],
        s.gen / "make-eltwise.log",
    )
    with open(s.gen / "gemv-host-default.out", "w") as f:
        subprocess.run(
            [str(ROOT / GEMV / "bin/host/lpddr6/pim-gemv")], cwd=ROOT, stdout=f
        )
    # A custom layout: its header, and the program built against it
    # (libpim_custom.a is shared, so this stays sequential).
    hdr = s.gen / "v16-RoRaBaCoCh.h"
    subprocess.run(
        [
            sys.executable,
            "libpim/gen_pim_layout.py",
            "--standard",
            "lpddr6",
            "--policy",
            "RoRaBaCoCh",
            "-o",
            str(hdr),
        ],
        cwd=ROOT,
        check=True,
    )
    s.sh(
        [
            "make",
            "-C",
            f"{GEMV}/src",
            "PIM_STD=lpddr6",
            f"PIM_LAYOUT={hdr}",
            "LAYOUT_NAME=p4r17-RoRaBaCoCh",
        ],
        s.gen / "make-gemv-custom.log",
    )
    # DDR5 rank counts. Headers for each system, and
    # pim-gemv built against each (sequential: libpim_custom.a is shared);
    # the 1-rank x 2 header is only for the refused mismatch.
    for name, extra in DDR5_RANK_LAYOUTS.items():
        hdr = s.gen / f"v16-ddr5-{name}.h"
        subprocess.run(
            [
                sys.executable,
                "libpim/gen_pim_layout.py",
                "--standard",
                "ddr5",
                "-o",
                str(hdr),
            ]
            + extra,
            cwd=ROOT,
            check=True,
        )
        if name == "r1x2":
            continue
        s.sh(
            [
                "make",
                "-C",
                f"{GEMV}/src",
                "PIM_STD=ddr5",
                f"PIM_LAYOUT={hdr}",
                f"LAYOUT_NAME=dcfg-ddr5-{name}",
            ],
            s.gen / f"make-gemv-ddr5-{name}.log",
        )


ELTWISE = "tests/test-progs/pim-eltwise"
ELTWISE_REF = "tests/gem5/pim/ref/pim-eltwise"
# DDR5 systems by rank count and channels: 2 ranks x 2
# (32 GiB), and the two 128 GiB systems, 8 single-rank DIMMs (1 rank x 16)
# and 4 dual-rank DIMMs (2 ranks x 8); gen_pim_layout.py options.
DDR5_RANK_LAYOUTS = {
    "r2x2": ["--channels", "2", "--ranks", "2"],
    "r1x2": ["--channels", "2"],
    "r1x16": ["--channels", "16"],
    "r2x8": ["--channels", "8", "--ranks", "2"],
}


def v16_runs(s):
    host = s.gen / "gemv-host-default.out"
    runs = []
    for cpu in ("atomic", "timing", "minor", "o3"):
        runs.append((f"v16-default-{cpu}", ["--cpu", cpu], host))
        args = (ROOT / GEMV_REF / "lpddr6.args").read_text().split()
        runs.append(
            (
                f"v16-ref-{cpu}",
                ["--cpu", cpu, "--args"] + args,
                ROOT / GEMV_REF / "lpddr6.out",
            )
        )
    # ClosedCAP closes PIM rows too: SB columns with
    # auto-precharge in pim-eltwise's SB kind.
    args = (ROOT / ELTWISE_REF / "lpddr6.args").read_text().split()
    runs.append(
        (
            "v16-closedcap",
            [
                "--row-policy",
                "ClosedCAP",
                "--binary",
                f"{ELTWISE}/bin/riscv/lpddr6/pim-eltwise",
                "--args",
            ]
            + args,
            ROOT / ELTWISE_REF / "lpddr6.out",
        )
    )
    # AESPA keeps rows open between jobs.
    args = (ROOT / GEMV_REF / "lpddr6.args").read_text().split()
    runs.append(
        (
            "v16-ref-aespa",
            ["--ab-mode", "aespa", "--args"] + args,
            ROOT / GEMV_REF / "lpddr6.out",
        )
    )
    for k in ("ddr5", "gddr7", "hbm4"):
        args = (ROOT / GEMV_REF / f"{k}.args").read_text().split()
        runs.append(
            (
                f"v16-ref-{k}",
                ["--memory", k, "--args"] + args,
                ROOT / GEMV_REF / f"{k}.out",
            )
        )
    runs.append(
        (
            "v16-custom-RoRaBaCoCh",
            [
                "--policy",
                "RoRaBaCoCh",
                "--layout",
                s.gen / "v16-RoRaBaCoCh.h",
                "--binary",
                f"{GEMV}/bin/riscv/p4r17-RoRaBaCoCh/pim-gemv",
            ],
            host,
        )
    )
    # DDR5 rank counts (the output does not depend on the
    # layout, so ddr5.out is the host build's for every one).
    args = (ROOT / GEMV_REF / "ddr5.args").read_text().split()
    for name, extra in DDR5_RANK_LAYOUTS.items():
        if name == "r1x2":
            continue
        runs.append(
            (
                f"v16-ddr5-{name}",
                ["--memory", "ddr5"]
                + extra
                + [
                    "--layout",
                    s.gen / f"v16-ddr5-{name}.h",
                    "--binary",
                    f"{GEMV}/bin/riscv/dcfg-ddr5-{name}/pim-gemv",
                    "--args",
                ]
                + args,
                ROOT / GEMV_REF / "ddr5.out",
            )
        )
    return runs


def tasks_v16(s):
    for name, a, _ in v16_runs(s):
        if s.want(name) or s.want(f"{name}-u"):
            # The atomic runs trace every PIM request's latency.
            flags = "PIM" if name.endswith("-atomic") else None
            s.task(
                name,
                lambda name=name, a=a, flags=flags: s.g5(
                    name, [EXAMPLE] + a, flags=flags, redirect=True
                ),
            )
    if s.want("v16-novalues"):
        s.task(
            "v16-novalues",
            lambda: s.g5(
                "v16-novalues", [EXAMPLE, "--no-values"], redirect=True
            ),
        )
    neg = [
        (
            "v16-refuse-policy",
            ["--policy", "RoRaBaCoCh"],
            "these options change the PIM layout",
        ),
        (
            "v16-refuse-layout",
            ["--layout", s.gen / "v16-RoRaBaCoCh.h"],
            "--layout needs --binary",
        ),
        (
            "v16-refuse-binary",
            ["--binary", "/nonexistent/pim-gemv"],
            "no program /nonexistent/pim-gemv",
        ),
        (
            "v16-refuse-mismatch",
            [
                "--policy",
                "RoRaBaCoCh",
                "--layout",
                "include/gem5/pim/pim_layout_lpddr6.h",
                "--binary",
                f"{GEMV}/bin/riscv/lpddr6/pim-gemv",
            ],
            "PIM layout mismatch: POLICY",
        ),
        # A 1-rank header on a 2-rank memory; --ranks off DDR5.
        (
            "v16-refuse-ranks-header",
            [
                "--memory",
                "ddr5",
                "--channels",
                "2",
                "--ranks",
                "2",
                "--layout",
                s.gen / "v16-ddr5-r1x2.h",
                "--binary",
                f"{GEMV}/bin/riscv/dcfg-ddr5-r2x2/pim-gemv",
            ],
            "PIM layout mismatch: RANKS",
        ),
        (
            "v16-refuse-ranks-lpddr6",
            [
                "--ranks",
                "2",
                "--layout",
                "include/gem5/pim/pim_layout_lpddr6.h",
                "--binary",
                f"{GEMV}/bin/riscv/lpddr6/pim-gemv",
            ],
            "ranks is a DDR5 option",
        ),
    ]
    s.v16_neg = neg
    for name, a, _ in neg:
        if s.want(name):
            s.task(
                name,
                lambda name=name, a=a: s.g5(
                    name, [EXAMPLE] + a, redirect=True
                ),
            )
    if s.want("v16-refuse-noram") and NATIVE.exists():
        s.task(
            "v16-refuse-noram",
            lambda: s.g5(
                "v16-refuse-noram", [EXAMPLE], binary=NATIVE, redirect=True
            ),
        )


def checks_v16(s):
    for name, _, want in v16_runs(s):

        def fn(name=name, want=want):
            out = s.runs / name / "program.out"
            same = out.exists() and out.read_bytes() == Path(want).read_bytes()
            st = s.stats(name)
            reqs = st.get(
                "board.memory.pim_memory.ramulator2.pim.reqs::total", "0"
            )
            lat = float(
                st.get(
                    "board.memory.pim_memory.ramulator2.pim.atomicLat::total",
                    "0",
                )
            )
            atomic = "atomic" in name
            ok = s.rc(name) == 0 and same and (not atomic or lat > 0)
            note = ""
            if name == "v16-closedcap":
                ap = sum(
                    float(v)
                    for n, v in st.items()
                    if n.endswith("controller_plugin.pim_ap_cols")
                )
                ok = ok and ap > 0
                note = f"; auto-precharge PIM columns {ap:.0f}"
            if name == "v16-ref-aespa":
                # Rows the job finds open are used: fewer ACTs than columns.
                pre = "board.memory.pim_memory.ramulator2.channel"
                tot = {
                    k: sum(
                        float(v)
                        for n, v in st.items()
                        if n.startswith(pre)
                        and n.endswith(f"controller_plugin.{k}")
                    )
                    for k in ("pim_acts", "pim_col_cmds_AbMac")
                }
                ok = ok and tot["pim_acts"] < tot["pim_col_cmds_AbMac"]
                tick = re.search(r"Exiting @ tick (\d+)", s.text(name))
                note = (
                    f"; ACTs {tot['pim_acts']:.0f}, AB MAC columns "
                    f"{tot['pim_col_cmds_AbMac']:.0f}; final tick "
                    f"{tick.group(1) if tick else None}"
                )
            return ok, (
                f"rc={s.rc(name)} output equal to {Path(want).name}: "
                f"{same}; {reqs} PIM requests"
                + (f"; atomic latency {lat:.0f} ticks" if atomic else "")
                + note
            )

        s.check(name, fn)

    def novalues():
        t = s.runs / "v16-novalues" / "program.out"
        tol = re.search(
            r"outside bf16 tolerance: (\d+)",
            t.read_text() if t.exists() else "",
        )
        return (
            s.rc("v16-novalues") == 1 and tol and int(tol.group(1)) > 0,
            f"rc={s.rc('v16-novalues')} "
            f"{tol.group(0) if tol else 'no tolerance line'}",
        )

    s.check("v16-novalues", novalues)
    for name, _, text in getattr(s, "v16_neg", []):
        s.check(name, lambda name=name, text=text: nonzero_with(s, name, text))
    if NATIVE.exists():
        s.check(
            "v16-refuse-noram",
            lambda: nonzero_with(
                s, "v16-refuse-noram", "built without Ramulator 2.1"
            ),
        )
    else:
        s.check("v16-refuse-noram", native_missing)
    for cpu_run in ("v16-default-atomic", "v16-ref-atomic"):
        s.check(f"{cpu_run}-u", lambda n=cpu_run: v16_atomic_u(s, n))


def v16_atomic_u(s, name):
    """Every PIM request's atomic latency (the wrapper's
    ``atomic`` trace lines) equals U of timing_lpddr6.json plus the issue
    term, as C2 and C26 check CPU-less (riscv-pim.py's default all-bank
    mode is Baseline)."""
    pt = timing_model()
    t = pt.load("LPDDR6")
    cb = t["geometry"]["col_bytes"]
    mul = t["clock"]["tick_multiplier"]
    issue = t["derived"]["issue_offset"]
    p = s.runs / name / "trace.out"
    lines = re.findall(
        r"atomic (\w+) at 0x[0-9a-f]+ size (\d+): U (\d+) ticks",
        p.read_text() if p.exists() else "",
    )
    bad, first = 0, []
    for op, size, u in lines:
        if op in SCALAR_SB:
            bad += 1  # pim-gemv issues no scalar form
            continue
        cls = pt.OP_CLASS[op]
        mode = "baseline" if cls in pt.AB_CLASSES else None
        n = -(-int(size) // cb)
        # The model counts controller clocks (C26); the trace has ticks.
        clocks = (pt.unloaded(t, cls, mode, n) + issue) * mul
        want = clocks * t["clock"]["tCK_ps"] // mul
        if int(u) != want:
            bad += 1
            if len(first) < 3:
                first.append(f"{op} size {size}: {u} != {want}")
    return (
        bool(lines) and bad == 0,
        f"{len(lines)} atomic PIM requests against timing_lpddr6.json, "
        f"{bad} differ" + (f" ({'; '.join(first)})" if first else ""),
    )


# ---------------------------------------------------------------------
# fs-boot: V16's full-system boot.
# ---------------------------------------------------------------------


def tasks_fs_boot(s):
    if s.want("v16-fs-timing"):
        s.task(
            "v16-fs-timing",
            lambda: s.g5(
                "v16-fs-timing",
                [EXAMPLE, "--mode", "fs", "--fs-exit", "kernel-booted"],
                binary=FAST,
                redirect=True,
            ),
        )


def checks_fs_boot(s):
    def fn():
        n = "v16-fs-timing"
        t = s.text(n)
        booted = "First exit: kernel booted" in t
        dts = s.runs / n / "device.dts"
        nodes = re.findall(
            r"memory@([0-9a-f]+)", dts.read_text() if dts.exists() else ""
        )
        st = s.stats(n)
        pim = [
            float(v)
            for k, v in st.items()
            if re.search(
                r"pim_memory\.ramulator2\.channel\d+\.num_(read|write)_reqs$",
                k,
            )
        ]
        ok = (
            s.rc(n) == 0
            and booted
            and nodes == ["80000000"]
            and pim
            and not any(pim)
        )
        return ok, (
            f"rc={s.rc(n)} kernel booted {booted}; memory nodes "
            f"{nodes}; PIM memory requests {sum(pim):.0f} on "
            f"{len(pim) // 2} channels"
        )

    s.check("v16-fs-timing", fn)


# ---------------------------------------------------------------------
# decode: the PIM instruction decoder on the native build.
# ---------------------------------------------------------------------

DECODE_CFG = "tests/gem5/pim/configs/pim_decode_se.py"
DECODE_BIN = "tests/test-progs/pim-decode/bin/riscv/linux/pim-decode"
DECODE_FLAGS = (
    "ExecEnable,ExecUser,ExecMacro,ExecMicro,ExecOpClass,ExecEffAddr,"
    "ExecFlags,Decode,Cache,PIM"
)
DECODE_MODES = ("scalar", "vstore", "vload", "all")
# The decode matrix: (CPU, VLEN).
DECODE_MATRIX = (
    ("atomic", 256),
    ("timing", 256),
    ("timing", 512),
    ("minor", 256),
    ("minor", 512),
    ("minor", 1024),
    ("o3", 256),
    ("o3", 512),
    ("o3", 1024),
)
# Negative runs: 16 reserved encodings and the two guards.
DECODE_NEG = [
    (f"bad{i}", f"bad {i}", "Unknown instruction") for i in range(16)
]
DECODE_NEG += [(f"guard{i}", f"guard {i}", "Fault Address") for i in (0, 1)]


def generated():
    """The native build's generated decoder."""
    return NATIVE.parent / "arch/riscv/generated/decoder-ns.cc.inc"


def native_missing():
    return None, f"SKIP: {NATIVE} not found (build it, or give --native)"


def prepare_decode(s):
    s.sh(
        ["make", "-C", "tests/test-progs/pim-decode/src"],
        s.gen / "make-pim-decode.log",
    )


def tasks_decode(s):
    if not NATIVE.exists():
        return
    for mode in DECODE_MODES:
        for cpu, vlen in DECODE_MATRIX:
            name = f"decode-{mode}-{cpu}-{vlen}"
            if not s.want(name):
                continue

            def run(name=name, mode=mode, cpu=cpu, vlen=vlen):
                s.g5(
                    name,
                    [DECODE_CFG, "--cpu", cpu, "--vlen", vlen, "--mode", mode],
                    binary=NATIVE,
                    flags=DECODE_FLAGS,
                )
                trace = s.runs / name / "trace.out"
                rc = s.sh(
                    [
                        sys.executable,
                        "tests/gem5/pim/check_exec_trace.py",
                        "--trace",
                        trace,
                        "--binary",
                        DECODE_BIN,
                        "--mode",
                        mode,
                        "--vlen",
                        vlen,
                        "--cpu",
                        cpu,
                    ],
                    s.runs / f"{name}.check",
                )
                (s.runs / f"{name}.checkrc").write_text(f"{rc}\n")
                # The traces are large; keep them compressed.
                if trace.exists():
                    subprocess.run(["gzip", "-f", str(trace)])

            s.task(name, run)
    for case, mode, _ in DECODE_NEG:
        name = f"decode-neg-{case}"
        if s.want(name):
            s.task(
                name,
                lambda name=name, mode=mode: s.g5(
                    name,
                    [DECODE_CFG, "--cpu", "atomic", "--mode", mode],
                    binary=NATIVE,
                ),
            )


def checks_decode(s):
    def static():
        if not generated().exists():
            return None, f"SKIP: no {generated()}"
        p = subprocess.run(
            [sys.executable, "tests/gem5/pim/check_generated.py", generated()],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        (s.runs / "decode-static.check").write_text(p.stdout + p.stderr)
        n = p.stdout.count("\nPASS ") + p.stdout.startswith("PASS ")
        fails = p.stdout.count("FAIL ")
        return p.returncode == 0, f"{n} PASS, {fails} FAIL"

    s.check("decode-static", static)
    for mode in DECODE_MODES:
        for cpu, vlen in DECODE_MATRIX:
            name = f"decode-{mode}-{cpu}-{vlen}"
            if not NATIVE.exists():
                s.check(name, native_missing)
                continue

            def fn(name=name):
                lines = (s.runs / f"{name}.check").read_text().splitlines()
                crc = int((s.runs / f"{name}.checkrc").read_text())
                done = "PIM-DECODE done" in s.text(name)
                return (
                    s.rc(name) == 0 and done and crc == 0,
                    f"rc={s.rc(name)} done={done} "
                    f"{lines[-1] if lines else '(no checker output)'}",
                )

            s.check(name, fn)
    for case, _, want in DECODE_NEG:
        name = f"decode-neg-{case}"
        if not NATIVE.exists():
            s.check(name, native_missing)
            continue

        def neg(name=name, want=want):
            text = s.text(name)
            hit = want in text
            ran = "PIM-DECODE end" in text
            m = re.search(r"(panic|fatal): .{0,90}", text)
            return (
                hit and not ran,
                f"{want!r} {'seen' if hit else 'missing'}; "
                f"{'ran to the end' if ran else 'stopped'}"
                + (f"; {m.group(0)}" if m else ""),
            )

        s.check(name, neg)


# ---------------------------------------------------------------------
# fsbm: pim-fs-bm's F1-F10 in FS bare metal on the native build.
# ---------------------------------------------------------------------

FS_BM = "tests/test-progs/pim-fs-bm/bin/riscv/bare/pim-fs-bm"
FS_BM_NOPIM = "tests/test-progs/pim-fs-bm/bin/riscv/bare/pim-fs-bm-nopim"
FS_PY = "configs/deprecated/example/fs.py"
FS_PMA = "tests/gem5/pim/configs/pim_fs_bm_pma.py"
FSBM_FLAGS = "PIM,Cache,ExecEnable,ExecKernel,ExecEffAddr"
# (name, checker CPU, config, fs.py options, PMA run); the checker has
# 18 checks on Atomic (no caches), 24 on Minor and 25 on the others.
FSBM_RUNS = (
    ("atomic", "atomic", FS_PY, ["--cpu-type=RiscvAtomicSimpleCPU"], False),
    ("timing", "timing", FS_PY, ["--cpu-type=RiscvTimingSimpleCPU"], False),
    ("minor", "minor", FS_PY, ["--cpu-type=RiscvMinorCPU"], False),
    ("o3", "o3", FS_PY, ["--cpu-type=RiscvO3CPU"], False),
    (
        "pma-timing",
        "timing",
        FS_PMA,
        ["--cpu-type=RiscvTimingSimpleCPU"],
        True,
    ),
    ("pma-o3", "o3", FS_PMA, ["--cpu-type=RiscvO3CPU"], True),
)


FSBM_NOPIM_CPUS = {
    "atomic": "RiscvAtomicSimpleCPU",
    "timing": "RiscvTimingSimpleCPU",
    "minor": "RiscvMinorCPU",
    "o3": "RiscvO3CPU",
}
# pim-fs-bm-sv48 (F1-F18) with Sv48 on each model;
# the checker has 23 checks on Atomic, 29 on Minor and 30 on the others.
# Without the parameter satp refuses Sv48 and only F14 fails (mask 1 << 14).
FS_BM_SV48 = "tests/test-progs/pim-fs-bm/bin/riscv/bare/pim-fs-bm-sv48"
SV48_PARAM = 'system.cpu[:].isa[:].max_paging_mode="sv48"'
FSBM_SV48_OFF_MASK = 1 << 14


def fsbm_cases():
    """(name, checker CPU, config, kernel, fs.py options, checker options,
    PASS lines wanted) of every fsbm run that check_fs_bm.py checks."""
    for case, cpu, cfg, opts, pma in FSBM_RUNS:
        # F12's request check is not made on Minor (check_fs_bm.py).
        want = {"atomic": 18, "minor": 24}.get(cpu, 25)
        yield (
            f"fsbm-{case}",
            cpu,
            cfg,
            FS_BM,
            opts,
            ["--pma"] if pma else [],
            want,
        )
    for cpu, cpu_type in FSBM_NOPIM_CPUS.items():
        want = {"atomic": 23, "minor": 29}.get(cpu, 30)
        yield (
            f"fsbm-sv48-{cpu}",
            cpu,
            FS_PY,
            FS_BM_SV48,
            [f"--cpu-type={cpu_type}", "--param", SV48_PARAM],
            ["--sv48"],
            want,
        )


def prepare_fsbm(s):
    s.sh(
        ["make", "-C", "tests/test-progs/pim-fs-bm/src"],
        s.gen / "make-pim-fs-bm.log",
    )


def tasks_fsbm(s):
    if not NATIVE.exists():
        return
    for name, cpu, cfg, kernel, opts, copts, _ in fsbm_cases():
        if not s.want(name):
            continue
        caches = [] if cpu == "atomic" else ["--caches", "--l2cache"]

        def run(
            name=name,
            cpu=cpu,
            cfg=cfg,
            kernel=kernel,
            opts=opts,
            copts=copts,
            c=caches,
        ):
            s.g5(
                name,
                [cfg, "--bare-metal", f"--kernel={ROOT / kernel}"]
                + ["--mem-size=512MiB"]
                + opts
                + c,
                binary=NATIVE,
                flags=FSBM_FLAGS,
            )
            rc = s.sh(
                [
                    sys.executable,
                    "tests/gem5/pim/check_fs_bm.py",
                    "--dir",
                    s.runs / name,
                    "--log",
                    s.runs / f"{name}.log",
                    "--binary",
                    kernel,
                    "--cpu",
                    cpu,
                ]
                + copts,
                s.runs / f"{name}.check",
            )
            (s.runs / f"{name}.checkrc").write_text(f"{rc}\n")
            trace = s.runs / name / "trace.out"
            if trace.exists():
                subprocess.run(["gzip", "-f", str(trace)])

        s.task(name, run)

    # pim-fs-bm-nopim on every model: its F2 (a committed store sets D) and
    # F11 (a wrong-path store leaves D = 0: the page-table walker sets the
    # dirty bit only for committed stores); the
    # exit code is the failure mask.
    for cpu in FSBM_NOPIM_CPUS:
        name = f"fsbm-nopim-{cpu}"
        if not s.want(name):
            continue
        caches = [] if cpu == "atomic" else ["--caches", "--l2cache"]
        s.task(
            name,
            lambda name=name, cpu=cpu, c=caches: s.g5(
                name,
                [
                    FS_PY,
                    "--bare-metal",
                    f"--kernel={ROOT / FS_BM_NOPIM}",
                    "--mem-size=512MiB",
                    f"--cpu-type={FSBM_NOPIM_CPUS[cpu]}",
                ]
                + c,
                binary=NATIVE,
            ),
        )

    # pim-fs-bm-sv48 without the parameter: satp stays Bare.
    name = "fsbm-sv48-off-timing"
    if s.want(name):
        s.task(
            name,
            lambda name=name: s.g5(
                name,
                [
                    FS_PY,
                    "--bare-metal",
                    f"--kernel={ROOT / FS_BM_SV48}",
                    "--mem-size=512MiB",
                    "--cpu-type=RiscvTimingSimpleCPU",
                    "--caches",
                    "--l2cache",
                ],
                binary=NATIVE,
            ),
        )


def fsbm_mask(s, name):
    """(reached m5_fail, the program's failure mask) of a bare-metal run."""
    text = (s.runs / f"{name}.log").read_text(errors="replace")
    m = re.search(r"Exit code is (\d+)", text)
    ended = "because m5_fail instruction encountered" in text
    return ended, int(m.group(1)) if m else 0


def checks_fsbm(s):
    for cpu in FSBM_NOPIM_CPUS:
        name = f"fsbm-nopim-{cpu}"
        if not NATIVE.exists():
            s.check(name, native_missing)
            continue

        def nopim(name=name):
            ended, mask = fsbm_mask(s, name)
            return (
                ended and mask == 0,
                f"failure mask {mask}" if ended else "did not reach m5_fail",
            )

        s.check(name, nopim)
    name = "fsbm-sv48-off-timing"
    if not NATIVE.exists():
        s.check(name, native_missing)
    else:

        def off(name=name):
            ended, mask = fsbm_mask(s, name)
            return (
                ended and mask == FSBM_SV48_OFF_MASK,
                (
                    (
                        f"failure mask {mask:#x} (want {FSBM_SV48_OFF_MASK:#x}: "
                        "F14 only, satp refuses Sv48)"
                    )
                    if ended
                    else "did not reach m5_fail"
                ),
            )

        s.check(name, off)
    for name, cpu, *_, want in fsbm_cases():
        if not NATIVE.exists():
            s.check(name, native_missing)
            continue

        def fn(name=name, want=want):
            text = (s.runs / f"{name}.check").read_text()
            crc = int((s.runs / f"{name}.checkrc").read_text())
            npass = len(re.findall(r"^PASS", text, re.M))
            fails = re.findall(r"^FAIL (.*)$", text, re.M)
            return (
                crc == 0 and npass == want and not fails,
                f"{npass}/{want} PASS"
                + (f"; FAIL: {'; '.join(fails)}" if fails else ""),
            )

        s.check(name, fn)


# ---------------------------------------------------------------------
# cache: CPU-less cache cases and an end-to-end check on a PimMemory.
# ---------------------------------------------------------------------

CACHE_CFG = "tests/gem5/pim/configs/pim_cache_cpuless.py"
CACHE_FLAGS = "PIM,CoherentXBar"
GB_DATA = "ab" * 32
ORDER_PIM = (
    f"100000 pim:WrGb 0x0 32 {GB_DATA}\n"
    "0 pim:SbMovGbufbk 0x1000 32\n"
    f"100000 rd 0x1000 32 - {GB_DATA}\n"
)
# The scripts of the cache cases (tester A issues PIM requests, B is the
# peer) and the end-to-end script (a normal write, then PIM operations on
# that block, which see the written data: its dirty line is written back
# before the PIM request reaches the memory).
CACHE_SCRIPTS = {
    "p2_a": f"0 wr 0x1000 8 1122334455667788\n"
    f"100000 pim:WrGb 0x1000 32 {GB_DATA}\n",
    "pimwr_a": f"100000 pim:WrGb 0x1000 32 {GB_DATA}\n",
    "pimrd_a": "100000 pim:RdGb 0x1000 32\n100000 pim:RdGb 0x1000 32\n",
    "peer_dirty": "0 wr 0x1000 8 1122334455667788\n",
    "peer_clean": "0 rd 0x1000 8\n",
    "peer_evict": "0 wr 0x1000 8 1122334455667788\n"
    "0 wr 0x1080 8 1122334455667788\n",
    # A PIM write delayed behind its block's writeback must not be
    # overtaken by a later PIM write
    # of the same tick to another block; the control touches another
    # block first.
    "order_dirty": f"0 wr 0x0 32 {'5a' * 32}\n" + ORDER_PIM,
    "order_clean": "0 rd 0x0 32\n" + ORDER_PIM,
    "order_other": f"0 wr 0x40000 32 {'5a' * 32}\n" + ORDER_PIM,
    "dpk_e2e": "0 wr 0x0 32 030a11181f262d343b424950575e656c737a81888f969d"
    "a4abb2b9c0c7ced5dc\n"
    "100000 pim:SbMovBkgbuf 0x0 32\n"
    "100000 pim:RdGb 0x0 32 - 030a11181f262d343b424950575e656c737a8188"
    "8f969da4abb2b9c0c7ced5dc\n",
}
# D-3: the PIM request meets the peer's dirty line in its write buffer
# (a 128-byte direct-mapped peer cache evicting 0x1000); 46 ns reaches it
# and the sweep checks that some time in 0-60 ns does.
D3_AT = 46
D3_SWEEP = range(0, 61)
# (name, expected exit status, options); per memory mode for P2-P6.
CACHE_RUNS = []
for _m in ("timing", "atomic"):
    CACHE_RUNS += [
        (f"cache-p2-{_m}", 0, ["--mode", _m, "--a", "p2_a"]),
        (
            f"cache-p4-{_m}",
            1,
            ["--mode", _m, "--a", "pimwr_a", "--b", "peer_dirty"],
        ),
        (
            f"cache-p5f-{_m}",
            1,
            ["--mode", _m, "--a", "pimrd_a", "--b", "peer_dirty"],
        ),
        (
            f"cache-p5w-{_m}",
            0,
            ["--mode", _m, "--a", "pimrd_a", "--b", "peer_dirty"]
            + ["--dirty-read", "warn"],
        ),
        (
            f"cache-p6-{_m}",
            0,
            ["--mode", _m, "--a", "pimwr_a", "--b", "peer_clean"],
        ),
        (
            f"cache-dpk-{_m}",
            0,
            ["--memory", "pim", "--mode", _m, "--a", "dpk_e2e"],
        ),
    ]
for _k in ("r", "w"):
    for _d in ("warn", "fatal"):
        CACHE_RUNS.append(
            (
                f"cache-d3{_k}-{_d}",
                0 if (_k, _d) == ("r", "warn") else 1,
                ["--a", f"d3{_k}_a", "--b", "peer_evict"]
                + ["--l1-size", "128B", "--l1-assoc", "1"]
                + ["--dirty-read", _d],
            )
        )
CACHE_RUNS.append(("cache-nc", 1, ["--a", "pimwr_a", "--noncoherent"]))
CACHE_RUNS += [
    (f"cache-order-{k}", 0, ["--memory", "pim", "--a", f"order_{k}"])
    for k in ("dirty", "clean", "other")
]
CACHE_RUNS += [
    (
        f"cache-d3-sweep-{t}",
        0,
        ["--a", f"d3s{t}_a", "--b", "peer_evict"]
        + ["--l1-size", "128B", "--l1-assoc", "1", "--dirty-read", "warn"],
    )
    for t in D3_SWEEP
]


def prepare_cache(s):
    d = s.gen / "cache"
    d.mkdir(exist_ok=True)
    scripts = dict(CACHE_SCRIPTS)
    scripts["d3r_a"] = f"{D3_AT * 1000} pim:RdGb 0x1000 32\n"
    scripts["d3w_a"] = f"{D3_AT * 1000} pim:WrGb 0x1000 32 {GB_DATA}\n"
    for t in D3_SWEEP:
        scripts[f"d3s{t}_a"] = f"{t * 1000} pim:RdGb 0x1000 32\n"
    for name, text in scripts.items():
        (d / f"{name}.txt").write_text(text)


def cache_argv(s, opts):
    out = [CACHE_CFG]
    for i, o in enumerate(opts):
        if i and opts[i - 1] in ("--a", "--b"):
            o = str(s.gen / "cache" / f"{o}.txt")
        out.append(o)
    return out


def tasks_cache(s):
    for name, _, opts in CACHE_RUNS:
        if s.want(name):
            s.task(
                name,
                lambda name=name, opts=opts: s.g5(
                    name, cache_argv(s, opts), flags=CACHE_FLAGS
                ),
            )


def cache_counters(s, name):
    return {
        k.split(".", 1)[1]: int(float(v))
        for k, v in s.stats(name).items()
        if re.search(
            r"pim(ValidBlk|DirtyBlk|SnoopValidBlk|DirtyPeerReads)$", k
        )
    }


def cache_trace(s, name):
    p = s.runs / name / "trace.out"
    return p.read_text(errors="replace") if p.exists() else ""


def checks_cache(s):
    # (case, message in the output, counters that must equal).
    want = {
        "p2": (None, {"l1a.pimValidBlk": 1, "l1a.pimDirtyBlk": 1}),
        "p4": ("PIM WrGb request for", {}),
        "p5f": ("PIM RdGb request for", {}),
        "p5w": (
            "warn: PIM RdGb",
            {"l1b.pimSnoopValidBlk": 2, "l1b.pimDirtyPeerReads": 2},
        ),
        "p6": (None, {"l1b.pimSnoopValidBlk": 1}),
        "dpk": (None, {}),
        "d3r-warn": ("warn: PIM RdGb", {"l1b.pimDirtyPeerReads": 1}),
        "d3r-fatal": ("PIM RdGb request for", {}),
        "d3w-warn": ("PIM WrGb request for", {}),
        "d3w-fatal": ("PIM WrGb request for", {}),
        "nc": ("reached a non-coherent", {}),
        "order-dirty": (None, {}),
        "order-clean": (None, {}),
        "order-other": (None, {}),
    }
    for name, rc_want, _ in CACHE_RUNS:
        if name.startswith("cache-d3-sweep-"):
            continue
        case = re.sub(r"^cache-|-(timing|atomic)$", "", name)
        msg, counters = want[case]

        def fn(name=name, rc_want=rc_want, msg=msg, counters=counters):
            text = s.text(name)
            got = cache_counters(s, name)
            bad = {
                k: got.get(k) for k, v in counters.items() if got.get(k) != v
            }
            ok = s.rc(name) == rc_want and not bad
            detail = f"rc={s.rc(name)} (want {rc_want})"
            if msg:
                ok = ok and msg in text
                detail += f"; {msg!r} {'seen' if msg in text else 'missing'}"
            if counters:
                detail += f"; counters {counters if not bad else bad}"
            if "-d3" in name:
                # D-3: the peer's dirty line was in its write buffer.
                hit = "found a dirty writeback" in cache_trace(s, name)
                ok = ok and hit
                detail += f"; write-buffer hit {hit}"
            if name == "cache-p2-timing":
                # The writeback reaches the bus before the PIM
                # write.
                order = []
                for line in cache_trace(s, name).splitlines():
                    m = re.search(
                        r"membus: recvTimingReq: src [^ ]+ packet "
                        r"(WritebackDirty|PimWrGbReq)",
                        line,
                    )
                    if m and "BUSY" not in line and m.group(1) not in order:
                        order.append(m.group(1))
                ok = ok and order == ["WritebackDirty", "PimWrGbReq"]
                detail += f"; bus order {order}"
            return ok, detail

        s.check(name, fn)

    def sweep():
        hits = [
            t
            for t in D3_SWEEP
            if "found a dirty writeback"
            in cache_trace(s, f"cache-d3-sweep-{t}")
        ]
        bad = [t for t in D3_SWEEP if s.rc(f"cache-d3-sweep-{t}") != 0]
        return (
            bool(hits) and not bad,
            f"{len(D3_SWEEP)} times, write-buffer hit at {hits} ns; "
            f"exit status not 0 at {bad or 'none'}",
        )

    s.check("cache-d3-sweep", sweep)


# ---------------------------------------------------------------------
# se-misc: the request path on native memory; the /dev/mem unmap check.
# ---------------------------------------------------------------------

PATH_NATIVE_CFG = [
    "tests/gem5/pim/configs/pim_path_se.py",
    "--memory",
    "native",
]
# pim-devmem unmap: unmap and map the /dev/mem window again; data
# kept, and a malloc beyond the 64 MiB OS memory runs out of memory. (base,
# size, span below the register rows) of the LPDDR6 PIM memory with 16
# channels (32 GiB, the default) and 1 channel (2 GiB).
DPN_RUNS = (
    ("32g", [], "0x800000000", "0x800000000", "0x7fff00000"),
    ("2g", ["--channels", "1"], "0x80000000", "0x80000000", "0x7ff00000"),
)


# pim-order's two PIM writes per
# iteration, with the row's line loaded first (warm) or not (cold).
ORDER_BIN = "tests/test-progs/pim-order/bin/riscv/lpddr6/pim-order"
ORDER_RUNS = (
    ("O3", "warm"),
    ("O3", "cold"),
    ("TIMING", "warm"),
    ("MINOR", "warm"),
)


class NativePath(Suite):
    """The ``path`` group's runs on ``build/RISCV/gem5.opt`` with native
    memory: same cases and checks, names prefixed ``native-``, outputs in
    ``runs/native``."""

    def __init__(self, s):
        self.__dict__.update(s.__dict__)
        self.runs = s.runs / "native"
        self.runs.mkdir(exist_ok=True)
        self.parent = s

    def want(self, name):
        return self.parent.want("native-" + name)

    def task(self, name, fn):
        self.parent.task("native-" + name, fn)

    def check(self, name, fn):
        self.parent.check("native-" + name, fn)

    def g5(self, name, argv, binary=None, flags=None, redirect=False):
        return Suite.g5(self, name, argv, NATIVE, flags, redirect)


def prepare_se_misc(s):
    s.sh(
        ["make", "-C", "tests/test-progs/pim-devmem/src"],
        s.gen / "make-devmem.log",
    )
    prepare_decode(s)
    s.sh(
        ["make", "-C", "tests/test-progs/pim-order/src", "PIM_STD=lpddr6"],
        s.gen / "make-pim-order.log",
    )


def tasks_se_misc(s):
    if NATIVE.exists():
        tasks_path(NativePath(s), PATH_NATIVE_CFG)
    for cpu, mode in ORDER_RUNS:
        name = f"order-{cpu.lower()}-{mode}"
        if s.want(name):
            argv = [
                "tests/gem5/pim/configs/pim_lib_se.py",
                "--std",
                "LPDDR6",
                "--layout-file",
                "include/gem5/pim/pim_layout_lpddr6.h",
                "--binary",
                ORDER_BIN,
                "--cpu",
                cpu,
                "--args",
                mode,
            ]
            s.task(name, lambda name=name, argv=argv: s.g5(name, argv))
    # Ruby stops on a PIM request.
    for cpu in ("timing", "atomic"):
        name = f"ruby-{cpu}"
        if ALL_OPT.exists() and s.want(name):
            argv = [
                "configs/deprecated/example/se.py",
                "--ruby",
                "--protocol=MESI_Two_Level",
                f"--cpu-type={cpu.capitalize()}SimpleCPU",
                f"--cmd={DECODE_BIN}",
                "--options=scalar",
            ]
            s.task(
                name,
                lambda name=name, argv=argv: s.g5(name, argv, binary=ALL_OPT),
            )
    for size, extra, base, wsize, span in DPN_RUNS:
        for cpu in ("ATOMIC", "TIMING"):
            name = f"dpn-{cpu.lower()}-{size}"
            if s.want(name):
                argv = [SE_PIM, "--os-size", "64MiB", "--cpu", cpu]
                argv += extra + ["--binary", DEVMEM, "--args", "unmap"]
                argv += [base, wsize, "128", span]
                s.task(
                    name,
                    lambda name=name, argv=argv: s.g5(
                        name, argv, redirect=True
                    ),
                )


def checks_se_misc(s):
    if NATIVE.exists():
        checks_path(NativePath(s))
    else:
        s.check("native-path", native_missing)
    for cpu, mode in ORDER_RUNS:
        name = f"order-{cpu.lower()}-{mode}"

        def order(name=name):
            p = s.runs / name / "program.out"
            out = p.read_text().strip() if p.exists() else "(no output)"
            return (
                s.rc(name) == 0 and re.search(r": 0 of \d+ wrong", out),
                f"rc={s.rc(name)} {out}",
            )

        s.check(name, order)
    for cpu in ("timing", "atomic"):
        name = f"ruby-{cpu}"
        if not ALL_OPT.exists():
            s.check(name, lambda: (None, f"SKIP: {ALL_OPT} not found"))
            continue
        s.check(
            name,
            lambda name=name: nonzero_with(
                s, name, "reached Ruby; PIM requests need the classic"
            ),
        )
    for size, *_ in DPN_RUNS:
        for cpu in ("atomic", "timing"):
            name = f"dpn-{cpu}-{size}"

            def fn(name=name):
                t = s.text(name)
                m = re.search(r"devmem: (\d+) failures", t)
                oom = "Out of memory" in t
                return (
                    m is not None and m.group(1) == "0" and oom,
                    f"{m.group(0) if m else 'no result'}; malloc beyond "
                    f"the OS memory: {'out of memory' if oom else 'NOT out'}",
                )

            s.check(name, fn)


# ---------------------------------------------------------------------
# r2suites: Ramulator 2.1's own test suites and K6; xcheck: model and
# address cross-checks.
# ---------------------------------------------------------------------

R2_SUITES = (
    ("device_timings", 140),
    ("controller_scheduling", 225),
    ("unit_tests", 78),
)
# K6: SHA-256 of the issued-command trace of the stock standards
# (tests/gem5/pim/k6_trace.py), taken before the PIM changes to
# Ramulator 2.1.
K6_SHA256 = "b99f072149b79ab653bee37105f437360d44eb7fd384e89dd1df4c7829880f8e"
XMODEL = "tests/gem5/pim/xmodel"
VECTORS = sorted(str(p) for p in (ROOT / "libpim/tests/vectors").glob("*.txt"))
TEST_MODEL = ROOT / "libpim/build/native/test/bin/test_model_lpddr6"


def r2_python():
    py = ROOT / ".venv/bin/python"
    return py if py.exists() else Path(sys.executable)


def r2_scratch(s):
    """An offline scratch build of ext/ramulator2 with its Python bindings,
    shared by the c28, r2suites and xcheck groups; its directory, or None
    when the build failed (logs in gen/r2-*.log)."""
    with s.lock:
        if hasattr(s, "r2_dir"):
            return s.r2_dir
        sc = s.out / "r2scratch"
        ok = sc / "build.ok"
        if not ok.exists() and not s.args.checks_only:
            shutil.rmtree(sc, ignore_errors=True)
            shutil.copytree(ROOT / "ext/ramulator2", sc / "ramulator2")
            env = dict(
                os.environ,
                http_proxy="http://127.0.0.1:9",
                https_proxy="http://127.0.0.1:9",
            )
            rc = s.sh(
                [
                    "cmake",
                    "-S",
                    sc / "ramulator2",
                    "-B",
                    sc / "build",
                    "-G",
                    "Ninja",
                    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
                    "-DRAMULATOR_PYTHON_BINDINGS=ON",
                    f"-DPython_EXECUTABLE={r2_python()}",
                ],
                s.gen / "r2-cmake.log",
                env=env,
            )
            rc = rc or s.sh(
                ["ninja", "-C", sc / "build", f"-j{s.args.jobs}"],
                s.gen / "r2-ninja.log",
                env=env,
            )
            if rc == 0:
                ok.write_text("ok\n")
        s.r2_dir = sc / "ramulator2" if ok.exists() else None
        return s.r2_dir


def r2_env():
    return dict(os.environ, PYTHONPATH="python:.")


def r2_missing(s):
    return None, "SKIP: the scratch build of ext/ramulator2 failed " + (
        f"(see {s.gen / 'r2-ninja.log'})"
    )


def prepare_r2suites(s):
    r2_scratch(s)


def tasks_r2suites(s):
    d = r2_scratch(s)
    if d is None:
        return
    for suite, _ in R2_SUITES:
        name = f"r2-{suite}"
        if s.want(name):
            s.task(
                name,
                lambda name=name, suite=suite: s.sh(
                    [
                        r2_python(),
                        "-m",
                        "pytest",
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        f"tests/{suite}",
                    ],
                    s.runs / f"{name}.log",
                    cwd=d,
                    env=r2_env(),
                ),
            )
    if s.want("r2-k6"):
        s.task(
            "r2-k6",
            lambda: s.sh(
                [r2_python(), HERE / "k6_trace.py", s.runs / "r2-k6.json"],
                s.runs / "r2-k6.log",
                cwd=d,
                env=r2_env(),
            ),
        )


def checks_r2suites(s):
    d = r2_scratch(s)
    for suite, least in R2_SUITES:
        name = f"r2-{suite}"
        if d is None:
            s.check(name, lambda: r2_missing(s))
            continue

        def fn(name=name, least=least):
            text = (s.runs / f"{name}.log").read_text(errors="replace")
            last = (text.strip().splitlines() or [""])[-1]
            m = re.search(r"(\d+) passed", last)
            n = int(m.group(1)) if m else 0
            bad = re.search(r"failed|error", last)
            return (
                n >= least and not bad,
                f"{last.strip('= ')} (at least {least})",
            )

        s.check(name, fn)
    if d is None:
        s.check("r2-k6", lambda: r2_missing(s))
        return

    def k6():
        import hashlib

        p = s.runs / "r2-k6.json"
        h = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""
        return (
            h == K6_SHA256,
            f"trace SHA-256 {h[:16]} (want {K6_SHA256[:16]})",
        )

    s.check("r2-k6", k6)


def prepare_xcheck(s):
    x = s.gen / "xmodel"
    s.sh(["sh", f"{XMODEL}/build.sh", x], s.gen / "xmodel-build.log")
    if not TEST_MODEL.exists():
        s.sh(
            [
                "scons",
                "-C",
                "libpim",
                str(TEST_MODEL.relative_to(ROOT / "libpim")),
            ],
            s.gen / "test-model-build.log",
        )
    for name, extra in (("random", []), ("random_nonan", ["--nonan"])):
        with open(x / f"{name}.txt", "w") as f:
            subprocess.run(
                [sys.executable, f"{XMODEL}/gen_random.py", "25"] + extra,
                cwd=ROOT,
                stdout=f,
                check=True,
            )
    with open(x / "geom.txt", "w") as f:
        subprocess.run([str(x / "geom_c")], stdout=f, check=False)
    s.sh(
        [
            sys.executable,
            "libpim/tests/ref_decode.py",
            "vectors",
            x / "decode.txt",
        ],
        s.gen / "ref-decode-vectors.log",
    )


def xrun(s, name, cmd, out=None, cwd=ROOT, env=None):
    """Run a command of the xcheck group into runs/<name>.<out or log>."""
    p = subprocess.run(
        [str(c) for c in cmd],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    (s.runs / f"{name}.{out or 'log'}").write_text(p.stdout + p.stderr)
    return p.returncode


def tasks_xcheck(s):
    x = s.gen / "xmodel"
    py = [sys.executable, REF]
    gem5m, cm = x / "test_model_gem5", x / "test_model_c"

    def vectors():
        a = xrun(s, "xv-c", [TEST_MODEL, "--dump"] + VECTORS, "dump")
        b = xrun(s, "xv-py", py + ["--dump"] + VECTORS, "dump")
        c = xrun(s, "xv-py-checks", py + VECTORS)
        (s.runs / "xcheck-vectors.rc").write_text(f"{a} {b} {c}\n")

    def model():
        rcs = [
            xrun(s, "xm-gem5-builtin", [gem5m]),
            xrun(s, "xm-gem5-vectors", [gem5m] + VECTORS),
            xrun(s, "xm-gem5", [gem5m, "--dump"] + VECTORS, "dump"),
            xrun(s, "xm-c", [cm, "--dump"] + VECTORS, "dump"),
            xrun(
                s,
                "xr-gem5-nonan",
                [gem5m, "--dump", x / "random_nonan.txt"],
                "dump",
            ),
            xrun(
                s, "xr-c-nonan", [cm, "--dump", x / "random_nonan.txt"], "dump"
            ),
            xrun(s, "xr-gem5", [gem5m, "--dump", x / "random.txt"], "dump"),
            xrun(s, "xr-py", py + ["--dump", x / "random.txt"], "dump"),
        ]
        (s.runs / "xcheck-model.rc").write_text(" ".join(map(str, rcs)) + "\n")

    def geom():
        rc = xrun(
            s,
            "xcheck-geom",
            [sys.executable, f"{XMODEL}/geom_py.py", x / "geom.txt"],
        )
        (s.runs / "xcheck-geom.rc").write_text(f"{rc}\n")

    def decode():
        d = r2_scratch(s)
        if d is None:
            return
        rc = xrun(
            s,
            "xcheck-decode",
            [
                r2_python(),
                ROOT / XMODEL / "xcheck_ref_decode.py",
                x / "decode.txt",
            ],
            cwd=d,
            env=r2_env(),
        )
        (s.runs / "xcheck-decode.rc").write_text(f"{rc}\n")

    for name, fn in (
        ("xcheck-vectors", vectors),
        ("xcheck-model", model),
        ("xcheck-geom", geom),
        ("xcheck-decode", decode),
    ):
        if s.want(name):
            s.task(name, fn)


def checks_xcheck(s):
    def same(a, b):
        pa, pb = s.runs / a, s.runs / b
        return pa.exists() and pa.read_bytes() == pb.read_bytes() != b""

    def last(name):
        p = s.runs / name
        t = p.read_text().strip().splitlines() if p.exists() else []
        return t[-1] if t else "(no output)"

    def vectors():
        if not TEST_MODEL.exists():
            return None, f"SKIP: no {TEST_MODEL.relative_to(ROOT)}"
        rcs = (s.runs / "xcheck-vectors.rc").read_text().split()
        eq = same("xv-c.dump", "xv-py.dump")
        n = len((s.runs / "xv-py.dump").read_text().splitlines())
        return (
            eq and rcs == ["0", "0", "0"],
            f"pim_ref.py --dump equals test_model --dump: {eq} ({n} lines); "
            f"{last('xv-py-checks.log')}",
        )

    s.check("xcheck-vectors", vectors)

    def model():
        rcs = (s.runs / "xcheck-model.rc").read_text().split()
        v = same("xm-gem5.dump", "xm-c.dump")
        r1 = same("xr-gem5-nonan.dump", "xr-c-nonan.dump")
        r2 = same("xr-gem5.dump", "xr-py.dump")
        n = len((s.runs / "xr-gem5.dump").read_text().splitlines())
        return (
            v and r1 and r2 and set(rcs) == {"0"},
            f"gem5 model: {last('xm-gem5-builtin.log')}, "
            f"{last('xm-gem5-vectors.log')}; vectors gem5 = C: {v}; "
            f"300 random programs gem5 = C (no NaN): {r1}, gem5 = pim_ref "
            f"(with NaN): {r2} ({n} states)",
        )

    s.check("xcheck-model", model)
    s.check(
        "xcheck-geom",
        lambda: (
            (s.runs / "xcheck-geom.rc").read_text().strip() == "0",
            last("xcheck-geom.log"),
        ),
    )
    if r2_scratch(s) is None:
        s.check("xcheck-decode", lambda: r2_missing(s))
    else:
        s.check(
            "xcheck-decode",
            lambda: (
                (s.runs / "xcheck-decode.rc").read_text().strip() == "0",
                last("xcheck-decode.log"),
            ),
        )


# ---------------------------------------------------------------------
# r6: the default builds stay free of Ramulator 2.1.
# ---------------------------------------------------------------------

DEFAULT_BUILDS = {"RISCV": RISCV_OPT, "ALL": ALL_OPT}
FORBIDDEN = re.compile(r"Ramulator|PimRamulator2|PimTester|PimSequencer")
# The relink noise of `size` text (seen: up to about 15 KB on ALL, and
# 23 KB between a clean and an incremental ALL build).
SIZE_NOISE = 32 * 1024


def text_size(binary):
    out = subprocess.run(
        ["size", str(binary)], capture_output=True, text=True
    ).stdout.splitlines()
    return int(out[1].split()[0]) if len(out) > 1 else None


def tasks_r6(s):
    for k, b in DEFAULT_BUILDS.items():
        name = f"r6-symbols-{k}"
        if b.exists() and s.want(name):

            def nm(name=name, b=b):
                p = subprocess.run(
                    ["nm", "-C", str(b)], capture_output=True, text=True
                )
                hits = sorted(
                    {
                        line.split(None, 2)[-1]
                        for line in p.stdout.splitlines()
                        if FORBIDDEN.search(line)
                    }
                )
                (s.runs / f"{name}.txt").write_text(
                    f"{len(p.stdout.splitlines())} symbols\n"
                    + "".join(h + "\n" for h in hits)
                )

            s.task(name, nm)
    if ALL_OPT.exists() and s.want("r6-all-objects"):
        s.task(
            "r6-all-objects",
            lambda: s.g5(
                "r6-all-objects",
                [
                    "-c",
                    "import m5.objects as o; "
                    "print('Ramulator2VectorPorts', "
                    "hasattr(o, 'Ramulator2VectorPorts'))",
                ],
                binary=ALL_OPT,
            ),
        )


def checks_r6(s):
    for k, b in DEFAULT_BUILDS.items():
        if not b.exists():
            s.check(f"r6-{k}", lambda b=b: (None, f"SKIP: {b} not found"))
            continue

        def objects(k=k):
            objs = [
                str(p.relative_to(ROOT))
                for p in (ROOT / "build" / k).rglob("*ramulator2*")
                if p.suffix in (".o", ".do", ".fo", ".a")
            ]
            return not objs, f"ramulator2 objects in build/{k}: {objs or 0}"

        s.check(f"r6-objects-{k}", objects)

        def symbols(k=k):
            t = (s.runs / f"r6-symbols-{k}.txt").read_text().splitlines()
            return (
                len(t) == 1,
                f"{t[0]}; forbidden: {t[1:4] or 'none'}"
                + (" ..." if len(t) > 4 else ""),
            )

        s.check(f"r6-symbols-{k}", symbols)

        def size(k=k, b=b):
            ref = Path(s.args.size_ref)
            got = text_size(b)
            # The first run on a machine records the sizes it measures; later
            # runs compare with them (a fresh build of another compiler
            # differs by more than the relink noise).
            first = not ref.exists() or not any(
                x.startswith(f"{k} ") for x in ref.read_text().splitlines()
            )
            if s.args.accept_sizes or first:
                old = ref.read_text() if ref.exists() else ""
                kept = [x for x in old.splitlines() if not x.startswith(k)]
                ref.parent.mkdir(parents=True, exist_ok=True)
                ref.write_text(
                    "".join(x + "\n" for x in kept + [f"{k} {got}"])
                )
            vals = dict(
                x.split()
                for x in (ref.read_text() if ref.exists() else "").splitlines()
            )
            d = got - int(vals[k])
            if first:
                return True, f"text {got} B, recorded as accepted in {ref}"
            return (
                abs(d) <= SIZE_NOISE,
                f"text {got} B, accepted {vals[k]} B ({d:+d} B; noise "
                f"{SIZE_NOISE} B)",
            )

        s.check(f"r6-size-{k}", size)

    def all_objects():
        if not ALL_OPT.exists():
            return None, f"SKIP: {ALL_OPT} not found"
        hit = "Ramulator2VectorPorts False" in s.text("r6-all-objects")
        return hit, f"Ramulator2VectorPorts absent from ALL: {hit}"

    s.check("r6-all-objects", all_objects)


# ---------------------------------------------------------------------
# fs-way1: the Way 1 request-path run, FS Linux with the stock kernel.
# ---------------------------------------------------------------------

FS_WAY1_LAYOUT = "include/gem5/pim/pim_layout_lpddr6.h"
FS_WAY1_PROGRAM = "tests/test-progs/pim-path/bin/riscv/linux/pim-path"
# (name suffix, CPU after the switch, boot CPU or None for no switch,
# mapping: "regions" maps the bank buffer and the register rows;
# "whole" the whole PIM memory)
FS_WAY1_RUNS = (
    ("atomic", "atomic", None, "regions"),
    ("timing", "timing", "atomic", "regions"),
    ("minor", "minor", "atomic", "regions"),
    ("o3", "o3", "atomic", "regions"),
)
FS_WAY1_WHOLE_RUNS = (("atomic-whole", "atomic", None, "whole"),)


def fs_way1_layout():
    text = (ROOT / FS_WAY1_LAYOUT).read_text()
    return {
        m.group(1): int(m.group(2), 0)
        for m in re.finditer(
            r"#define PIM_LAYOUT_(\w+) (0x[0-9a-fA-F]+|\d+)", text
        )
    }


def prepare_fs_way1(s):
    s.sh(
        ["make", "-C", "tests/test-progs/pim-path/src"],
        s.gen / "make-pim-path.log",
    )


def tasks_fs_way1(s, runs=FS_WAY1_RUNS):
    if not OPT.exists():
        return
    lay = fs_way1_layout()
    for suffix, cpu, boot, mapping in runs:
        pim_args = [
            hex(lay["WINDOW_BASE_FS"]),
            hex(lay["WINDOW_SIZE"]),
            "all",
            hex(lay["REG_ROW0"] << lay["ROW_SHIFT"]),
            mapping,
        ]
        name = f"fs-way1-{suffix}"
        if not s.want(name):
            continue
        argv = [
            EXAMPLE,
            "--mode",
            "fs",
            "--boot",
            "no-systemd",
            "--cpu",
            cpu,
            "--fs-program",
            str(ROOT / FS_WAY1_PROGRAM),
            "--debug-from",
            "work",
        ]
        if boot:
            argv += ["--boot-cpu", boot]
        argv += ["--args"] + pim_args
        s.task(
            name,
            lambda name=name, argv=argv: s.g5(
                name,
                argv,
                binary=OPT,
                flags="PIM",
                redirect=True,
            ),
        )


def checks_fs_way1(s, runs=FS_WAY1_RUNS):
    for suffix, cpu, _, mapping in runs:
        name = f"fs-way1-{suffix}"
        if not OPT.exists():
            s.check(name, lambda: (None, f"SKIP: {OPT} not found"))
            continue

        def fn(name=name, cpu=cpu, mapping=mapping):
            d = s.runs / name
            vlen = re.search(r"VLEN (\d+)\)", s.text(name))
            cmd = [
                sys.executable,
                "tests/gem5/pim/check_fs_linux.py",
                "--dir",
                str(d),
                "--vlen",
                vlen.group(1) if vlen else "256",
                "--cpu",
                cpu,
                "--layout",
                FS_WAY1_LAYOUT,
            ] + (["--regions"] if mapping == "regions" else [])
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            (s.runs / f"{name}.check").write_text(p.stdout + p.stderr)
            fails = re.findall(r"^FAIL (.*)$", p.stdout, re.M)
            npass = len(re.findall(r"^PASS", p.stdout, re.M))
            trace = d / "trace.out"
            if trace.exists():
                subprocess.run(["gzip", "-f", str(trace)])
            return (
                s.rc(name) == 0 and p.returncode == 0,
                f"rc={s.rc(name)} {npass} PASS"
                + (f"; FAIL: {'; '.join(fails)}" if fails else ""),
            )

        s.check(name, fn)


# ---------------------------------------------------------------------
# fs-protocol: the full flush protocol under Linux.
# ---------------------------------------------------------------------

# util/pim-kernel/build.sh's outputs: Linux 6.8.12 with NUMA and
# user cache-block operations, and the image copy with its modules.
PIM_KERNEL_WORK = Path(
    os.environ.get("PIM_KERNEL_WORK", Path.home() / "pim-kernel")
)
PIM_KERNEL = PIM_KERNEL_WORK / "vmlinux"
PIM_IMAGE = PIM_KERNEL_WORK / "riscv-ubuntu-24.04-img-pim"
# One Atomic no_systemd boot per FS configuration,
# checkpointed when after_boot.sh starts; every case restores it.
LIB_REF = HERE / "ref"
CKPT_ROOT = ROOT / "build_logs" / "p6" / "ckpt"
FSP_CKPT = "lpddr6-pimkernel-2c"
FSP_BOOT = [
    EXAMPLE,
    "--mode",
    "fs",
    "--boot",
    "no-systemd",
    "--cores",
    "2",
    "--boot-cpu",
    "atomic",
    "--kernel",
    str(PIM_KERNEL),
    "--disk-image",
    str(PIM_IMAGE),
]
FSP_PROGS = {
    "gemv": "tests/test-progs/pim-gemv",
    "eltwise": "tests/test-progs/pim-eltwise",
}
FSP_CPUS = ("timing", "o3")
# pim-path's protocol modes P1-P6 and P5 with the
# dirty-read diagnostic as a warning.
FSP_PROTO = ("p1", "p2", "p3", "p4", "p5", "p5w", "p6")
FSP_PROTO_FLAGS = "Cache,PIM,CoherentXBar"
# A restore fills the whole 64 GiB of guest memory (OS and PIM memories)
# in host memory, so at most this many run at once.
FSP_PARALLEL = threading.Semaphore(int(os.environ.get("PIM_FS_PARALLEL", "4")))
CKPT_GB = "123456789abcdef 1111222233334444 5555666677778888 99990000aaaabbbb"
CKPT_REG = (
    "3f8000003f800000 4000000040400000 4080000040a00000 40c0000040e00000"
)


# Bumped when riscv-pim.py's FS defaults change the booted state without a
# change of the arguments (2: Sv48 by default).
CKPT_EPOCH = 2


def ckpt_key(argv):
    """The inputs a boot checkpoint depends on: its arguments, the kernel,
    the image and CKPT_EPOCH. A change of the board, the memory or the CPUs'
    ISA configuration shows in a restored run as a device tree or an
    ``isa`` section of config.ini that differs from the checkpoint's
    (fsp_same_dt)."""
    h = hashlib.sha256()
    h.update(f"epoch {CKPT_EPOCH}".encode())
    h.update(json.dumps([str(a) for a in argv]).encode())
    h.update(PIM_KERNEL.read_bytes())
    st = PIM_IMAGE.stat()
    h.update(f"{st.st_size} {st.st_mtime_ns}".encode())
    return h.hexdigest()


def boot_checkpoint(s, name, argv):
    """The checkpoint directory of configuration <name>, booted (gem5.opt,
    Atomic, no_systemd) when absent or when its inputs changed."""
    d = CKPT_ROOT / name
    key = ckpt_key(argv)
    if (d / "key.txt").exists() and (d / "key.txt").read_text() == key:
        if (d / "ckpt" / "m5.cpt").exists():
            return d / "ckpt"
    if d.exists():
        shutil.rmtree(d)
    d.parent.mkdir(parents=True, exist_ok=True)
    print(f"  booting {name} for its checkpoint (about an hour)", flush=True)
    cmd = (
        [OPT, "-re", "-d", d]
        + list(argv)
        + [
            "--cpu",
            "timing",
            "--switch-at",
            "marker",
            "--checkpoint-at",
            "after-boot",
        ]
    )
    rc = s.sh(cmd, CKPT_ROOT / f"{name}.log")
    if rc != 0 or not (d / "ckpt" / "m5.cpt").exists():
        return None
    (d / "key.txt").write_text(key)
    return d / "ckpt"


def fsp_layout():
    return fs_way1_layout()


def fsp_path_args(mode):
    lay = fsp_layout()
    return [
        hex(lay["WINDOW_BASE_FS"]),
        hex(lay["WINDOW_SIZE"]),
        mode,
        hex(lay["REG_ROW0"] << lay["ROW_SHIFT"]),
        "regions",
    ]


def fsp_ready():
    return PIM_KERNEL.exists() and PIM_IMAGE.exists()


def prepare_fs_protocol(s):
    prepare_fs_way1(s)
    for prog, d in FSP_PROGS.items():
        for extra in ([], ["FS_MARKERS=1"], ["TARGET=host"]):
            s.sh(
                ["make", "-C", f"{d}/src", "PIM_STD=lpddr6"] + extra,
                s.gen / f"make-{prog}-{'-'.join(extra) or 'se'}.log",
            )
    s.fsp_ckpt = None
    if OPT.exists() and fsp_ready():
        s.fsp_ckpt = boot_checkpoint(s, FSP_CKPT, FSP_BOOT)


def fsp_names():
    names = ["fsp-ckpt-take"] + [f"fsp-ckpt-{c}" for c in FSP_CPUS]
    names += [f"fsp-{p}-{c}" for p in FSP_PROGS for c in FSP_CPUS]
    names += [f"fsp-{m}-{c}" for m in FSP_PROTO for c in FSP_CPUS]
    return names


def tasks_fs_protocol(s):
    ckpt = getattr(s, "fsp_ckpt", None)
    if not ckpt:
        return
    path_prog = str(ROOT / FS_WAY1_PROGRAM)

    def restore(cpu, ck=ckpt):
        return FSP_BOOT + ["--cpu", cpu, "--restore", str(ck)]

    def g5(name, argv, flags=None):
        with FSP_PARALLEL:
            return s.g5(name, argv, flags=flags, redirect=True)

    def run(name, argv, flags=None):
        if s.want(name):
            s.task(name, lambda: g5(name, argv, flags))

    # The checkpoint case: PIM state written on
    # Atomic, checkpointed after the PIM work, read back on Timing and O3.
    take = s.runs / "fsp-ckpt-take" / "ckpt"

    def ckpt_chain():
        rc = g5(
            "fsp-ckpt-take",
            restore("timing")
            + [
                "--switch-at",
                "marker",
                "--checkpoint-at",
                "marker",
                "--fs-program",
                path_prog,
                "--args",
            ]
            + fsp_path_args("ckpt"),
        )
        if rc != 0 or not (take / "m5.cpt").exists():
            return
        with concurrent.futures.ThreadPoolExecutor(len(FSP_CPUS)) as ex:
            list(
                ex.map(
                    lambda cpu: g5(
                        f"fsp-ckpt-{cpu}",
                        restore(cpu, take)
                        + [
                            "--switch-at",
                            "marker",
                            "--fs-program",
                            path_prog,
                            "--args",
                        ]
                        + fsp_path_args("ckpt"),
                    ),
                    FSP_CPUS,
                )
            )

    if any(s.want(n) for n in fsp_names()[:3]):
        s.task("fsp-ckpt", ckpt_chain)

    # The library's programs, built with the markers.
    for prog, d in FSP_PROGS.items():
        binary = ROOT / d / "bin" / "riscv-fs" / "lpddr6" / f"pim-{prog}"
        args = (LIB_REF / f"pim-{prog}" / "lpddr6.args").read_text().split()
        for cpu in FSP_CPUS:
            run(
                f"fsp-{prog}-{cpu}",
                restore(cpu)
                + [
                    "--switch-at",
                    "marker",
                    "--fs-program",
                    str(binary),
                    "--fs-env",
                    "PIM_MODE=fs",
                    "--args",
                ]
                + args,
            )
    # pim-path's P1-P6 under Linux, two cores.
    for mode in FSP_PROTO:
        for cpu in FSP_CPUS:
            extra = ["--dirty-read", "warn"] if mode == "p5w" else []
            run(
                f"fsp-{mode}-{cpu}",
                restore(cpu)
                + extra
                + [
                    "--switch-at",
                    "marker",
                    "--debug-from",
                    "switch",
                    "--fs-program",
                    path_prog,
                    "--fs-env",
                    "PIM_PATH_MARKERS=1",
                    "--args",
                ]
                + fsp_path_args(mode[:2]),
                flags=FSP_PROTO_FLAGS,
            )


def fsp_console(s, name):
    p = s.runs / name / "board.platform.terminal"
    return p.read_text(errors="replace") if p.exists() else ""


def fsp_trace(s, name):
    p = s.runs / name / "trace.out"
    if p.exists():
        return p.read_text(errors="replace")
    p = p.with_suffix(".out.gz")
    if p.exists():
        import gzip

        return gzip.open(p, "rt", errors="replace").read()
    return ""


def isa_sections(path):
    """The ``isa`` sections of a config.ini (the CPUs' ISA settings, for
    example ``max_paging_mode``), as one text."""
    if not path.exists():
        return None
    out, keep = [], False
    for line in path.read_text().splitlines():
        if line.startswith("["):
            keep = line.endswith(".isa]")
        if keep:
            out.append(line)
    return "\n".join(out)


def fsp_same_dt(s, name, ckpt=FSP_CKPT):
    """The restored run's device tree and CPUs' ISA settings are the boot
    checkpoint's: the board, the memory and the harts the kernel booted on
    (the checkpoint is re-taken on a change)."""
    a = s.runs / name / "device.dts"
    b = CKPT_ROOT / ckpt / "device.dts"
    isa_a = isa_sections(s.runs / name / "config.ini")
    isa_b = isa_sections(CKPT_ROOT / ckpt / "config.ini")
    return (
        a.exists()
        and b.exists()
        and a.read_bytes() == b.read_bytes()
        and isa_a is not None
        and isa_a == isa_b
    )


def fsp_switched(s, name, cpu):
    return f"PIM-FS switched from atomic to {cpu}" in s.text(name)


def fs_lib_fails(s, n, prog, cpu, host, args, ref, lay):
    """The checks of an FS library run <n> (``--switch-at
    marker``): the program's console output equals the reference, the CPU
    switched at the first marker and the run ended at the second, the PIM
    requests and column commands equal the host build's prediction (the
    statistics from the first marker on), no software error, no fatal."""
    import run_lib_se as L

    fails = []
    con = fsp_console(s, n)
    first = ref.splitlines()[0].split()[0]
    i = con.find(first + " ")
    j = con.find("outside bf16 tolerance:", i)
    got = con[i : con.find("\n", j) + 1] if i >= 0 <= j else ""
    got = got.replace("\r\n", "\n")
    # The library's own diagnostics (PIM_VERBOSE, on stderr) share the
    # console.
    got = "".join(
        line for line in got.splitlines(True) if not line.startswith("pim: ")
    )
    if got != ref:
        fails.append("output differs from the reference")
    if not fsp_switched(s, n, cpu):
        fails.append("no switch at the marker")
    if "PIM-FS marker end" not in s.text(n):
        fails.append("no end marker")
    _, _, pred = L.predict(host, args, lay["COL_BYTES"], s.gen, f"fsp-{prog}")
    st = {
        k: float(v)
        for k, v in s.stats(n).items()
        if re.match(r"^-?[\d.]+(e[-+]?\d+)?$", v)
    }
    pm = "board.memory.pim_memory.ramulator2"
    for op, want in pred["reqs"].items():
        if st.get(f"{pm}.pim.reqs::{op}", 0) != want:
            fails.append(
                f"pim.reqs::{op} "
                f"{st.get(f'{pm}.pim.reqs::{op}', 0):g}, "
                f"predicted {want}"
            )
    for (ch, cls), want in pred["cols"].items():
        k = f"{pm}.channel{ch}.controller_plugin.pim_col_cmds_{cls}"
        if st.get(k) != want:
            fails.append(f"channel{ch} {cls} {st.get(k)}, predicted {want}")
    if st.get(f"{pm}.pim.swErrors", 0):
        fails.append("PIM software errors")
    for w in ("fatal:", "panic:"):
        if w in s.text(n):
            fails.append(f"gem5 printed {w}")
    return fails


def checks_fs_protocol(s):
    names = fsp_names()
    if not getattr(s, "fsp_ckpt", True):
        for n in names:
            s.check(
                n,
                lambda: (
                    False,
                    f"no boot checkpoint: see {CKPT_ROOT / FSP_CKPT}.log",
                ),
            )
        return
    if not OPT.exists() or not fsp_ready():
        why = (
            f"{OPT} not found"
            if not OPT.exists()
            else (
                f"no {PIM_KERNEL} or {PIM_IMAGE}: run util/pim-kernel/build.sh"
            )
        )
        for n in names:
            s.check(n, lambda why=why: (None, f"SKIP: {why}"))
        return
    lay = fsp_layout()
    base, size = lay["WINDOW_BASE_FS"], lay["WINDOW_SIZE"]

    def check(name, fn):
        """Every case restores the boot checkpoint: its device tree must be
        the one the kernel booted on."""

        def wrapped():
            if not fsp_same_dt(s, name):
                return False, (
                    f"device.dts or the ISA settings differ from {FSP_CKPT}'s: "
                    "the board, memory or CPUs changed; delete the "
                    "checkpoint to re-take it"
                )
            return fn()

        s.check(name, wrapped)

    ckpt_re = re.compile(
        r"PIM-PATH ckpt (before|after) gb ((?:\S+ ){3}\S+) reg ((?:\S+ ){3}\S+)"
    )

    def ckpt_lines(name):
        return {
            m.group(1): (m.group(2), m.group(3))
            for m in ckpt_re.finditer(fsp_console(s, name))
        }

    def take():
        n = "fsp-ckpt-take"
        before = ckpt_lines(n).get("before")
        saved = (s.runs / n / "ckpt" / "m5.cpt").exists()
        ok = s.rc(n) == 0 and saved and before == (CKPT_GB, CKPT_REG)
        return ok, (
            f"rc={s.rc(n)} checkpoint {saved}; read back before it: "
            f"{before}"
        )

    check("fsp-ckpt-take", take)
    for cpu in FSP_CPUS:

        def restored(cpu=cpu):
            n = f"fsp-ckpt-{cpu}"
            got = ckpt_lines(n).get("after")
            want = ckpt_lines("fsp-ckpt-take").get("before")
            ok = (
                s.rc(n) == 0
                and fsp_switched(s, n, cpu)
                and want is not None
                and got == want
                and "PIM-PATH done ckpt" in fsp_console(s, n)
            )
            return ok, (
                f"rc={s.rc(n)} switched {fsp_switched(s, n, cpu)}; "
                f"after the restore on {cpu}: {got}, before: {want}"
            )

        check(f"fsp-ckpt-{cpu}", restored)

    # The library's output and counts, and the cacheability of normal
    # accesses to the PIM memory.
    for prog, d in FSP_PROGS.items():
        host = ROOT / d / "bin" / "host" / "lpddr6" / f"pim-{prog}"
        args = (LIB_REF / f"pim-{prog}" / "lpddr6.args").read_text().split()
        ref_text = (LIB_REF / f"pim-{prog}" / "lpddr6.out").read_text()
        for cpu in FSP_CPUS:

            def lib(prog=prog, cpu=cpu, host=host, args=args, ref=ref_text):
                n = f"fsp-{prog}-{cpu}"
                fails = fs_lib_fails(s, n, prog, cpu, host, args, ref, lay)
                # Normal accesses to the PIM memory are cacheable: every
                # normal read and write the PIM memory saw
                # was a whole cache line (a fill or a write-back), as an
                # uncacheable access would carry only its own bytes.
                pm = "board.memory.pim_memory.ramulator2"
                st = s.stats(n)
                v = {
                    k: float(st.get(f"{pm}.{k}::total", 0))
                    for k in (
                        "numReads",
                        "bytesRead",
                        "numWrites",
                        "bytesWritten",
                    )
                }
                lines = (
                    v["numReads"] > 0
                    and v["bytesRead"] == 64 * v["numReads"]
                    and v["bytesWritten"] == 64 * v["numWrites"]
                )
                if not lines:
                    fails.append(f"normal accesses to the PIM memory: {v}")
                return s.rc(n) == 0 and not fails, (
                    f"rc={s.rc(n)} "
                    + ("; ".join(fails) if fails else "output, counts equal")
                    + f"; {v['numReads']:.0f} normal reads and "
                    f"{v['numWrites']:.0f} writes at the PIM memory, all "
                    "whole lines"
                )

            check(f"fsp-{prog}-{cpu}", lib)

    # P1-P6 under Linux: the path group's criteria, on physical addresses.
    def inwin(n, cmd):
        return sum(
            1
            for m in re.finditer(
                r"\b" + cmd + r" \[([0-9a-f]+):", fsp_trace(s, n)
            )
            if base <= int(m.group(1), 16) < base + size
        )

    def st(n, pattern):
        return {
            k: float(v) for k, v in s.stats(n).items() if re.search(pattern, k)
        }

    def done(n):
        return "PIM-PATH done" in fsp_console(s, n)

    line = base + 4096  # pim_path.c's LINE_OFF
    for cpu in FSP_CPUS:
        n = lambda m, cpu=cpu: f"fsp-{m}-{cpu}"

        def p1(n=n("p1")):
            a, b = inwin(n, "CleanSharedReq"), inwin(n, "CleanInvalidReq")
            kept = "PIM-PATH inval kept 8/8 zero 8/8" in fsp_console(s, n)
            return (
                s.rc(n) == 0 and done(n) and a > 0 and b > 0 and kept,
                f"rc={s.rc(n)} CleanSharedReq {a} CleanInvalidReq {b} "
                f"inval kept and zero: {kept}",
            )

        def p2(n=n("p2")):
            order = []
            for t in fsp_trace(s, n).splitlines():
                m = re.search(
                    r"membus: recv(?:Timing|Atomic).* packet "
                    r"(WritebackDirty|PimSbMacReq) \[([0-9a-f]+):",
                    t,
                )
                if m and "SF" in t and line <= int(m.group(2), 16) < line + 64:
                    if not order or order[-1] != m.group(1):
                        order.append(m.group(1))
            # Under Linux, system activity between the store and the PIM
            # write can move the dirty line from L1D to L2 (seen on O3):
            # the PIM write finds it there and L2 writes it back first.
            dirty = st(n, r"(l1d|l2).*pimDirtyBlk$")
            ok = order[:2] == ["WritebackDirty", "PimSbMacReq"] and any(
                v == 1 for v in dirty.values()
            )
            return (
                s.rc(n) == 0 and done(n) and ok,
                f"rc={s.rc(n)} order {order[:2]} "
                f"pimDirtyBlk {list(dirty.values())}",
            )

        def p3(n=n("p3")):
            v = st(n, r"pimSnoopValidBlk")
            return (
                s.rc(n) == 0 and done(n) and not any(v.values()),
                f"rc={s.rc(n)} pimSnoopValidBlk {v or 'none'}",
            )

        def p5w(n=n("p5w")):
            w = s.text(n).count("warn: PIM RdGb")
            v = st(n, r"pimDirtyPeerReads$")
            return (
                s.rc(n) == 0 and done(n) and w == 1 and sum(v.values()) == 1,
                f"rc={s.rc(n)} warnings {w} pimDirtyPeerReads "
                f"{sum(v.values())}",
            )

        def p6(n=n("p6")):
            v = st(n, r"l1d.*pimSnoopValidBlk$")
            return (
                s.rc(n) == 0 and done(n) and any(x == 1 for x in v.values()),
                f"rc={s.rc(n)} pimSnoopValidBlk {list(v.values())}",
            )

        for m, fn in (
            ("p1", p1),
            ("p2", p2),
            ("p3", p3),
            ("p5w", p5w),
            ("p6", p6),
        ):
            check(n(m), fn)
        check(
            n("p4"),
            lambda n=n("p4"): nonzero_with(s, n, "PIM SbMac request for"),
        )
        check(
            n("p5"),
            lambda n=n("p5"): nonzero_with(s, n, "PIM RdGb request for"),
        )


# ---------------------------------------------------------------------
# fs-way2: the library under Linux on Way 2.
# ---------------------------------------------------------------------

FSW2_CKPT = "lpddr6-way2-pimkernel-2c"
FSW2_BOOT = FSP_BOOT + ["--way", "2"]
NODE_HP = (
    "/sys/devices/system/node/node{}/hugepages/hugepages-2048kB/"
    "nr_hugepages"
)
# Huge pages for the programs on the PIM node;
# the wrong-node case also reserves some on the OS node.
FSW2_SETUP = [f"echo 64 > {NODE_HP.format(1)}"]
FSW2_SETUP_NODE0 = FSW2_SETUP + [f"echo 16 > {NODE_HP.format(0)}"]


def fsw2_names():
    return [f"fsw2-{p}-{c}" for p in FSP_PROGS for c in FSP_CPUS] + [
        "fsw2-wrongnode"
    ]


def prepare_fs_way2(s):
    for prog, d in FSP_PROGS.items():
        for extra in (["FS_MARKERS=1"], ["TARGET=host"]):
            s.sh(
                ["make", "-C", f"{d}/src", "PIM_STD=lpddr6"] + extra,
                s.gen / f"make-w2-{prog}-{'-'.join(extra)}.log",
            )
    s.fsw2_ckpt = None
    if OPT.exists() and fsp_ready():
        s.fsw2_ckpt = boot_checkpoint(s, FSW2_CKPT, FSW2_BOOT)


def tasks_fs_way2(s):
    ckpt = getattr(s, "fsw2_ckpt", None)
    if not ckpt:
        return

    def run(name, prog, cpu, setup, env=()):
        binary = (
            ROOT
            / FSP_PROGS[prog]
            / "bin"
            / "riscv-fs"
            / "lpddr6"
            / f"pim-{prog}"
        )
        args = (LIB_REF / f"pim-{prog}" / "lpddr6.args").read_text().split()
        argv = FSW2_BOOT + [
            "--cpu",
            cpu,
            "--restore",
            str(ckpt),
            "--switch-at",
            "marker",
            "--fs-program",
            str(binary),
        ]
        for e in ["PIM_MODE=fs", "PIM_WAY=2", "PIM_VERBOSE=1", *env]:
            argv += ["--fs-env", e]
        for c in setup:
            argv += ["--fs-setup", c]
        argv += ["--args"] + args
        if s.want(name):

            def go():
                with FSP_PARALLEL:
                    return s.g5(name, argv, redirect=True)

            s.task(name, go)

    for prog in FSP_PROGS:
        for cpu in FSP_CPUS:
            run(f"fsw2-{prog}-{cpu}", prog, cpu, FSW2_SETUP)
    # Bound to the OS node, the pagemap check refuses the page.
    run("fsw2-wrongnode", "gemv", "timing", FSW2_SETUP_NODE0, ["PIM_NODE=0"])


def checks_fs_way2(s):
    names = fsw2_names()
    if not OPT.exists() or not fsp_ready():
        for n in names:
            s.check(n, lambda: (None, "SKIP: no gem5.opt or no PIM kernel"))
        return
    if not getattr(s, "fsw2_ckpt", True):
        for n in names:
            s.check(
                n,
                lambda: (
                    False,
                    f"no boot checkpoint: see {CKPT_ROOT / FSW2_CKPT}.log",
                ),
            )
        return
    lay = fsp_layout()
    base = lay["WINDOW_BASE_FS"]
    regs = lay["REG_ROW0"] << lay["ROW_SHIFT"]
    hp = 2 << 20
    dts = (CKPT_ROOT / FSW2_CKPT / "device.dts").read_text()
    m = re.search(
        rf"memory@{base:x} \{{[^}}]*numa-node-id = <(0x[0-9a-f]+)>", dts
    )
    board_node = int(m.group(1), 16) if m else None

    def frames(n):
        return [
            int(f, 16)
            for f in re.findall(
                r"pim: way 2 page \S+ frame (0x[0-9a-f]+)", fsp_console(s, n)
            )
        ]

    def check(name, fn):
        def wrapped():
            if not fsp_same_dt(s, name, FSW2_CKPT):
                return False, (
                    f"device.dts differs from {FSW2_CKPT}'s: delete the "
                    "checkpoint to re-take it"
                )
            return fn()

        s.check(name, wrapped)

    for prog, d in FSP_PROGS.items():
        host = ROOT / d / "bin" / "host" / "lpddr6" / f"pim-{prog}"
        args = (LIB_REF / f"pim-{prog}" / "lpddr6.args").read_text().split()
        ref = (LIB_REF / f"pim-{prog}" / "lpddr6.out").read_text()
        for cpu in FSP_CPUS:

            def lib(prog=prog, cpu=cpu, host=host, args=args, ref=ref):
                n = f"fsw2-{prog}-{cpu}"
                fails = fs_lib_fails(s, n, prog, cpu, host, args, ref, lay)
                node = re.search(
                    r"pim: way 2, node (\d+) \((.*?)\)", fsp_console(s, n)
                )
                if not node or int(node.group(1)) != board_node:
                    fails.append(
                        f"node {node.groups() if node else None}, the "
                        f"board's {board_node}"
                    )
                elif node.group(2) != "device tree":
                    fails.append(f"node from {node.group(2)}")
                fr = frames(n)
                bad = [
                    hex(f)
                    for f in fr
                    if f % hp
                    or not base <= f < base + regs
                    or f + hp > base + regs
                ]
                if not fr or bad:
                    fails.append(
                        f"{len(fr)} huge pages, outside the PIM node or "
                        f"unaligned: {bad[:3]}"
                    )
                return s.rc(n) == 0 and not fails, (
                    f"rc={s.rc(n)} "
                    + ("; ".join(fails) if fails else "output, counts equal")
                    + f"; node {board_node} from the device tree; "
                    f"{len(fr)} huge pages on it, 2 MiB-aligned"
                )

            check(f"fsw2-{prog}-{cpu}", lib)

    def wrongnode():
        n = "fsw2-wrongnode"
        con = fsp_console(s, n)
        refused = "not in the PIM memory" in con
        ex = re.search(r"PIM-FS exit (\d+)", con)
        ok = refused and ex is not None and ex.group(1) == "2"
        line = re.search(r"[^\n]*not in the PIM memory[^\n]*", con)
        return ok, (
            f"exit {ex.group(1) if ex else '?'}; "
            f"{line.group(0).strip() if line else 'no refusal'}"
        )

    check("fsw2-wrongnode", wrongnode)


# ---------------------------------------------------------------------
# sv48: Linux in Sv48 and the Way 1 request-path run under it.
# ---------------------------------------------------------------------

SV48_RUN = "sv48-fs-timing"
# Every hart's /proc/cpuinfo line, printed after the pim-path program.
SV48_CPUINFO = "grep '^mmu' /proc/cpuinfo"


def prepare_sv48(s):
    prepare_fs_way1(s)
    sys.path.insert(0, str(HERE))
    from fs_script import make_script

    lay = fs_way1_layout()
    pim_args = [
        hex(lay["WINDOW_BASE_FS"]),
        hex(lay["WINDOW_SIZE"]),
        "all",
        hex(lay["REG_ROW0"] << lay["ROW_SHIFT"]),
        "regions",
    ]
    (s.gen / "sv48-script.sh").write_text(
        make_script(ROOT / FS_WAY1_PROGRAM, pim_args, commands=SV48_CPUINFO)
    )


def tasks_sv48(s):
    if not OPT.exists() or not s.want(SV48_RUN):
        return
    s.task(
        SV48_RUN,
        lambda: s.g5(
            SV48_RUN,
            [
                EXAMPLE,
                "--mode",
                "fs",
                "--boot",
                "no-systemd",
                "--boot-cpu",
                "atomic",
                "--cpu",
                "timing",
                "--paging-mode",
                "sv48",
                "--fs-script",
                str(s.gen / "sv48-script.sh"),
                "--debug-from",
                "work",
            ],
            binary=OPT,
            flags="PIM",
            redirect=True,
        ),
    )


def checks_sv48(s):
    if not OPT.exists():
        s.check(SV48_RUN, lambda: (None, f"SKIP: {OPT} not found"))
        return

    def mmu():
        term = s.runs / SV48_RUN / "board.platform.terminal"
        text = term.read_text(errors="replace") if term.exists() else ""
        modes = re.findall(r"^mmu\s*:\s*(\S+)", text, re.M)
        cfg = s.runs / SV48_RUN / "config.ini"
        cfg = cfg.read_text() if cfg.exists() else ""
        params = set(re.findall(r"^max_paging_mode=(\S+)", cfg, re.M))
        return (
            s.rc(SV48_RUN) == 0
            and modes
            and set(modes) == {"sv48"}
            and params == {"sv48"},
            f"rc={s.rc(SV48_RUN)}; /proc/cpuinfo mmu: {modes}; "
            f"max_paging_mode {sorted(params)}",
        )

    s.check(f"{SV48_RUN}-mmu", mmu)

    def d6():
        d = s.runs / SV48_RUN
        vlen = re.search(r"VLEN (\d+)\)", s.text(SV48_RUN))
        p = subprocess.run(
            [
                sys.executable,
                "tests/gem5/pim/check_fs_linux.py",
                "--dir",
                str(d),
                "--vlen",
                vlen.group(1) if vlen else "256",
                "--cpu",
                "timing",
                "--layout",
                FS_WAY1_LAYOUT,
                "--regions",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        (s.runs / f"{SV48_RUN}.check").write_text(p.stdout + p.stderr)
        fails = re.findall(r"^FAIL (.*)$", p.stdout, re.M)
        npass = len(re.findall(r"^PASS", p.stdout, re.M))
        trace = d / "trace.out"
        if trace.exists():
            subprocess.run(["gzip", "-f", str(trace)])
        return (
            p.returncode == 0,
            f"request-path checker {npass} PASS"
            + (f"; FAIL: {'; '.join(fails)}" if fails else ""),
        )

    s.check(f"{SV48_RUN}-d6", d6)


# ---------------------------------------------------------------------
# fs-ddr5: the 128 GiB DDR5 system in FS.
# ---------------------------------------------------------------------


def layout_header(path):
    """A layout header's PIM_LAYOUT_<NAME> values."""
    return {
        m.group(1): int(m.group(2), 0)
        for m in re.finditer(
            r"#define PIM_LAYOUT_(\w+) (0x[0-9a-fA-F]+|\d+)",
            Path(path).read_text(),
        )
    }


# The 128 GiB systems are explicit; 8 single-rank DIMMs, each
# memory DDR5_16Gb_x8 x 16 channels. The header sits beside the boot
# checkpoints, so that the checkpoint's arguments stay the same.
FSD5_LAYOUT = CKPT_ROOT / "pim_layout_ddr5x16.h"
FSD5_LAYOUT_NAME = "fsddr5-r1x16"
FSD5_CKPT = "ddr5x16-pimkernel-1c"
FSD5_SYSTEM = ["--memory", "ddr5", "--channels", "16", "--layout"]
FSD5_BOOT = (
    [
        EXAMPLE,
        "--mode",
        "fs",
        "--boot",
        "no-systemd",
        "--boot-cpu",
        "atomic",
        "--kernel",
        str(PIM_KERNEL),
        "--disk-image",
        str(PIM_IMAGE),
    ]
    + FSD5_SYSTEM
    + [str(FSD5_LAYOUT)]
)
# Linux's memory map of the OS memory [0x80000000, 0x2080000000): all of it
# under Sv48 (riscv-pim.py's FS default), up to 0x1F7FFFFFFF under Sv39
# (the end of Linux's Sv39 linear map).
FSD5_MAP = {"sv48": 0x207FFFFFFF, "sv39": 0x1F7FFFFFFF}
FSD5_MAP_RE = re.compile(
    r"Initmem setup node 0 \[mem (0x[0-9a-f]+)-(0x[0-9a-f]+)\]"
)
# A restore of this system fills 256 GiB of host memory: one at a time.
FSD5_PARALLEL = threading.Semaphore(1)


def fsd5_names():
    return ["fsd5-map-sv48", "fsd5-map-sv39", "fsd5-d6", "fsd5-gemv"]


def prepare_fs_ddr5(s):
    prepare_fs_way1(s)
    if not FSD5_LAYOUT.exists():
        FSD5_LAYOUT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "libpim/gen_pim_layout.py",
                "--standard",
                "ddr5",
                "--channels",
                "16",
                "-o",
                str(FSD5_LAYOUT),
            ],
            cwd=ROOT,
            check=True,
        )
    for extra in (["FS_MARKERS=1"], ["TARGET=host"]):
        s.sh(
            [
                "make",
                "-C",
                f"{GEMV}/src",
                "PIM_STD=ddr5",
                f"PIM_LAYOUT={FSD5_LAYOUT}",
                f"LAYOUT_NAME={FSD5_LAYOUT_NAME}",
            ]
            + extra,
            s.gen / f"make-d5-gemv-{'-'.join(extra)}.log",
        )
    s.fsd5_ckpt = None
    if OPT.exists() and fsp_ready():
        s.fsd5_ckpt = boot_checkpoint(s, FSD5_CKPT, FSD5_BOOT)


def tasks_fs_ddr5(s):
    if not OPT.exists() or not fsp_ready():
        return
    # Sv39: boot to the booted kernel only (its memory map).
    if s.want("fsd5-map-sv39"):
        s.task(
            "fsd5-map-sv39",
            lambda: s.g5(
                "fsd5-map-sv39",
                FSD5_BOOT
                + [
                    "--cpu",
                    "atomic",
                    "--paging-mode",
                    "sv39",
                    "--fs-exit",
                    "kernel-booted",
                ],
                binary=FAST,
                redirect=True,
            ),
        )
    ck = getattr(s, "fsd5_ckpt", None)
    if not ck:
        return
    lay = layout_header(FSD5_LAYOUT)
    restore = FSD5_BOOT + ["--cpu", "timing", "--restore", str(ck)]

    def run(name, argv, flags=None):
        if s.want(name):

            def go():
                with FSD5_PARALLEL:
                    return s.g5(name, argv, flags=flags, redirect=True)

            s.task(name, go)

    # The Way 1 request-path run (regions), as fs-way1's, on Timing.
    run(
        "fsd5-d6",
        restore
        + [
            "--switch-at",
            "work",
            "--debug-from",
            "switch",
            "--fs-program",
            str(ROOT / FS_WAY1_PROGRAM),
            "--args",
            hex(lay["WINDOW_BASE_FS"]),
            hex(lay["WINDOW_SIZE"]),
            "all",
            hex(lay["REG_ROW0"] << lay["ROW_SHIFT"]),
            "regions",
        ],
        flags="PIM",
    )
    # pim-gemv with the DDR5 arguments (pim_init maps the 128 GiB PIM
    # memory on Atomic, before the switch).
    binary = ROOT / GEMV / "bin" / "riscv-fs" / FSD5_LAYOUT_NAME / "pim-gemv"
    args = (LIB_REF / "pim-gemv" / "ddr5.args").read_text().split()
    run(
        "fsd5-gemv",
        restore
        + [
            "--switch-at",
            "marker",
            "--fs-program",
            str(binary),
            "--fs-env",
            "PIM_MODE=fs",
            "--args",
        ]
        + args,
    )


def checks_fs_ddr5(s):
    if not OPT.exists() or not fsp_ready():
        for n in fsd5_names():
            s.check(n, lambda: (None, "SKIP: no gem5.opt or no PIM kernel"))
        return

    def mapped(text, mode):
        m = FSD5_MAP_RE.search(text)
        end = int(m.group(2), 16) if m else None
        ok = m is not None and int(m.group(1), 16) == 0x80000000
        ok = ok and end == FSD5_MAP[mode]
        return ok, (
            f"Linux maps [{m.group(1)}-{m.group(2)}] under {mode}, wanted "
            f"up to {FSD5_MAP[mode]:#x}"
            if m
            else f"no memory map in the console under {mode}"
        )

    def map48():
        p = CKPT_ROOT / FSD5_CKPT / "board.platform.terminal"
        return mapped(
            p.read_text(errors="replace") if p.exists() else "", "sv48"
        )

    def map39():
        ok, why = mapped(fsp_console(s, "fsd5-map-sv39"), "sv39")
        booted = "kernel booted" in s.text("fsd5-map-sv39")
        return ok and booted, why + f"; kernel booted: {booted}"

    s.check("fsd5-map-sv48", map48)
    s.check("fsd5-map-sv39", map39)
    if not getattr(s, "fsd5_ckpt", True):
        for n in ("fsd5-d6", "fsd5-gemv"):
            s.check(
                n,
                lambda: (False, f"no boot checkpoint: see {FSD5_CKPT}.log"),
            )
        return

    def check(name, fn):
        def wrapped():
            if not fsp_same_dt(s, name, FSD5_CKPT):
                return False, (
                    f"device.dts or the ISA settings differ from "
                    f"{FSD5_CKPT}'s: delete the checkpoint to re-take it"
                )
            return fn()

        s.check(name, wrapped)

    def d6():
        n = "fsd5-d6"
        vlen = re.search(r"VLEN (\d+)\)", s.text(n))
        p = subprocess.run(
            [
                sys.executable,
                "tests/gem5/pim/check_fs_linux.py",
                "--dir",
                str(s.runs / n),
                "--vlen",
                vlen.group(1) if vlen else "512",
                "--cpu",
                "timing",
                "--layout",
                str(FSD5_LAYOUT),
                "--regions",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        (s.runs / f"{n}.check").write_text(p.stdout + p.stderr)
        fails = re.findall(r"^FAIL (.*)$", p.stdout, re.M)
        npass = len(re.findall(r"^PASS", p.stdout, re.M))
        trace = s.runs / n / "trace.out"
        if trace.exists():
            subprocess.run(["gzip", "-f", str(trace)])
        return s.rc(n) == 0 and p.returncode == 0, (
            f"rc={s.rc(n)} {npass} PASS"
            + (f"; FAIL: {'; '.join(fails)}" if fails else "")
        )

    def gemv():
        n = "fsd5-gemv"
        host = ROOT / GEMV / "bin" / "host" / FSD5_LAYOUT_NAME / "pim-gemv"
        args = (LIB_REF / "pim-gemv" / "ddr5.args").read_text().split()
        ref = (LIB_REF / "pim-gemv" / "ddr5.out").read_text()
        lay = layout_header(FSD5_LAYOUT)
        fails = fs_lib_fails(s, n, "gemv-ddr5", "timing", host, args, ref, lay)
        return s.rc(n) == 0 and not fails, f"rc={s.rc(n)} " + (
            "; ".join(fails) if fails else "output, counts equal"
        )

    check("fsd5-d6", d6)
    check("fsd5-gemv", gemv)


# ---------------------------------------------------------------------

PLAN = {
    "cpuless": (prepare_cpuless, tasks_cpuless, checks_cpuless),
    "refresh": (prepare_refresh, tasks_refresh, checks_refresh),
    "c28": (prepare_c28, tasks_c28, checks_c28),
    "path": (None, tasks_path, checks_path),
    "config": (prepare_config, tasks_config, checks_config),
    "v16": (prepare_v16, tasks_v16, checks_v16),
    "r6": (None, tasks_r6, checks_r6),
    "decode": (prepare_decode, tasks_decode, checks_decode),
    "fsbm": (prepare_fsbm, tasks_fsbm, checks_fsbm),
    "cache": (prepare_cache, tasks_cache, checks_cache),
    "se-misc": (prepare_se_misc, tasks_se_misc, checks_se_misc),
    "r2suites": (prepare_r2suites, tasks_r2suites, checks_r2suites),
    "xcheck": (prepare_xcheck, tasks_xcheck, checks_xcheck),
    "fs-boot": (None, tasks_fs_boot, checks_fs_boot),
    "fs-way1": (prepare_fs_way1, tasks_fs_way1, checks_fs_way1),
    "fs-way1-whole": (
        prepare_fs_way1,
        lambda s: tasks_fs_way1(s, FS_WAY1_WHOLE_RUNS),
        lambda s: checks_fs_way1(s, FS_WAY1_WHOLE_RUNS),
    ),
    "fs-protocol": (
        prepare_fs_protocol,
        tasks_fs_protocol,
        checks_fs_protocol,
    ),
    "fs-way2": (prepare_fs_way2, tasks_fs_way2, checks_fs_way2),
    "sv48": (prepare_sv48, tasks_sv48, checks_sv48),
    "fs-ddr5": (prepare_fs_ddr5, tasks_fs_ddr5, checks_fs_ddr5),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    ap.add_argument(
        "--groups",
        default=",".join(
            g
            for g in GROUPS
            if g
            not in (
                "fs-boot",
                "fs-way1",
                "fs-way1-whole",
                "fs-protocol",
                "fs-way2",
                "sv48",
                "fs-ddr5",
            )
        ),
        help=f"comma-separated, of {', '.join(GROUPS)}",
    )
    ap.add_argument("--only", help="regular expression on case names")
    ap.add_argument(
        "--native",
        help="the gem5 binary without Ramulator 2.1 for the native cases "
        f"(default {RISCV_OPT.relative_to(ROOT)}; r6 always uses that one)",
    )
    ap.add_argument(
        "--size-ref",
        default=str(ROOT / "build_logs/r6-size-ref.txt"),
        help="accepted text sizes of the default builds (group r6)",
    )
    ap.add_argument(
        "--accept-sizes",
        action="store_true",
        help="record the default builds' text sizes as accepted",
    )
    ap.add_argument(
        "--checks-only",
        action="store_true",
        help="evaluate the verdicts of an earlier run",
    )
    args = ap.parse_args()
    s = Suite(args)
    for g in s.groups - set(GROUPS):
        ap.error(f"unknown group {g}")
    if args.native:
        global NATIVE
        NATIVE = Path(args.native).resolve()
    if not OPT.exists() and s.groups - NATIVE_GROUPS:
        ap.error(f"{OPT} not found: build build/RISCV_RAMULATOR2/gem5.opt")
    for path in ("tests/test-progs/pim-path/src",):
        s.sh(["make", "-C", path], s.gen / "make-pim-path.log")
    for g in GROUPS:
        if g in s.groups and PLAN[g][0] and not args.checks_only:
            print(f"prepare {g}", flush=True)
            PLAN[g][0](s)
        elif g in s.groups and g in ("cpuless", "refresh", "c28"):
            # Verdicts need the geometries.
            for k, a in CPULESS_GEO.items():
                s.geometry(k, *a)
        if g == "config" and g in s.groups and args.checks_only:
            names = []
            for line in (
                (ROOT / "include/gem5/pim/pim_layout_lpddr6.h")
                .read_text()
                .splitlines()
            ):
                m = re.match(
                    r"^#define PIM_LAYOUT_(\w+) (0x[0-9a-f]+|\d+)$", line
                )
                if m:
                    names.append(m.group(1))
            s.v15_names = names
    for g in GROUPS:
        if g in s.groups:
            PLAN[g][1](s)
    if not args.checks_only:
        print(f"{len(s.tasks)} tasks, {args.jobs} at a time", flush=True)
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
            futs = {ex.submit(fn): name for name, fn in s.tasks}
            for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
                f.result()
                if i % 20 == 0 or i == len(futs):
                    print(f"  {i}/{len(futs)} done", flush=True)
    for g in GROUPS:
        if g in s.groups:
            PLAN[g][2](s)
    results = []
    for name, fn in s.checks:
        try:
            ok, detail = fn()
        except Exception as e:  # a missing file is a failure
            ok, detail = False, f"{type(e).__name__}: {e}"
        verdict = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        results.append(f"{verdict} {name}: {detail}")
    n = {
        v: sum(r.startswith(v) for r in results)
        for v in ("PASS", "FAIL", "SKIP")
    }
    summary = "\n".join(results) + (
        f"\nrun_p4r: {n['PASS']} PASS, {n['FAIL']} FAIL, {n['SKIP']} SKIP\n"
    )
    (s.out / "summary.txt").write_text(summary)
    print(summary, end="")
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
