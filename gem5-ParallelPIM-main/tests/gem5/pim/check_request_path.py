#!/usr/bin/env python3
"""Check a run of the PIM request-path test in SE.

The run is tests/gem5/pim/configs/pim_path_se.py with a decode mode
(scalar, vstore, vload, all) and
--debug-flags=PIM,Cache,MemCtrl,ExecEnable,ExecMicro,ExecEffAddr. Against the decode test's kernel table
(check_exec_trace.py), split at cache lines, it checks:
1. flags: every processing-in-memory (PIM) translation line has the
   expected operation and direction, "PIM UC SO" and no fault, one per
   request piece (Minor and O3: at least one, as they translate squashed
   micro-ops too);
2. linear translation: pa - window base == va - map base;
3. commands: every packet in the window in the Cache trace is a PIM
   request or response of an expected operation and carries UC; no
   other packet reaches the window; with caches, the window's memory
   sees the same PIM requests and nothing else (native: its MemCtrl
   trace; a PimMemory: the wrapper's PIM trace, and no request in
   Ramulator 2.1's own counters, which never count PIM requests);
4. counts: per operation, the memory bus's transDist::Pim<Op>Req and
   Pim<Op>Resp, and with caches the L1D's and L2's mshrUncacheable,
   equal the expected pieces;
5. no walker in SE: every walker's walk counts are 0.

The program's "PIM-PATH reg" line gives the address of the register
forms (pim.wr.reg, pim.rd.reg); without it they use the window's start.

Modes s1, s2 and s3 are the speculation cases on a PIM memory, run with
--debug-flags=Cache,PIM,CoherentXBar; the memory bus's request lines
give the order:
  s1  the load got the global buffer's zeros, not the store's data (no
      forwarding), and the store's write-back reached the memory bus
      before the pim.rd.gb request;
  s2  the training line got one PIM write per training iteration, the
      wrong path's line was translated (the case reached the wrong path)
      but got no request;
  s3  the sixteen pim.rd.gb requests reached the memory bus, and the PIM
      memory accepted them, in program order.

Usage: check_request_path.py --dir <run dir> --log <run log> --mode M
       --vlen V --cpu C [--caches]
       check_request_path.py --dir <run dir> --log <run log> --mode s1|s2|s3
Exit status 0 if every check passes, 1 otherwise.
"""

import argparse
import collections
import gzip
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_exec_trace import (  # noqa: E402
    MODES,
    expected_lines,
)

LINE = 64
TLB_RE = re.compile(
    r"PIM (\w+) ([RW]) va=(0x[0-9a-f]+) pa=(0x[0-9a-f]+) size=(\d+) "
    r"flags=(.*?) fault=(\S+)"
)
PKT_RE = re.compile(r"\b(\w+(?:Req|Resp)) \[([0-9a-f]+):([0-9a-f]+)\]([^\n]*)")
# The memory controllers' request lines: timing ("recvTimingReq: request
# <cmd> addr <a> size <n>") and atomic ("recvAtomic: <cmd> <a>").
CTRL_RE = re.compile(
    r"(\w+)_memory\.mem_ctrl: recv\w+: (?:request )?(\w+(?:Req|Resp)) "
    r"(?:addr )?(0x[0-9a-f]+)"
)
# A PimMemory's accepted PIM requests: timing ("accept <Op> at <a> size
# <n> (seq ...") and atomic ("atomic <Op> at <a> size <n>: U ...").
PIMMEM_RE = re.compile(
    r"pim_memory\.ramulator2: (?:accept|atomic) (\w+) at (0x[0-9a-f]+) "
    r"size \d+"
)
SPECULATIVE = ("minor", "o3")


def open_trace(run_dir):
    """trace.out, or the gzipped trace.out.gz."""
    path = os.path.join(run_dir, "trace.out")
    if os.path.exists(path):
        return open(path)
    return gzip.open(path + ".gz", "rt")


def pim_op(mnemonic):
    """pim_sb_ewadd_gbuf[_micro] -> SbEwaddGbuf."""
    parts = mnemonic[len("pim_") :].replace("_micro", "").split("_")
    return "".join(p.capitalize() for p in parts)


def pieces(mode, vlen, map_base, reg):
    """[(op, R/W, va, bytes)] of the expected requests, split at lines."""
    out = []
    for mnem, kind, va, size in expected_lines(mode, vlen, map_base, reg):
        rw = "R" if kind == "vl" else "W"
        end = va + size
        while va < end:
            stop = min(end, (va // LINE + 1) * LINE)
            out.append((pim_op(mnem), rw, va, stop - va))
            va = stop
    return out


def stats(path):
    values = {}
    for line in open(path):
        f = line.split()
        if len(f) >= 2 and not f[0].startswith("-"):
            values[f[0]] = f[1]
    return values


SPEC_MODES = ("s1", "s2", "s3")
BUS_RE = re.compile(
    r"membus: recvTimingReq: .* packet (\w+) \[([0-9a-f]+):[0-9a-f]+\]"
    r".* SF "
)


def check_spec(args):
    """The speculation cases s1, s2, s3."""
    fails = []

    def check(ok, what):
        print(("PASS " if ok else "FAIL ") + what)
        if not ok:
            fails.append(what)

    log = open(args.log).read()
    m = re.search(r"PIM window (0x[0-9a-f]+) size", log)
    done = f"PIM-PATH done {args.mode}" in log
    check(bool(m) and done, f"the program ran mode {args.mode}")
    if not m:
        return 1
    base = int(m.group(1), 16)
    bus, tlb, accepted = [], [], []
    for line in open_trace(args.dir):
        b = BUS_RE.search(line)
        if b:
            bus.append((b.group(1), int(b.group(2), 16) - base))
            continue
        t = TLB_RE.search(line)
        if t:
            tlb.append(t.groups())
            continue
        a = PIMMEM_RE.search(line)
        if a:
            accepted.append((a.group(1), int(a.group(2), 16) - base))
    if args.mode == "s1":
        v = re.search(r"PIM-PATH s1 v8 (\S+) (\S+) (\S+) (\S+)", log)
        check(
            v is not None and all(x == "0" for x in v.groups()),
            f"s1: pim.rd.gb got the global buffer (zeros), not the store "
            f"({' '.join(v.groups()) if v else 'no output'})",
        )
        line = [c for c, o in bus if 0x1000 <= o < 0x1040]
        wb = [i for i, c in enumerate(line) if c.startswith("Writeback")]
        rd = [i for i, c in enumerate(line) if c == "PimRdGbReq"]
        check(
            len(rd) == 1 and wb and wb[0] < rd[0],
            f"s1: the store's write-back reached the bus before the PIM "
            f"read ({', '.join(line)})",
        )
    elif args.mode == "s2":
        r = re.search(
            r"PIM-PATH s2 train (0x[0-9a-f]+) clean (0x[0-9a-f]+)", log
        )
        train, clean = (int(g, 16) for g in r.groups()) if r else (0, 0)
        n_train = sum(1 for c, o in bus if c == "PimSbMacReq" and o == train)
        n_clean = sum(1 for c, o in bus if c.startswith("Pim") and o == clean)
        mapped = [
            t for t in tlb if t[0] == "SbMac" and int(t[3], 16) - base == clean
        ]
        check(n_train == 31, f"s2: {n_train} PIM writes to the training line")
        check(
            mapped and n_clean == 0,
            f"s2: the wrong path's PIM write was translated "
            f"{len(mapped)} times and sent {n_clean} requests",
        )
    else:
        r = re.search(r"PIM-PATH s3 a (0x[0-9a-f]+) b (0x[0-9a-f]+)", log)
        a, b = (int(g, 16) for g in r.groups()) if r else (0, 0)
        want = [x + 64 * i for i in range(8) for x in (a, b)]
        got = [o for c, o in bus if c == "PimRdGbReq"]
        acc = [o for op, o in accepted if op == "RdGb"]
        check(got == want, f"s3: bus order of {len(got)} pim.rd.gb")
        check(acc == want, f"s3: PIM memory order of {len(acc)} pim.rd.gb")
    print(f"check_request_path: {len(fails)} failures")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--mode", required=True, choices=(*MODES, *SPEC_MODES))
    ap.add_argument("--vlen", type=int)
    ap.add_argument("--cpu")
    ap.add_argument("--caches", action="store_true")
    args = ap.parse_args()
    if args.mode in SPEC_MODES:
        return check_spec(args)
    if args.vlen is None or args.cpu is None:
        ap.error("--vlen and --cpu are required for the decode modes")
    caches = args.caches or args.cpu != "atomic"
    fails = []

    def check(ok, what):
        print(("PASS " if ok else "FAIL ") + what)
        if not ok:
            fails.append(what)

    log = open(args.log).read()
    m = re.search(
        r"PIM-PATH map (0x[0-9a-f]+) base (0x[0-9a-f]+) size "
        r"(0x[0-9a-f]+)",
        log,
    )
    if not m:
        check(False, "the program mapped the window")
        return 1
    map_base, win_base, win_size = (int(g, 16) for g in m.groups())
    r = re.search(r"PIM-PATH reg (0x[0-9a-f]+) offset", log)
    reg = int(r.group(1), 16) if r else map_base
    want = pieces(args.mode, args.vlen, map_base, reg)
    want_ops = collections.Counter(p[0] for p in want)

    tlb, packets, ctrl, outside = [], [], [], []
    pimmem = False
    for line in open_trace(args.dir):
        t = TLB_RE.search(line)
        if t:
            tlb.append(t.groups())
            continue
        c = CTRL_RE.search(line)
        if c:
            ctrl.append((c.group(1), c.group(2), int(c.group(3), 16)))
            continue
        c = PIMMEM_RE.search(line)
        if c:
            pimmem = True
            ctrl.append(("window", f"Pim{c.group(1)}Req", int(c.group(2), 16)))
            continue
        for p in PKT_RE.finditer(line):
            lo, hi = int(p.group(2), 16), int(p.group(3), 16)
            if win_base <= lo < win_base + win_size:
                packets.append((p.group(1), lo, hi, p.group(4)))
            elif p.group(1).startswith("Pim"):
                outside.append((p.group(1), lo))

    # 1. Translation lines.
    bad = [t for t in tlb if t[5].strip() != "PIM UC SO" or t[6] != "none"]
    check(
        tlb and not bad,
        f"1. {len(tlb)} PIM translations, all "
        f"'PIM UC SO', no fault{' ' + str(bad[:2]) if bad else ''}",
    )

    def in_map(t):
        return map_base <= int(t[2], 16) < map_base + win_size

    got = collections.Counter(
        (t[0], t[1], int(t[2], 16)) for t in tlb if in_map(t)
    )
    stray = [t for t in tlb if not in_map(t)]
    exp = collections.Counter((op, rw, va) for op, rw, va, size in want)
    if args.cpu in SPECULATIVE:
        # Minor and O3 translate squashed micro-ops too: again when one is
        # replayed (after a vsetvli, or a load at the head of the ROB), on
        # a wrong path, with the operands of that path, possibly outside
        # the window, and with a stale vector length, so more micro-ops of
        # an expected instruction (O3 has been seen to translate four
        # AbMac micro-ops, then the two that committed). Only committed ones
        # issue requests (checks 3 and 4). So every piece is translated,
        # and other lines in the window only for an expected operation and
        # direction.
        kinds = {(op, rw) for op, rw, va in exp}
        extra = [k for k in got if k not in exp]
        ok = all(got[k] >= n for k, n in exp.items()) and all(
            k[:2] in kinds for k in extra
        )
        rule = (
            f"at least one per request piece, others in the window only of "
            f"an expected operation ({args.cpu}; {len(extra)} other "
            f"addresses inside, {len(stray)} squashed translations outside)"
        )
    else:
        ok = got == exp and not stray
        rule = "exactly one per request piece, none outside the window"
    check(
        ok,
        f"1. translations: {rule}: {sum(got.values())} lines for "
        f"{sum(exp.values())} pieces",
    )

    # 2. Linear translation.
    lin = [
        t
        for t in tlb
        if in_map(t) and int(t[3], 16) - win_base != int(t[2], 16) - map_base
    ]
    check(
        not lin,
        "2. pa - window base == va - map base for every line in " "the window",
    )

    # 3. Commands.
    if caches:
        reqs = [p for p in packets if p[0].endswith("Req")]
        other = sorted({p[0] for p in packets if not p[0].startswith("Pim")})
        check(
            not other,
            f"3. no non-PIM packet in the window"
            f"{' (' + ', '.join(other) + ')' if other else ''}",
        )
        check(
            all(" UC" in p[3] for p in reqs),
            "3. every PIM request in " "the window carries UC",
        )
        ops = {p[0][3:-3] for p in reqs}
        check(
            ops == set(want_ops),
            f"3. the window's packets are exactly "
            f"the expected operations ({len(ops)})",
        )
        check(
            not outside,
            f"3. no PIM packet outside the window"
            f"{' ' + str(outside[:3]) if outside else ''}",
        )
        wctrl = [c for c in ctrl if c[0] == "window" and c[1].endswith("Req")]
        cops = collections.Counter(
            c[1][3:-3] for c in wctrl if c[1].startswith("Pim")
        )
        cother = [c for c in wctrl if not c[1].startswith("Pim")]
        os_pim = [c for c in ctrl if c[0] == "os" and c[1].startswith("Pim")]
        check(not os_pim, "3. the OS memory's controller sees no PIM request")
        check(
            cops == want_ops and not cother,
            f"3. the window's memory "
            f"controller sees {sum(cops.values())} PIM requests of the "
            f"expected operations and nothing else",
        )

    st = stats(os.path.join(args.dir, "stats.txt"))
    if pimmem:
        # Ramulator 2.1 counts only normal requests.
        normal = {
            k: v
            for k, v in st.items()
            if re.search(
                r"pim_memory\.ramulator2\.channel\d+\.num_"
                r"(read|write)_reqs$",
                k,
            )
        }
        check(
            normal and all(float(v) == 0 for v in normal.values()),
            f"3. no normal request at the PIM memory ({len(normal)} "
            f"Ramulator 2.1 request counters)",
        )
        sw = [v for k, v in st.items() if k.endswith("pim.swErrors")]
        check(
            all(float(v) == 0 for v in sw),
            "3. no PIM software error",
        )

    # 4. Counts.
    bus = (
        "board.cache_hierarchy.membus"
        if caches
        else "board.cache_hierarchy.membus"
    )

    def stat(key):
        hits = [v for k, v in st.items() if k.endswith(key)]
        return int(float(hits[0])) if hits else 0

    wrong = []
    for op, n in want_ops.items():
        for key in (
            f"membus.transDist::Pim{op}Req",
            f"membus.transDist::Pim{op}Resp",
        ):
            if stat(key) != n:
                wrong.append((key, stat(key), n))
        # Atomic-mode caches do not count uncacheable accesses as MSHR
        # uncacheable (BaseCache::handleAtomicReqMiss is not reached).
        if caches and args.cpu != "atomic":
            for c in ("l1d-cache-0", "l2-cache-0"):
                key = f"{c}.Pim{op}Req.mshrUncacheable::total"
                if stat(key) != n:
                    wrong.append((key, stat(key), n))
    check(
        not wrong,
        f"4. per-operation bus and cache counts equal the "
        f"expected pieces{' ' + str(wrong[:3]) if wrong else ''}",
    )

    # 5. No walker in SE.
    walks = {
        k: v
        for k, v in st.items()
        if ".walker.num_" in k and "walks" in k and float(v) != 0
    }
    check(not walks, f"5. no page walk{' ' + str(walks) if walks else ''}")
    print(f"check_request_path: {len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
