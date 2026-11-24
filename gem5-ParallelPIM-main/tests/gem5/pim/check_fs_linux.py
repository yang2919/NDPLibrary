#!/usr/bin/env python3
"""Check a run of the PIM FS Linux Way 1 test.

``configs/example/gem5_library/riscv-pim.py --mode fs`` runs
``pim-path <WINDOW_BASE_FS> <WINDOW_SIZE> all <REG_ROW0 << ROW_SHIFT>
[regions]`` as root under the stock kernel (``--fs-program``), with the
program's statistics in their own dump (hypercalls 4 and 5) and
``--debug-flags=PIM --debug-from work``. Mapping the whole 32 GiB PIM
memory makes Linux write, and at exit clear, 8.4 million page-table
entries (a packet trace of that run would be tens of GB, and the run takes
1-3 hours); with ``regions`` the program maps
only its bank buffer and the register rows, as two mappings. Either way
the bus checks use the program's statistics. Against the decode test's
kernel table (check_exec_trace.py, as check_request_path.py), it checks:

0. the program ran: the console shows its "PIM-PATH map", "PIM-PATH done
   all" and "PIM-FS exit 0" lines; it mapped the window from the layout
   header (whole), or (``--regions``) the bank buffer at its base and the
   register rows from the register offset to its end;
1. every PIM translation line of ``TLB::tracePim`` shows "PIM UC SO" and
   no fault; one per request piece (Atomic, Timing), at least one per
   piece and others in the mapping only of an expected operation (Minor,
   O3: replays, and squashed micro-ops of a stale vector length);
2. linear translation: pa - mapping's base == va - mapping's address,
   for each mapping (Linux's ``/dev/mem`` mappings are linear);
3. commands: per operation, the memory bus carried and the PIM memory
   accepted (its PIM trace, all in the window) exactly the expected
   requests, so none went elsewhere; with timing caches, the L1 data and
   L2 caches counted each as an uncacheable miss; Ramulator 2.1 counted no
   normal request at the PIM memory; no PIM software error;
4. F1 in the program's dump: dtb pimAccesses > 0, pimMisses > 0 and data
   walker walks > 0;
5. F2: Linux maps ``/dev/mem`` pages without D, so on every page with a
   store-like PIM request a store-like translation walked (the walker
   writes D back): a line marked "walk" on the
   timing models; on Atomic, whose walks are not marked, a store first on
   each page and at least one TLB miss per page;
6. F10: the program's dump counts at least as many PIM lookups as pieces
   (Atomic and Timing: exactly as many).

Usage: check_fs_linux.py --dir <run dir> --vlen V --cpu C
       --layout <pim_layout_<std>.h> [--regions]
Exit status 0 if every check passes, 1 otherwise.
"""

import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_request_path import (  # noqa: E402
    PIMMEM_RE,
    SPECULATIVE,
    TLB_RE,
    open_trace,
    pieces,
)

PAGE = 4096
WALK_RE = re.compile(r" walk\s*$")


def layout(path):
    vals = {}
    for m in re.finditer(
        r"#define PIM_LAYOUT_(\w+) (0x[0-9a-fA-F]+|\d+)", open(path).read()
    ):
        vals[m.group(1)] = int(m.group(2), 0)
    return vals


def dumps(path):
    """The statistics dumps, each as {name: value}."""
    out = []
    for part in open(path).read().split("Begin Simulation Statistics")[1:]:
        vals = {}
        for line in part.splitlines():
            f = line.split()
            if len(f) >= 2 and not f[0].startswith("-"):
                vals[f[0]] = f[1]
        out.append(vals)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--vlen", type=int, required=True)
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument(
        "--regions",
        action="store_true",
        help="the program mapped only the bank buffer and register rows",
    )
    args = ap.parse_args()
    fails = []

    def check(ok, what):
        print(("PASS " if ok else "FAIL ") + what)
        if not ok:
            fails.append(what)

    lay = layout(args.layout)
    win_base, win_size = lay["WINDOW_BASE_FS"], lay["WINDOW_SIZE"]
    console = open(
        os.path.join(args.dir, "board.platform.terminal"), errors="replace"
    ).read()
    maps = [
        tuple(int(g, 16) for g in m.groups())
        for m in re.finditer(
            r"PIM-PATH map (0x[0-9a-f]+) base (0x[0-9a-f]+) size "
            r"(0x[0-9a-f]+)",
            console,
        )
    ]
    done = "PIM-PATH done all" in console
    ex = re.search(r"PIM-FS exit (\d+)", console)
    check(
        bool(maps) and done and ex is not None and ex.group(1) == "0",
        f"0. the program ran as root and ended with status "
        f"{ex.group(1) if ex else '?'} ({len(maps)} mappings, done {done})",
    )
    if not maps:
        return 1
    map_base = maps[0][0]
    r = re.search(r"PIM-PATH reg (0x[0-9a-f]+) offset (0x[0-9a-f]+)", console)
    reg = int(r.group(1), 16) if r else map_base
    reg_off = int(r.group(2), 16) if r else 0
    if args.regions:
        want_maps = [
            (win_base, maps[0][2]),
            (win_base + reg_off, win_size - reg_off),
        ]
        shape = (
            f"the bank buffer at {win_base:#x} and the register rows "
            f"{win_base + reg_off:#x}-{win_base + win_size:#x}"
        )
    else:
        want_maps = [(win_base, win_size)]
        shape = f"WINDOW_BASE_FS {win_base:#x}, WINDOW_SIZE {win_size:#x}"
    check(
        [(pa, n) for va, pa, n in maps] == want_maps
        and (not args.regions or maps[0][2] < reg_off),
        f"0. it mapped {shape} "
        f"({', '.join(f'{pa:#x}+{n:#x}' for va, pa, n in maps)})",
    )
    want = pieces("all", args.vlen, map_base, reg)
    want_ops = collections.Counter(p[0] for p in want)
    tlb, accepted = [], []
    for line in open_trace(args.dir):
        t = TLB_RE.search(line)
        if t:
            tlb.append(t.groups() + (bool(WALK_RE.search(line)),))
            continue
        c = PIMMEM_RE.search(line)
        if c:
            accepted.append((c.group(1), int(c.group(2), 16)))

    # 1. Translation lines.
    bad = [t for t in tlb if t[5].strip() != "PIM UC SO" or t[6] != "none"]
    check(
        tlb and not bad,
        f"1. {len(tlb)} PIM translations, all 'PIM UC SO', no fault"
        f"{' ' + str(bad[:2]) if bad else ''}",
    )

    def mapping(va):
        """The mapping holding va, or None."""
        for m in maps:
            if m[0] <= va < m[0] + m[2]:
                return m
        return None

    def in_map(t):
        return mapping(int(t[2], 16)) is not None

    def pa_of(va):
        m = mapping(va)
        return m[1] + va - m[0]

    got = collections.Counter(
        (t[0], t[1], int(t[2], 16)) for t in tlb if in_map(t)
    )
    stray = [t for t in tlb if not in_map(t)]
    exp = collections.Counter((op, rw, va) for op, rw, va, size in want)
    if args.cpu in SPECULATIVE:
        # As check_request_path.py: squashed micro-ops (replays, wrong
        # paths, a stale vector length) may add translations of expected
        # operations; only committed ones issue requests (check 3).
        kinds = {(op, rw) for op, rw, va in exp}
        extra = [k for k in got if k not in exp]
        ok = all(got[k] >= n for k, n in exp.items()) and all(
            k[:2] in kinds for k in extra
        )
        rule = (
            f"at least one per request piece, others in the mapping only of "
            f"an expected operation ({len(extra)} other addresses inside, "
            f"{len(stray)} squashed translations outside)"
        )
    else:
        ok = got == exp and not stray
        rule = "exactly one per request piece, none outside the mapping"
    check(
        ok,
        f"1. translations: {rule}: {sum(got.values())} lines for "
        f"{sum(exp.values())} pieces",
    )

    # 2. Linear translation.
    lin = [
        t for t in tlb if in_map(t) and int(t[3], 16) != pa_of(int(t[2], 16))
    ]
    check(
        tlb and not lin,
        f"2. pa - mapping's base == va - mapping's address for every "
        f"translation ({len(maps)} mappings)",
    )

    # 3. Commands, from the program's dump (the first) and the last one.
    all_dumps = dumps(os.path.join(args.dir, "stats.txt"))
    st = all_dumps[0] if all_dumps else {}
    last = all_dumps[-1] if all_dumps else {}

    def stat(suffix, d=st):
        return sum(float(v) for k, v in d.items() if k.endswith(suffix))

    ops = set(want_ops) | {a[0] for a in accepted}
    bus = {op: stat(f"membus.transDist::Pim{op}Req") for op in ops}
    check(
        all(bus[op] == want_ops.get(op, 0) for op in ops),
        f"3. the memory bus carried the expected PIM requests per operation "
        f"({sum(bus.values()):.0f} for {sum(want_ops.values())} pieces)",
    )
    check(
        collections.Counter(a[0] for a in accepted) == want_ops
        and all(win_base <= a[1] < win_base + win_size for a in accepted),
        f"3. the PIM memory accepted {len(accepted)} PIM requests of the "
        f"expected operations, all in the window",
    )
    if args.cpu != "atomic":
        # Atomic-mode caches do not count uncacheable misses.
        wrong = [
            (c, op, stat(f"{c}.Pim{op}Req.mshrUncacheable::total"), n)
            for op, n in want_ops.items()
            for c in ("l1d-cache-0", "l2-cache-0")
            if stat(f"{c}.Pim{op}Req.mshrUncacheable::total") != n
        ]
        check(
            not wrong,
            f"3. L1D and L2 counted every PIM request as an uncacheable "
            f"miss{' ' + str(wrong[:2]) if wrong else ''}",
        )
    normal = {
        k: v
        for k, v in last.items()
        if re.search(
            r"pim_memory\.ramulator2\.channel\d+\.num_(read|write)_reqs$", k
        )
    }
    check(
        normal and all(float(v) == 0 for v in normal.values()),
        f"3. no normal request at the PIM memory ({len(normal)} Ramulator "
        f"2.1 request counters, last dump)",
    )
    # pim.swErrors is nozero: absent when 0.
    sw = [v for d in all_dumps for k, v in d.items() if "pim.swErrors" in k]
    check(all(float(v) == 0 for v in sw), "3. no PIM software error")

    # 4. F1: walks in the program's dump.
    acc, miss = stat("dtb.pimAccesses"), stat("dtb.pimMisses")
    walks = sum(
        float(v)
        for k, v in st.items()
        if re.search(r"dtb\.walker\.num_\w+_walks$", k)
    )
    check(
        acc > 0 and miss > 0 and walks > 0,
        f"4. F1: pimAccesses {acc:.0f}, pimMisses {miss:.0f}, data walks "
        f"{walks:.0f}",
    )

    # 5. F2: a store-like translation walked on every page stored to.
    pages = {pa_of(va) // PAGE for op, rw, va, size in want if rw == "W"}
    if args.cpu == "atomic":
        first = {}
        for t in tlb:
            if in_map(t):
                first.setdefault(int(t[3], 16) // PAGE, t[1])
        ok = all(first.get(p) == "W" for p in pages) and miss >= len(first)
        check(
            pages and ok,
            f"5. F2 (Atomic): a store first on each of the {len(pages)} "
            f"pages stored to; {miss:.0f} TLB misses for {len(first)} pages",
        )
    else:
        walked = {
            int(t[3], 16) // PAGE
            for t in tlb
            if in_map(t) and t[1] == "W" and t[7]
        }
        check(
            pages and pages <= walked,
            f"5. F2: store-like translations walked on all {len(pages)} "
            f"pages with store-like requests (missing "
            f"{[hex(p * PAGE) for p in sorted(pages - walked)]})",
        )

    # 6. F10: lookups.
    n = sum(exp.values())
    ok = acc >= n if args.cpu in SPECULATIVE else acc == n
    check(ok, f"6. F10: {acc:.0f} PIM lookups for {n} pieces")
    print(f"check_fs_linux: {len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
