#!/usr/bin/env python3
"""Check a run of the PIM FS bare-metal translation test.

The run is configs/deprecated/example/fs.py --bare-metal with
tests/test-progs/pim-fs-bm (cases F1-F8 in S-mode) and
--debug-flags=PIM,Cache,ExecEnable,ExecKernel,ExecEffAddr. The program
reports its own checks (the fault cause of each case, F2's dirty bit) in
the m5_fail exit code, one bit per case; the trace, split into cases at
the program's marker stores, and the first statistics dump (case F1
alone) give the rest:
  F1  PIM translations of the 4 KiB and 2 MiB pages walk (timing models
      mark the line "walk") and give pa = WIN_PA + (va - WIN_VA); the
      walker counted 4 KiB and 2 MiB walks; pimMisses >= 1
  F2  the program saw D = 1 after a store-like PIM instruction to a page
      with D = 0
  F3  store page fault (15), F4 load page fault (13), F5 store page
      fault on a user page (15), F6 store and load access faults (7, 5):
      each with no PIM packet in that case
  F7  normal accesses to window page 7 are uncacheable only in the PMA
      run (--pma); PIM translations keep "PIM UC SO"
  F8  the guards' misaligned faults (6), before any TLB lookup
  F9  no packet to the page tables is a PIM packet or uncacheable
  F10 in the F1 dump, the data TLB's read and write accesses equal the
      PIM lookups (2 load-like, 3 store-like; O3 and Minor may translate
      a load again)
  F12 a PMP fault in micro-op 1 of pim.ab.mac: the program saw vstart =
      the micro-op's first element and 0 after the re-execution; each
      micro-op's request on page 6 goes
      out once (Atomic: translations without a fault; Timing, O3: L1
      packets). Not on Minor, which decodes the re-executed instruction
      with a stale vl (one micro-op), with or without the restart
  F13 the program saw U-mode cache-block operations fault (illegal
      instruction) with senvcfg = 0 and run with its bits set, and an
      S-mode cbo.flush run with senvcfg = 0
  F14-F18 (--sv48, pim-fs-bm-sv48): the program's own
      checks of Sv48 (satp mode, a 512 GiB page, a misaligned one, the
      upper half, two non-canonical addresses); F9 also covers l3

Usage: check_fs_bm.py --dir <run dir> --log <log> --binary <elf> --cpu C
       [--pma] [--sv48]
Exit status 0 if every check passes, 1 otherwise.
"""

import argparse
import collections
import gzip
import os
import re
import subprocess
import sys

WIN_VA = 0x40000000
WIN_PA = 0x10000000
PAGE = 4096
MEGA2 = 0x200000
CASES = {
    1: "F1",
    2: "F2",
    3: "F3",
    4: "F4",
    5: "F5",
    6: "F6",
    7: "F7",
    8: "F8",
    12: "F12",
    13: "F13",
}
SV48_CASES = {14: "F14", 15: "F15", 16: "F16", 17: "F17", 18: "F18"}

TLB_RE = re.compile(
    r"^\s*(\d+): \S+: PIM (\w+) ([RW]) va=(0x[0-9a-f]+) pa=(\S+) size=(\d+) "
    r"flags=(.*?) fault=(\S+)( walk)?"
)
PKT_RE = re.compile(r"\b(\w+(?:Req|Resp)) \[([0-9a-f]+):([0-9a-f]+)\]([^\n]*)")
EXEC_RE = re.compile(
    r"^\s*(\d+): \S+: 0x[0-9a-f]+[^:]*: (\S+)[^\n]*A=(0x[0-9a-f]+)"
)


def symbols(binary):
    out = subprocess.run(
        ["riscv64-linux-gnu-nm", binary], capture_output=True, text=True
    ).stdout
    return {
        p[2]: int(p[0], 16)
        for p in (l.split() for l in out.splitlines())
        if len(p) == 3
    }


def open_trace(d):
    p = os.path.join(d, "trace.out")
    return open(p) if os.path.exists(p) else gzip.open(p + ".gz", "rt")


def first_dump(path):
    text = open(path).read().split("Begin Simulation Statistics")
    vals = {}
    for line in text[1].splitlines() if len(text) > 1 else []:
        f = line.split()
        if len(f) >= 2 and not f[0].startswith("-"):
            vals[f[0]] = f[1]
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--pma", action="store_true")
    ap.add_argument("--sv48", action="store_true")
    args = ap.parse_args()
    if args.sv48:
        CASES.update(SV48_CASES)
    fails = []

    def check(ok, what):
        print(("PASS " if ok else "FAIL ") + what)
        if not ok:
            fails.append(what)

    sym = symbols(args.binary)
    markers = sym["markers"]
    nmarkers = 32 if args.sv48 else 16  # pim_fs_bm.c's markers[]
    tables = [sym[n] for n in ("root", "l1w", "l0a", "l3") if n in sym]
    log = open(args.log).read()
    m = re.search(r"Exit code is (\d+)", log)
    code = int(m.group(1)) if m else (0 if "m5_fail" in log else None)
    check(
        code is not None,
        f"the program ended through m5_fail (exit code " f"{code})",
    )
    if code is None:
        return 1
    for c, name in CASES.items():
        check(
            not code & (1 << c),
            f"{name}: the program's own check "
            f"(fault cause{', dirty bit' if c == 2 else ''}"
            f"{', vstart' if c == 12 else ''}"
            f"{', senvcfg' if c == 13 else ''})",
        )

    starts, tlb, packets = {}, [], []  # case markers by commit tick
    for line in open_trace(args.dir):
        e = EXEC_RE.match(line)
        if (
            e
            and e.group(2) in ("sd", "c_sd")
            and markers <= int(e.group(3), 16) < markers + 8 * nmarkers
        ):
            starts.setdefault(
                (int(e.group(3), 16) - markers) // 8, int(e.group(1))
            )
            continue
        t = TLB_RE.match(line)
        if t:
            tlb.append(t.groups())
            continue
        if args.cpu != "atomic":
            tick = line.split(":", 1)[0].strip()
            # One line per packet: the L1 data cache's access, or for the
            # page tables any line (the walker may use another port).
            l1 = "dcache: access for" in line
            for p in PKT_RE.finditer(line):
                if p.group(1).startswith("Pim") and not l1:
                    continue
                packets.append(
                    (
                        int(tick) if tick.isdigit() else 0,
                        p.group(1),
                        int(p.group(2), 16),
                        p.group(4),
                    )
                )
    check(sorted(starts) == sorted(CASES), f"case markers {sorted(starts)}")

    # Translations.
    def pa_ok(t):
        va = int(t[3], 16)
        return t[4] == "-" or int(t[4], 16) == WIN_PA + (va - WIN_VA)

    check(
        all(pa_ok(t) for t in tlb),
        f"{len(tlb)} PIM translations: pa = " "WIN_PA + (va - WIN_VA)",
    )
    check(
        all(t[6].strip() == "PIM UC SO" for t in tlb),
        "every PIM " "translation keeps 'PIM UC SO' (F7 included)",
    )
    if args.cpu != "atomic":
        # A younger access to the same page may walk first (O3).
        walked = {int(t[3], 16) for t in tlb if t[8]}
        w4k = any(WIN_VA <= v < WIN_VA + PAGE for v in walked)
        w2m = any(WIN_VA + MEGA2 <= v < WIN_VA + 2 * MEGA2 for v in walked)
        check(
            w4k and w2m,
            "F1: a PIM translation of the 4 KiB page and one "
            "of the 2 MiB page walk",
        )
    st = first_dump(os.path.join(args.dir, "stats.txt"))

    def s(key):
        v = [x for k, x in st.items() if k.endswith(key)]
        return float(v[0]) if v else 0

    check(
        s("dtb.walker.num_4kb_walks") > 0
        and s("dtb.walker.num_2mb_walks") > 0
        and s("dtb.pimMisses") >= 1,
        f"F1: walks 4 KiB "
        f"{s('dtb.walker.num_4kb_walks'):.0f}, 2 MiB "
        f"{s('dtb.walker.num_2mb_walks'):.0f}, pimMisses "
        f"{s('dtb.pimMisses'):.0f}",
    )
    rd, wr, pa = (
        s("dtb.readAccesses"),
        s("dtb.writeAccesses"),
        s("dtb.pimAccesses"),
    )
    if args.cpu in ("minor", "o3"):
        # Speculative models also translate squashed and wrong-path
        # accesses, PIM and normal.
        ok = wr >= 3 and rd >= 2 and pa >= 5
    else:
        ok = wr == 3 and rd == 2 and pa == 5
    check(
        ok,
        f"F10: dtb read {rd:.0f}, write {wr:.0f}, PIM {pa:.0f} lookups "
        "in the F1 dump (3 store-like, 2 load-like)",
    )

    # The cases use their own pages (pim_fs_bm.c), so packets and
    # translations are attributed by address: an out-of-order model
    # translates younger instructions before a marker store commits and
    # drains PIM stores after it.
    def page(n):
        return WIN_PA + n * PAGE

    if args.cpu != "atomic":
        pim = [
            p
            for p in packets
            if p[1].startswith("Pim") and p[1].endswith("Req")
        ]
        for c, n in ((3, 2), (4, 3), (5, 4)):
            hit = [p for p in pim if page(n) <= p[2] < page(n + 1)]
            check(not hit, f"{CASES[c]}: no PIM packet to its page ({n})")
        # F6 re-uses page 0 at offset 0, which F1 writes once with
        # pim.sb.mac and does not read.
        sbmac0 = sum(
            1 for p in pim if p[1] == "PimSbMacReq" and p[2] == page(0)
        )
        rdgb0 = sum(1 for p in pim if p[1] == "PimRdGbReq" and p[2] == page(0))
        check(
            sbmac0 == 1 and rdgb0 == 0,
            f"F6: no PIM packet from the PMP "
            f"cases (page 0 offset 0: {sbmac0} pim.sb.mac from F1, {rdgb0} "
            "pim.rd.gb)",
        )
        normal = [
            p
            for p in packets
            if not p[1].startswith("Pim") and page(7) <= p[2] < page(8)
        ]
        uc = [p for p in normal if " UC" in p[3]]
        if args.pma:
            check(
                normal and len(uc) == len(normal),
                f"F7 (PMA run): "
                f"{len(normal)} normal packets to page 7, all UC",
            )
        else:
            check(
                normal and not uc,
                f"F7: {len(normal)} normal packets to " "page 7, none UC",
            )
        bad = [
            p
            for p in packets
            if any(t <= p[2] < t + PAGE for t in tables)
            and (p[1].startswith("Pim") or " UC" in p[3])
        ]
        n = sum(
            1 for p in packets if any(t <= p[2] < t + PAGE for t in tables)
        )
        check(
            n > 0 and not bad,
            f"F9: {n} packets to the page tables, none " "PIM or UC",
        )
    if args.cpu == "atomic":
        f12 = collections.Counter(
            int(t[3], 16) - WIN_VA - 6 * PAGE
            for t in tlb
            if t[1] == "AbMac"
            and t[7] == "none"
            and WIN_VA + 6 * PAGE <= int(t[3], 16) < WIN_VA + 7 * PAGE
        )
    else:
        f12 = collections.Counter(
            p[2] - WIN_PA - 6 * PAGE
            for p in packets
            if p[1] == "PimAbMacReq" and page(6) <= p[2] < page(7)
        )
    if args.cpu != "minor":
        check(
            len(f12) == 2 and set(f12.values()) == {1},
            f"F12: one request per micro-op on page 6 "
            f"(offsets: {dict(sorted(f12.items()))})",
        )
    f8 = [t for t in tlb if int(t[3], 16) in (WIN_VA + 4, WIN_VA + PAGE - 16)]
    check(
        not f8,
        f"F8: no PIM translation of the guarded addresses " f"({len(f8)})",
    )
    print(f"check_fs_bm: {len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
