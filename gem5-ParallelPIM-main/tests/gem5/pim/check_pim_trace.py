#!/usr/bin/env python3
"""Checks of the DRAM command traces of a CPU-less PIM run.

Reads the ``cmd_trace.csv.ch<i>`` files that Ramulator 2.1's stock
``CmdTraceRecorder`` writes (``pim_r2_cpuless.py --trace``): one row per
issued command, ``clock,command,<levels>,type,source``, where the type is
0 for a host read, 1 for a host write, 2 for a PIM command and -1 for
maintenance.

Checks, each printing one line and failing the exit status when it does
not hold:

  after-host-write CH       the first PIM column command of channel CH
                            follows a host write column command to the
                            same bank and row (C3)
  count CH TYPE CMD OP N    commands named CMD (a prefix ending in *) of
                            TYPE on channel CH compare to N; OP is =, >, <
  switches CH UNTIL         on channel CH, until clock UNTIL, every change
                            between host and PIM commands follows a
                            closing command (PRE, RDA, WRA or an
                            auto-precharge PIM column; C9 and C29)
"""

import csv
import os
import re
import sys

CLOSING = re.compile(r"^(PRE.*|.*RDA.*|.*WRA.*|PIM_.*A)$")


def load(outdir, ch):
    path = os.path.join(outdir, "cmd_trace.csv.ch%d" % ch)
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    head = rows[0]
    levels = head[2:-2]
    out = []
    for r in rows[1:]:
        out.append(
            dict(
                clk=int(r[0]),
                cmd=r[1],
                addr=dict(zip(levels, (int(v) for v in r[2:-2]))),
                type=int(r[-2]),
            )
        )
    return out


def is_column(cmd):
    return cmd.startswith(("RD", "WR", "PIM_"))


def bank_row(a):
    return tuple(v for k, v in a.items() if k not in ("Column",))


def after_host_write(outdir, ch):
    rows = load(outdir, ch)
    first = next(
        (r for r in rows if r["type"] == 2 and r["cmd"].startswith("PIM_")),
        None,
    )
    if first is None:
        return False, "no PIM column command"
    target = bank_row(first["addr"])
    wr = [
        r
        for r in rows
        if r["type"] == 1
        and r["cmd"].startswith("WR")
        and bank_row(r["addr"]) == target
        and r["clk"] < first["clk"]
    ]
    if not wr:
        return (
            False,
            "first PIM column at clock %d has no earlier host write"
            % (first["clk"]),
        )
    return True, "first PIM column at clock %d after a host write at %d" % (
        first["clk"],
        wr[-1]["clk"],
    )


def count(outdir, ch, typ, cmd, op, n):
    rows = load(outdir, ch)
    pre = cmd[:-1] if cmd.endswith("*") else None
    k = sum(
        1
        for r in rows
        if r["type"] == typ
        and (r["cmd"].startswith(pre) if pre is not None else r["cmd"] == cmd)
    )
    ok = {"=": k == n, ">": k > n, "<": k < n}[op]
    return ok, "%d %s commands of type %d (expected %s %d)" % (
        k,
        cmd,
        typ,
        op,
        n,
    )


def switches(outdir, ch, until):
    rows = [
        r
        for r in load(outdir, ch)
        if r["type"] >= 0 and r["cmd"] != "CAS" and r["clk"] < until
    ]
    changes = bad = 0
    first_bad = None
    for prev, cur in zip(rows, rows[1:]):
        if (prev["type"] == 2) == (cur["type"] == 2):
            continue
        changes += 1
        if not CLOSING.match(prev["cmd"]):
            bad += 1
            if first_bad is None:
                first_bad = (prev["clk"], prev["cmd"], cur["cmd"])
    if changes == 0:
        return False, "no change of side before clock %d" % until
    msg = "%d side changes, %d not after a closing command" % (changes, bad)
    if first_bad:
        msg += " (first: %s at %d, then %s)" % (
            first_bad[1],
            first_bad[0],
            first_bad[2],
        )
    return bad == 0, msg


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    outdir, check = argv[1], argv[2]
    a = argv[3:]
    if check == "after-host-write":
        ok, msg = after_host_write(outdir, int(a[0]))
    elif check == "count":
        ok, msg = count(outdir, int(a[0]), int(a[1]), a[2], a[3], int(a[4]))
    elif check == "switches":
        ok, msg = switches(outdir, int(a[0]), int(a[1]))
    else:
        print("unknown check %s" % check)
        return 2
    print(
        "check_pim_trace {}: {} ({})".format(
            check, msg, "PASS" if ok else "FAIL"
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
