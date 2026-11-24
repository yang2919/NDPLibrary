#!/usr/bin/env python3
"""Checks of a CPU-less processing-in-memory (PIM) run.

Reads the PimTester results (``pim_tester.csv``) and the statistics of a
gem5 output directory:

- every script line completed, each response at or after its request
  (``--positive-latency``: every PIM line's after it);
- no line with an error response or unexpected data;
- optional statistic conditions on the first dump of ``stats.txt``:
  ``--stat NAME=VALUE``, ``--stat NAME>VALUE``, ``--stat NAME<VALUE`` (a
  name with ``*`` is a pattern; every match must hold).

Exit status 0 if every check holds, 1 otherwise.

Usage:
  check_pim_ctrl.py OUTDIR [--csv NAME] [--lines N] [--stat COND]...
"""

import argparse
import csv
import fnmatch
import os
import re
import sys

MAX_TICK = 2**64 - 1


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_stats(path):
    """The first dump of stats.txt as {name: value string}."""
    stats = {}
    begun = False
    with open(path) as f:
        for line in f:
            if line.startswith("---------- Begin"):
                if begun:
                    break
                begun = True
                continue
            if line.startswith("---------- End"):
                break
            parts = line.split()
            if len(parts) >= 2 and not line.startswith(" "):
                stats[parts[0]] = parts[1]
    return stats


def check_rows(rows, lines, positive=False):
    errors = []
    if lines is not None and len(rows) != lines:
        errors.append("%d lines in the CSV, %d expected" % (len(rows), lines))
    for r in rows:
        send, resp = int(r["send_tick"]), int(r["resp_tick"])
        where = "line {} ({})".format(r["line"], r["kind"])
        if send == MAX_TICK or resp == MAX_TICK:
            errors.append("%s did not complete" % where)
        elif resp < send:
            errors.append("%s: response before its request" % where)
        elif positive and r["kind"].startswith("pim:") and resp <= send:
            errors.append(
                "%s: latency %d, not positive" % (where, resp - send)
            )
        if r["result"] == "mismatch":
            errors.append("%s: error response or unexpected data" % where)
    return errors


def check_stat(stats, cond):
    m = re.fullmatch(r"([^=<>]+)([=<>])(.+)", cond)
    if not m:
        return ["bad condition %r" % cond]
    name, rel, want = m.group(1), m.group(2), float(m.group(3))
    if "*" in name:
        names = [n for n in stats if fnmatch.fnmatchcase(n, name)]
    else:
        names = [name] if name in stats else []
    if not names:
        return ["no statistic %s" % name]
    errors = []
    for n in names:
        try:
            got = float(stats[n])
        except ValueError:
            errors.append(f"{n} is {stats[n]}, not a number")
            continue
        ok = {"=": got == want, "<": got < want, ">": got > want}[rel]
        if not ok:
            errors.append(f"{n} = {stats[n]}, expected {rel}{m.group(3)}")
    return errors


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("outdir")
    p.add_argument("--csv", default="pim_tester.csv")
    p.add_argument("--lines", type=int, help="expected number of lines")
    p.add_argument(
        "--positive-latency",
        action="store_true",
        help="every PIM line's latency is above 0 (atomic mode, C2)",
    )
    p.add_argument(
        "--stat", action="append", default=[], help="NAME=V, NAME>V or NAME<V"
    )
    a = p.parse_args(argv)
    errors = []
    rows = read_csv(os.path.join(a.outdir, a.csv))
    errors += check_rows(rows, a.lines, a.positive_latency)
    if a.stat:
        stats = read_stats(os.path.join(a.outdir, "stats.txt"))
        for cond in a.stat:
            errors += check_stat(stats, cond)
    for e in errors:
        print("FAIL: %s" % e)
    print(
        "check_pim_ctrl: %d lines, %d checks failed%s"
        % (len(rows), len(errors), "" if errors else " (PASS)")
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
