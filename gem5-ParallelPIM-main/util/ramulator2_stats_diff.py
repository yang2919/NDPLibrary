#!/usr/bin/env python3
"""Compare the statistics of gem5 runs with a Ramulator 2.1 memory.

Checks that a candidate run, for example of a workload without
processing-in-memory (PIM) instructions, produces the same statistics as
its reference run.

Usage:
    ramulator2_stats_diff.py [options] <reference> <candidate>

<reference> and <candidate> are two run directories (each holding a stats.txt),
or two trees of run directories whose runs are matched by directory name.

Compared, per run:
- stats.txt, split into dumps at "Begin Simulation Statistics": every
  statistic's name and value, per dump. Host statistics (host*) are ignored.
- every Ramulator 2.1 statistics file, <object>.ramulator_stats.yaml and
  <object>.ramulator_stats.<tick>.yaml, paired by file name, so the dump
  ticks must match exactly. Loaded as YAML and flattened; controllers are
  keyed by their id ("Channel 0", ...). Values are compared exactly.

A key present only in the candidate is accepted if its name matches
--allow-new (default: "(?i)pim") and its value is 0: a PIM counter that
reads 0 is the only difference allowed. Every other difference counts.

To compare a run on a PimSystemMemory with a run on its operating-system
memory alone:
- --path-map REF=CAND renames, in the candidate, every occurrence of CAND
  in a statistic's name and in a statistics file's name to REF (for
  example board.memory=board.memory.os_memory);
- --zero-subtree PATH leaves out of the comparison every candidate
  statistic whose name contains PATH, and every candidate statistics file
  whose name starts with PATH, but requires each of them whose name
  matches --zero-keys to read 0 (the PIM memory must see no request).
Both options repeat.

Exit status: 0 if all compared runs are equal, 1 if any differs, 2 on a usage
error. Needs Python 3 and PyYAML.
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

YAML_GLOB = "*ramulator_stats*.yaml"
HOST = re.compile(r"(^|\.)host")


def read_stats(path):
    """A list of {name: value} dicts, one per dump, host statistics out."""
    dumps = []
    text = path.read_text(errors="replace")
    for dump in text.split("Begin Simulation Statistics")[1:]:
        values = {}
        for line in dump.splitlines():
            parts = line.split()
            if len(parts) < 2 or parts[0].startswith("-"):
                continue
            if HOST.search(parts[0]):
                continue
            values[parts[0]] = parts[1]
        dumps.append(values)
    return dumps


def flatten(node, prefix, out):
    """Flatten a statistics tree; list items are keyed by their id."""
    if isinstance(node, dict):
        for key, child in node.items():
            flatten(child, f"{prefix}.{key}" if prefix else str(key), out)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            key = item.get("id", i) if isinstance(item, dict) else i
            flatten(item, f"{prefix}[{key}]", out)
    else:
        out[prefix] = node


def read_yaml(path):
    out = {}
    flatten(yaml.safe_load(path.read_text()) or {}, "", out)
    return out


def is_zero(value):
    try:
        return float(value) == 0
    except (TypeError, ValueError):
        return False


def compare_maps(ref, cand, allow_new, where):
    """Differences between two {key: value} maps, as text lines."""
    diffs = []
    for key in sorted(set(ref) | set(cand)):
        if key not in cand:
            diffs.append(f"{where}: {key}: only in the reference")
        elif key not in ref:
            if not (allow_new.search(key) and is_zero(cand[key])):
                diffs.append(
                    f"{where}: {key}: only in the candidate " f"({cand[key]})"
                )
        elif ref[key] != cand[key]:
            diffs.append(f"{where}: {key}: {ref[key]} != {cand[key]}")
    return diffs


class Tier2:
    """--path-map and --zero-subtree."""

    def __init__(self, path_maps=(), subtrees=(), zero_keys=None):
        self.maps = []
        for m in path_maps:
            ref, sep, cand = m.partition("=")
            if not sep or not ref or not cand:
                raise ValueError(f"--path-map {m}: REF=CAND expected")
            self.maps.append((ref, cand))
        self.subtrees = list(subtrees)
        self.zero_keys = zero_keys

    def rename(self, name):
        for ref, cand in self.maps:
            name = name.replace(cand, ref)
        return name

    def in_subtree(self, name, files=False):
        if files:
            return any(name.startswith(s) for s in self.subtrees)
        return any(s in name for s in self.subtrees)

    def split(self, values, where, inside=False):
        """(values outside the subtrees, renamed; differences inside).
        With inside, every value is inside (a file of a subtree)."""
        kept, diffs = {}, []
        for key, value in values.items():
            if inside or self.in_subtree(key):
                if self.zero_keys.search(key) and not is_zero(value):
                    diffs.append(
                        f"{where}: {key}: {value}, but the zero-request "
                        "subtree must read 0"
                    )
            else:
                kept[self.rename(key)] = value
        return kept, diffs


NO_TIER2 = Tier2()


def compare_run(ref_dir, cand_dir, allow_new, tier2=NO_TIER2):
    """Differences between two runs, as text lines."""
    diffs = []
    ref_stats = ref_dir / "stats.txt"
    cand_stats = cand_dir / "stats.txt"
    if not ref_stats.exists() or not cand_stats.exists():
        missing = ref_stats if not ref_stats.exists() else cand_stats
        return [f"{missing}: missing"]
    ref_dumps = read_stats(ref_stats)
    cand_dumps = read_stats(cand_stats)
    if len(ref_dumps) != len(cand_dumps):
        diffs.append(
            f"stats.txt: {len(ref_dumps)} dumps in the reference, "
            f"{len(cand_dumps)} in the candidate"
        )
    for n, (a, b) in enumerate(zip(ref_dumps, cand_dumps)):
        where = f"stats.txt dump {n}"
        b, inside = tier2.split(b, where)
        diffs += inside
        diffs += compare_maps(a, b, allow_new, where)

    ref_yaml = {p.name: p for p in ref_dir.glob(YAML_GLOB)}
    cand_yaml = {}
    for p in cand_dir.glob(YAML_GLOB):
        if tier2.in_subtree(p.name, files=True):
            _, inside = tier2.split(read_yaml(p), p.name, inside=True)
            diffs += inside
        else:
            cand_yaml[tier2.rename(p.name)] = p
    for name in sorted(set(ref_yaml) - set(cand_yaml)):
        diffs.append(f"{name}: only in the reference")
    for name in sorted(set(cand_yaml) - set(ref_yaml)):
        diffs.append(f"{name}: only in the candidate")
    for name in sorted(set(ref_yaml) & set(cand_yaml)):
        diffs += compare_maps(
            read_yaml(ref_yaml[name]),
            read_yaml(cand_yaml[name]),
            allow_new,
            name,
        )
    return diffs


def runs_of(tree):
    return {p.name: p for p in tree.iterdir() if (p / "stats.txt").exists()}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--allow-new",
        default="(?i)pim",
        help="regular expression for candidate-only keys that are accepted "
        "when their value is 0 (default: %(default)s)",
    )
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="REF=CAND",
        help="rename CAND to REF in the candidate's statistic and file "
        "names (repeats)",
    )
    parser.add_argument(
        "--zero-subtree",
        action="append",
        default=[],
        metavar="PATH",
        help="leave the candidate's statistics under PATH out, but require "
        "those matching --zero-keys to be 0 (repeats)",
    )
    parser.add_argument(
        "--zero-keys",
        default=r"(?i)^(?!.*maintenance).*(reqs|requests|pktcount|pktsize|"
        r"bytes|numreads|numwrites|numother)",
        help="regular expression for the request counters of a "
        "--zero-subtree; Ramulator 2.1's own maintenance (refresh) requests "
        "are not requests of the system (default: %(default)s)",
    )
    parser.add_argument(
        "--ignore-unmatched",
        action="store_true",
        help="with two trees, compare only the runs both hold",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=10,
        help="differences printed per run (default: %(default)s; 0: all)",
    )
    try:
        args = parser.parse_args()
    except SystemExit as e:
        return 2 if e.code else 0
    try:
        allow_new = re.compile(args.allow_new)
        zero_keys = re.compile(args.zero_keys)
    except re.error as e:
        print(f"error: --allow-new or --zero-keys: {e}", file=sys.stderr)
        return 2
    try:
        tier2 = Tier2(args.path_map, args.zero_subtree, zero_keys)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for path in (args.reference, args.candidate):
        if not path.is_dir():
            print(f"error: {path} is not a directory", file=sys.stderr)
            return 2

    single = (args.reference / "stats.txt").exists() and (
        args.candidate / "stats.txt"
    ).exists()
    if single:
        pairs = {args.candidate.name: (args.reference, args.candidate)}
        unmatched = []
    else:
        ref_runs, cand_runs = runs_of(args.reference), runs_of(args.candidate)
        if not ref_runs or not cand_runs:
            print(
                "error: give two run directories or two trees of runs",
                file=sys.stderr,
            )
            return 2
        pairs = {
            name: (ref_runs[name], cand_runs[name])
            for name in sorted(set(ref_runs) & set(cand_runs))
        }
        unmatched = sorted(set(ref_runs) ^ set(cand_runs))

    differing = 0
    for name, (ref, cand) in pairs.items():
        diffs = compare_run(ref, cand, allow_new, tier2)
        if diffs:
            differing += 1
            print(f"{name}: {len(diffs)} differences")
            shown = diffs if args.max_lines == 0 else diffs[: args.max_lines]
            for line in shown:
                print(f"    {line}")
            if len(shown) < len(diffs):
                print(f"    ... {len(diffs) - len(shown)} more")
        else:
            print(f"{name}: equal")
    for name in unmatched:
        print(f"{name}: in only one of the two trees")
    print(
        f"{len(pairs)} runs compared, {differing} differ, "
        f"{len(unmatched)} unmatched"
    )
    if args.ignore_unmatched:
        unmatched_fail = False
    else:
        unmatched_fail = bool(unmatched)
    return 1 if differing or unmatched_fail else 0


if __name__ == "__main__":
    sys.exit(main())
