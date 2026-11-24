#!/usr/bin/env python3
"""Static check of the generated decoder code for the PIM instructions.

Reads build/<X>/arch/riscv/generated/decoder-ns.cc.inc and checks, for the
instruction groups given with --groups (scalar: custom-1, vstore:
custom-2, vload: custom-0):
- every processing-in-memory (PIM) class exists, plus a *Micro class for
  each vector one;
- every scalar and micro-op constructor sets memAccessFlags to
  PIM|UNCACHEABLE|STRICT_ORDER and its operation (scalar: plus the
  alignment flag), and every PIM constructor sets flags[IsPim];
- the totals of flags[IsPim] and of memAccessFlags = (Request::PIM are
  those of the groups (all three: 31 and 18);
- no constructor outside the PIM classes mentions IsPim or Request::PIM.

Exit status 0 if every check passes, 1 otherwise.
"""

import argparse
import re
import sys

# (mnemonic, PimOp, group) of every PIM instruction.
PIM_INSTS = [
    ("pim_sb_ewadd", "SbEwadd", "scalar"),
    ("pim_sb_ewmul", "SbEwmul", "scalar"),
    ("pim_sb_ewadd_gbuf", "SbEwaddGbuf", "scalar"),
    ("pim_sb_ewmul_gbuf", "SbEwmulGbuf", "scalar"),
    ("pim_sb_mac", "SbMac", "scalar"),
    ("pim_sb_mov_bkgbuf", "SbMovBkgbuf", "vstore"),
    ("pim_sb_mov_gbufbk", "SbMovGbufbk", "vstore"),
    ("pim_ab_ewadd", "AbEwadd", "vstore"),
    ("pim_ab_ewmul", "AbEwmul", "vstore"),
    ("pim_ab_ewadd_gbuf", "AbEwaddGbuf", "vstore"),
    ("pim_ab_ewmul_gbuf", "AbEwmulGbuf", "vstore"),
    ("pim_ab_mac", "AbMac", "vstore"),
    ("pim_ab_mov_regbk", "AbMovRegbk", "vstore"),
    ("pim_ab_mov_bkreg", "AbMovBkreg", "vstore"),
    ("pim_rd_gb", "RdGb", "vload"),
    ("pim_wr_gb", "WrGb", "vstore"),
    ("pim_rd_reg", "RdReg", "vload"),
    ("pim_wr_reg", "WrReg", "vstore"),
]
GROUPS = ("scalar", "vstore", "vload")


def class_name(mnemonic):
    # The ISA parser capitalizes the first letter (vle64_v -> Vle64_v).
    return mnemonic[0].upper() + mnemonic[1:]


def constructors(text):
    """{class name: constructor text} for every constructor definition,
    from its signature to the brace that closes its body."""
    out = {}
    pattern = re.compile(r"^[ \t]*(\w+)::\1\(", re.M)
    for m in pattern.finditer(text):
        start = text.find("{", m.end())
        depth, pos = 0, start
        while pos < len(text):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        out.setdefault(m.group(1), "")
        out[m.group(1)] += text[m.start() : pos + 1]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "decoder_ns",
        nargs="?",
        default="build/RISCV/arch/riscv/generated/decoder-ns.cc.inc",
    )
    ap.add_argument(
        "--groups",
        default=",".join(GROUPS),
        help="comma-separated groups that must be decoded (default: all)",
    )
    args = ap.parse_args()
    groups = set(args.groups.split(",")) if args.groups else set()
    if groups - set(GROUPS):
        ap.error(f"unknown group in {args.groups}")
    text = open(args.decoder_ns).read()
    ctors = constructors(text)
    fails = []

    def check(ok, what):
        print(("PASS " if ok else "FAIL ") + what)
        if not ok:
            fails.append(what)

    want_ispim = want_mem = 0
    pim_classes = set()
    for mnem, op, group in PIM_INSTS:
        cls = class_name(mnem)
        vector = group != "scalar"
        names = [cls, cls + "Micro"] if vector else [cls]
        present = [n for n in names if n in ctors]
        if group not in groups:
            check(not present, f"{cls}: absent (group {group} not decoded)")
            continue
        pim_classes.update(names)
        check(len(present) == len(names), f"{cls}: classes {names} exist")
        flags = (
            r"memAccessFlags = \(Request::PIM\)\|\(Request::UNCACHEABLE\)"
            r"\|\(Request::STRICT_ORDER\)\|\(Request::pimOpFlags\(PimOp::"
            + op
            + r"\)\)"
        )
        if not vector:
            flags += r"\|MMU::DoubleWordAlign;"
        else:
            flags += ";"
        mem_cls = cls + "Micro" if vector else cls
        body = ctors.get(mem_cls, "")
        check(
            re.search(flags, body) is not None,
            f"{mem_cls}: memAccessFlags PIM|UNCACHEABLE|STRICT_ORDER|{op}",
        )
        for n in names:
            check(
                "flags[IsPim] = true;" in ctors.get(n, ""),
                f"{n}: flags[IsPim]",
            )
        want_ispim += len(names)
        want_mem += 1

    n_ispim = text.count("flags[IsPim] = true;")
    n_mem = text.count("memAccessFlags = (Request::PIM)")
    check(
        n_ispim == want_ispim,
        f"flags[IsPim] {n_ispim} times, want " f"{want_ispim}",
    )
    check(
        n_mem == want_mem,
        f"memAccessFlags = (Request::PIM {n_mem} " f"times, want {want_mem}",
    )
    leaks = [
        c
        for c, body in ctors.items()
        if c not in pim_classes and ("IsPim" in body or "Request::PIM" in body)
    ]
    check(
        not leaks,
        f"no other constructor mentions IsPim or Request::PIM"
        f"{' (' + ', '.join(leaks) + ')' if leaks else ''}",
    )
    print(f"check_generated: {len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
