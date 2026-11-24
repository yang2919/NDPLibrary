#!/usr/bin/env python3
"""Check the trace of a processing-in-memory (PIM) decode test run.

The run is tests/gem5/pim/configs/pim_decode_se.py with
--debug-flags=ExecEnable,ExecUser,ExecMacro,ExecMicro,ExecOpClass,
ExecEffAddr,ExecFlags,Decode,Cache,PIM. Against a table that mirrors
tests/test-progs/pim-decode/src/pim_kernels.S, it checks:
1. every PIM word decodes to the class its fields name;
2. the PIM scalar and micro-op lines of the Exec trace come in program
   order;
3. each has IsPim and no serializing flag, and no other line has IsPim;
4. the OpClass of each;
5. the micro-op count of each vector instruction, ceil(vl / (VLEN /
   EEW)), and its addresses rs1 + k * VLENB (none for vl = 0);
6. with caches (--cpu other than atomic): every PIM packet the Cache
   trace shows carries UC and lies in one cache line of pim_buf, the
   line-crossing store gives two packets meeting at the line boundary,
   and no other packet is uncacheable.

Usage: check_exec_trace.py --trace T --binary B --mode M --vlen V --cpu C
Exit status 0 if every check passes, 1 otherwise.
"""

import argparse
import math
import re
import subprocess
import sys

LINE = 64
SERIALIZING = (
    "IsSerializing",
    "IsSerializeBefore",
    "IsSerializeAfter",
    "IsNonSpeculative",
)
# Opcode -> {select value: mnemonic}.
SCALAR = [
    "pim_sb_ewadd",
    "pim_sb_ewmul",
    "pim_sb_ewadd_gbuf",
    "pim_sb_ewmul_gbuf",
    "pim_sb_mac",
]
VSTORE = [
    "pim_sb_mov_bkgbuf",
    "pim_sb_mov_gbufbk",
    "pim_ab_ewadd",
    "pim_ab_ewmul",
    "pim_ab_ewadd_gbuf",
    "pim_ab_ewmul_gbuf",
    "pim_ab_mac",
    "pim_ab_mov_regbk",
    "pim_ab_mov_bkreg",
    "pim_wr_gb",
    "pim_wr_reg",
]
VLOAD = ["pim_rd_gb", "pim_rd_reg"]
REGISTER_FORMS = ("pim_wr_reg", "pim_rd_reg")
OPCLASS = {
    "s": "MemWrite",
    "vs": "SimdUnitStrideStore",
    "vl": "SimdUnitStrideLoad",
}


def word_name(word):
    """The PIM mnemonic of a 32-bit word, or None."""
    op = word & 0x7F
    funct3 = (word >> 12) & 7
    sel = (word >> 20) & 0x1F
    funct7 = word >> 25
    if op == 0x2B:
        return SCALAR[funct3] if funct3 < 5 else None
    if op in (0x5B, 0x0B):
        if funct7 != 1 or funct3 not in (0, 5, 6, 7):
            return None
        table = VSTORE if op == 0x5B else VLOAD
        return table[sel] if sel < len(table) else None
    return None


# The kernels (pim_kernels.S): per group, a list of entries
# (mnemonic, kind, eew, vl rule, offset from pim_buf). vl rule: an int
# (elements), "m1" or "m2" (VLMAX at that LMUL). The register forms take
# their own pointer (REGISTER_FORMS); the decode test passes pim_buf.
def groups(vlen):
    g = {}
    g["scalar"] = [(m, "s", 0, None, 0) for m in SCALAR] + [
        ("pim_sb_mac", "s", 0, None, 0)
    ]
    g["vstore"] = [(m, "vs", 64, "m1", 0) for m in VSTORE]
    vm = []
    for eew, rule in ((64, "m2"), (8, "m1"), (16, "m1"), (32, "m1"), (64, 0)):
        vm += [
            ("pim_ab_mac", "vs", eew, rule, 0),
            ("pim_wr_gb", "vs", eew, rule, 0),
        ]
    g["vmulti"] = vm
    g["vsplit"] = [("pim_wr_gb", "vs", 64, 4, 48)]
    g["vload"] = []
    for rule in ("m1", "m2"):
        g["vload"] += [(m, "vl", 64, rule, 0) for m in VLOAD]
    return g


MODES = {
    "scalar": ["scalar"],
    "vstore": ["vstore", "vmulti", "vsplit"],
    "vload": ["vload"],
    "all": ["scalar", "vstore", "vmulti", "vsplit", "vload"],
}


def expected_lines(mode, vlen, buf, reg=None):
    """[(mnemonic, kind, address, bytes)] of the Exec lines, in order;
    the register forms at reg (default buf)."""
    vlenb = vlen // 8
    out = []
    for grp in MODES[mode]:
        for mnem, kind, eew, rule, off in groups(vlen)[grp]:
            base = buf if reg is None or mnem not in REGISTER_FORMS else reg
            base += off
            if kind == "s":
                out.append((mnem, kind, base, 8))
                continue
            if isinstance(rule, int):
                vl = rule
            else:
                vl = int(rule[1]) * vlen // eew
            per_reg = vlen // eew
            nmicro = math.ceil(vl / per_reg) if vl else 0
            for k in range(nmicro):
                elems = min(per_reg, vl - k * per_reg)
                out.append(
                    (mnem + "_micro", kind, base + k * vlenb, elems * eew // 8)
                )
    return out


EXEC_RE = re.compile(
    r"^\s*\d+: \S+: (0x[0-9a-f]+)(?:\.\s*\d+)?\s*(?:@\S+)?\s*: (\S+)([^:]*): "
    r"(\S+) :(?:\s+A=(0x[0-9a-f]+))?(?:\s+flags=\(([^)]*)\))?"
)
DECODE_RE = re.compile(r"Decode: Decoded (\S+) instruction: 0x([0-9a-f]+)")
PKT_RE = re.compile(r"\b(\w+(?:Req|Resp)) \[([0-9a-f]+):([0-9a-f]+)\]([^\n]*)")
TLB_RE = re.compile(r"PIM \w+ [RW] va=(0x[0-9a-f]+) pa=(0x[0-9a-f]+)")


def symbol(binary, name):
    out = subprocess.run(
        ["riscv64-linux-gnu-nm", binary], capture_output=True, text=True
    ).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == name:
            return int(parts[0], 16)
    raise SystemExit(f"{binary}: no symbol {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--vlen", type=int, required=True)
    ap.add_argument("--cpu", required=True)
    args = ap.parse_args()
    buf = symbol(args.binary, "pim_buf")
    fails = []

    def check(ok, what):
        print(("PASS " if ok else "FAIL ") + what)
        if not ok:
            fails.append(what)

    decodes, execs, packets, va2pa = [], [], [], {}
    with open(args.trace) as f:
        for line in f:
            m = DECODE_RE.search(line)
            if m:
                decodes.append((m.group(1), int(m.group(2), 16) & 0xFFFFFFFF))
                continue
            m = TLB_RE.search(line)
            if m:
                va2pa[int(m.group(1), 16) >> 12] = int(m.group(2), 16) >> 12
                continue
            m = EXEC_RE.match(line)
            if m and " : " in line:
                execs.append(
                    dict(
                        mnem=m.group(2),
                        opclass=m.group(4),
                        addr=int(m.group(5), 16) if m.group(5) else None,
                        flags=(m.group(6) or "").split("|"),
                    )
                )
                continue
            if args.cpu != "atomic":
                for p in PKT_RE.finditer(line):
                    packets.append(
                        (
                            p.group(1),
                            int(p.group(2), 16),
                            int(p.group(3), 16),
                            p.group(4),
                        )
                    )

    # 1. Decode.
    pim_decodes = [(n, w) for n, w in decodes if n.startswith("pim_")]
    bad = [(n, hex(w)) for n, w in pim_decodes if word_name(w) != n]
    check(
        pim_decodes and not bad,
        f"1. {len(pim_decodes)} PIM decodes name "
        f"the class of their word{' ' + str(bad[:3]) if bad else ''}",
    )
    leak = [
        (n, hex(w))
        for n, w in decodes
        if not n.startswith("pim_") and word_name(w)
    ]
    check(
        not leak,
        f"1. no PIM word decodes as anything else{' ' + str(leak[:3]) if leak else ''}",
    )

    # 2-5. Exec lines.
    want = expected_lines(args.mode, args.vlen, buf)
    vector_macros = set(VSTORE + VLOAD)
    got = [
        e
        for e in execs
        if e["mnem"].startswith("pim_") and e["mnem"] not in vector_macros
    ]
    check(
        [g["mnem"] for g in got] == [w[0] for w in want],
        f"2. {len(got)} PIM scalar and micro-op lines in program order "
        f"(want {len(want)})",
    )
    for g, (mnem, kind, addr, size) in zip(got, want):
        if g["mnem"] != mnem:
            break
        ok = "IsPim" in g["flags"] and not set(SERIALIZING) & set(g["flags"])
        if not ok:
            check(False, f"3. {mnem} flags {g['flags']}")
        if g["opclass"] != OPCLASS[kind]:
            check(
                False,
                f"4. {mnem} OpClass {g['opclass']}, want {OPCLASS[kind]}",
            )
        if g["addr"] != addr:
            check(
                False,
                f"5. {mnem} A={g['addr'] and hex(g['addr'])}, want {hex(addr)}",
            )
    check(
        all(
            "IsPim" in g["flags"] and not set(SERIALIZING) & set(g["flags"])
            for g in got
        ),
        "3. every PIM line has IsPim and no serializing flag",
    )
    check(
        all(g["opclass"] == OPCLASS[w[1]] for g, w in zip(got, want)),
        "4. OpClasses MemWrite / SimdUnitStrideStore / SimdUnitStrideLoad",
    )
    check(
        all(g["addr"] == w[2] for g, w in zip(got, want)),
        "5. micro-op counts and addresses rs1 + k * VLENB",
    )
    others = [
        e["mnem"]
        for e in execs
        if "IsPim" in e["flags"] and not e["mnem"].startswith("pim_")
    ]
    check(
        not others,
        f"3. no other line has IsPim{' ' + str(others[:3]) if others else ''}",
    )

    # 6. Packets.
    if args.cpu != "atomic":
        bufpages = {va2pa.get((buf >> 12) + i) for i in range(3)} - {None}
        pim = [
            p
            for p in packets
            if p[0].startswith("Pim") and p[0].endswith("Req")
        ]
        check(
            bool(pim) and bool(bufpages),
            f"6. {len(pim)} PIM packet lines, "
            f"pim_buf at physical pages {sorted(hex(p << 12) for p in bufpages)}",
        )
        check(
            all(" UC" in p[3] for p in pim), "6. every PIM packet carries UC"
        )
        check(
            all(
                p[1] >> 12 in bufpages and p[1] // LINE == p[2] // LINE
                for p in pim
            ),
            "6. every PIM packet lies in one line of pim_buf",
        )
        if "vsplit" in MODES[args.mode]:
            pa = (va2pa[(buf + 48) >> 12] << 12) | ((buf + 48) & 0xFFF)
            pieces = {(p[1], p[2]) for p in pim if p[0] == "PimWrGbReq"}
            check(
                (pa, pa + 15) in pieces and (pa + 16, pa + 31) in pieces,
                "6. the line-crossing pim.wr.gb gives two packets meeting at "
                "the line",
            )
        uc_other = {
            p[0]
            for p in packets
            if " UC" in p[3] and not p[0].startswith("Pim")
        }
        check(
            not uc_other,
            f"6. no other packet is uncacheable"
            f"{' ' + str(sorted(uc_other)) if uc_other else ''}",
        )
    print(f"check_exec_trace: {len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
