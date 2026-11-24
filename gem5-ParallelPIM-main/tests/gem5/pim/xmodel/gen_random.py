"""Random PIM programs in the libpim/tests/vectors format, for the
cross-check of gem5's functional model against the C reference model (both
run with --dump; the expect values are placeholders). No NaN or infinity
inputs: there host floats (C model) and SoftFloat (gem5) legitimately differ
in NaN payloads and signs."""

import random
import sys

PAIRS = [
    ("bf16", "fp32"),
    ("bf16", "bf16"),
    ("fp16", "fp32"),
    ("fp16", "fp16"),
    ("fp16", "bf16"),
    ("fp32", "fp32"),
    ("int8", "int8"),
    ("int8", "int16"),
    ("int8", "int32"),
    ("int16", "int16"),
    ("int16", "int32"),
    ("int32", "int32"),
]
SIZE = {"bf16": 2, "fp16": 2, "fp32": 4, "int8": 1, "int16": 2, "int32": 4}
COLB, COLS, ROWS, BGS, BANKS, UNITS, CH = 16, 4, 16, 2, 2, 2, 2
ROWB = COLB * COLS
SCALAR = [
    "pim.sb.ewadd",
    "pim.sb.ewmul",
    "pim.sb.ewadd.gbuf",
    "pim.sb.ewmul.gbuf",
    "pim.sb.mac",
]
VECTOR = [
    "pim.sb.mov.bkgbuf",
    "pim.sb.mov.gbufbk",
    "pim.ab.ewadd",
    "pim.ab.ewmul",
    "pim.ab.ewadd.gbuf",
    "pim.ab.ewmul.gbuf",
    "pim.ab.mac",
    "pim.ab.mov.regbk",
    "pim.ab.mov.bkreg",
]


def value(rng, fmt):
    if fmt.startswith("int"):
        return "0x%x" % rng.getrandbits(8 * SIZE[fmt])
    sign = rng.getrandbits(1)
    if fmt == "bf16":
        e = 0 if rng.random() < 0.1 else rng.randint(127 - 20, 127 + 10)
        return "0x%x" % (sign << 15 | e << 7 | rng.getrandbits(7))
    if fmt == "fp16":
        e = 0 if rng.random() < 0.15 else rng.randint(1, 19)
        return "0x%x" % (sign << 15 | e << 10 | rng.getrandbits(10))
    e = 0 if rng.random() < 0.1 else rng.randint(127 - 30, 127 + 10)
    return "0x%x" % (sign << 31 | e << 23 | rng.getrandbits(23))


def vals(rng, fmt, n):
    return " ".join([fmt] + [value(rng, fmt) for _ in range(n)])


def bank(rng):
    return dict(
        ch=rng.randrange(CH),
        unit=rng.randrange(UNITS),
        bg=rng.randrange(BGS),
        bank=rng.randrange(BANKS),
    )


def coord(c, **kw):
    d = dict(c, **kw)
    return " ".join(f"{k}={v}" for k, v in d.items())


def program(rng, num, acc, out):
    e = SIZE[num]
    epr = ROWB // e
    out.append(
        "geom policy=0 channels=%d intlv=%d units=%d sids=1 bgs=%d banks=%d rows=%d "
        "cols=%d col_bytes=%d num_regs=1 num_fmt=%s acc_fmt=%s vlenb=%d"
        % (CH, ROWB, UNITS, BGS, BANKS, ROWS, COLS, COLB, num, acc, ROWB // 8)
    )
    for ch in range(CH):
        for u in range(UNITS):
            for bg in range(BGS):
                for bk in range(BANKS):
                    for row in range(1, 4):
                        c = dict(ch=ch, unit=u, bg=bg, bank=bk, row=row)
                        out.append(f"row {coord(c)} {vals(rng, num, epr)}")
            out.append("gb ch=%d unit=%d %s" % (ch, u, vals(rng, num, epr)))
    for _ in range(40):
        b = bank(rng)
        row = rng.randint(1, 3)
        kind = rng.random()
        if kind < 0.4:
            op = rng.choice(SCALAR)
            if op in ("pim.sb.ewadd", "pim.sb.ewmul"):
                b["bank"] &= ~1
            col = rng.randrange(COLS)
            out.append(
                "cmd %s %s ncols=%d"
                % (op, coord(b, row=row, col=col), rng.randint(0, COLS - col))
            )
        elif kind < 0.85:
            op = rng.choice(VECTOR)
            if op in ("pim.ab.ewadd", "pim.ab.ewmul"):
                b["bank"] &= ~1
            lo = rng.randrange(epr) * e
            hi = rng.randint(lo // e + 1, epr) * e
            out.append(
                "cmd %s %s bytes=%d"
                % (
                    op,
                    coord(b, row=row, col=lo // COLB, byte=lo % COLB),
                    hi - lo,
                )
            )
        elif kind < 0.93:
            lo = rng.randrange(ROWB)
            n = rng.randint(1, ROWB - lo)
            if NONAN:
                lo -= lo % e
                n = max(e, n - n % e)
                n = min(n, ROWB - lo)
                data = vals(rng, num, n // e)
            else:
                data = "int8 " + " ".join(
                    "0x%x" % rng.getrandbits(8) for _ in range(n)
                )
            out.append(
                "cmd pim.wr.gb ch=%d unit=%d col=%d byte=%d %s"
                % (b["ch"], b["unit"], lo // COLB, lo % COLB, data)
            )
        else:
            n = rng.randint(1, 8)
            data = (
                vals(rng, num, max(1, n // e))
                if NONAN
                else "int8 "
                + " ".join("0x%x" % rng.getrandbits(8) for _ in range(n))
            )
            out.append(
                "cmd pim.wr.reg %s %s"
                % (coord(b, row="reg0", col=rng.randrange(COLS)), data)
            )
        # Dump the state of the unit the command touched.
        for bg in range(BGS):
            for bk in range(BANKS):
                c = dict(ch=b["ch"], unit=b["unit"], bg=bg, bank=bk)
                out.append(
                    "expect reg %s j=0 %s"
                    % (coord(c), " ".join([num] + ["0"] * epr))
                )
                out.append(f"expect acc {coord(c)} {acc} 0")
                out.append(
                    "expect row %s %s"
                    % (coord(c, row=row), " ".join([num] + ["0"] * epr))
                )
        out.append(
            "expect gb ch=%d unit=%d %s"
            % (b["ch"], b["unit"], " ".join([num] + ["0"] * epr))
        )


NONAN = "--nonan" in sys.argv


def main():
    seeds = int(sys.argv[1])
    out = []
    for s in range(seeds):
        for num, acc in PAIRS:
            program(random.Random("%d-%s-%s" % (s, num, acc)), num, acc, out)
    print("\n".join(out))


main()
