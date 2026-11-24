#!/usr/bin/env python3
"""Generate the Processing-in-Memory (PIM) layout header.

The header ``include/gem5/pim/pim_layout_<std>.h`` holds the constants of
one PIM memory configuration: the dynamic random-access memory (DRAM)
organization, the address-mapping policy, the channel interleave, the bit
position of every DRAM level, the largest address-contiguous piece of a
bank row, the PIM memory's base in Syscall Emulation (SE) and Full System
(FS) mode, the register rows and the number formats. The PIM library
compiles them in; gem5 reads the same file at start-up and stops on a
mismatch. Every constant is written as ``#define PIM_LAYOUT_<NAME>
<number>``, the only form gem5's reader accepts.

The organization comes from the Ramulator 2.1 Python package embedded in
``ext/ramulator2/python``; nothing is written there (no bytecode).

Examples::

    util/pim/gen_pim_layout.py --defaults          # the four headers
    util/pim/gen_pim_layout.py --check             # are they up to date?
    util/pim/gen_pim_layout.py --standard lpddr6 --policy RoBaRaCoCh \\
        --interleave 64 -o my_layout.h
"""

import argparse
import hashlib
import os
import re
import sys

sys.dont_write_bytecode = True

GEM5_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
RAMULATOR_PYTHON = os.path.join(GEM5_ROOT, "ext", "ramulator2", "python")
DEFAULT_OUTDIR = os.path.join(GEM5_ROOT, "include", "gem5", "pim")

# The regular expression gem5's start-up check reads the header with.
LAYOUT_LINE_RE = re.compile(
    r"^\s*#\s*define\s+PIM_LAYOUT_(\w+)\s+(0x[0-9a-fA-F]+|\d+)[uUlL]*\s*$"
)

# The header's identifier of each PIM standard (PIM_LAYOUT_STD).
STD_IDS = {"ddr5": 1, "lpddr6": 2, "gddr7": 3, "hbm4": 4}


def _ramulator_dram():
    if RAMULATOR_PYTHON not in sys.path:
        sys.path.insert(0, RAMULATOR_PYTHON)
    import ramulator.dram

    return ramulator.dram


def _standards():
    """The PIM standards: identifier, Ramulator 2.1 class, organization and
    timing presets, organization overrides and the default channel count,
    from the per-standard descriptors of Ramulator 2.1's
    python/ramulator/dram/pim.py (docs/pim/05b-architecture-timing-plan.md
    Q4 (b))."""
    _ramulator_dram()
    from ramulator.dram.pim import DESCRIPTORS

    out = {}
    for name, std_id in STD_IDS.items():
        p = DESCRIPTORS[name.upper()].presets
        out[name] = dict(
            std_id=std_id,
            cls=name.upper(),
            org=p["org"],
            timing=p["timing"],
            overrides=dict(p["org_overrides"]),
            channels=p["channels"],
        )
    return out


STANDARDS = _standards()

# gem5's three policies in the order of its AddrMap enum, then Ramulator
# 2.1's three; PIM_LAYOUT_POLICY is the index.
POLICIES = (
    "RoRaBaChCo",
    "RoRaBaCoCh",
    "RoCoRaBaCh",
    "RoBaRaCoCh",
    "ChRaBaRoCo",
    "MOP4CLXOR",
)
DEFAULT_POLICY = "RoRaBaChCo"

# Number formats in the order of the simulator's PimNumFormat values:
# (name, bytes, floating point).
FORMATS = (
    ("bf16", 2, True),
    ("fp16", 2, True),
    ("fp32", 4, True),
    ("int8", 1, False),
    ("int16", 2, False),
    ("int32", 4, False),
)
FORMAT_NAMES = tuple(f[0] for f in FORMATS)

CACHE_LINE_BYTES = 64
SE_OS_BASE = 0x0  # SimpleBoard's memory starts at 0
FS_OS_BASE = 0x80000000  # RiscvBoard's memory starts at 0x80000000

# Ramulator 2.1 level names and the coordinate field each one becomes.
LEVEL_FIELDS = {
    "Rank": "unit",
    "PseudoChannel": "unit",
    "Sid": "sid",
    "BankGroup": "bg",
    "Bank": "bank",
    "Row": "row",
    "Column": "col",
}
HIERARCHY = ("unit", "sid", "bg", "bank", "row")


class LayoutError(ValueError):
    """A configuration the PIM memory does not accept."""


def _log2(value, what):
    if value <= 0 or value & (value - 1):
        raise LayoutError(f"{what} must be a power of two, got {value}")
    return value.bit_length() - 1


def _align_up(value, alignment):
    return -(-value // alignment) * alignment


def resolve_organization(standard):
    """The organization of a standard's PIM preset, from Ramulator 2.1.

    Returns a dict with the level counts (one per coordinate field, 0 for
    a level the standard does not have), the transaction (column) bytes
    and the column count in transactions, computed as the stdlib's
    ``channel_capacity`` does.
    """
    spec = STANDARDS[standard]
    cls = getattr(_ramulator_dram(), spec["cls"])
    device = cls(
        org_preset=spec["org"],
        timing_preset=spec["timing"],
        **spec["overrides"],
    )
    org, _ = device.resolve()
    levels = list(cls.levels)
    if levels[0] != "Channel" or levels[-1] != "Column":
        raise LayoutError(f"{cls.name}: unexpected levels {levels}")
    counts = {field: 0 for field in HIERARCHY}
    fields = []
    for level in levels[1:-1]:
        if level not in LEVEL_FIELDS:
            raise LayoutError(f"{cls.name}: unknown level {level}")
        field = LEVEL_FIELDS[level]
        counts[field] = org[level.lower()]
        fields.append(field)
    if fields != [f for f in HIERARCHY if f in fields]:
        raise LayoutError(f"{cls.name}: unexpected level order {levels}")
    prefetch = cls.internal_prefetch_size
    tx = cls.data_payload_bytes or (prefetch * org["channel_width"] // 8)
    counts["col"] = org["column"] // prefetch
    return dict(
        name=cls.name,
        levels=levels,
        counts=counts,
        present=set(fields),
        col_bytes=tx,
    )


class Layout:
    """One PIM memory configuration and its address bit map."""

    def __init__(
        self,
        standard,
        channels=None,
        policy=DEFAULT_POLICY,
        interleave=None,
        num_regs=1,
        num_fmt="bf16",
        acc_fmt="fp32",
        os_size=None,
        organization=None,
    ):
        if standard not in STANDARDS:
            raise LayoutError(
                f"unknown standard {standard}; choose one of "
                + ", ".join(STANDARDS)
            )
        if policy not in POLICIES:
            raise LayoutError(
                f"unknown policy {policy}; choose one of "
                + ", ".join(POLICIES)
            )
        self.standard = standard
        self.spec = STANDARDS[standard]
        self.org = organization or resolve_organization(standard)
        self.channels = self.spec["channels"] if channels is None else channels
        self.policy = policy
        self.num_regs = num_regs
        self.num_fmt = num_fmt
        self.acc_fmt = acc_fmt
        self._check_counts()
        self.interleave = self._interleave(interleave)
        self.window_size = self.channels * self.channel_capacity
        self.os_size = self.window_size if os_size is None else os_size
        if self.os_size <= 0:
            raise LayoutError(f"OS memory size must be positive")
        self._check_regs_and_formats()
        self.fields = self._local_fields()
        self.bitmap = self._bitmap()
        self.constants = self._constants()

    # Organization ------------------------------------------------------

    @property
    def counts(self):
        return self.org["counts"]

    @property
    def col_bytes(self):
        return self.org["col_bytes"]

    @property
    def row_bytes(self):
        return self.counts["col"] * self.col_bytes

    def width(self, field):
        count = self.counts[field]
        return _log2(count, field) if count else 0

    @property
    def units(self):
        return self.counts["unit"] or 1

    @property
    def banks_per_unit(self):
        return (
            (self.counts["sid"] or 1)
            * (self.counts["bg"] or 1)
            * self.counts["bank"]
        )

    @property
    def channel_capacity(self):
        return (
            self.units * self.banks_per_unit * self.counts["row"]
        ) * self.row_bytes

    def _check_counts(self):
        _log2(self.channels, "channel count")
        _log2(self.col_bytes, "column bytes")
        for field in HIERARCHY + ("col",):
            if field in self.org["present"] or field in ("bank", "row", "col"):
                _log2(self.counts[field], f"{field} count")
        if self.policy == "MOP4CLXOR" and self.width("col") < 2:
            raise LayoutError("MOP4CLXOR needs at least two column bits")

    def _interleave(self, interleave):
        fixed = {
            "RoRaBaChCo": self.row_bytes,
            "ChRaBaRoCo": self.channel_capacity,
        }.get(self.policy)
        if fixed is not None:
            if interleave is not None and interleave != fixed:
                raise LayoutError(
                    f"{self.policy} fixes the channel interleave at {fixed} "
                    f"bytes"
                )
            return fixed
        value = CACHE_LINE_BYTES if interleave is None else interleave
        _log2(value, "channel interleave")
        if value < CACHE_LINE_BYTES or value < self.col_bytes:
            raise LayoutError(
                f"channel interleave {value} is below one cache line "
                f"({CACHE_LINE_BYTES} bytes) or one column"
            )
        if value > self.channel_capacity:
            raise LayoutError(
                f"channel interleave {value} is above one channel's "
                f"capacity ({self.channel_capacity} bytes)"
            )
        if self.policy in ("RoRaBaCoCh", "RoCoRaBaCh") and (
            value > self.row_bytes
        ):
            raise LayoutError(
                f"{self.policy} needs a channel interleave of at most one "
                f"row ({self.row_bytes} bytes), as gem5 does"
            )
        return value

    def _check_regs_and_formats(self):
        if self.num_regs < 1 or self.num_regs + 1 >= self.counts["row"]:
            raise LayoutError(f"bad register count {self.num_regs}")
        for name in (self.num_fmt, self.acc_fmt):
            if name not in FORMAT_NAMES:
                raise LayoutError(
                    f"unknown format {name}; choose one of "
                    + ", ".join(FORMAT_NAMES)
                )
        _, num_bytes, num_float = FORMATS[FORMAT_NAMES.index(self.num_fmt)]
        _, acc_bytes, acc_float = FORMATS[FORMAT_NAMES.index(self.acc_fmt)]
        if acc_bytes < num_bytes or acc_float != num_float:
            raise LayoutError(
                f"accumulator format {self.acc_fmt} must be at least as "
                f"wide as {self.num_fmt} and of the same family"
            )
        if self.col_bytes % num_bytes:
            raise LayoutError(
                f"{self.num_fmt} elements do not divide a column"
            )

    # Address bit map ---------------------------------------------------

    def _local_fields(self):
        """The fields of the channel-local transaction address.

        A list of (field, width, low bit of the field value), least
        significant first, as each policy's mapper decodes the address
        that remains once the channel mapper has removed the channel
        bits. Absent levels appear with width 0, so that their bit
        position is still defined.
        """
        w = {f: self.width(f) for f in HIERARCHY + ("col",)}

        def lvl(field):
            return (field, w[field], 0)

        col = w["col"]
        p = self.policy
        if p in ("RoRaBaChCo", "RoRaBaCoCh"):
            # src/mem/dram_interface.cc:856-872: column, flat bank, rank,
            # row. The flat bank puts the bank group lowest (:1140-1151),
            # then the bank and, on HBM4, the stack ID; the rank is the
            # PIM unit (rank or pseudo channel).
            return [("col", col, 0)] + [
                lvl(f) for f in ("bg", "bank", "sid", "unit", "row")
            ]
        if p == "RoCoRaBaCh":
            # src/mem/dram_interface.cc:873-895: the column bits of one
            # channel stripe, the flat bank, the rank, the other column
            # bits, the row. gem5 uses a stripe of one burst when the
            # range is not interleaved (src/mem/mem_interface.cc:64-65).
            stripe = 0
            if self.channels > 1:
                stripe = min(
                    _log2(self.interleave // self.col_bytes, "stripe"), col
                )
            return (
                [("col", stripe, 0)]
                + [lvl(f) for f in ("bg", "bank", "sid", "unit")]
                + [("col", col - stripe, stripe), lvl("row")]
            )
        if p == "RoBaRaCoCh":
            # ro_ba_ra_co_ch.cpp:17-26: column, then the levels from the
            # first below the channel down to the row.
            return [("col", col, 0)] + [lvl(f) for f in HIERARCHY]
        if p == "ChRaBaRoCo":
            # ch_ra_ba_ro_co.cpp:17-24: column, row, ..., rank.
            return [("col", col, 0)] + [lvl(f) for f in reversed(HIERARCHY)]
        # MOP4CLXOR, mop4clxor.cpp:20-43: two column bits, the levels
        # above the row, the other column bits, the row; the levels are
        # then XORed with column bits (see xor_masks()).
        return (
            [("col", 2, 0)]
            + [lvl(f) for f in HIERARCHY[:-1]]
            + [("col", col - 2, 2), lvl("row")]
        )

    def xor_sources(self):
        """For MOP4CLXOR: (field, low column bit) of each XORed level.

        mop4clxor.cpp:36-43 XORs every level with bits, in level order
        (unit, sid, bank group, bank, row), with the next column bits.
        """
        if self.policy != "MOP4CLXOR":
            return []
        out = []
        x = 0
        for field in HIERARCHY:
            if self.width(field) > 0:
                out.append((field, x))
                x += self.width(field)
        return out

    @property
    def tx_bits(self):
        return _log2(self.col_bytes, "column bytes")

    @property
    def ch_shift(self):
        return _log2(self.interleave, "channel interleave")

    @property
    def ch_bits(self):
        return _log2(self.channels, "channel count")

    def _physical(self, local_bit):
        """Physical offset bit of a channel-local byte-address bit."""
        if local_bit >= self.ch_shift:
            return local_bit + self.ch_bits
        return local_bit

    def _bitmap(self):
        """Physical offset bit -> (field, bit of the field value)."""
        total = _log2(self.window_size, "PIM memory size")
        bitmap = [None] * total
        for k in range(self.tx_bits):
            bitmap[k] = ("byte", k)
        for k in range(self.ch_bits):
            bitmap[self.ch_shift + k] = ("ch", k)
        pos = self.tx_bits
        for field, width, low in self.fields:
            for k in range(width):
                phys = self._physical(pos + k)
                if bitmap[phys] is not None:
                    raise LayoutError(f"bit {phys} mapped twice")
                bitmap[phys] = (field, low + k)
            pos += width
        if any(b is None for b in bitmap):
            raise LayoutError("address bits left unmapped")
        return bitmap

    def shift(self, field):
        """Physical position of bit 0 of a field (also for width 0)."""
        if field == "ch":
            return self.ch_shift
        pos = self.tx_bits
        for name, width, low in self.fields:
            if name == field and low == 0:
                return self._physical(pos)
            pos += width
        raise LayoutError(f"no field {field}")

    @property
    def contig_bytes(self):
        """The largest aligned range inside one bank row (DP-E (a)).

        Counted from the bit map: the run of column bits 0, 1, ... above
        the byte offset, ended by a channel bit, another level, or a
        column bit that MOP4CLXOR XORs into a level.
        """
        xored = set()
        for field, low in self.xor_sources():
            xored.update(range(low, low + self.width(field)))
        bit = self.tx_bits
        k = 0
        while (
            bit < len(self.bitmap)
            and self.bitmap[bit] == ("col", k)
            and k not in xored
        ):
            bit += 1
            k += 1
        return 1 << bit

    # Constants ---------------------------------------------------------

    def _constants(self):
        c = self.counts
        num = FORMATS[FORMAT_NAMES.index(self.num_fmt)]
        acc = FORMATS[FORMAT_NAMES.index(self.acc_fmt)]
        present = self.org["present"]
        unit_is_rank = "Rank" in self.org["levels"]
        unit_is_pc = "PseudoChannel" in self.org["levels"]
        window_base_se = _align_up(SE_OS_BASE + self.os_size, self.window_size)
        window_base_fs = _align_up(FS_OS_BASE + self.os_size, self.window_size)
        values = [
            ("STD", self.spec["std_id"]),
            ("POLICY", POLICIES.index(self.policy)),
            ("CHANNELS", self.channels),
            ("INTLV_BYTES", self.interleave),
            ("RANKS", c["unit"] if unit_is_rank else 0),
            ("PSEUDO_CHANNELS", c["unit"] if unit_is_pc else 0),
            ("SIDS", c["sid"] if "sid" in present else 0),
            ("BANK_GROUPS", c["bg"] if "bg" in present else 0),
            ("BANKS_PER_GROUP", c["bank"]),
            ("ROWS", c["row"]),
            ("COLS", c["col"]),
            ("COL_BYTES", self.col_bytes),
            ("ROW_BYTES", self.row_bytes),
            ("UNITS", self.units),
            ("BANKS_PER_UNIT", self.banks_per_unit),
            ("PUS_PER_BANK", 1),
            ("COL_SHIFT", self.shift("col")),
            ("CH_SHIFT", self.shift("ch")),
            ("UNIT_SHIFT", self.shift("unit")),
            ("SID_SHIFT", self.shift("sid")),
            ("BG_SHIFT", self.shift("bg")),
            ("BANK_SHIFT", self.shift("bank")),
            ("ROW_SHIFT", self.shift("row")),
            ("CONTIG_BYTES", self.contig_bytes),
            ("WINDOW_SIZE", self.window_size),
            ("WINDOW_ROW0", 0),
            ("WINDOW_BASE_SE", window_base_se),
            ("WINDOW_BASE_FS", window_base_fs),
            ("NUM_REGS", self.num_regs),
            ("REG_ROW0", c["row"] - self.num_regs - 1),
            ("NUM_FMT", FORMAT_NAMES.index(self.num_fmt)),
            ("ACC_FMT", FORMAT_NAMES.index(self.acc_fmt)),
            ("ELEM_BYTES", num[1]),
            ("ELEMS_PER_COL", self.col_bytes // num[1]),
            ("ACC_BYTES", acc[1]),
        ]
        return dict(values)

    @property
    def layout_id(self):
        """An identifier derived from every constant (DP-B (a))."""
        text = "".join(f"{k}={v}\n" for k, v in self.constants.items())
        return (
            "pim_layout_id_" + hashlib.sha256(text.encode()).hexdigest()[:12]
        )

    def command_line(self, output=None):
        args = [
            "util/pim/gen_pim_layout.py",
            f"--standard {self.standard}",
            f"--channels {self.channels}",
            f"--policy {self.policy}",
        ]
        if self.policy not in ("RoRaBaChCo", "ChRaBaRoCo"):
            args.append(f"--interleave {self.interleave}")
        if self.num_regs != 1:
            args.append(f"--num-regs {self.num_regs}")
        if self.num_fmt != "bf16":
            args.append(f"--num-fmt {self.num_fmt}")
        if self.acc_fmt != "fp32":
            args.append(f"--acc-fmt {self.acc_fmt}")
        if self.os_size != self.window_size:
            args.append(f"--os-size {self.os_size}")
        if output:
            args.append(f"-o {output}")
        return args

    def describe(self):
        """The bit map as text, most significant field first."""
        runs = []
        for bit, (field, k) in enumerate(self.bitmap):
            if runs and runs[-1][0] == field and runs[-1][3] + 1 == k:
                runs[-1][2] = bit
                runs[-1][3] = k
            else:
                runs.append([field, bit, bit, k, k])
        parts = []
        for field, lo, hi, _, k0 in reversed(runs):
            label = field if field != "col" else f"col[{k0}+]"
            parts.append(f"{label} {hi}-{lo}" if hi != lo else f"{label} {lo}")
        return " | ".join(parts)

    def header_text(self, default=False):
        guard = (
            f"__GEM5_PIM_PIM_LAYOUT_{self.standard.upper()}_H__"
            if default
            else "__GEM5_PIM_PIM_LAYOUT_CUSTOM_H__"
        )
        regen = (
            "util/pim/gen_pim_layout.py --defaults"
            if default
            else " ".join(self.command_line())
        )
        hexed = ("WINDOW_SIZE", "WINDOW_BASE_SE", "WINDOW_BASE_FS")
        lines = [
            "/*",
            " * Generated by util/pim/gen_pim_layout.py; do not edit.",
            f" * Regenerate with: {regen}",
            " *",
            f" * {self.org['name']} ({self.spec['org']}) x {self.channels} "
            f"channels, policy {self.policy},",
            f" * channel interleave {self.interleave} bytes, "
            f"{self.num_regs} register(s), {self.num_fmt} operands,",
            f" * {self.acc_fmt} accumulator, OS memory of "
            f"{self.os_size} bytes.",
            " * Physical offset bits, most significant first:",
        ]
        text = self.describe()
        while text:
            cut = len(text) if len(text) <= 72 else text.rfind(" | ", 0, 72)
            lines.append(f" *   {text[:cut]}")
            text = text[cut + 3 :] if cut < len(text) else ""
        lines += [
            " *",
            " * Every PIM_LAYOUT_ line has the form gem5's start-up check",
            " * reads; include <gem5/pim/pim_layout.h> instead of this file.",
            " */",
            "",
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
        ]
        for name, value in self.constants.items():
            text = f"{value:#x}" if name in hexed else str(value)
            lines.append(f"#define PIM_LAYOUT_{name} {text}")
        lines += [
            "",
            "/* Names the archive built for this layout (libpim_<std>.a). */",
            f"#define GEM5_PIM_LAYOUT_ID {self.layout_id}",
            "",
            f"#endif // {guard}",
            "",
        ]
        return "\n".join(lines)


def default_layout(standard):
    return Layout(standard)


def default_header_path(standard, outdir=DEFAULT_OUTDIR):
    return os.path.join(outdir, f"pim_layout_{standard}.h")


def _size(text):
    """Bytes from a number with an optional KiB/MiB/GiB/TiB suffix."""
    units = {"KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30, "TiB": 1 << 40}
    for suffix, scale in units.items():
        if text.endswith(suffix):
            return int(text[: -len(suffix)], 0) * scale
    return int(text, 0)


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--defaults",
        action="store_true",
        help="write the four default headers into --outdir",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="exit with status 1 if a default header in --outdir is not "
        "what the generator writes",
    )
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--standard", choices=sorted(STANDARDS))
    parser.add_argument(
        "--channels", type=int, help="default: the standard's default"
    )
    parser.add_argument("--policy", choices=POLICIES, default=DEFAULT_POLICY)
    parser.add_argument(
        "--interleave",
        type=_size,
        help="channel interleave in bytes (default: fixed by the policy, "
        "or one cache line)",
    )
    parser.add_argument("--num-regs", type=int, default=1)
    parser.add_argument("--num-fmt", choices=FORMAT_NAMES, default="bf16")
    parser.add_argument("--acc-fmt", choices=FORMAT_NAMES, default="fp32")
    parser.add_argument(
        "--os-size",
        type=_size,
        help="OS memory size (default: the PIM memory's size)",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="print the address bit map instead of a header",
    )
    parser.add_argument("-o", "--output", help="default: standard output")
    args = parser.parse_args(argv)

    try:
        if args.defaults or args.check:
            stale = []
            for standard in STANDARDS:
                text = default_layout(standard).header_text(default=True)
                path = default_header_path(standard, args.outdir)
                if args.defaults:
                    _write(path, text)
                    continue
                try:
                    with open(path, encoding="utf-8") as f:
                        current = f.read()
                except OSError:
                    current = None
                if current != text:
                    stale.append(path)
            for path in stale:
                print(f"out of date: {path}", file=sys.stderr)
            return 1 if stale else 0
        if not args.standard:
            parser.error("--standard is required without --defaults/--check")
        layout = Layout(
            args.standard,
            channels=args.channels,
            policy=args.policy,
            interleave=args.interleave,
            num_regs=args.num_regs,
            num_fmt=args.num_fmt,
            acc_fmt=args.acc_fmt,
            os_size=args.os_size,
        )
    except LayoutError as e:
        print(f"gen_pim_layout.py: error: {e}", file=sys.stderr)
        return 2
    if args.describe:
        text = layout.describe() + "\n"
    else:
        text = layout.header_text()
    if args.output:
        _write(args.output, text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
