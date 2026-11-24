#!/usr/bin/env python3
"""Tests of util/pim/gen_pim_layout.py and the committed layout headers.

Run with ``python3 -B util/pim/tests/test_gen_layout.py`` or through
``scons -C util/pim build/native/test``.
"""

import os
import re
import subprocess
import sys
import unittest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import gen_pim_layout as g  # noqa: E402

GiB = 1 << 30

# P-DRAM section 6.5: field widths under RoBaRaCoCh with the interleave
# equal to one row: (transaction, column, unit, SID, bank group, bank,
# row) bits.
PDRAM_6_5 = {
    "ddr5": (6, 7, 1, 0, 3, 2, 17),
    "lpddr6": (5, 6, 0, 0, 2, 2, 16),
    "gddr7": (5, 6, 0, 0, 0, 4, 16),
    "hbm4": (5, 5, 1, 2, 1, 3, 14),
}

# Every name the generator writes, in order.
NAMES = (
    "STD POLICY CHANNELS INTLV_BYTES RANKS PSEUDO_CHANNELS SIDS BANK_GROUPS "
    "BANKS_PER_GROUP ROWS COLS COL_BYTES ROW_BYTES UNITS BANKS_PER_UNIT "
    "PUS_PER_BANK COL_SHIFT CH_SHIFT UNIT_SHIFT SID_SHIFT BG_SHIFT "
    "BANK_SHIFT ROW_SHIFT CONTIG_BYTES WINDOW_SIZE WINDOW_ROW0 "
    "WINDOW_BASE_SE WINDOW_BASE_FS NUM_REGS REG_ROW0 NUM_FMT ACC_FMT "
    "ELEM_BYTES ELEMS_PER_COL ACC_BYTES"
).split()

_ORGS = {}


def layout(standard, **kw):
    """A Layout, resolving each organization once."""
    if standard not in _ORGS:
        _ORGS[standard] = g.resolve_organization(standard)
    return g.Layout(standard, organization=_ORGS[standard], **kw)


def read_header(standard):
    with open(g.default_header_path(standard), encoding="utf-8") as f:
        return f.read()


def expected_contig(lay):
    """PIM_CONTIG_BYTES by DP-E (a), in closed form."""
    r, c, i = lay.row_bytes, lay.col_bytes, lay.interleave
    p = lay.policy
    if p in ("RoRaBaChCo", "ChRaBaRoCo"):
        return r
    if p == "MOP4CLXOR":
        return c
    if lay.channels == 1:
        return c if p == "RoCoRaBaCh" else r
    return min(i, r)


class CommittedHeaders(unittest.TestCase):
    def test_lines_match_gem5_regex(self):
        for std in g.STANDARDS:
            text = read_header(std)
            seen = []
            for line in text.splitlines():
                if not re.match(r"^\s*#\s*define\b", line):
                    continue
                if re.match(r"^\s*#\s*define\s+PIM_LAYOUT_", line):
                    m = g.LAYOUT_LINE_RE.match(line)
                    self.assertIsNotNone(m, f"{std}: {line!r}")
                    seen.append(m.group(1))
                else:
                    self.assertRegex(
                        line,
                        r"^#define (__GEM5_PIM_PIM_LAYOUT_[A-Z0-9]+_H__|"
                        r"GEM5_PIM_LAYOUT_ID pim_layout_id_[0-9a-f]{12})$",
                    )
            self.assertEqual(seen, NAMES, std)

    def test_regenerating_changes_nothing(self):
        self.assertEqual(g.main(["--check"]), 0)
        for std in g.STANDARDS:
            self.assertEqual(
                read_header(std), layout(std).header_text(default=True)
            )

    def test_layout_ids_differ(self):
        ids = {layout(s).layout_id for s in g.STANDARDS}
        ids.add(layout("lpddr6", policy="RoBaRaCoCh").layout_id)
        self.assertEqual(len(ids), 5)


class DefaultValues(unittest.TestCase):
    def test_lpddr6_x8(self):
        # P-DRAM section 9 and P4R sections 6.2-6.3.
        c = layout("lpddr6").constants
        expect = dict(
            STD=2,
            POLICY=0,
            CHANNELS=8,
            INTLV_BYTES=2048,
            RANKS=1,
            PSEUDO_CHANNELS=0,
            SIDS=0,
            BANK_GROUPS=4,
            BANKS_PER_GROUP=4,
            ROWS=65536,
            COLS=64,
            COL_BYTES=32,
            ROW_BYTES=2048,
            UNITS=1,
            BANKS_PER_UNIT=16,
            PUS_PER_BANK=1,
            COL_SHIFT=5,
            CH_SHIFT=11,
            BG_SHIFT=14,
            BANK_SHIFT=16,
            ROW_SHIFT=18,
            CONTIG_BYTES=2048,
            WINDOW_SIZE=16 * GiB,
            WINDOW_ROW0=0,
            WINDOW_BASE_SE=0x400000000,
            WINDOW_BASE_FS=0x800000000,
            NUM_REGS=1,
            REG_ROW0=65534,
            NUM_FMT=0,
            ACC_FMT=2,
            ELEM_BYTES=2,
            ELEMS_PER_COL=16,
            ACC_BYTES=4,
        )
        for name, value in expect.items():
            self.assertEqual(c[name], value, name)

    def test_other_defaults(self):
        # DP-D (b): DDR5 x 2, the others x 8.
        d = layout("ddr5").constants
        self.assertEqual(
            (d["CHANNELS"], d["RANKS"], d["ROW_BYTES"], d["WINDOW_SIZE"]),
            (2, 2, 8192, 128 * GiB),
        )
        self.assertEqual(d["WINDOW_BASE_SE"], 0x2000000000)
        self.assertEqual(d["WINDOW_BASE_FS"], 0x4000000000)
        self.assertEqual(d["REG_ROW0"], 131070)
        for std, row, size in (("gddr7", 2048, 16), ("hbm4", 1024, 16)):
            c = layout(std).constants
            self.assertEqual(c["CHANNELS"], 8)
            self.assertEqual(c["ROW_BYTES"], row)
            self.assertEqual(c["WINDOW_SIZE"], size * GiB)
            self.assertEqual(c["WINDOW_BASE_SE"], 0x400000000)
            self.assertEqual(c["WINDOW_BASE_FS"], 0x800000000)
        h = layout("hbm4").constants
        self.assertEqual(
            (h["PSEUDO_CHANNELS"], h["SIDS"], h["UNITS"], h["BANKS_PER_UNIT"]),
            (2, 4, 2, 64),
        )

    def test_hbm4_gem5_policy_order(self):
        # DP-A (a): column, bank group, bank, SID, pseudo channel, row.
        for policy in ("RoRaBaChCo", "RoRaBaCoCh"):
            lay = layout("hbm4", policy=policy, interleave=1024)
            order = []
            for field, _ in lay.bitmap:
                if field not in order:
                    order.append(field)
            self.assertEqual(
                order,
                ["byte", "col", "ch", "bg", "bank", "sid", "unit", "row"],
            )


class BitPositions(unittest.TestCase):
    def test_p_dram_6_5(self):
        for std, widths in PDRAM_6_5.items():
            row_bytes = 1 << (widths[0] + widths[1])
            lay = layout(
                std, channels=8, policy="RoBaRaCoCh", interleave=row_bytes
            )
            c = lay.constants
            tx, col, unit, sid, bg, bank, row = widths
            pos = tx + col + 3  # 8 channels
            self.assertEqual(c["COL_SHIFT"], tx, std)
            self.assertEqual(c["CH_SHIFT"], tx + col, std)
            for name, width in (
                ("UNIT", unit),
                ("SID", sid),
                ("BG", bg),
                ("BANK", bank),
            ):
                self.assertEqual(c[name + "_SHIFT"], pos, f"{std} {name}")
                pos += width
            self.assertEqual(c["ROW_SHIFT"], pos, std)
            self.assertEqual(1 << (pos + row), lay.window_size, std)

    def test_am_4_9_figure(self):
        # AM section 4.9: LPDDR6 x 8, RoBaRaCoCh, 64 B interleave.
        lay = layout("lpddr6", policy="RoBaRaCoCh", interleave=64)
        self.assertEqual(
            lay.describe(),
            "row 33-18 | bank 17-16 | bg 15-14 | col[1+] 13-9 | ch 8-6 | "
            "col[0+] 5 | byte 4-0",
        )

    def test_every_bit_once(self):
        for std in g.STANDARDS:
            for policy in g.POLICIES:
                lay = layout(std, policy=policy)
                seen = {}
                for field, k in lay.bitmap:
                    seen.setdefault(field, set()).add(k)
                for field in ("col", "bg", "bank", "sid", "unit", "row"):
                    width = lay.width(field)
                    self.assertEqual(
                        seen.get(field, set()), set(range(width)), field
                    )
                self.assertEqual(seen["ch"], set(range(lay.ch_bits)))


class Rules(unittest.TestCase):
    def test_interleave_rules(self):
        # P4R section 6.1 and AM8 (b).
        for std in g.STANDARDS:
            lay = layout(std)
            self.assertEqual(lay.interleave, lay.row_bytes)
            self.assertEqual(
                layout(std, policy="ChRaBaRoCo").interleave,
                lay.channel_capacity,
            )
            for policy in ("RoRaBaCoCh", "RoCoRaBaCh", "RoBaRaCoCh"):
                self.assertEqual(layout(std, policy=policy).interleave, 64)
            with self.assertRaises(g.LayoutError):
                layout(std, interleave=64)
            with self.assertRaises(g.LayoutError):
                layout(std, policy="ChRaBaRoCo", interleave=64)
            for bad in (32, 96, 0):
                with self.assertRaises(g.LayoutError):
                    layout(std, policy="RoBaRaCoCh", interleave=bad)
            for policy in ("RoRaBaCoCh", "RoCoRaBaCh"):
                with self.assertRaises(g.LayoutError):
                    layout(std, policy=policy, interleave=2 * lay.row_bytes)
            # Ramulator 2.1's policies accept more than one row.
            self.assertEqual(
                layout(
                    std, policy="MOP4CLXOR", interleave=4 * lay.row_bytes
                ).interleave,
                4 * lay.row_bytes,
            )

    def test_base_rule(self):
        # P4R section 6.2: the first multiple of the size at or above the
        # end of the OS memory.
        for std in g.STANDARDS:
            for os_size in (None, 1 << 20, 3 * GiB, 100 * GiB):
                lay = layout(std, os_size=os_size)
                s = lay.window_size
                os_end_se = lay.os_size
                os_end_fs = 0x80000000 + lay.os_size
                for name, end in (
                    ("WINDOW_BASE_SE", os_end_se),
                    ("WINDOW_BASE_FS", os_end_fs),
                ):
                    base = lay.constants[name]
                    self.assertEqual(base % s, 0)
                    self.assertGreaterEqual(base, end)
                    self.assertLess(base - end, s)

    def test_contig(self):
        # DP-E (a), against the closed form.
        for std in g.STANDARDS:
            for policy in g.POLICIES:
                for channels in (1, 2, 8):
                    for intlv in (None, 64, 128, 256, 1024, 2048, 8192):
                        try:
                            lay = layout(
                                std,
                                channels=channels,
                                policy=policy,
                                interleave=intlv,
                            )
                        except g.LayoutError:
                            continue
                        self.assertEqual(
                            lay.contig_bytes,
                            expected_contig(lay),
                            f"{std} {policy} {channels} {intlv}",
                        )

    def test_registers_and_formats(self):
        lay = layout("lpddr6", num_regs=3)
        self.assertEqual(lay.constants["REG_ROW0"], 65536 - 4)
        with self.assertRaises(g.LayoutError):
            layout("lpddr6", num_regs=0)
        c = layout("ddr5", num_fmt="fp32", acc_fmt="fp32").constants
        self.assertEqual((c["ELEM_BYTES"], c["ELEMS_PER_COL"]), (4, 16))
        c = layout("lpddr6", num_fmt="int8", acc_fmt="int32").constants
        self.assertEqual((c["NUM_FMT"], c["ACC_FMT"]), (3, 5))
        for num, acc in (
            ("fp32", "bf16"),
            ("bf16", "int32"),
            ("int16", "fp32"),
        ):
            with self.assertRaises(g.LayoutError):
                layout("lpddr6", num_fmt=num, acc_fmt=acc)
        with self.assertRaises(g.LayoutError):
            layout("lpddr6", channels=3)


class Tooling(unittest.TestCase):
    def test_command_line_and_ext_untouched(self):
        root = g.GEM5_ROOT
        ext = os.path.join(root, "ext", "ramulator2")

        def status():
            return subprocess.run(
                ["git", "status", "--porcelain", "--ignored", ext],
                cwd=root,
                capture_output=True,
                text=True,
            ).stdout

        try:
            before = status()
        except OSError:
            self.skipTest("git not available")
        out = subprocess.run(
            [
                sys.executable,
                "-B",
                os.path.join(root, "util", "pim", "gen_pim_layout.py"),
                "--standard",
                "hbm4",
                "--policy",
                "MOP4CLXOR",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertIn("#define PIM_LAYOUT_POLICY 5", out)
        self.assertIn("--policy MOP4CLXOR --interleave 64", out)
        self.assertEqual(status(), before)
        bad = subprocess.run(
            [
                sys.executable,
                "-B",
                os.path.join(root, "util", "pim", "gen_pim_layout.py"),
                "--standard",
                "lpddr6",
                "--interleave",
                "64",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(bad.returncode, 2)
        self.assertIn("fixes the channel interleave", bad.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
