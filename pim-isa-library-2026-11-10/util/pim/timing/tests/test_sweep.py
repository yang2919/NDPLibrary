#!/usr/bin/env python3
"""Tests of util/pim/timing/sweep.py and the committed sensitivity report.

Run with ``python3 -B util/pim/timing/tests/test_sweep.py`` or through
``scons -C util/pim build/native/test``.
"""

import os
import sys
import unittest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import pim_timing as pt  # noqa: E402
import sweep  # noqa: E402

COL = {c[0]: i for i, c in enumerate(sweep.COLUMNS)}


class Report(unittest.TestCase):
    def test_check(self):
        """The committed report equals the committed tables' sweep."""
        self.assertEqual(sweep.main(["--check"]), 0)

    def test_base(self):
        """The base latencies are the tables' U."""
        for std in pt.STANDARDS:
            t = pt.load(std)
            cols = str(t["geometry"]["cols_per_row"])
            u = t["U_clocks"]
            _, base = sweep.sweep(std)
            self.assertEqual(
                base,
                [
                    u["ab_rd_baseline"][cols],
                    u["ab_rd_ideal"][cols],
                    u["ab_rd_baseline"]["1"],
                    u["sb_rd"][cols],
                    u["sb_wr"][cols],
                    u["nb_ld"]["1"],
                ],
                std,
            )

    def test_rows_left_out(self):
        text = sweep.report()
        for std in pt.STANDARDS:
            for name, v, up, pct in sweep.sweep(std)[0]:
                line = f"| {std} | `{name}` {v} → {up} |"
                shown = max(abs(p) for p in pct) >= sweep.THRESHOLD
                self.assertEqual(line in text, shown, (std, name))


class Sweep(unittest.TestCase):
    def test_monotone(self):
        """Raising a timing value never shortens a request."""
        for std in pt.STANDARDS:
            for name, v, up, pct in sweep.sweep(std)[0]:
                self.assertGreater(up, v)
                self.assertTrue(all(p >= 0 for p in pct), (std, name))

    def test_what_moves(self):
        """P5b section 8: SB rows follow the same-bank read gap, non-bank
        loads the read latency, Baseline's opening the ACT spacing; none
        of them moves the others."""
        for std in pt.STANDARDS:
            f = pt.FAMILY[std]
            rows = {n: pct for n, _, _, pct in sweep.sweep(std)[0]}
            s = rows[f["s_rd"]]
            self.assertGreater(s[COL["SB RD row"]], 15, std)
            self.assertEqual(s[COL["NB load 1"]], 0, std)
            cl = rows[f["cl"]]
            self.assertGreater(cl[COL["NB load 1"]], 15, std)
            self.assertEqual(cl[COL["AB row Ideal"]], 0, std)
            rrd = rows[f["rrd_s"]]
            self.assertGreater(rrd[COL["AB 1 Baseline"]], 0, std)
            self.assertEqual(rrd[COL["AB row Ideal"]], 0, std)
            self.assertEqual(rrd[COL["SB RD row"]], 0, std)

    def test_swept_names(self):
        for std in pt.STANDARDS:
            t = pt.load(std)
            names = sweep.swept(t)
            self.assertTrue(set(names) <= set(t["host"]))
            for term in ("rcd_rd", "s_rd", "burst", "cl"):
                self.assertIn(pt.FAMILY[std][term], names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
