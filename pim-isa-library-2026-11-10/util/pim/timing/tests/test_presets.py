#!/usr/bin/env python3
"""Tests of the PIM timing values of gem5's derived Ramulator 2.1 standards
(``ext/ramulator2/python/ramulator/dram/pim.py``, ``*_pim.py``) on the PIM
presets: P5b section 3.5's final values, P5b Q1 (a) (PIM defaults follow the
host values in use) and Q13 (b), and the stock standards unchanged.

Run with ``python3 -B util/pim/timing/tests/test_presets.py`` or through
``scons -C util/pim build/native/test``.
"""

import hashlib
import json
import os
import sys
import unittest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import pim_timing as pt  # noqa: E402

# P5b section 3.5 (clocks; HBM4 in clocks): DDR5, LPDDR6, GDDR7, HBM4.
FINAL_3_5 = {
    "nRCD_AB": (34, 48, 30, 32),
    "nRCDW_AB": (34, 22, 19, 19),
    "nRAS_AB": (77, 54, 60, 58),
    "nRP_AB": (34, 115, 30, 32),
    "nRRD_AB": (8, 10, 4, 4),
    "nCCD_AB": (12, 10, 4, 4),
    "nCCDW_AB": (48, 10, 4, 4),
    "nAB2PRE_RD": (18, 14, 4, 12),
    "nAB2PRE_WR": (112, 70, 38, 54),
    "nCL_GB": (34, 56, 24, 32),
    "nCL_REG": (34, 56, 24, 32),
    "nCWL_GB": (32, 26, 6, 10),
    "nCWL_REG": (32, 26, 6, 10),
    "nCCD_NB": (8, 6, 2, 2),
    "nEX_AB_RD": (18, 14, 4, 12),
    "nEX_AB_WR": (40, 32, 8, 12),
    "nSB_RD_DONE": (18, 14, 4, 12),
    "nSB_WR_DONE": (40, 32, 8, 12),
}

# SHA-256 of json.dumps(to_config(), sort_keys=True) of the stock standards
# on the PIM presets, taken before P5b.1 (P-DRAM T1: gem5's PIM work leaves
# them unchanged).
STOCK_CONFIG_SHA256 = {
    "DDR5": "27b96e704dfc7be36ab5bf9cb683cc858957be4d58695abce67633fa346dd4d0",
    "LPDDR6": "adfe82337e2a7183e4b40ea408268acf12c325f8ab181ec56a6651ba27058d6e",
    "GDDR7": "250a9ed00827447186b6e0fd631d31cd4e75c554329636a92b18c7228f4f541d",
    "HBM4": "0eaa4873b6bd86591fe5b8ab0ba51ccb9cfbe581520ddad333d7ad060fe5923b",
}


def make(std, pim=True, **overrides):
    p = pt.preset(std)
    kw = dict(p["org_overrides"])
    kw.update({k: v[0] for k, v in pt.HOST_OVERRIDES[std].items()})
    kw.update(overrides)
    dram = pt._ramulator_dram()
    cls = getattr(dram, std + "_PIM" if pim else std)
    return cls(org_preset=p["org"], timing_preset=p["timing"], **kw)


def gaps(cfg, commands):
    """The largest single-command row of every command pair (window 1, not
    sibling), in ticks."""
    out = {}
    for e in cfg["timing_constraints"]:
        if len(e) > 4 and (e[4] != 1 or e[5]):
            continue
        for p in e[1]:
            for f in e[2]:
                k = (commands[p], commands[f])
                out[k] = max(out.get(k, -1), e[3])
    return out


class Values(unittest.TestCase):
    def test_3_5(self):
        for i, std in enumerate(pt.STANDARDS):
            _, t = make(std).resolve()
            got = {k: t[k] for k in FINAL_3_5}
            want = {k: v[i] for k, v in FINAL_3_5.items()}
            self.assertEqual(got, want, std)

    def test_tables_hold_the_values(self):
        for i, std in enumerate(pt.STANDARDS):
            pim = pt.load(std)["pim"]
            for k, v in FINAL_3_5.items():
                self.assertEqual(pim[k]["value"], v[i], (std, k))

    def test_to_config(self):
        """Every preset configures, with the PIM values in ticks."""
        for i, std in enumerate(pt.STANDARDS):
            d = make(std)
            cfg = d.to_config()
            m = type(d).tick_multiplier
            names = list(type(d).timing_params)
            for k, v in FINAL_3_5.items():
                self.assertEqual(cfg["timing"][names.index(k)], v[i] * m)


class HostOverrides(unittest.TestCase):
    """Q1 (a): the PIM defaults are evaluated after the host overrides."""

    def test_override_reaches_the_pim_default(self):
        _, t = make("LPDDR6", nRCDr=60, nCCDL=13).resolve()
        self.assertEqual((t["nRCD_AB"], t["nCCD_AB"]), (60, 13))
        _, t = make("GDDR7", nRCDWR=20, nRRD=11).resolve()
        self.assertEqual((t["nRCDW_AB"], t["nRRD_AB"]), (20, 11))
        _, t = make("HBM4", nCWL=13).resolve()
        got = [t[k] for k in ("nCWL_GB", "nCWL_REG", "nAB2PRE_WR")]
        self.assertEqual(got, [13, 13, 57])
        self.assertEqual((t["nEX_AB_WR"], t["nSB_WR_DONE"]), (15, 15))
        _, t = make("DDR5", nCCDL_WR=60, nRTP=20).resolve()
        got = [t[k] for k in ("nCCDW_AB", "nSB_RD_DONE", "nEX_AB_RD")]
        self.assertEqual(got, [60, 20, 20])

    def test_explicit_pim_value_wins(self):
        _, t = make("LPDDR6", nRCDr=60, nRCD_AB=40).resolve()
        self.assertEqual((t["nRCDr"], t["nRCD_AB"]), (60, 40))
        _, t = make("DDR5", nCCD_AB=9).resolve()
        self.assertEqual(t["nCCD_AB"], 9)
        make("DDR5", nCCD_AB=9).to_config()

    def test_host_values_unchanged_by_the_pim_standard(self):
        """The derived standard resolves every host value as the stock one,
        overrides included."""
        for std in pt.STANDARDS:
            rtp = "nRTPSB" if std == "GDDR7" else "nRTP"
            for ovr in ({}, {rtp: 30}):
                _, a = make(std, **ovr).resolve()
                _, b = make(std, pim=False, **ovr).resolve()
                self.assertEqual({k: a[k] for k in b}, b, (std, ovr))

    def test_model_follows_the_same_rule(self):
        """pim_timing.build with host overrides gives the derived
        standard's PIM values."""
        _, t = make("HBM4", nCWL=13, nCCDL=5).resolve()
        b = pt.build("HBM4", host={"nCWL": 13, "nCCDL": 5})
        self.assertEqual(
            {k: e["value"] for k, e in b["pim"].items()},
            {k: t[k] for k in FINAL_3_5},
        )


class Stock(unittest.TestCase):
    def test_stock_to_config_unchanged(self):
        for std in pt.STANDARDS:
            cfg = make(std, pim=False).to_config()
            text = json.dumps(cfg, sort_keys=True)
            got = hashlib.sha256(text.encode()).hexdigest()
            self.assertEqual(got, STOCK_CONFIG_SHA256[std], std)


class Q13Rows(unittest.TestCase):
    """Q13 (b): an AB column meets the bank and bank-group rows of SB columns
    and of AB columns of the other class (ticks; P5b section 12, Q13)."""

    def test_values(self):
        want = {
            # (AB write -> SB read, AB read -> AB write, SB read -> AB read)
            "DDR5": (64, 14, 12),
            "LPDDR6": (70, 54, 10),
            "GDDR7": (19, 24, 4),
            "HBM4": (50, 60, 8),  # read to write: nRTW 30, the branch
        }
        for std, w in want.items():
            d = make(std)
            g = gaps(d.to_config(), type(d).commands)
            got = (
                g[("PIM_AB_REGBK", "PIM_SB_RD")],
                g[("PIM_AB_MAC", "PIM_AB_REGBK")],
                g[("PIM_SB_RD", "PIM_AB_MAC")],
            )
            self.assertEqual(got, w, std)

    def test_ab_like_sb(self):
        """Each pair equals the same pair of SB columns of one bank."""
        sb = {"PIM_AB_MAC": "PIM_SB_RD", "PIM_AB_REGBK": "PIM_SB_WR"}
        for std in pt.STANDARDS:
            d = make(std)
            g = gaps(d.to_config(), type(d).commands)
            for p in ("PIM_AB_MAC", "PIM_AB_REGBK", "PIM_SB_RD", "PIM_SB_WR"):
                for f in ("PIM_SB_RD", "PIM_SB_WR"):
                    self.assertEqual(
                        g[(p, f)], g[(sb.get(p, p), f)], (std, p, f)
                    )
                    self.assertEqual(
                        g[(f, p)], g[(f, sb.get(p, p))], (std, f, p)
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
