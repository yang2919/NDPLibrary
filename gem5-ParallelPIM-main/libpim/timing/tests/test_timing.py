#!/usr/bin/env python3
"""Tests of libpim/timing/pim_timing.py and the committed timing tables.

Run with ``python3 -B libpim/timing/tests/test_timing.py`` or through
``scons -C libpim build/native/test``.
"""

import os
import sys
import unittest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import pim_timing as pt  # noqa: E402

# Idle 64-byte reads measured in gem5 on Ramulator 2.1, refresh off, on
# the stock presets (src/python/gem5/components/memory/ramulator2.py,
# _PRESETS), in clocks from the first command: hit, miss, conflict.
E5N = {
    "DDR4": (
        (
            "DDR4",
            "DDR4_8Gb_x8",
            "DDR4_2400R",
            dict(rank=2, nCL=17, nRCD=17, nRP=17),
        ),
        (21, 38, 55),
    ),
    "LPDDR6": ("LPDDR6", (69, 117, 222)),
}

# A(k): the last per-bank ACT of one unit, clocks after the first command;
# Ramulator 2.1's flat order, and first-ready with bank groups alternating
# (for comparison). On the default devices HBM4 8Hi has 32 banks per unit
# (16Hi: 64, last ACT 492).
A_FLAT = {"DDR5": 344, "LPDDR6": 152, "GDDR7": 60, "HBM4": 236}
A_FIRST_READY = {"DDR5": 248, "LPDDR6": 152, "GDDR7": 60, "HBM4": 236}

# The earliest legal cycles on an idle device (clocks; HBM4 in ticks): T5
# is the same-bank column gap. T10 follows the default devices' refresh
# (DDR5_16Gb_x8 nRFC 708, GDDR7_32Gb_x8 nRFCab 473, derived from 16 Gb's
# tRFCab, HBM4_32Gb_8Hi nRFC 900), T12 HBM4's 32 banks per unit.
T5 = {"DDR5": 12, "LPDDR6": 10, "GDDR7": 4, "HBM4": 8}
DEVICE_6_1 = {
    "DDR5": (34, 34, 19, 78, 113, 33, 8, 707, 344),
    "LPDDR6": (48, 22, 14, 54, 70, 115, 10, 1014, 152),
    "GDDR7": (30, 19, 5, 61, 39, 29, 4, 472, 60),
    "HBM4": (65, 39, 25, 118, 109, 62, 8, 1798, 236),
}
DEVICE_KEYS = (
    "T4 ACTab->AB_RD",
    "T4 ACTab->AB_REGBK",
    "T7 AB_RD->PREab",
    "T7 ACTab->PREab",
    "T7 AB_REGBK->PREab",
    "T8 PREab->ACTab",
    "T8 ACTab->ACT",
    "T10 REFab->ACTab",
    "T12 last ACT",
)

# The commands of each device row, to read it from Ramulator 2.1's table.
DEVICE_PAIRS = {
    "T4 ACTab->AB_RD": ("ACTab", "PIM_AB_MAC"),
    "T4 ACTab->AB_REGBK": ("ACTab", "PIM_AB_REGBK"),
    "T5 AB->AB": ("PIM_AB_MAC", "PIM_AB_MAC"),
    "T7 AB_RD->PREab": ("PIM_AB_MAC", "PREab"),
    "T7 ACTab->PREab": ("ACTab", "PREab"),
    "T7 AB_REGBK->PREab": ("PIM_AB_REGBK", "PREab"),
    "T8 PREab->ACTab": ("PREab", "ACTab"),
    "T8 ACTab->ACT": ("ACTab", "ACT"),
    "T10 REFab->ACTab": ("REFab", "ACTab"),
}

# The request classes each PIM parameter times. Keys are
# (class, mode); a parameter outside this map times no request (it
# closes rows: nRAS_AB, nRP_AB, nRRD_AB, nAB2PRE_*).
AB_RD = ("ab_rd", "ab_ew")
ISSUED = ("ideal", "aim", "baseline")
USES = {
    "nRCD_AB": {(c, m) for c in AB_RD for m in ("ideal", "aim")},
    "nRCDW_AB": {("ab_wr", m) for m in ("ideal", "aim")},
    "nCCD_AB": {(c, m) for c in AB_RD for m in ISSUED},
    "nCCDW_AB": {("ab_wr", m) for m in ISSUED},
    "nEX_AB_RD": {(c, m) for c in AB_RD for m in ISSUED},
    "nEX_AB_WR": {("ab_wr", m) for m in ISSUED},
    "nSB_RD_DONE": {
        ("sb_rd", None),
        ("sb_rd2", None),
        ("ab_rd", "aespa"),
        ("ab_ew", "aespa"),
    },
    "nSB_WR_DONE": {("sb_wr", None), ("ab_wr", "aespa")},
    "nCL_GB": {("gb_rd", None)},
    "nCL_REG": {("reg_rd", None)},
    "nCWL_GB": {("gb_wr", None)},
    "nCWL_REG": {("reg_wr", None)},
    "nCCD_NB": {(c, None) for c in ("gb_rd", "gb_wr", "reg_rd", "reg_wr")},
}


def all_requests(t):
    """U of every (class, mode) at n = 1, 2 and a full row."""
    cols = t["geometry"]["cols_per_row"]
    out = {}
    for cls in pt.CLASSES:
        modes = pt.MODES if cls in pt.AB_CLASSES else (None,)
        for m in modes:
            out[(cls, m)] = tuple(
                pt.unloaded(t, cls, m, n) for n in (1, 2, cols)
            )
    return out


class CommittedTables(unittest.TestCase):
    def test_check(self):
        """--check: the committed tables equal the tree's descriptors."""
        self.assertEqual(pt.check(), 0)
        self.assertEqual(pt.main(["--check"]), 0)

    def test_load_equals_build(self):
        for std in pt.STANDARDS:
            self.assertEqual(pt.load(std), pt.build(std), std)

    def test_format(self):
        keys = (
            "format standard presets clock geometry command_cycles "
            "read_latency host host_overrides pim derived U_clocks"
        ).split()
        for std in pt.STANDARDS:
            t = pt.load(std)
            self.assertEqual(list(t), keys, std)
            self.assertEqual(t["format"], 1)
            self.assertEqual(len(t["pim"]), 18, std)
            for e in t["pim"].values():
                self.assertEqual(set(e), {"value", "rule", "source"})
            self.assertEqual(
                [k for k, _, _ in pt.U_ROWS], list(t["U_clocks"]), std
            )
        g = pt.load("GDDR7")["host_overrides"]
        self.assertEqual(
            {k: (v["value"], v["was"]) for k, v in g.items()},
            {
                "nRFCab": (473, None),
                "nRFCpb": (106, None),
                "nRFMab": (473, None),
                "nRFMpb": (106, None),
            },
        )

    def test_model_reads_the_table_alone(self):
        """U and the derived terms follow from the JSON, without Ramulator
        2.1 (for host tools)."""
        for std in pt.STANDARDS:
            t = pt.load(std)
            self.assertEqual(pt.u_table(t), t["U_clocks"], std)
            self.assertEqual(pt.derived(t), t["derived"], std)
            self.assertEqual(pt.with_values(t), t, std)

    def test_pim_defaults_are_the_descriptors(self):
        for std in pt.STANDARDS:
            t = pt.load(std)
            cls = pt._pim_class(std)
            for k, e in t["pim"].items():
                self.assertEqual(e["rule"], cls.pim_descriptor.defaults[k])
                self.assertEqual(
                    e["value"], cls._eval_expr(e["rule"], t["host"]), k
                )


class HostClosedForms(unittest.TestCase):
    def test_e5n(self):
        """The model's host part against the measured idle reads (each
        case one clock below the run: the controller's issue offset)."""
        for name, (src, want) in E5N.items():
            got = tuple(
                pt.host_read(src if name == "DDR4" else pt.load(src), case)
                for case in ("hit", "miss", "conflict")
            )
            self.assertEqual(got, want, name)


class ActSequence(unittest.TestCase):
    def test_flat(self):
        for std in pt.STANDARDS:
            seq = pt.act_sequence(pt.load(std))
            t = pt.load(std)
            self.assertEqual(len(seq), t["geometry"]["banks_per_unit"])
            self.assertEqual([b for b, _ in seq], list(range(len(seq))))
            self.assertEqual(seq[-1][1], A_FLAT[std], std)
            times = [a for _, a in seq]
            self.assertEqual(times, sorted(times))

    def test_first_ready(self):
        for std in pt.STANDARDS:
            last = pt.act_sequence(pt.load(std), "first-ready")[-1][1]
            self.assertEqual(last, A_FIRST_READY[std], std)


class DeviceRows(unittest.TestCase):
    def test_6_1(self):
        for std in pt.STANDARDS:
            rows = pt.device_rows(pt.load(std))
            got = tuple(rows[k] for k in DEVICE_KEYS)
            self.assertEqual(got, DEVICE_6_1[std], std)
            self.assertEqual(rows["T5 AB->AB"], T5[std], std)

    def test_against_ramulator_rows(self):
        """Each row equals the largest single-command row Ramulator 2.1's
        to_config() gives that command pair."""
        for std in pt.STANDARDS:
            cfg = config(std)
            cmds = pt._pim_class(std).commands
            two = "ACTab2" in cmds
            rows = pt.device_rows(pt.load(std))
            for key, (p, f) in DEVICE_PAIRS.items():
                if two:
                    p = {"ACTab": "ACTab2", "ACT": "ACT2"}.get(p, p)
                    f = {"ACTab": "ACTab2", "ACT": "ACT2"}.get(f, f)
                lat = [
                    e[3]
                    for e in cfg["timing_constraints"]
                    if cmds.index(p) in e[1]
                    and cmds.index(f) in e[2]
                    and (len(e) < 5 or (e[4] == 1 and not e[5]))
                ]
                self.assertEqual(rows[key], max(lat), (std, key))


def config(std):
    p = pt.preset(std)
    kw = dict(p["org_overrides"])
    kw.update({k: v[0] for k, v in pt.HOST_OVERRIDES[std].items()})
    cls = pt._pim_class(std)
    return cls(
        org_preset=p["org"], timing_preset=p["timing"], **kw
    ).to_config()


class ParameterUse(unittest.TestCase):
    def test_each_parameter_times_only_its_classes(self):
        """Raising one PIM parameter by 25% changes U of exactly the
        classes that use it."""
        for std in pt.STANDARDS:
            t = pt.load(std)
            base = all_requests(t)
            for k, e in t["pim"].items():
                up = -(-e["value"] * 5 // 4)
                if up == e["value"]:
                    up += 1
                got = all_requests(pt.with_values(t, pim={k: up}))
                changed = {key for key in base if got[key] != base[key]}
                self.assertEqual(changed, USES.get(k, set()), (std, k))
                for key in changed:
                    self.assertTrue(
                        all(a >= b for a, b in zip(got[key], base[key])),
                        (std, k, key),
                    )

    def test_pim_defaults_follow_the_host(self):
        """A raised host value reaches its PIM default, an explicit PIM
        value wins."""
        t = pt.load("LPDDR6")
        r = pt.with_values(t, host={"nRCDr": 60})
        self.assertEqual(r["pim"]["nRCD_AB"]["value"], 60)
        r = pt.with_values(t, host={"nRCDr": 60}, pim={"nRCD_AB": 40})
        self.assertEqual(r["pim"]["nRCD_AB"]["value"], 40)
        b = pt.build("LPDDR6", host={"nRCDr": 60})
        self.assertEqual(b["pim"]["nRCD_AB"]["value"], 60)
        self.assertEqual(b["U_clocks"], r_host(t, {"nRCDr": 60}))


def r_host(t, host):
    return pt.with_values(t, host=host)["U_clocks"]


# The U table (clocks from the first command; n = 1, 2, a full row). The
# AESPA rows follow Ramulator 2.1's rows and the PIM sequencer, as
# measured in gem5, same-bank-group column gaps included. On the default
# devices a DDR5_16Gb_x8 row has 64 columns (x4: 128) and nCCDL_WR 24 (x4:
# 48; nCCDM_WR stays 48, and binds only the other banks of a group, so
# AESPA's writes take 770 clocks for two columns, not x4's 854); HBM4 8Hi
# has 32 banks per unit (16Hi: 64).
U_5_4 = {
    "sb_rd": ("52/64/808", "64/74/694", "34/38/286", "45/49/169"),
    "sb_rd2": ("64/76/820", "74/84/704", "38/42/290", "49/53/173"),
    "sb_wr": ("74/98/1586", "56/66/686", "27/31/279", "32/36/156"),
    "ab_rd_ideal": ("52/64/808", "64/74/694", "34/38/286", "45/49/169"),
    "ab_wr_ideal": ("74/98/1586", "56/66/686", "27/31/279", "32/36/156"),
    "ab_rd_baseline": (
        "396/408/1152",
        "214/224/844",
        "94/98/346",
        "281/285/405",
    ),
    "ab_wr_baseline": (
        "418/442/1930",
        "206/216/836",
        "87/91/339",
        "268/272/392",
    ),
    "ab_rd_aespa": (
        "396/580/16452",
        "214/314/6286",
        "94/101/2084",
        "281/297/2153",
    ),
    "ab_ew_aespa": (
        "396/408/8272",
        "214/224/3216",
        "94/98/1068",
        "281/285/1133",
    ),
    "ab_wr_aespa": (
        "526/770/18746",
        "206/302/6298",
        "87/93/2077",
        "275/291/2142",
    ),
    "nb_ld": ("42/50/546", "64/70/442", "27/29/153", "34/36/96"),
    "nb_st": ("40/48/544", "34/40/412", "8/10/134", "12/14/74"),
}

# Ideal's AB row cycle and row switch (DDR5: 64 columns
# on DDR5_16Gb_x8; 1610 with x4's 128).
ROW_CYCLE = {
    "DDR5": (842, 86),
    "LPDDR6": (807, 177),
    "GDDR7": (316, 64),
    "HBM4": (201, 77),
}


class FinalValues(unittest.TestCase):
    def test_5_4(self):
        for i, std in enumerate(pt.STANDARDS):
            u = pt.load(std)["U_clocks"]
            for key, cells in U_5_4.items():
                got = "/".join(str(v) for v in u[key].values())
                self.assertEqual(got, cells[i], (std, key))

    def test_row_cycle(self):
        for std in pt.STANDARDS:
            self.assertEqual(pt.row_cycle(pt.load(std)), ROW_CYCLE[std], std)

    def test_sb_and_ab_differ_in_bank_scope_only(self):
        """An SB request of one bank and an AB request in Ideal or AiM of
        the same size take equal time."""
        for std in pt.STANDARDS:
            t = pt.load(std)
            for n in range(1, t["geometry"]["cols_per_row"] + 1):
                for mode in ("ideal", "aim"):
                    self.assertEqual(
                        pt.unloaded(t, "sb_rd", None, n),
                        pt.unloaded(t, "ab_rd", mode, n),
                    )
                    self.assertEqual(
                        pt.unloaded(t, "sb_wr", None, n),
                        pt.unloaded(t, "ab_wr", mode, n),
                    )

    def test_aespa_equals_baseline_at_one_column(self):
        """SB reads complete at tRTP, as AB reads: AESPA equals Baseline
        for reads on every standard; for writes on LPDDR6 and GDDR7 (DDR5,
        nCCDM_WR 48 on x4 and x8, and HBM4 space SB writes to the banks of
        one bank group, which one AB write does not meet)."""
        for std in pt.STANDARDS:
            u = pt.load(std)["U_clocks"]
            for cls in ("ab_rd", "ab_ew"):
                self.assertEqual(
                    u[cls + "_aespa"]["1"], u["ab_rd_baseline"]["1"], std
                )
            same = u["ab_wr_aespa"]["1"] == u["ab_wr_baseline"]["1"]
            self.assertEqual(same, std in ("LPDDR6", "GDDR7"), std)

    def test_gddr7_nb_load_read_path(self):
        """A GDDR7 non-bank load completes at the host read
        latency, one clock after nCL_GB + burst."""
        t = pt.load("GDDR7")
        self.assertEqual(t["derived"]["nb_rd_extra"], 1)
        self.assertEqual(
            t["U_clocks"]["nb_ld"]["1"], t["read_latency"]["value"]
        )
        for std in ("DDR5", "LPDDR6", "HBM4"):
            self.assertEqual(pt.load(std)["derived"]["nb_rd_extra"], 0)


class Model(unittest.TestCase):
    def test_one_column(self):
        """With one column the spacing terms do not enter."""
        for std in pt.STANDARDS:
            t = pt.load(std)
            d = t["derived"]
            p = {k: e["value"] for k, e in t["pim"].items()}
            u = t["U_clocks"]
            a0 = d["act_phase"]
            self.assertEqual(
                u["sb_rd"]["1"], a0 + d["rcd_rd"] + p["nSB_RD_DONE"], std
            )
            self.assertEqual(
                u["ab_rd_ideal"]["1"],
                a0 + d["rcd_ab"] + p["nEX_AB_RD"],
                std,
            )
            self.assertEqual(
                u["ab_rd_baseline"]["1"],
                A_FLAT[std] + d["rcd_rd"] + p["nEX_AB_RD"],
                std,
            )
            self.assertEqual(
                u["nb_st"]["1"],
                d["nb_sync"] + p["nCWL_GB"] + d["burst"],
                std,
            )

    def test_aim_equals_ideal(self):
        for std in pt.STANDARDS:
            t = pt.load(std)
            for cls in pt.AB_CLASSES:
                for n in (1, 2, t["geometry"]["cols_per_row"]):
                    self.assertEqual(
                        pt.unloaded(t, cls, "aim", n),
                        pt.unloaded(t, cls, "ideal", n),
                    )

    def test_pair_flag(self):
        t = pt.load("DDR5")
        self.assertEqual(
            pt.unloaded(t, "sb_rd", n=3, pair=True),
            pt.unloaded(t, "sb_rd2", n=3),
        )

    def test_row_cycle(self):
        for std in pt.STANDARDS:
            t = pt.load(std)
            cycle, switch = pt.row_cycle(t)
            self.assertGreater(cycle, t["U_clocks"]["ab_rd_ideal"]["1"])
            self.assertGreater(switch, 0)

    def test_errors(self):
        t = pt.load("LPDDR6")
        with self.assertRaises(pt.TimingError):
            pt.unloaded(t, "ab_rd", None, 1)
        with self.assertRaises(pt.TimingError):
            pt.unloaded(t, "sb_rd", None, 0)
        with self.assertRaises(pt.TimingError):
            pt.unloaded(t, "nope", None, 1)
        with self.assertRaises(pt.TimingError):
            pt.row_cycle(t, "baseline")


if __name__ == "__main__":
    unittest.main(verbosity=2)
