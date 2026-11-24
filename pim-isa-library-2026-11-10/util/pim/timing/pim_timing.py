#!/usr/bin/env python3
"""Processing-in-Memory (PIM) timing tables and unloaded-latency model.

The PIM timing values of the four PIM standards (DDR5, LPDDR6, GDDR7,
HBM4) are defined by gem5's derived Ramulator 2.1 standards
(``ext/ramulator2/python/ramulator/dram/pim.py``, the per-standard
descriptors, and ``*_pim.py``). This tool resolves them through the
Ramulator 2.1 Python package embedded in ``ext/ramulator2/python`` (as
``util/pim/gen_pim_layout.py`` does; nothing is written there, no
bytecode) and writes one committed table per standard,
``util/pim/timing/timing_<std>.json`` (docs/pim/05b-architecture-timing-
plan.md, "P5b", section 4.3), for host tools that cannot import gem5 or
Ramulator 2.1, such as a compiler cost model.

It also holds P5b section 5's analytic model of the unloaded latency U of
one PIM request: from the clock of the request's first command to its
Done, on an idle unit with every bank closed, without refresh, host
traffic or gem5's static latencies. gem5's own U (``PimUnloaded``,
``src/mem/ramulator2/pim/pim_unloaded.cc``) replays the request's commands
against the timing table; this model is its independent cross-check, as
closed forms over named parameters. gem5's U adds the issue term (the
clocks from acceptance to the first command; ``derived.issue_offset``).

Values are in DRAM clocks. HBM4 counts half-clock ticks internally
(``tick_multiplier`` 2); the results are converted back to clocks.

Examples::

    util/pim/timing/pim_timing.py --write        # the four tables
    util/pim/timing/pim_timing.py --check        # are they up to date?
    util/pim/timing/pim_timing.py --show lpddr6  # U per class and size
"""

import argparse
import copy
import json
import os
import sys

sys.dont_write_bytecode = True

TIMING_DIR = os.path.dirname(os.path.abspath(__file__))
PIM_ROOT = os.path.dirname(TIMING_DIR)
if PIM_ROOT not in sys.path:
    sys.path.insert(0, PIM_ROOT)

import gen_pim_layout as layout  # noqa: E402

FORMAT = 1
STANDARDS = ("DDR5", "LPDDR6", "GDDR7", "HBM4")
MODES = ("ideal", "aim", "baseline", "aespa")
ORDERS = ("flat", "first-ready")

# Q9 (b) (P5b section 12): a non-bank load completes after the host read
# path, nCL_x + burst + (read_latency - host CAS latency - burst), as the
# PIM sequencer does (ext/ramulator2/src/ramulator/pim/pim_sequencer.cpp,
# m_done_lat): one clock more on GDDR7, nothing on the others.
NB_LOAD_READ_PATH = True

# Where each PIM parameter's default rule was decided (P5b section 3.5).
SOURCES = {
    "nRCD_AB": "D8.3",
    "nRCDW_AB": "D8.3",
    "nRAS_AB": "D8.3",
    "nRP_AB": "D8.3",
    "nRRD_AB": "D8.3; P-DRAM Q15 (a)",
    "nCCD_AB": "Q12 (b)",
    "nCCDW_AB": "Q12 (b)",
    "nAB2PRE_RD": "D8.3",
    "nAB2PRE_WR": "D8.3",
    "nCL_GB": "P-DRAM Q20 (a)",
    "nCL_REG": "P-DRAM Q20 (a)",
    "nCWL_GB": "P-DRAM Q20 (a)",
    "nCWL_REG": "P-DRAM Q20 (a)",
    "nCCD_NB": "P-DRAM Q20 (a)",
    "nEX_AB_RD": "DP-H (b)",
    "nEX_AB_WR": "Q11 (c)",
    "nSB_RD_DONE": "Q6 (b)",
    "nSB_WR_DONE": "P4R section 4.5",
}


def _descriptors():
    layout._ramulator_dram()
    from ramulator.dram.pim import DESCRIPTORS

    return DESCRIPTORS


# Host overrides of the PIM presets, name: (value, reason): values Ramulator
# 2.1 does not give (amended P-DRAM Q8 (c), decisions.md D11.1), from the
# per-standard descriptors (P5b Q4 (b)), as gem5's
# src/python/gem5/components/memory/pim.py (PIM_PRESETS) applies them.
HOST_OVERRIDES = {
    std: dict(_descriptors()[std].host_overrides) for std in STANDARDS
}

# The host parameters behind each model term, per standard family (P5b
# section 5.2): ACT to RD and WR, the same-bank read and write column
# gaps, the different-group column gap, the burst, the ACT spacings in a
# different and the same bank group, the four-activate window, the
# precharge time, the CAS (or read) latency, the clock-start command
# before a non-bank column (LPDDR6's write clock) and the level whose
# banks share a bank-group ACT spacing.
FAMILY = {
    "DDR4": dict(
        rcd_rd="nRCD",
        rcd_wr="nRCD",
        s_rd="nCCDL",
        s_wr="nCCDL",
        sg_rd="nCCDL",
        sg_wr="nCCDL",
        ccd_s="nCCDS",
        burst="nBL",
        rrd_s="nRRDS",
        rrd_l="nRRDL",
        faw="nFAW",
        rp="nRP",
        rfc="nRFC",
        cl="nCL",
        sync=None,
        unit="Rank",
    ),
    "DDR5": dict(
        rcd_rd="nRCD",
        rcd_wr="nRCD",
        s_rd="nCCDL",
        s_wr="nCCDL_WR",
        sg_rd="nCCDM",  # a sibling row at Bank: the bank group's others
        sg_wr="nCCDM_WR",
        ccd_s="nCCDS",
        burst="nBL",
        rrd_s="nRRDS",
        rrd_l="nRRDL",
        faw="nFAW",
        rp="nRP",
        rfc="nRFC",
        cl="nCL",
        sync=None,
        unit="Rank",
    ),
    "LPDDR6": dict(
        rcd_rd="nRCDr",
        rcd_wr="nRCDw",
        s_rd="nCCDL",
        s_wr="nCCDL_WR",
        sg_rd="nCCDL",
        sg_wr="nCCDL_WR",
        ccd_s="nCCDS",
        burst="nBL_min",
        rrd_s="nRRD",
        rrd_l="nRRD",
        faw="nFAW",
        rp="nRP",
        rfc="nRFC",
        cl="nRL",
        sync="CAS",
        unit="Rank",
    ),
    "GDDR7": dict(
        rcd_rd="nRCDRD",
        rcd_wr="nRCDWR",
        s_rd="nCCDSB",
        s_wr="nCCDSB",
        sg_rd=None,  # no bank groups
        sg_wr=None,
        ccd_s="nCCD",
        burst="nBL",
        rrd_s="nRRD",
        rrd_l="nRRD",
        faw=None,
        rp="nRP",
        rfc="nRFCab",
        cl="nRL",
        sync=None,  # RCK always on (gddr7_controller.cpp, default mode)
        unit="Channel",
    ),
    "HBM4": dict(
        rcd_rd="nRCDRD",
        rcd_wr="nRCDWR",
        s_rd="nCCDL",
        s_wr="nCCDL",
        sg_rd="nCCDL",
        sg_wr="nCCDL",
        ccd_s="nCCDS",
        burst="nBL",
        rrd_s="nRRDS",
        rrd_l="nRRDL",
        faw="nFAW",
        rp="nRP",
        rfc="nRFC",
        cl="nCL",
        sync=None,
        unit="PseudoChannel",
    ),
}

# Request classes of the model (P5b section 5.3) and the gem5 PIM
# operations of each (P4R section 4.2; D8.2's class map).
CLASSES = (
    "sb_rd",  # PIM_SB_RD: one bank
    "sb_rd2",  # PIM_SB_RD2: a bank pair (SB element-wise)
    "sb_wr",  # PIM_SB_WR
    "ab_rd",  # PIM_AB_MAC, PIM_AB_BKREG; PIM_AB_EW outside AESPA
    "ab_ew",  # PIM_AB_EW: under AESPA one SB pair command per bank pair
    "ab_wr",  # PIM_AB_REGBK
    "gb_rd",
    "gb_wr",
    "reg_rd",
    "reg_wr",
)
AB_CLASSES = ("ab_rd", "ab_ew", "ab_wr")
OP_CLASS = {
    "SbMac": "sb_rd",
    "SbEwaddGbuf": "sb_rd",
    "SbEwmulGbuf": "sb_rd",
    "SbMovBkgbuf": "sb_rd",
    "SbEwadd": "sb_rd2",
    "SbEwmul": "sb_rd2",
    "SbMovGbufbk": "sb_wr",
    "AbEwadd": "ab_ew",
    "AbEwmul": "ab_ew",
    "AbEwaddGbuf": "ab_ew",
    "AbEwmulGbuf": "ab_ew",
    "AbMac": "ab_rd",
    "AbMovBkreg": "ab_rd",
    "AbMovRegbk": "ab_wr",
    "RdGb": "gb_rd",
    "WrGb": "gb_wr",
    "RdReg": "reg_rd",
    "WrReg": "reg_wr",
}

# The rows of P5b section 5.4's table: (key, class, mode).
U_ROWS = (
    ("sb_rd", "sb_rd", None),
    ("sb_rd2", "sb_rd2", None),
    ("sb_wr", "sb_wr", None),
    ("ab_rd_ideal", "ab_rd", "ideal"),
    ("ab_wr_ideal", "ab_wr", "ideal"),
    ("ab_rd_baseline", "ab_rd", "baseline"),
    ("ab_wr_baseline", "ab_wr", "baseline"),
    ("ab_rd_aespa", "ab_rd", "aespa"),
    ("ab_ew_aespa", "ab_ew", "aespa"),
    ("ab_wr_aespa", "ab_wr", "aespa"),
    ("nb_ld", "gb_rd", None),
    ("nb_st", "gb_wr", None),
)


class TimingError(ValueError):
    """A request or configuration the model does not cover."""


def _ramulator_dram():
    return layout._ramulator_dram()


def preset(std):
    """The PIM preset of std: organization and timing preset names,
    organization overrides and the default channel count."""
    p = _descriptors()[std].presets
    return dict(
        org=p["org"],
        timing=p["timing"],
        org_overrides=dict(p["org_overrides"]),
        channels=p["channels"],
    )


def _pim_class(std):
    return getattr(_ramulator_dram(), std + "_PIM")


def resolve_values(std, host=None, pim=None, base=None):
    """The resolved host and PIM values of std, in clocks.

    host: host timing overrides (on top of the preset's host overrides);
    every PIM parameter not given in pim then follows its default rule on
    the host values in use (P5b Q1 (a)). pim: explicit PIM values. base:
    a stock standard (for example "DDR4") with (org, timing, overrides),
    for the host-only closed forms.

    Returns (cls, org, host_values, pim_values).
    """
    dram = _ramulator_dram()
    if base is not None:
        name, org_preset, timing_preset, ovr = base
        cls = getattr(dram, name)
        kw = dict(ovr)
        kw.update(host or {})
        org, tim = cls(
            org_preset=org_preset, timing_preset=timing_preset, **kw
        ).resolve()
        return cls, org, tim, {}
    p = preset(std)
    cls = _pim_class(std)
    desc = cls.pim_descriptor
    kw = dict(p["org_overrides"])
    kw.update({k: v[0] for k, v in HOST_OVERRIDES[std].items()})
    kw.update(host or {})
    kw.update(pim or {})
    org, tim = cls(
        org_preset=p["org"], timing_preset=p["timing"], **kw
    ).resolve()
    names = list(desc.defaults)
    hostv = {k: v for k, v in tim.items() if k not in names}
    pimv = {}
    for k in names:
        if pim and k in pim:
            pimv[k] = pim[k]
        else:
            pimv[k] = cls._eval_expr(desc.defaults[k], hostv)
    return cls, org, hostv, pimv


def _geometry(std, cls, org):
    levels = list(cls.levels)
    unit = FAMILY[std]["unit"]
    prefetch = cls.internal_prefetch_size
    col_bytes = cls.data_payload_bytes or (
        prefetch * org["channel_width"] // 8
    )
    below = levels[levels.index(unit) + 1 : levels.index("Bank")]
    groups = 1
    for lv in below:
        groups *= org[lv.lower()]
    return dict(
        unit=unit,
        units_per_channel=1 if unit == "Channel" else org[unit.lower()],
        banks_per_unit=groups * org["bank"],
        bank_groups=groups,
        cols_per_row=org["column"] // prefetch,
        col_bytes=col_bytes,
    )


def _cycles(cls):
    """Command cycles in clocks (Ramulator 2.1 counts 1 when not given)."""
    cc = cls.command_cycles
    two = "ACT1" in cls.commands
    act = "ACT2" if two else "ACT"
    rd = cls.pim_descriptor.host_rd if hasattr(cls, "pim_descriptor") else "RD"
    out = {
        "ACT": cc.get(act, 1),
        "act_phases": 2 if two else 1,
        "column": cc.get(rd, 1),
        "PREpb": cc.get("PREpb", cc.get("PRE", 1)),
        "PREab": cc.get("PREab", cc.get("PREA", 1)),
        "REFab": cc.get("REFab", 1),
    }
    if "ACTab" in cls.commands or "ACTab2" in cls.commands:
        out["ACTab"] = cc.get("ACTab2" if two else "ACTab", 1)
    if "CAS" in cls.commands:
        out["CAS"] = cc.get("CAS", 1)
    return out


def _num(x):
    return int(x) if float(x).is_integer() else x


def _host_family(cls):
    for name in FAMILY:
        if cls.name in (name, name + "_PIM"):
            return name
    raise TimingError(f"no model terms for {cls.name}")


def build(std, host=None, pim=None):
    """The table of std (the content of timing_<std>.json), resolved from
    the tree's descriptor; host and pim are overrides (see
    resolve_values)."""
    cls, org, hostv, pimv = resolve_values(std, host, pim)
    desc = cls.pim_descriptor
    stock = getattr(_ramulator_dram(), std)
    p = preset(std)
    _, stock_t = stock(
        org_preset=p["org"], timing_preset=p["timing"], **p["org_overrides"]
    ).resolve()
    tab = {
        "format": FORMAT,
        "standard": std,
        "presets": p,
        "clock": {
            "tCK_ps": hostv["tCK_ps"],
            "tick_multiplier": cls.tick_multiplier,
        },
        "geometry": _geometry(std, cls, org),
        "command_cycles": _cycles(cls),
        "read_latency": {
            "rule": cls.read_latency,
            "value": cls._eval_expr(cls.read_latency, hostv),
        },
        "host": dict(hostv),
        "host_overrides": {
            k: {
                "value": v,
                "was": None if stock_t.get(k, -1) == -1 else stock_t[k],
                "comment": c,
            }
            for k, (v, c) in HOST_OVERRIDES[std].items()
        },
        "pim": {
            k: {
                "value": pimv[k],
                "rule": desc.defaults[k],
                "source": SOURCES[k],
            }
            for k in desc.defaults
        },
    }
    tab["derived"] = derived(tab)
    tab["U_clocks"] = u_table(tab)
    return tab


def table_path(std):
    return os.path.join(TIMING_DIR, f"timing_{std.lower()}.json")


def load(std):
    """The committed table of std."""
    with open(table_path(std)) as f:
        return json.load(f)


def _table(t):
    return load(t) if isinstance(t, str) else t


def with_values(t, host=None, pim=None):
    """A copy of table t with host and PIM values changed in the model only
    (no Ramulator 2.1 import): PIM values not given follow their rule on
    the new host values (P5b Q1 (a))."""
    t = copy.deepcopy(_table(t))
    t["host"].update(host or {})
    names = dict(t["host"])
    for k, e in t["pim"].items():
        if pim and k in pim:
            e["value"] = pim[k]
        else:
            e["value"] = eval(e["rule"], {"__builtins__": {}}, names)
    rl = t["read_latency"]
    rl["value"] = eval(rl["rule"], {"__builtins__": {}}, names)
    t["derived"] = derived(t)
    t["U_clocks"] = u_table(t)
    return t


# ----------------------------------------------------------------------
# The model, in ticks (clocks x tick_multiplier).


class _Terms:
    """P5b section 5.2's per-standard terms of table t, in ticks."""

    def __init__(self, t):
        std = t["standard"]
        f = FAMILY[std]
        h = t["host"]
        self.std = std
        m = self.m = t["clock"]["tick_multiplier"]
        self.pim = {k: e["value"] * m for k, e in t["pim"].items()}
        cc = t["command_cycles"]
        cyc = lambda c: int(c * m)  # noqa: E731
        self.col = cyc(cc["column"])
        self.act = cyc(cc["ACT"])
        self.actab = cyc(cc.get("ACTab", cc["ACT"]))
        self.phases = cc["act_phases"]
        # HBM4: every command but PRE on a rising edge (odd tick); clock
        # 0 is one (hbm34_controller.cpp).
        self.edges = std == "HBM4"
        # One PIM command per controller cycle (P4R section 4.10): a
        # command occupies the bus for its cycles.
        self.slot = self.col
        # LPDDR6: rows count from the last ACT phase; the first phase is
        # one command earlier.
        self.a0 = self.act * (self.phases - 1)
        self.act_slot = self.act * self.phases
        hv = lambda n: h[n] * m  # noqa: E731
        # ACT to column as the device's ready clock, then after the edge.
        self.rcd_rd_dev = self.adj(hv(f["rcd_rd"]), self.act, self.col)
        self.rcd_wr_dev = self.adj(hv(f["rcd_wr"]), self.act, self.col)
        self.rcd_rd = self.edge(self.rcd_rd_dev)
        self.rcd_wr = self.edge(self.rcd_wr_dev)
        self.rcd_ab = self.edge(
            self.adj(self.pim["nRCD_AB"], self.actab, self.col)
        )
        self.rcdw_ab = self.edge(
            self.adj(self.pim["nRCDW_AB"], self.actab, self.col)
        )
        self.s_rd = hv(f["s_rd"])
        self.s_wr = hv(f["s_wr"])
        self.sg_rd = hv(f["sg_rd"]) if f["sg_rd"] else 0
        self.sg_wr = hv(f["sg_wr"]) if f["sg_wr"] else 0
        # GDDR7 and HBM4 have a row and a column command bus, the others
        # one (spec.py, _generate_bus_constraints).
        self.dual_bus = std in ("GDDR7", "HBM4")
        self.aad = h.get("nAAD", 0) * m
        self.burst = hv(f["burst"])
        # The PIM branch (P-DRAM Q28 (a), rule 0b): one burst per column.
        self.b = max(self.burst, hv(f["ccd_s"]))
        self.rrd_s = hv(f["rrd_s"])
        self.rrd_l = hv(f["rrd_l"])
        self.faw = hv(f["faw"]) if f["faw"] else 0
        self.nb0 = int(cc["CAS"] * m) if f["sync"] else 0
        self.c_nb = max(self.pim["nCCD_NB"], self.burst, hv(f["ccd_s"]))
        self.c_nb = max(self.c_nb, self.slot)
        rl = t["read_latency"]["value"] * m
        self.nb_rd_extra = (
            rl - hv(f["cl"]) - self.burst if NB_LOAD_READ_PATH else 0
        )
        geo = t["geometry"]
        self.B = geo["banks_per_unit"]
        self.groups = geo["bank_groups"]
        self.cols = geo["cols_per_row"]

    @staticmethod
    def adj(nominal, pre_cycles, fol_cycles):
        """Ramulator 2.1 records a command at its first cycle
        (ext/ramulator2/python/ramulator/dram/spec.py, to_config)."""
        return nominal + (pre_cycles - 1) - (fol_cycles - 1)

    def edge(self, x):
        if self.edges and x % self.m:
            return x + self.m - x % self.m
        return x

    def sp(self, x):
        """A PIM column spacing: the row's gap, the branch and the slot."""
        return max(x, self.b, self.slot)


def _acts(T, order):
    """Last-phase ACT clock of each bank of a unit when nothing else is
    issued (Baseline; P5b section 5.3's A(k)), as (bank, tick) in issue
    order."""
    B, G = T.B, T.groups
    per = B // G
    if order == "flat":
        seq = list(range(B))
    elif order == "first-ready":
        seq = [g * per + k for k in range(per) for g in range(G)]
    else:
        raise TimingError(f"unknown order {order}")
    times, last = [], {}
    for i, bank in enumerate(seq):
        g = bank // per
        if i == 0:
            a = T.a0
        else:
            a = times[-1] + max(T.rrd_s, T.act_slot)
        if g in last:
            a = max(a, last[g] + T.rrd_l)
        if T.faw and i >= 4:
            a = max(a, times[i - 4] + T.faw)
        a = T.edge(a)
        times.append(a)
        last[g] = a
    return list(zip(seq, times))


def act_sequence(t, order="flat"):
    """P5b section 5.3's A(k): the clock of each per-bank ACT (its last
    phase on LPDDR6) of one unit, from the request's first command, in
    Ramulator 2.1's flat bank order (Q8 (a), as the PIM sequencer issues
    them) or first-ready with bank groups alternating (for the record).
    Returns [(bank, clock)]."""
    T = _Terms(_table(t))
    return [(b, _num(a / T.m)) for b, a in _acts(T, order)]


def _aespa(T, cls, n, order):
    """AESPA's walk (P5b section 5.3), cycle by cycle as the PIM sequencer
    issues it: a pending second ACT phase first (LPDDR6); else the first
    legal SB column of an open bank (bank pair for AB EW), banks in
    activation order; else the next bank's ACT, banks in flat order (Q8
    (a)). Done is the last column plus its SB completion. Ticks."""
    if order != "flat":
        raise TimingError("AESPA is modelled in flat bank order (Q8 (a))")
    wr = cls == "ab_wr"
    # Legality is checked at the device's ready clocks (on HBM4 a falling
    # edge may choose a command that issues on the next rising edge).
    rcd = T.rcd_wr_dev if wr else T.rcd_rd_dev
    s = T.s_wr if wr else T.s_rd
    sg = T.sg_wr if wr else T.sg_rd
    per = T.B // T.groups
    if cls == "ab_ew":
        units = [(2 * k, 2 * k + 1) for k in range(T.B // 2)]
    else:
        units = [(b,) for b in range(T.B)]
    opened = {}  # bank: its (last-phase) ACT
    acts = []
    grp_act = {}
    col_bank, col_grp = {}, {}
    last_col = None
    row_bus = col_bus = 0  # the clock each bus is free again
    pend = None  # LPDDR6: the bank whose ACTsb1 was issued
    order_open = []
    left = [n] * len(units)
    nxt_bank = 0
    done = None

    def act_rows_ok(b, y):
        g = b // per
        if acts and y < acts[-1] + T.rrd_s:
            return False
        if g in grp_act and y < grp_act[g] + T.rrd_l:
            return False
        return not (T.faw and len(acts) >= 4 and y < acts[-4] + T.faw)

    def col_ok(k, x):
        if x < col_bus or (last_col is not None and x < last_col + T.b):
            return False
        for b in units[k]:
            if x < opened[b] + rcd or x < col_bank.get(b, -(10**9)) + s:
                return False
            if x < col_grp.get(b // per, -(10**9)) + sg:
                return False
        return True

    def note_act(b, y):
        nonlocal row_bus, col_bus, order_open
        opened[b] = y
        acts.append(y)
        grp_act[b // per] = y
        row_bus = y + T.act
        if not T.dual_bus:
            col_bus = row_bus
        order_open += [
            k
            for k, u in enumerate(units)
            if k not in order_open and all(c in opened for c in u)
        ]

    # The sequencer chooses one command per controller clock, legal at that
    # clock, and the controller issues it; on HBM4 a command chosen on a
    # falling edge waits for the next rising edge, and nothing else is
    # chosen meanwhile.
    x = 0
    while any(left):
        y = x + (x % T.m if T.edges else 0)  # issue clock
        if pend is not None:
            if x >= row_bus and act_rows_ok(pend, x):
                b, pend = pend, None
                note_act(b, y)
                x = y
            x += 1
            continue
        for k in order_open:
            if left[k] and col_ok(k, x):
                for b in units[k]:
                    col_bank[b] = y
                    col_grp[b // per] = y
                last_col = done = y
                col_bus = y + T.col
                if not T.dual_bus:
                    row_bus = col_bus
                left[k] -= 1
                x = y
                break
        else:
            b = nxt_bank
            if b < T.B and x >= row_bus:
                if T.phases == 2:
                    # ACTsb1 only when ACTsb2 is legal within nAAD.
                    if act_rows_ok(b, x + T.aad):
                        pend = b
                        nxt_bank += 1
                        row_bus = y + T.act
                        col_bus = max(col_bus, row_bus)
                        x = y
                elif act_rows_ok(b, x):
                    nxt_bank += 1
                    note_act(b, y)
                    x = y
        x += 1
    return done + (T.pim["nSB_WR_DONE"] if wr else T.pim["nSB_RD_DONE"])


def _ticks(T, cls, mode, n, order):
    if n < 1:
        raise TimingError("a request has at least one column")
    p = T.pim
    if cls == "sb_rd":
        return T.a0 + T.rcd_rd + (n - 1) * T.sp(T.s_rd) + p["nSB_RD_DONE"]
    if cls == "sb_rd2":
        # The pair's second ACT, in the same bank group.
        a1 = T.edge(T.a0 + max(T.rrd_l, T.rrd_s, T.act_slot))
        return a1 + T.rcd_rd + (n - 1) * T.sp(T.s_rd) + p["nSB_RD_DONE"]
    if cls == "sb_wr":
        return T.a0 + T.rcd_wr + (n - 1) * T.sp(T.s_wr) + p["nSB_WR_DONE"]
    if cls in AB_CLASSES:
        if mode not in MODES:
            raise TimingError(f"an AB request needs a mode of {MODES}")
        wr = cls == "ab_wr"
        c = T.sp(p["nCCDW_AB"] if wr else p["nCCD_AB"])
        ex = p["nEX_AB_WR"] if wr else p["nEX_AB_RD"]
        if mode in ("ideal", "aim"):
            first = T.a0 + (T.rcdw_ab if wr else T.rcd_ab)
            return first + (n - 1) * c + ex
        if mode == "baseline":
            last = _acts(T, order)[-1][1]
            first = last + (T.rcd_wr if wr else T.rcd_rd)
            return first + (n - 1) * c + ex
        return _aespa(T, cls, n, order)
    if cls in ("gb_rd", "reg_rd"):
        cl = p["nCL_GB" if cls == "gb_rd" else "nCL_REG"]
        return T.nb0 + (n - 1) * T.c_nb + cl + T.burst + T.nb_rd_extra
    if cls in ("gb_wr", "reg_wr"):
        cwl = p["nCWL_GB" if cls == "gb_wr" else "nCWL_REG"]
        return T.nb0 + (n - 1) * T.c_nb + cwl + T.burst
    raise TimingError(f"unknown class {cls}; choose one of {CLASSES}")


def unloaded(t, cls, mode=None, n=1, pair=False, order="flat"):
    """U of one request (P5b section 5.3) in clocks from its first command:
    cls one of CLASSES (pair=True turns sb_rd into sb_rd2), mode one of
    MODES for AB classes, n columns. gem5's U adds
    derived["issue_offset"]."""
    if pair and cls == "sb_rd":
        cls = "sb_rd2"
    T = _Terms(_table(t))
    return _num(_ticks(T, cls, mode, n, order) / T.m)


def aespa(t, cls, n, order="flat"):
    """U of an AB request of class cls and n columns under AESPA."""
    return unloaded(t, cls, "aespa", n, order=order)


def per_column(t, cls, mode=None):
    """Clocks one more column adds to a request of class cls (for AESPA the
    mean over a full row)."""
    t = _table(t)
    if mode == "aespa":
        cols = t["geometry"]["cols_per_row"]
        u1 = unloaded(t, cls, mode, 1)
        return _num((unloaded(t, cls, mode, cols) - u1) / (cols - 1))
    return _num(unloaded(t, cls, mode, 2) - unloaded(t, cls, mode, 1))


def row_cycle(t, mode="ideal", n=None):
    """Ideal's AB row cycle (P-COMP section 7.2's T_row): from an ACTab
    (its last phase), n RD-class AB columns (default a full row), PREab
    and the next ACTab. Returns (cycle, switch) in clocks; switch is the
    last column to the next row's first column."""
    if mode != "ideal":
        raise TimingError("row_cycle models Ideal (ACTab, PREab) only")
    t = _table(t)
    T = _Terms(t)
    n = n or T.cols
    cc = t["command_cycles"]
    preab = int(cc["PREab"] * T.m)
    col0 = T.rcd_ab
    last = col0 + (n - 1) * T.sp(T.pim["nCCD_AB"])
    pre = max(
        last + T.adj(T.pim["nAB2PRE_RD"], T.col, preab),
        T.adj(T.pim["nRAS_AB"], T.actab, preab),
    )
    act = T.edge(pre + T.adj(T.pim["nRP_AB"], preab, T.actab))
    return _num(act / T.m), _num((act + col0 - last) / T.m)


def device_rows(t):
    """P5b section 6.1's earliest legal cycles on an idle device, after
    Ramulator 2.1's command-cycle adjustment: in clocks, on HBM4 in ticks
    (T12, the last per-bank ACT, in clocks on every standard)."""
    t = _table(t)
    T = _Terms(t)
    cc = t["command_cycles"]
    preab = int(cc["PREab"] * T.m)
    refab = int(cc["REFab"] * T.m)
    p, adj = T.pim, T.adj
    rfc = t["host"][FAMILY[t["standard"]]["rfc"]] * T.m
    return {
        "T4 ACTab->AB_RD": adj(p["nRCD_AB"], T.actab, T.col),
        "T4 ACTab->AB_REGBK": adj(p["nRCDW_AB"], T.actab, T.col),
        # AB to AB: nCCD_AB, with the PIM branch as its floor (rule 0b).
        "T5 AB->AB": max(adj(p["nCCD_AB"], T.col, T.col), T.b),
        "T7 AB_RD->PREab": adj(p["nAB2PRE_RD"], T.col, preab),
        "T7 ACTab->PREab": adj(p["nRAS_AB"], T.actab, preab),
        "T7 AB_REGBK->PREab": adj(p["nAB2PRE_WR"], T.col, preab),
        "T8 PREab->ACTab": adj(p["nRP_AB"], preab, T.actab),
        "T8 ACTab->ACT": adj(p["nRRD_AB"], T.actab, T.act),
        "T10 REFab->ACTab": adj(rfc, refab, T.actab),
        "T12 last ACT": act_sequence(t)[-1][1],
    }


def derived(t):
    """P5b section 5.2's terms, in clocks, as the table stores them."""
    T = _Terms(t)
    c = lambda x: _num(x / T.m)  # noqa: E731
    return {
        "act_phase": c(T.a0),
        "rcd_rd": c(T.rcd_rd),
        "rcd_wr": c(T.rcd_wr),
        "rcd_ab": c(T.rcd_ab),
        "rcdw_ab": c(T.rcdw_ab),
        # ACT to column beyond the nominal tRCD: the command-cycle
        # adjustment and, on HBM4, the rising edge.
        "act_to_col_edge": c(
            T.rcd_rd - t["host"][FAMILY[t["standard"]]["rcd_rd"]] * T.m
        ),
        "sb_rd_spacing": c(T.sp(T.s_rd)),
        "sb_wr_spacing": c(T.sp(T.s_wr)),
        "pim_branch": c(T.b),
        "ab_rd_spacing": c(T.sp(T.pim["nCCD_AB"])),
        "ab_wr_spacing": c(T.sp(T.pim["nCCDW_AB"])),
        "pair_act": c(T.edge(T.a0 + max(T.rrd_l, T.rrd_s, T.act_slot)) - T.a0),
        "nb_sync": c(T.nb0),
        "nb_spacing": c(T.c_nb),
        "nb_rd_extra": c(T.nb_rd_extra),
        "burst": c(T.burst),
        "command_slot": c(T.slot),
        "issue_offset": ISSUE_OFFSET[t["standard"]],
    }


# Clocks from a PIM request's acceptance to its first command (P4R.13's
# PimUnloaded: two controller clocks to the first push; on HBM4 two ticks,
# one clock). Measured by P4R.13's C26.
ISSUE_OFFSET = {"DDR5": 2, "LPDDR6": 2, "GDDR7": 2, "HBM4": 1}


def u_table(t):
    """P5b section 5.4's table: U for n = 1, 2 and a full row."""
    cols = t["geometry"]["cols_per_row"]
    out = {}
    for key, cls, mode in U_ROWS:
        out[key] = {str(n): unloaded(t, cls, mode, n) for n in (1, 2, cols)}
    return out


# ----------------------------------------------------------------------
# Host closed forms (P5b section 7, E5n): one idle 64-byte read.


def host_read(t, case, line_bytes=64):
    """Clocks from the first command to the data of an idle read of one
    line, averaged over its transactions (LPDDR6: two of 32 bytes), for
    case "hit", "miss" or "conflict". t is a table or a resolved
    (cls, org, host) of a stock standard (see host_terms)."""
    h = t if isinstance(t, dict) and "rcd" in t else host_terms(t)
    first = {
        "hit": h["sync"],
        "miss": max(h["sync"], h["a0"] + h["rcd"]),
        "conflict": max(h["sync"], h["rp"] + h["rcd"]),
    }[case]
    k = max(1, line_bytes // h["col_bytes"])
    done = [first + i * h["s_rd"] + h["rl"] for i in range(k)]
    return _num(sum(done) / k)


def host_terms(t):
    """The host-read terms of a table, or of a stock standard given as
    (name, org preset, timing preset, overrides)."""
    if isinstance(t, (tuple, list)):
        cls, org, hostv, _ = resolve_values(None, base=tuple(t))
        fam = _host_family(cls)
        cc = cls.command_cycles
        two = "ACT1" in cls.commands
        act = cc.get("ACT2" if two else "ACT", 1)
        pre = cc.get("PREpb", cc.get("PRE", 1))
        col = cc.get("RD_S" if two else "RD", 1)
        sync = cc.get("CAS", 1) if FAMILY[fam]["sync"] else 0
        prefetch = cls.internal_prefetch_size
        col_bytes = cls.data_payload_bytes or (
            prefetch * org["channel_width"] // 8
        )
        rl = cls._eval_expr(cls.read_latency, hostv)
    else:
        t = _table(t)
        fam = t["standard"]
        hostv = t["host"]
        cc = t["command_cycles"]
        act, pre, col = cc["ACT"], cc["PREpb"], cc["column"]
        two = cc["act_phases"] == 2
        sync = cc.get("CAS", 0) if FAMILY[fam]["sync"] else 0
        col_bytes = t["geometry"]["col_bytes"]
        rl = t["read_latency"]["value"]
    f = FAMILY[fam]
    adj = _Terms.adj
    return dict(
        sync=sync,
        a0=act if two else 0,
        rcd=adj(hostv[f["rcd_rd"]], act, col),
        rp=adj(hostv[f["rp"]], pre, act),
        s_rd=hostv[f["s_rd"]],
        rl=rl,
        col_bytes=col_bytes,
    )


# ----------------------------------------------------------------------
# The committed tables.


def dumps(tab):
    return json.dumps(tab, indent=2, sort_keys=False) + "\n"


def write(stds=STANDARDS):
    for std in stds:
        with open(table_path(std), "w") as f:
            f.write(dumps(build(std)))
        print(f"wrote {os.path.relpath(table_path(std))}")


def check(stds=STANDARDS):
    """0 if every committed table equals the tree's descriptors."""
    bad = 0
    for std in stds:
        want = dumps(build(std))
        try:
            with open(table_path(std)) as f:
                have = f.read()
        except OSError:
            have = None
        if have != want:
            print(f"{os.path.relpath(table_path(std))} is out of date")
            bad += 1
    return 1 if bad else 0


def show(std):
    t = build(std)
    print(
        f"{std}: B={t['geometry']['banks_per_unit']} columns per row "
        f"{t['geometry']['cols_per_row']}; U in clocks (n = 1, 2, row), "
        f"gem5 adds issue_offset {t['derived']['issue_offset']}"
    )
    for key, vals in t["U_clocks"].items():
        print(f"  {key:15s} " + " / ".join(str(v) for v in vals.values()))
    print(
        "  A(k) flat, last:",
        act_sequence(t)[-1][1],
        "first-ready, last:",
        act_sequence(t, "first-ready")[-1][1],
    )
    print("  Ideal row cycle, switch:", row_cycle(t))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--write",
        action="store_true",
        help="write util/pim/timing/timing_<std>.json",
    )
    g.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if a committed table is out of date",
    )
    g.add_argument("--show", metavar="STD", help="print U of a standard")
    args = ap.parse_args(argv)
    if args.write:
        write()
        return 0
    if args.check:
        rc = check()
        if rc == 0:
            print("timing tables up to date")
        return rc
    show(args.show.upper())
    return 0


if __name__ == "__main__":
    sys.exit(main())
