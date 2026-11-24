"""Earliest-legal-cycle tests of gem5's PIM standards (T1-T18).

Added for gem5; the rules under test are those of ramulator.dram.pim. Each
case takes a standard name ("DDR5", "LPDDR6", "GDDR7", "HBM4") and is
run by test_pim_<std>.py. Expected cycles come from the same device: a PIM
command is compared with the stock command whose rows it copies, and a PIM
parameter is checked by raising it by DELTA and seeing the gate move by
exactly that much (both absorb the command-cycle offsets of
spec.py:to_config).
"""

import itertools
import random

import ramulator
import tests.device_timings.harness as device_timings
from ramulator.dram.pim import DESCRIPTORS

# The PIM presets, from the descriptors (the one source): organization and timing
# presets, and the organization and host timing overrides (GDDR7 32 Gb's refresh
# values). DDR5 runs with two ranks (the two-rank configuration): its default has
# one, and the cases that check that another unit is not delayed (T5, T17, K4, K7,
# rank switches) need a second unit per channel, as HBM4's pseudo channels give.
TEST_ORG_OVERRIDES = {"DDR5": {"rank": 2}}
PRESETS = {
    std: (d.presets["org"], d.presets["timing"],
          {**d.presets["org_overrides"], **TEST_ORG_OVERRIDES.get(std, {}),
           **{k: v for k, (v, _) in d.host_overrides.items()}})
    for std, d in DESCRIPTORS.items()
}
# Same-group read-to-read spacing inside the array (the internal row an AB
# column keeps with host reads, rule 2).
INTERNAL_RD_RD = {"DDR5": "nCCDL", "LPDDR6": "nCCDL", "GDDR7": "nCCDSB", "HBM4": "nCCDL"}
DELTA = 7
SMALLER = 3  # clocks below the default (the PIM parameter alone governs)
GAP = 600  # ticks: past every ACT-to-column and ACT-to-PRE gate of the presets


def make_dut(std, pim=True, **overrides):
    org, timing, kw = PRESETS[std]
    cls = getattr(ramulator.dram, f"{std}_PIM" if pim else std)
    dram = cls(org_preset=org, timing_preset=timing, **{**kw, **overrides})
    return device_timings.DeviceUnderTest(dram)


class Std:
    """Names and addresses of one PIM standard."""

    def __init__(self, std, dut):
        self.std = std
        self.dut = dut
        self.desc = DESCRIPTORS[std]
        self.levels = list(dut.level_names)
        u = self.levels.index(self.desc.unit)
        self.below_unit = self.levels[u + 1:self.levels.index("Row")]  # the bank levels of a unit
        self.rd, self.wr = self.desc.host_rd, self.desc.host_wr
        self.act = self.desc.act
        self.actab = self.desc.actab_phases
        self.actsb = self.desc.actsb_phases or self.desc.act_phases
        self.two_phase = len(self.desc.act_phases) == 2

    def t(self, name):
        return self.dut.timings[name]

    def bank(self, unit=0, row=5, col=0, **levels):
        """A bank of a unit; levels name its bank group, bank, stack ID."""
        args = {self.desc.unit: unit, "Row": row, "Column": col}
        args.update({n: levels.get(n, 0) for n in self.below_unit})
        return self.dut.addr_vec(**args)

    def unit_addr(self, unit=0, row=5, col=0):
        """An all-bank address: every bank level of the unit is -1."""
        args = {self.desc.unit: unit, "Row": row, "Column": col}
        args.update({n: self.dut.ALL for n in self.below_unit})
        return self.dut.addr_vec(**args)

    def banks(self, unit=0, row=5):
        """Every bank of the unit, in flat bank order."""
        sizes = [self.dut.org[n.lower()] for n in self.below_unit]
        for idx in itertools.product(*[range(s) for s in sizes]):
            yield self.bank(unit=unit, row=row, **dict(zip(self.below_unit, idx)))

    def other_group(self, unit=0, row=5):
        """A bank in another bank group (another bank on GDDR7) of the unit."""
        level = "BankGroup" if "BankGroup" in self.below_unit else "Bank"
        return self.bank(unit=unit, row=row, **{level: 1})

    def same_group_other_bank(self, unit=0, row=5):
        other = {"Bank": 2} if "BankGroup" in self.below_unit else {"Bank": 4}
        return self.bank(unit=unit, row=row, **other)

    # Issue helpers: each returns the cycle it issued at.
    def first(self, cmd, addr, start):
        return self.dut.get_first_ready_clk(cmd, addr, start)

    def issue(self, cmd, addr, start=0):
        clk = self.first(cmd, addr, start)
        self.dut.issue(cmd, addr, clk)
        return clk

    def open_bank(self, addr, start=0):
        """Open a row with the per-bank ACT(s); returns the last phase's cycle."""
        clk = start
        for phase in (self.desc.act_phases if not self.two_phase else self.actsb):
            clk = self.issue(phase, addr, clk)
        return clk

    def open_bank_stock(self, addr, start=0):
        clk = start
        for phase in self.desc.act_phases:
            clk = self.issue(phase, addr, clk)
        return clk

    def open_unit(self, unit=0, row=5, start=0):
        clk = start
        for phase in self.actab:
            clk = self.issue(phase, self.unit_addr(unit=unit, row=row), clk)
        return clk


def _first_unit_act(s, start):
    """The earliest cycle of ACTab's last phase (the one the rows name)."""
    if s.two_phase:
        start = s.issue(s.actab[0], s.unit_addr(), start)
    return s.first(s.actab[-1], s.unit_addr(), start)


def case(fn):
    CASES.append(fn)
    return fn


CASES = []


@case
def t1_host_timing_untouched(std):
    """Host-only command sequences time alike on the stock and the PIM standard."""
    stock, pim = Std(std, make_dut(std, pim=False)), Std(std, make_dut(std))
    for name, v in stock.dut.timings.items():
        assert pim.dut.timings[name] == v, name
    rng = random.Random(1)
    clk = 0
    for step in range(400):
        cmd = rng.choice([stock.rd, stock.wr, stock.rd, "PREpb", stock.desc.host_rda])
        levels = {n: rng.randrange(min(2, stock.dut.org[n.lower()])) for n in stock.below_unit}
        addr = stock.bank(row=rng.randrange(3), **levels)
        if rng.random() < 0.05:
            cmd, addr = "PREab", stock.unit_addr(row=0)
            addr[stock.levels.index("Row")] = stock.dut.ALL
        probe = stock.dut.probe(cmd, addr, clk)
        if probe.preq != cmd:
            cmd = probe.preq
        if cmd in stock.desc.act_phases and stock.two_phase:
            cmd = stock.desc.act_phases[0]
        a = stock.first(cmd, addr, clk)
        b = pim.first(cmd, addr, clk)
        assert a == b, (step, cmd, addr)
        stock.dut.issue(cmd, addr, a)
        pim.dut.issue(cmd, addr, a)
        if cmd == stock.desc.act_phases[0] and stock.two_phase:
            a = stock.issue(stock.desc.act_phases[1], addr, a)
            assert pim.first(stock.desc.act_phases[1], addr, a) == a
            pim.dut.issue(stock.desc.act_phases[1], addr, a)
        clk = a


@case
def t2_sb_columns_follow_host_rows(std):
    """PIM_SB_RD/WR keep the host's internal rows and drop the bus rows (rules 0, 1)."""
    s = Std(std, make_dut(std))
    a, g, o = s.bank(), s.same_group_other_bank(), s.other_group()
    t = s.open_bank(a)
    t = max(t, s.open_bank(g, t), s.open_bank(o, t))
    # After ACT: the SB column waits exactly as the host column.
    for sb, host in (("PIM_SB_RD", s.rd), ("PIM_SB_WR", s.wr)):
        assert s.first(sb, a, 0) == s.first(host, a, 0), sb
    c = s.issue(s.rd, a, t + GAP)
    # Same bank group, other bank: the internal same-group row holds.
    assert s.first("PIM_SB_RD", g, c) == s.first(s.rd, g, c)
    # Another bank group: the host waits for the data bus, the SB column only
    # for the column command bus slot (on GDDR7 both are one slot).
    host_other = s.first(s.rd, o, c)
    sb_other = s.first("PIM_SB_RD", o, c)
    slot = type(s.dut.dram).command_cycles.get(s.rd, 1) * s.dut.tick_multiplier
    assert sb_other - c == slot, (sb_other - c, slot)
    assert sb_other <= host_other
    # SB to SB in another group: the PIM branch spaces them as host bursts.
    c2 = s.issue("PIM_SB_RD", a, max(c, s.first("PIM_SB_RD", a, c)))
    assert s.first("PIM_SB_RD", o, c2) == c2 + (host_other - c)


@case
def t3_pair_waits_for_both_banks(std):
    """PIM_SB_RD2 after an ACT of bank 2k + 1 only: never before both banks are open."""
    s = Std(std, make_dut(std))
    even, odd = s.bank(Bank=0), s.bank(Bank=1)
    t0 = s.open_bank(even)
    t1 = s.open_bank(odd, t0 + 20)
    assert s.first("PIM_SB_RD2", even, 0) == s.first(s.rd, odd, 0)
    assert s.first("PIM_SB_RD2", even, 0) > s.first(s.rd, even, 0)
    assert t1 > t0


@case
def t4_actab_to_ab_columns(std):
    """ACTab to RD-class AB columns nRCD_AB, to PIM_AB_REGBK nRCDW_AB; the
    parameter alone governs, so a smaller value takes effect."""
    for param, cmd, host in (("nRCD_AB", "PIM_AB_MAC", "rd"), ("nRCDW_AB", "PIM_AB_REGBK", "wr")):
        s = Std(std, make_dut(std))
        t = s.open_unit()
        base = s.first(cmd, s.unit_addr(), t)
        ref = Std(std, make_dut(std))
        r = ref.open_bank(ref.bank())
        assert base - t == ref.first(getattr(ref, host), ref.bank(), r) - r, cmd
        for delta in (DELTA, -SMALLER):
            d = Std(std, make_dut(std, **{param: s.t(param) // s.dut.tick_multiplier + delta}))
            t = d.open_unit()
            want = base + delta * s.dut.tick_multiplier
            assert d.first(cmd, d.unit_addr(), t) == want, (param, delta)

@case
def t5_ab_column_spacing(std):
    """AB to AB nCCD_AB and the PIM branch; SB to another group; no wait across units."""
    s = Std(std, make_dut(std))
    t = s.open_unit()
    c = s.issue("PIM_AB_MAC", s.unit_addr(), t)
    nxt = s.first("PIM_AB_MAC", s.unit_addr(), c)
    assert s.first("PIM_AB_EW", s.unit_addr(), c) == nxt
    d = Std(std, make_dut(std, nCCD_AB=s.t("nCCD_AB") // s.dut.tick_multiplier + DELTA))
    tc = d.issue("PIM_AB_MAC", d.unit_addr(), d.open_unit())
    assert d.first("PIM_AB_MAC", d.unit_addr(), tc) - tc == nxt - c + DELTA * s.dut.tick_multiplier
    if std in ("DDR5", "HBM4"):
        # The other unit (rank, pseudo channel) of the channel is not delayed.
        u1 = Std(std, make_dut(std))
        t1 = u1.open_unit(unit=0)
        t1 = max(t1, u1.open_unit(unit=1, start=t1))
        c1 = u1.issue("PIM_AB_MAC", u1.unit_addr(unit=0), t1)
        other = u1.first("PIM_AB_MAC", u1.unit_addr(unit=1), c1)
        assert other - c1 < nxt - c, (other - c1, nxt - c)


@case
def t6_ab_and_host_columns(std):
    """AB to host columns: the internal rows only, no data-bus or turnaround row."""
    s = Std(std, make_dut(std))
    t = s.open_unit()
    c = s.issue("PIM_AB_MAC", s.unit_addr(), t + GAP)
    rd = s.first(s.rd, s.bank(), c)
    ref = Std(std, make_dut(std))
    # The bank sharing the internal row: another bank of the same group, or on
    # GDDR7 (no groups) the same bank.
    a = ref.bank()
    g = ref.same_group_other_bank() if "BankGroup" in ref.below_unit else a
    r = max(ref.open_bank(a), ref.open_bank(g) if g != a else 0)
    rc = ref.issue(ref.rd, g, r + GAP)
    # As a host read to the same bank group: the internal same-group row.
    assert rd - c == ref.first(ref.rd, a, rc) - rc
    assert rd - c >= s.t(INTERNAL_RD_RD[std]) - 2 * s.dut.tick_multiplier
    # Host WR after AB: never later than after a host read of the same group.
    wr = s.first(s.wr, s.bank(), c)
    assert wr - c <= ref.first(ref.wr, a, rc) - rc


@case
def t7_ab_to_preab(std):
    """AB to PREab: nAB2PRE_RD, PIM_AB_REGBK nAB2PRE_WR; ACTab to PREab
    nRAS_AB; each alone, so a smaller value takes effect."""
    for param, cmd in (("nAB2PRE_RD", "PIM_AB_MAC"), ("nAB2PRE_WR", "PIM_AB_REGBK")):
        s = Std(std, make_dut(std))
        c = s.issue(cmd, s.unit_addr(), s.open_unit() + GAP)
        base = s.first("PREab", s.unit_addr(), c) - c
        for delta in (DELTA, -SMALLER):
            d = Std(std, make_dut(std, **{param: s.t(param) // s.dut.tick_multiplier + delta}))
            dc = d.issue(cmd, d.unit_addr(), d.open_unit() + GAP)
            want = base + delta * s.dut.tick_multiplier
            assert d.first("PREab", d.unit_addr(), dc) - dc == want, (param, delta)
    s = Std(std, make_dut(std))
    t = s.open_unit()
    base = s.first("PREab", s.unit_addr(), t) - t
    for delta in (DELTA, -SMALLER):
        d = Std(std, make_dut(std, nRAS_AB=s.t("nRAS_AB") // s.dut.tick_multiplier + delta))
        dt = d.open_unit()
        want = base + delta * s.dut.tick_multiplier
        assert d.first("PREab", d.unit_addr(), dt) - dt == want, ("nRAS_AB", delta)

@case
def t8_all_bank_act_spacing(std):
    """PREab to ACTab nRP_AB; a per-bank ACT to ACTab nRRD_AB; each alone,
    so a smaller value takes effect."""
    s = Std(std, make_dut(std))
    p = s.issue("PREab", s.unit_addr(), s.open_unit() + GAP)
    base = _first_unit_act(s, p) - p
    for delta in (DELTA, -SMALLER):
        d = Std(std, make_dut(std, nRP_AB=s.t("nRP_AB") // s.dut.tick_multiplier + delta))
        dp = d.issue("PREab", d.unit_addr(), d.open_unit() + GAP)
        want = base + delta * s.dut.tick_multiplier
        assert _first_unit_act(d, dp) - dp == want, ("nRP_AB", delta)
    # A per-bank ACT, then ACTab of the unit.
    e = Std(std, make_dut(std))
    a = e.open_bank(e.bank())
    base = _first_unit_act(e, a) - a
    for delta in (DELTA, -2):
        f = Std(std, make_dut(std, nRRD_AB=e.t("nRRD_AB") // e.dut.tick_multiplier + delta))
        fa = f.open_bank(f.bank())
        want = base + delta * e.dut.tick_multiplier
        assert _first_unit_act(f, fa) - fa == want, ("nRRD_AB", delta)

@case
def t9_actab_outside_faw(std):
    """Four per-bank ACTs then ACTab, and ACTab then four ACTs: no tFAW delay."""
    has_faw = "nFAW" in make_dut(std).timings
    s = Std(std, make_dut(std, nFAW=400) if has_faw else make_dut(std))
    banks = list(s.banks())
    clk = 0
    acts = []
    for b in banks[:4]:
        clk = s.open_bank(b, clk)
        acts.append(clk)
    # A fifth per-bank ACT (its last phase) is held by the window.
    if has_faw:
        if s.two_phase:
            fifth = s.first(s.actsb[1], banks[4], s.issue(s.actsb[0], banks[4], clk))
        else:
            fifth = s.first(s.actsb[-1], banks[4], clk)
        assert fifth >= acts[0] + s.t("nFAW") - 4 * s.dut.tick_multiplier
    for b in banks[:4]:
        clk = max(clk, s.issue("PREpb", b, clk))
    # ACTab is not: it waits only for the PREs and nRRD_AB.
    tab = s.issue(s.actab[0], s.unit_addr(), clk)
    if s.two_phase:
        tab = s.first(s.actab[1], s.unit_addr(), tab)
    if has_faw:
        assert tab < acts[0] + s.t("nFAW")


@case
def t10_refresh_and_actab(std):
    """REFab to ACTab nRFC; ACTab to REFab as ACT to REFab."""
    s = Std(std, make_dut(std))
    ref_addr = s.unit_addr(row=0)
    ref_addr[s.levels.index("Row")] = s.dut.ALL
    r = s.issue("REFab", ref_addr, 0)
    tab = s.first(s.actab[0], s.unit_addr(), r)
    ref = Std(std, make_dut(std))
    rr = ref.issue("REFab", ref_addr, 0)
    assert tab == ref.first(ref.desc.act_phases[0], ref.bank(), rr)
    s2 = Std(std, make_dut(std))
    t = s2.open_unit()
    pre = s2.issue("PREab", s2.unit_addr(), t)
    ref2 = Std(std, make_dut(std))
    t2 = ref2.open_bank(ref2.bank())
    pre2 = ref2.issue("PREab", ref2.unit_addr(), t2)
    slack = 4 * s.dut.tick_multiplier
    assert s2.first("REFab", ref_addr, pre) - t >= ref2.first("REFab", ref_addr, pre2) - t2 - slack


@case
def t11_non_bank_commands(std):
    """PIM_GB_RD on a closed unit: ready at 0 (no preq); after a host read: the bus spacing."""
    s = Std(std, make_dut(std))
    assert s.dut.probe("PIM_GB_RD", s.unit_addr(), 0).ready
    a, o = s.bank(), s.other_group()
    t = max(s.open_bank(a), s.open_bank(o))
    c = s.issue(s.rd, a, t + GAP)
    assert s.first("PIM_GB_RD", s.unit_addr(), c) == s.first(s.rd, o, c)
    d = Std(std, make_dut(std, nCL_GB=s.t(s.desc.rl) // s.dut.tick_multiplier + DELTA))
    dt = max(d.open_bank(d.bank()), d.open_bank(d.other_group()))
    dc = d.issue("PIM_GB_RD", d.unit_addr(), dt + GAP)
    # A later GB read means a later data burst: the host read after it waits longer.
    s2 = Std(std, make_dut(std))
    st = max(s2.open_bank(s2.bank()), s2.open_bank(s2.other_group()))
    sc = s2.issue("PIM_GB_RD", s2.unit_addr(), st + GAP)
    want = s2.first(s2.rd, s2.other_group(), sc) - sc + DELTA * s.dut.tick_multiplier
    assert d.first(d.rd, d.other_group(), dc) - dc == want


@case
def t12_baseline_act_sequence(std):
    """Baseline: per-bank ACTs in bank order at the stock limits, then the AB column."""
    s = Std(std, make_dut(std))
    ref = Std(std, make_dut(std, pim=False))
    clk = rclk = 0
    opens = []
    for b in s.banks():
        clk = s.open_bank(b, clk)
        rclk = ref.open_bank_stock(b, rclk)
        assert clk == rclk
        opens.append(clk)
    col = s.first("PIM_AB_MAC", s.unit_addr(), clk)
    want = max(ref.first(ref.rd, b, o) for b, o in zip(s.banks(), opens))
    assert col == want
    return opens[-1]


@case
def t13_aim_closes_per_bank(std):
    """AiM: ACTab, columns, PREpb per bank, spaced by the stock PRE-to-PRE rows."""
    s = Std(std, make_dut(std))
    c = s.issue("PIM_AB_MAC", s.unit_addr(), s.open_unit())
    banks = list(s.banks())
    p0 = s.issue("PREpb", banks[0], c)
    p1 = s.first("PREpb", banks[1], p0)
    ref = Std(std, make_dut(std))
    r0 = max(ref.open_bank(banks[0]), ref.open_bank(banks[1]))
    q0 = ref.issue("PREpb", banks[0], r0 + 200)
    assert p1 - p0 == ref.first("PREpb", banks[1], q0) - q0


@case
def t14_pair_state_and_timing(std):
    """PIM_SB_RD2 updates banks 2k and 2k + 1; PIM_SB_RD2A closes both."""
    s = Std(std, make_dut(std))
    even, odd = s.bank(Bank=0), s.bank(Bank=1)
    t = max(s.open_bank(even), s.open_bank(odd))
    c = s.issue("PIM_SB_RD2", even, t)
    ref = Std(std, make_dut(std))
    rt = max(ref.open_bank(even), ref.open_bank(odd))
    rc = ref.issue(ref.rd, even, rt)
    # PRE readiness of the odd bank follows the pair's column, as the even bank's.
    assert s.first("PREpb", odd, c) - c == ref.first("PREpb", even, rc) - rc
    s.issue("PIM_SB_RD2A", even, s.first("PIM_SB_RD2A", even, c))
    for b in (even, odd):
        assert not s.dut.probe(s.rd, b, 10_000).row_open


@case
def t15_ab_columns_meet_array_rows_of_pim_columns(std):
    """Against SB columns and AB columns of the other class, an AB column
    waits what an SB column of the same bank waits (the bank and bank-group
    rows); AB to AB of one class stays nCCD_AB (T5)."""
    sb = {"PIM_AB_MAC": "PIM_SB_RD", "PIM_AB_REGBK": "PIM_SB_WR", "PIM_SB_RD": "PIM_SB_RD",
          "PIM_SB_WR": "PIM_SB_WR"}
    pairs = [("PIM_AB_REGBK", "PIM_SB_RD"), ("PIM_AB_MAC", "PIM_SB_WR"), ("PIM_AB_MAC", "PIM_SB_RD"),
             ("PIM_SB_WR", "PIM_AB_MAC"), ("PIM_SB_RD", "PIM_AB_REGBK"), ("PIM_AB_MAC", "PIM_AB_REGBK"),
             ("PIM_AB_REGBK", "PIM_AB_MAC")]
    gaps = {}
    for p, f in pairs:
        s = Std(std, make_dut(std))
        t = s.open_unit()
        a = s.bank()
        c = s.issue(p, s.unit_addr() if "_AB_" in p else a, t + GAP)
        gaps[(p, f)] = s.first(f, s.unit_addr() if "_AB_" in f else a, c) - c
        ref = Std(std, make_dut(std))
        rc = ref.issue(sb[p], a, ref.open_bank(a) + GAP)
        assert gaps[(p, f)] == ref.first(sb[f], a, rc) - rc, (p, f)
    if std == "LPDDR6":
        # Same-group write-to-read nWL + nBL_max + nWTRL and read-to-write
        # nRTW_L (were 49 and 48 through the unit's bus-row copies).
        assert gaps[("PIM_AB_REGBK", "PIM_SB_RD")] == 70
        assert gaps[("PIM_AB_MAC", "PIM_AB_REGBK")] == 54


@case
def t16_host_and_pim_acts_share_faw(std):
    """LPDDR6's host ACT-2 and PIM ACTsb2 count in one four-activate window.
    With nFAW 80, host, host, host, PIM, PIM puts the fifth ACT where five
    host ACTs put it."""
    if std != "LPDDR6":
        return

    def fifth(kinds):
        s = Std(std, make_dut(std, nFAW=80))
        banks = list(s.banks())
        clk = 0
        for b, kind in zip(banks, kinds):
            clk = (s.open_bank_stock if kind == "host" else s.open_bank)(b, clk)
        return clk

    host = fifth(["host"] * 5)
    assert fifth(["host"] * 3 + ["pim"] * 2) == host
    assert fifth(["pim"] * 5) == host


# The data dependencies inside a unit: preceding, following, the parameter
# the row is timed by. GB_WR's row is nCWL_GB plus the host write burst.
DEPENDENT = [
    # A column that writes the register file before a register read or write.
    ("PIM_AB_MAC", "PIM_REG_RD", "nEX_AB_RD"),
    ("PIM_AB_EW", "PIM_REG_WR", "nEX_AB_RD"),
    ("PIM_AB_BKREG", "PIM_REG_RD", "nEX_AB_RD"),
    ("PIM_SB_RD", "PIM_REG_RD", "nSB_RD_DONE"),
    ("PIM_SB_RD2", "PIM_REG_WR", "nSB_RD_DONE"),
    # A GB write before the columns that read the GB.
    ("PIM_GB_WR", "PIM_AB_MAC", "nCWL_GB"),
    ("PIM_GB_WR", "PIM_AB_EW", "nCWL_GB"),
    ("PIM_GB_WR", "PIM_SB_RD", "nCWL_GB"),
    ("PIM_GB_WR", "PIM_SB_WR", "nCWL_GB"),
    # Write after read: a GB or register write after a column that reads it.
    ("PIM_AB_MAC", "PIM_GB_WR", "nEX_AB_RD"),
    ("PIM_SB_RD", "PIM_GB_WR", "nSB_RD_DONE"),
    ("PIM_SB_WR", "PIM_GB_WR", "nSB_WR_DONE"),
    ("PIM_AB_REGBK", "PIM_REG_WR", "nEX_AB_WR"),
]
# Pairs that share no state: the parameter does not move them.
INDEPENDENT = [
    ("PIM_AB_REGBK", "PIM_REG_RD", "nEX_AB_WR"),
    ("PIM_GB_WR", "PIM_AB_BKREG", "nCWL_GB"),
    ("PIM_GB_WR", "PIM_AB_REGBK", "nCWL_GB"),
    ("PIM_AB_BKREG", "PIM_GB_WR", "nEX_AB_RD"),
    ("PIM_SB_RD2", "PIM_GB_WR", "nSB_RD_DONE"),
]


def _dep_gap(std, p, f, unit=0, **overrides):
    """Clocks (ticks on HBM4) from p to the earliest f, on unit 0's open row."""
    s = Std(std, make_dut(std, **overrides))
    t = s.open_unit()
    addr = lambda c, u: s.bank(unit=u) if "_SB_" in c else s.unit_addr(unit=u)  # noqa: E731
    c = s.issue(p, addr(p, 0), t + GAP)
    return s, s.first(f, addr(f, unit), c) - c


@case
def t17_unit_data_dependencies(std):
    """A non-bank command that reads or overwrites the register file or the
    GB waits for the column that writes or reads it by that column's
    execution time
    (PIM_AB_MAC -> PIM_REG_RD by nEX_AB_RD), a column that reads the GB for
    a GB write by write latency + burst; pairs that share no state do not."""
    tm = None
    for p, f, param in DEPENDENT + INDEPENDENT:
        s, gap = _dep_gap(std, p, f)
        tm = s.dut.tick_multiplier
        _, moved = _dep_gap(std, p, f, **{param: s.t(param) // tm + DELTA})
        if (p, f, param) in INDEPENDENT:
            assert moved == gap, (p, f, gap, moved)
            continue
        assert moved == gap + DELTA * tm, (p, f, gap, moved)
        cls = getattr(ramulator.dram, f"{std}_PIM")
        cc = lambda c: int(cls.command_cycles.get(c, 1) * tm)  # noqa: E731
        want = s.t(param) + (s.t(s.desc.bl) if p == "PIM_GB_WR" else 0) + cc(p) - cc(f)
        assert want <= gap < want + tm, (p, f, gap, want)
    # The rows are at the unit: another unit (DDR5 rank, HBM4 pseudo channel)
    # does not wait.
    s = Std(std, make_dut(std))
    if s.dut.org.get(s.desc.unit.lower(), 1) > 1:
        _, own = _dep_gap(std, "PIM_AB_MAC", "PIM_REG_RD")
        _, other = _dep_gap(std, "PIM_AB_MAC", "PIM_REG_RD", unit=1)
        assert other < own, (own, other)


@case
def t18_ddr5_presb_after_all_bank_commands(std):
    """DDR5's same-bank PREsb waits for ACTab by nRAS_AB
    and for an AB column by nAB2PRE_RD/nAB2PRE_WR, as PREpb does."""
    if std != "DDR5":
        return
    for p, param in (("ACTab", "nRAS_AB"), ("PIM_AB_MAC", "nAB2PRE_RD"),
                     ("PIM_AB_REGBK", "nAB2PRE_WR")):
        gaps = []
        for extra in (0, DELTA):
            s = Std(std, make_dut(std))
            d = Std(std, make_dut(std, **{param: s.t(param) // s.dut.tick_multiplier + extra}))
            t = d.open_unit()
            c = t if p == "ACTab" else d.issue(p, d.unit_addr(), t + GAP)
            gaps.append({pre: d.first(pre, d.bank(), c) - c for pre in ("PREsb", "PREpb")})
        assert gaps[1]["PREsb"] - gaps[0]["PREsb"] == DELTA, (p, gaps)
        assert gaps[0]["PREsb"] == gaps[0]["PREpb"], (p, gaps)
