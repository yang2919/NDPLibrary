"""gem5 processing-in-memory (PIM) commands for Ramulator 2.1 standards.

Added for gem5. A PIM standard (DDR5_PIM, LPDDR6_PIM, GDDR7_PIM, HBM4_PIM)
is a subclass of the stock standard, as DDR5_VRR is of DDR5. ``apply_pim``
gives it:

- the stock commands plus one command per PIM timing class and the
  all-bank and direct ACT commands;
- the stock timing parameters plus 18 PIM parameters, each defaulting to a
  host value, evaluated after the host overrides unless the PIM value
  itself is given (``resolve_pim_defaults``);
- a transformed deep copy of the stock rows plus new rows (``derive_pim``),
  so the stock standard is never changed;
- the stock presets.

Rules, applied in this order to the copied rows:

0. Bus rows exist because of the external data pins: a column-to-column row
   above the bank-group level (channel burst, different-group spacing,
   turnarounds, rank or stack switch), and every row of a clock-control
   command (LPDDR6 CAS, GDDR7 RCKSTRT/RCKSTOP). Single-bank (SB) and
   all-bank (AB) commands take none of them with host commands.
1. SB: PIM_SB_RD and PIM_SB_RD2 join every other row that names the host
   read, PIM_SB_WR every one that names the host write, and the
   auto-precharge forms every one that names the host's auto-precharge
   command, on the same side.
0b. The PIM branch, one per parallelism unit: every non-sibling
   column-to-column bus row is copied with PIM SB and AB column commands on
   both sides, at the parallelism unit, never paired with host commands.
3. All-bank ACT: ACTab (its last phase on LPDDR6) joins every row naming the
   standard's ACT, except the activation windows (such as nFAW) and the
   pairs a PIM parameter names (below). LPDDR6's direct per-bank
   ACTsb1/ACTsb2 join every row of ACT1/ACT2 first.
2. AB columns: RD-class AB commands join, paired with host commands only,
   every other row naming the host read, PIM_AB_REGBK every one naming the
   host write, except the pairs a PIM parameter names. At the bank and the
   bank group they are also paired with SB columns and with AB columns of
   the other class, as with host columns; AB to AB of one class stays
   nCCD_AB/nCCDW_AB alone.
4. Non-bank (GB, register) commands carry host data: they join the bus rows
   at the channel and the unit, and the clock-control rows. Where a row
   protects the data bus, its latency is shifted by the difference between
   the non-bank and the host data latency (0 with the defaults).

The PIM timing rows are added (all at the unit): ACTab to AB columns
nRCD_AB/nRCDW_AB, ACTab to any PRE nRAS_AB, PREab to ACTab nRP_AB,
ACTab to or from any ACT nRRD_AB, AB to PRE nAB2PRE_RD/nAB2PRE_WR, AB to AB
nCCD_AB/nCCDW_AB, non-bank to non-bank nCCD_NB. For the ACT, PRE and column
pairs the first six name, the parameter alone governs: rules 2 and 3 copy no
host row for them, so a value below the host's takes effect. AB to AB
keeps the PIM branch of rule 0b as its floor. An SB and an AB column differ
only in bank scope: AB columns are spaced by the same-bank column gaps.
nEX_AB_RD and nEX_AB_WR, the completion of RD- and WR-class AB columns, and
nSB_RD_DONE, nSB_WR_DONE are read by the PIM sequencer only: a read is done
when its data have left the sense amplifiers (tRTP), a write when its data
are in them (write latency + burst).

Data dependencies inside a unit are rows too, as the host's write-to-read
is: a column that writes the register file (registers and accumulators)
comes before a register read or write by its class's execution time
(nSB_RD_DONE, nEX_AB_RD), AB_REGBK, which reads registers, before a register
write by nEX_AB_WR; a column that reads the global buffer (GB) comes before
a GB write by its execution time, and a GB write before such a column by
write latency + burst (nCWL_GB plus the host write burst). Column to column
stays as above (AB columns pipeline).
"""

import copy

from ramulator.dram.spec import TimingConstraint as TC

# Column commands, one per timing class, and the three auto-precharge forms.
SB_RD = ["PIM_SB_RD", "PIM_SB_RD2"]
SB_RDA = ["PIM_SB_RDA", "PIM_SB_RD2A"]
SB_WR = ["PIM_SB_WR"]
SB_WRA = ["PIM_SB_WRA"]
AB_RD = ["PIM_AB_EW", "PIM_AB_MAC", "PIM_AB_BKREG"]
AB_WR = ["PIM_AB_REGBK"]
NB_RD = ["PIM_GB_RD", "PIM_REG_RD"]
NB_WR = ["PIM_GB_WR", "PIM_REG_WR"]
COLUMN_COMMANDS = SB_RD + SB_WR + SB_RDA + SB_WRA + AB_RD + AB_WR + NB_RD + NB_WR
BRANCH_RD = SB_RD + SB_RDA + AB_RD
BRANCH_WR = SB_WR + SB_WRA + AB_WR
# The unit's state each column class touches (a class holds several
# operations): PIM_SB_RD is SB EW with the
# GB, MAC and MOV_BKGBUF, PIM_SB_RD2 SB EW of a bank pair, PIM_SB_WR
# MOV_GBUFBK. The register file holds the registers and the accumulators.
GB_READERS = ["PIM_SB_RD", "PIM_SB_RDA", "PIM_SB_WR", "PIM_SB_WRA", "PIM_AB_EW", "PIM_AB_MAC"]
REG_WRITERS = SB_RD + SB_RDA + ["PIM_AB_EW", "PIM_AB_MAC", "PIM_AB_BKREG"]
REG_READERS = ["PIM_SB_RD", "PIM_SB_RDA", "PIM_AB_MAC", "PIM_AB_REGBK"]

# The PIM timing parameters, in this order in timing_params.
PIM_PARAMS = [
    "nRCD_AB", "nRCDW_AB", "nRAS_AB", "nRP_AB", "nRRD_AB",
    "nCCD_AB", "nCCDW_AB", "nAB2PRE_RD", "nAB2PRE_WR",
    "nCL_GB", "nCL_REG", "nCWL_GB", "nCWL_REG", "nCCD_NB", "nEX_AB_RD", "nEX_AB_WR",
    "nSB_RD_DONE", "nSB_WR_DONE",
]

# Levels whose column-to-column rows are internal to the array (rule 0).
INTERNAL_LEVELS = ("BankGroup", "Bank", "Row", "Column")


class PimDescriptor:
    """The PIM properties of one standard."""

    def __init__(self, *, unit, host_rd, host_wr, host_rda, host_wra, host_reads, host_writes,
                 act_phases, clock_cmds, rl, wl, bl, defaults, bus_rows, lists, presets,
                 host_overrides):
        self.unit = unit  # level name of the parallelism unit
        self.host_rd = host_rd  # host read and write commands
        self.host_wr = host_wr
        self.host_rda = host_rda
        self.host_wra = host_wra
        self.host_reads = host_reads  # every host read column command
        self.host_writes = host_writes  # every host write column command
        self.act_phases = act_phases  # the ACT command(s) ACTab follows
        self.clock_cmds = clock_cmds  # clock-control commands (rule 0)
        self.rl = rl  # host read and write data latency parameters
        self.wl = wl
        self.bl = bl  # the host write burst (the burst of nEX_AB_WR's default)
        self.defaults = defaults  # the 18 parameter defaults, as expressions
        self.bus_rows = bus_rows  # rule 0's rows, (level, latency, sibling)
        self.lists = lists  # which hard-coded controller lists each joins
        # The PIM preset (the one source read by gem5's
        # components/memory/pim.py, libpim/gen_pim_layout.py and
        # libpim/timing/pim_timing.py): organization and timing presets,
        # organization overrides, controller and default channel count; each
        # default PIM memory is 32 GiB.
        self.presets = presets
        # Host timing overrides: only values Ramulator 2.1 does not give,
        # name: (value, reason). Applied to every Ramulator 2.1 instance of a
        # PIM configuration.
        self.host_overrides = host_overrides

    @property
    def act(self):
        return self.act_phases[-1]

    @property
    def actab_phases(self):
        return ["ACTab1", "ACTab2"] if len(self.act_phases) == 2 else ["ACTab"]

    @property
    def actsb_phases(self):
        return ["ACTsb1", "ACTsb2"] if len(self.act_phases) == 2 else []


DESCRIPTORS = {
    "DDR5": PimDescriptor(
        unit="Rank", host_rd="RD", host_wr="WR", host_rda="RDA", host_wra="WRA",
        host_reads=["RD", "RDA"], host_writes=["WR", "WRA"],
        act_phases=["ACT"], clock_cmds=[], rl="nCL", wl="nCWL", bl="nBL",
        defaults={
            "nRCD_AB": "nRCD", "nRCDW_AB": "nRCD", "nRAS_AB": "nRAS", "nRP_AB": "nRP",
            "nRRD_AB": "nRRDS", "nCCD_AB": "nCCDL", "nCCDW_AB": "nCCDL_WR",
            "nAB2PRE_RD": "nRTP", "nAB2PRE_WR": "nCWL + nBL + nWR",
            "nCL_GB": "nCL", "nCL_REG": "nCL", "nCWL_GB": "nCWL", "nCWL_REG": "nCWL",
            "nCCD_NB": "nCCDS", "nEX_AB_RD": "nRTP",
            "nEX_AB_WR": "nCWL + nBL",
            "nSB_RD_DONE": "nRTP", "nSB_WR_DONE": "nCWL + nBL",
        },
        bus_rows={
            ("Channel", "nBL", False),
            ("Rank", "nCCDS", False), ("Rank", "nCCDS_WR", False), ("Rank", "nRTW", False),
            ("Rank", "nCWL + nBL + nWTRS", False),
            ("Rank", "nBL + nCS", True), ("Rank", "nCWL + nBL + nCS - nCL", True),
        },
        lists={},  # GenericDDR has no hard-coded command list
        # Two 1Rx8 16 GB DIMMs (4 sub-channels, 32 GiB); a rank count of 2 is an
        # organization override.
        presets=dict(org="DDR5_16Gb_x8", timing="DDR5_4800AN", org_overrides={},
                     controller="GenericDDR", channels=4),
        # Kept as they are: nCS 2, DDR4's 2, active with two ranks (the default has one).
        host_overrides={},
    ),
    "LPDDR6": PimDescriptor(
        unit="Rank", host_rd="RD_S", host_wr="WR_S", host_rda="RDA_S", host_wra="WRA_S",
        host_reads=["RD_S", "RDA_S", "RD_L", "RDA_L"],
        host_writes=["WR_S", "WRA_S", "WR_L", "WRA_L"],
        act_phases=["ACT1", "ACT2"], clock_cmds=["CAS"], rl="nRL", wl="nWL", bl="nBL_min",
        defaults={
            "nRCD_AB": "nRCDr", "nRCDW_AB": "nRCDw", "nRAS_AB": "nRAS", "nRP_AB": "nRPab",
            "nRRD_AB": "nRRD", "nCCD_AB": "nCCDL", "nCCDW_AB": "nCCDL_WR",
            "nAB2PRE_RD": "nRTP", "nAB2PRE_WR": "nWL + nBL_max + nWTP",
            "nCL_GB": "nRL", "nCL_REG": "nRL", "nCWL_GB": "nWL", "nCWL_REG": "nWL",
            "nCCD_NB": "nCCDS", "nEX_AB_RD": "nRTP",
            "nEX_AB_WR": "nWL + nBL_min",
            "nSB_RD_DONE": "nRTP", "nSB_WR_DONE": "nWL + nBL_min",
        },
        bus_rows={
            ("Channel", "nBL_min", False), ("Channel", "nBL_min + nBL_min", False),
            ("Bank", "nCAS", False),
            ("Rank", "nCCDS", False), ("Rank", "nCCDS_WR", False),
            ("Rank", "nRTW_S", False), ("Rank", "nRTW_S_L", False),
            ("Rank", "nWL + nBL_min + nWTRS", False), ("Rank", "nWL + nBL_min_L + nWTRS", False),
            ("Rank", "nBL_min + nCS", True), ("Rank", "nBL_min + nBL_min + nCS", True),
            ("Rank", "nRL + nBL_min + nCS - nWL", True),
            ("Rank", "nRL + nBL_min + nBL_min + nCS - nWL", True),
        },
        # LPDDRControllerBase::is_access_cmd and is_read_cmd (README.gem5, change 8).
        lists={"is_access_cmd": NB_RD + NB_WR, "is_read_cmd": NB_RD},
        # 16 sub-channels, 8 dies, 32 GiB.
        presets=dict(org="LPDDR6_16Gb_x12", timing="LPDDR6_10667_BL24", org_overrides={},
                     controller="LPDDR6", channels=16),
        # Kept as they are: nAAD 8 (LPDDR5, JESD209-5C 7.3.1); nRPST 0
        # (no reference); nCS 2 (inactive, one rank).
        host_overrides={},
    ),
    "GDDR7": PimDescriptor(
        unit="Channel", host_rd="RD", host_wr="WR", host_rda="RDA", host_wra="WRA",
        host_reads=["RD", "RDA"], host_writes=["WR", "WRA"],
        act_phases=["ACT"], clock_cmds=["RCKSTRT", "RCKSTOP"], rl="nRL", wl="nWL", bl="nBL",
        defaults={
            "nRCD_AB": "nRCDRD", "nRCDW_AB": "nRCDWR", "nRAS_AB": "nRAS", "nRP_AB": "nRP",
            "nRRD_AB": "nRRD", "nCCD_AB": "nCCDSB", "nCCDW_AB": "nCCDSB",
            "nAB2PRE_RD": "nRTPSB", "nAB2PRE_WR": "nWL + nBL + nWR",
            "nCL_GB": "nRL", "nCL_REG": "nRL", "nCWL_GB": "nWL", "nCWL_REG": "nWL",
            "nCCD_NB": "nCCD", "nEX_AB_RD": "nRTPSB",
            "nEX_AB_WR": "nWL + nBL",
            "nSB_RD_DONE": "nRTPSB", "nSB_WR_DONE": "nWL + nBL",
        },
        bus_rows={
            ("Channel", "nCCD", False), ("Channel", "nRTW", False),
            ("Channel", "nWL + nBL + nWTR", False),
            ("Channel", "nRCKSTRT2RD", False), ("Channel", "nRD2RCKSTOP", False),
            ("Channel", "nRCKSP2ST", False),
            ("Channel", "nRCKST2SP", False),
        },
        # GDDR7Controller::is_read_like (README.gem5, change 10).
        lists={"is_read_like": NB_RD},
        # 8 devices of 32 Gb (a 256-bit bus), 32 channels, 32 GiB.
        presets=dict(org="GDDR7_32Gb_x8", timing="GDDR7_28000_PAM3", org_overrides={},
                     controller="GDDR7", channels=32),
        # Earlier overrides, dropped since only values Ramulator 2.1 does not give are
        # overridden (its own values are used, guesses included), kept as the record
        # should a JEDEC or vendor value replace one: nRCDWR 20 (GDDR6_14000_1250mV_double, Samsung K4Z80325BC;
        # guess 19), nRRD 11 (GDDR6 nRRDS; guess 4), nRREFD 22 (guess 21), nWTR 10 and
        # nWTRSB 12 (GDDR6 nWTRS, nWTRL; guesses 9, 11). Kept: nRL 24, nWL 6, nRCDRD, nRP,
        # nRAS, nWR, nRTPSB equal GDDR6's; nDQERL 0 and the RCK values have no reference.
        # GDDR7_32Gb_x8 at 28 Gb/s resolves no refresh times (gddr7.py:193-209 gives them for
        # 16 Gb only, and JESD239D leaves them to vendors), so they are derived by
        # one density doubling up from 16 Gb: tRFCab x1.5 per doubling, tRFCpb kept at 60 ns
        # (GDDR7_64Gb_x8's values by that rule were 405 ns / 60 ns, nRFCab 710, nRFCpb 106).
        host_overrides={
            "nRFCab": (473, "tRFCab 270 ns = 16 Gb's 180 ns x1.5 per density doubling (as "
                            "from GDDR6 8 Gb to GDDR7 16 Gb), rounded up at 571 ps; "
                            "JESD239D leaves it to vendors"),
            "nRFCpb": (106, "tRFCpb 60 ns, as for 16 Gb, rounded up at 571 ps"),
            "nRFMab": (473, "Equal to nRFCab, Ramulator 2.1's own rule (gddr7.py:156-158); "
                            "refresh management inactive"),
            "nRFMpb": (106, "Equal to nRFCpb, Ramulator 2.1's own rule (gddr7.py:156-158); "
                            "refresh management inactive"),
        },
    ),
    "HBM4": PimDescriptor(
        unit="PseudoChannel", host_rd="RD", host_wr="WR", host_rda="RDA", host_wra="WRA",
        host_reads=["RD", "RDA"], host_writes=["WR", "WRA"],
        act_phases=["ACT"], clock_cmds=[], rl="nCL", wl="nCWL", bl="nBL",
        defaults={
            "nRCD_AB": "nRCDRD", "nRCDW_AB": "nRCDWR", "nRAS_AB": "nRAS", "nRP_AB": "nRP",
            "nRRD_AB": "nRRDS", "nCCD_AB": "nCCDL", "nCCDW_AB": "nCCDL",
            "nAB2PRE_RD": "nRTP", "nAB2PRE_WR": "nCWL + nBL + nWR",
            "nCL_GB": "nCL", "nCL_REG": "nCL", "nCWL_GB": "nCWL", "nCWL_REG": "nCWL",
            "nCCD_NB": "nCCDS", "nEX_AB_RD": "nRTP",
            "nEX_AB_WR": "nCWL + nBL",
            "nSB_RD_DONE": "nRTP", "nSB_WR_DONE": "nCWL + nBL",
        },
        bus_rows={
            ("PseudoChannel", "nBL", False), ("PseudoChannel", "nRTW", False),
            ("PseudoChannel", "nCWL + nBL + nWTRS", False),
            ("Sid", "nCCDS", False), ("Sid", "nCCDR", True), ("Sid", "nCCDS", True),
        },
        # HBM34Controller::is_all_bank_row_command (README.gem5, change 9).
        lists={"is_all_bank_row_command": ["ACTab"]},
        # One whole 8Hi stack, 32 channels, 32 GiB.
        presets=dict(org="HBM4_32Gb_8Hi", timing="HBM4_8000Mbps_folded_banks_baseline",
                     org_overrides={}, controller="HBM34", channels=32),
        # Earlier overrides, dropped since only values Ramulator 2.1 does not give are
        # overridden: nCWL 13 (HBM3_6400's 6.25 ns at 500 ps, rounded up; guess 10), and
        # with it nRTW 27 (Ramulator 2.1's formula gives 30 with its nCWL).
        # Kept: nRCDWR 19, nRTP 12, nWR 42, nWTRL 13, nWTRS 9 equal HBM3's in ns; nCCDR 4.
        host_overrides={},
    ),
}


def _names(row):
    return set(row.preceding) | set(row.following)


def is_bus_row(row, desc):
    """Rule 0: a row that exists because of the external data pins."""
    if _names(row) & set(desc.clock_cmds):
        return True
    host = set(desc.host_reads + desc.host_writes)
    col_to_col = set(row.preceding) & host and set(row.following) & host
    return bool(col_to_col) and row.level not in INTERNAL_LEVELS


def _join(side, mapping):
    """The commands of side plus the PIM commands mapped from its host ones."""
    out = list(side)
    for host, pims in mapping.items():
        if host in side:
            out += [p for p in pims if p not in out]
    return out


def _mapped(side, mapping):
    """Only the PIM commands mapped from the host commands of side."""
    out = []
    for host, pims in mapping.items():
        if host in side:
            out += [p for p in pims if p not in out]
    return out


def _row(level, preceding, following, latency, like=None):
    """A new row, with the window and sibling flags of like."""
    if like is None:
        return TC(level=level, preceding=preceding, following=following, latency=latency)
    return TC(level=level, preceding=preceding, following=following, latency=latency,
              window=like.window, sibling=like.sibling, shared_window=like.shared_window)


def derive_pim(base, desc):
    """The PIM standard's rows: a transformed copy of base's plus new rows."""
    rows = copy.deepcopy(base.timing_constraints)
    unit = desc.unit
    levels = list(base.levels)
    unit_idx = levels.index(unit)
    bank_idx = levels.index("Bank")
    act, actab = desc.act, desc.actab_phases[-1]
    acts = [act] + desc.actsb_phases[-1:]  # the ACT phases timing rows name

    found = {(r.level, r.latency, r.sibling) for r in rows if is_bus_row(r, desc)}
    if found != desc.bus_rows:
        raise ValueError(f"{base.name}: rule 0 bus rows {sorted(found)} differ from the "
                         f"descriptor's {sorted(desc.bus_rows)}")
    bus = [is_bus_row(r, desc) for r in rows]

    # LPDDR6's direct per-bank ACTs: the rows of ACT1 and ACT2 (the spike).
    # A joined window row (nFAW) keeps one history for host and PIM ACTs
    # as DDR5 and HBM4 do for their multi-command window rows.
    for one, two in zip(desc.act_phases, desc.actsb_phases):
        for r in rows:
            joined = one in r.preceding and r.window > 1
            r.preceding = _join(r.preceding, {one: [two]})
            r.following = _join(r.following, {one: [two]})
            if joined:
                r.shared_window = True

    # Rule 1: SB commands join the host's rows, except the bus rows.
    sb = {desc.host_rd: SB_RD, desc.host_wr: SB_WR, desc.host_rda: SB_RDA, desc.host_wra: SB_WRA}
    for r, b in zip(rows, bus):
        if not b:
            r.preceding = _join(r.preceding, sb)
            r.following = _join(r.following, sb)

    new = []
    # Rule 0b: the PIM branch, one per unit.
    # The host commands the PIM column commands resemble (on LPDDR6 the
    # short-burst family); rows of other host commands (RD_L) are
    # not copied.
    branch = {desc.host_rd: BRANCH_RD, desc.host_rda: BRANCH_RD,
              desc.host_wr: BRANCH_WR, desc.host_wra: BRANCH_WR}
    for r, b in zip(rows, bus):
        if not b or r.sibling or _names(r) & set(desc.clock_cmds):
            continue
        pre = _mapped(r.preceding, branch)
        fol = _mapped(r.following, branch)
        level = unit if levels.index(r.level) < unit_idx else r.level
        key = (level, tuple(pre), tuple(fol), r.latency)
        seen = {(n.level, tuple(n.preceding), tuple(n.following), n.latency) for n in new}
        if pre and fol and key not in seen:
            new.append(_row(level, pre, fol, r.latency))

    # Rule 3: the all-bank ACT joins the ACT rows, as new rows, without the
    # pairs nRAS_AB (to any PRE), nRP_AB (from PREab) and nRRD_AB (to and
    # from any ACT) name: those parameters alone govern.
    pres = {"PREpb", "PREab", "PREsb"}
    for r, b in list(zip(rows, bus)):
        if b or act not in _names(r) or r.window != 1 or r.shared_window:
            continue  # bus rows have no ACT; ACTab is not in the activation windows
        if act in r.preceding:
            fol = [c for c in r.following if c not in pres and c not in acts]
            if fol:
                new.append(_row(r.level, [actab], fol, r.latency, like=r))
        if act in r.following:
            pre = [c for c in r.preceding if c != "PREab" and c not in acts]
            if pre:
                new.append(_row(r.level, pre, [actab], r.latency, like=r))

    # Rule 2: AB columns paired with host commands (and the per-bank ACTs),
    # without the pairs nAB2PRE_* (to any PRE) names; ACTab to AB columns is
    # nRCD_AB/nRCDW_AB alone. At the bank and the bank group, SB columns
    # (joined by rule 1) and AB columns of the other class are paired like
    # host columns; AB to AB of one class stays nCCD_AB or nCCDW_AB alone.
    host_side = set(base.commands) | set(desc.actsb_phases)
    sb_cols = set(SB_RD + SB_RDA + SB_WR + SB_WRA)
    ab = {desc.host_rd: AB_RD, desc.host_wr: AB_WR}
    for r, b in zip(rows, bus):
        if b or not (_names(r) & set(ab)):
            continue
        array = r.level in ("Bank", "BankGroup")
        side = host_side | sb_cols if array else host_side
        hpre = [c for c in r.preceding if c in side]
        hfol = [c for c in r.following if c in side]
        apre = _mapped(r.preceding, ab)
        afol = _mapped(r.following, ab)
        hfol = [c for c in hfol if c not in pres]
        if apre and hfol:
            new.append(_row(r.level, apre, hfol, r.latency, like=r))
        if hpre and afol:
            new.append(_row(r.level, hpre, afol, r.latency, like=r))
        if array:
            for pc, fc in ((AB_RD, AB_WR), (AB_WR, AB_RD)):
                p = [c for c in apre if c in pc]
                f = [c for c in afol if c in fc]
                if p and f:
                    new.append(_row(r.level, p, f, r.latency, like=r))

    # Rule 4: non-bank commands on the data bus. Shift per data latency.
    shift = {c: "0" for c in base.commands}
    shift.update({"PIM_GB_RD": f"nCL_GB - {desc.rl}", "PIM_REG_RD": f"nCL_REG - {desc.rl}",
                  "PIM_GB_WR": f"nCWL_GB - {desc.wl}", "PIM_REG_WR": f"nCWL_REG - {desc.wl}"})
    nb = {desc.host_rd: NB_RD, desc.host_wr: NB_WR}
    for r, b in zip(rows, bus):
        if not b or not (r.level in (levels[0], unit) or _names(r) & set(desc.clock_cmds)):
            continue
        clock = bool(_names(r) & set(desc.clock_cmds))
        pre = _join([c for c in r.preceding if c in base.commands], nb)
        fol = _join([c for c in r.following if c in base.commands], nb)
        groups = {}
        for p in pre:
            for f in fol:
                if p in base.commands and f in base.commands:
                    continue  # the host row itself
                lat = r.latency if clock else f"({r.latency}) + ({shift[p]}) - ({shift[f]})"
                g = groups.setdefault(lat, ([], []))
                if p not in g[0]:
                    g[0].append(p)
                if f not in g[1]:
                    g[1].append(f)
        for lat, (p, f) in groups.items():
            new.append(_row(r.level, p, f, lat, like=r))

    # The PIM timing rows, at the unit. DDR5's same-bank PREsb is a PRE too.
    prs = [c for c in ("PREab", "PREpb", "PREsb") if c in base.commands]
    new += [
        _row(unit, [actab], AB_RD, "nRCD_AB"),
        _row(unit, [actab], AB_WR, "nRCDW_AB"),
        _row(unit, [actab], prs, "nRAS_AB"),
        _row(unit, ["PREab"], [actab], "nRP_AB"),
        _row(unit, [actab], acts + [actab], "nRRD_AB"),
        _row(unit, acts, [actab], "nRRD_AB"),
        _row(unit, AB_RD, AB_RD + AB_WR, "nCCD_AB"),
        _row(unit, AB_WR, AB_RD + AB_WR, "nCCDW_AB"),
        _row(unit, AB_RD, prs, "nAB2PRE_RD"),
        _row(unit, AB_WR, prs, "nAB2PRE_WR"),
        _row(unit, NB_RD + NB_WR, NB_RD + NB_WR, "nCCD_NB"),
    ]

    # Data dependencies inside a unit: rows between the bank
    # columns and the non-bank commands that touch the same GB or register
    # file, as the host's write-to-read is a row (nWTR), not a completion.
    # A column reads or writes them during its execution, so it is timed by
    # its class's: nSB_RD_DONE, nSB_WR_DONE, nEX_AB_RD, nEX_AB_WR; a non-bank
    # write's data are in after write latency + burst.
    for cols, ex in ((SB_RD + SB_RDA, "nSB_RD_DONE"), (SB_WR + SB_WRA, "nSB_WR_DONE"),
                     (AB_RD, "nEX_AB_RD"), (AB_WR, "nEX_AB_WR")):
        # Register-file writers (SB EW and MAC, AB EW, MAC and BKREG) before a
        # register read or write; the register reader AB_REGBK before a
        # register write (write after read).
        reg = [c for c in cols if c in REG_WRITERS]
        if reg:
            new.append(_row(unit, reg, ["PIM_REG_RD", "PIM_REG_WR"], ex))
        reg = [c for c in cols if c in REG_READERS and c not in REG_WRITERS]
        if reg:
            new.append(_row(unit, reg, ["PIM_REG_WR"], ex))
        # GB readers (SB EW with the GB, MAC, MOV_GBUFBK; AB EW, MAC) before a
        # GB write (write after read).
        gb = [c for c in cols if c in GB_READERS]
        if gb:
            new.append(_row(unit, gb, ["PIM_GB_WR"], ex))
    # A GB write before the columns that read the GB (read after write).
    new.append(_row(unit, ["PIM_GB_WR"], GB_READERS, f"nCWL_GB + {desc.bl}"))
    return rows + new


def resolve_pim_timings(cls, timing_dict):
    """Fill the PIM parameters with their host-value defaults (inside
    resolve_secondary_timings, so that a PIM override is accepted)."""
    for name in PIM_PARAMS:
        timing_dict[name] = cls._eval_expr(cls.pim_descriptor.defaults[name], timing_dict)


def resolve_pim_defaults(std, resolved):
    """Evaluate every PIM default on the host values in use, after
    the host overrides, unless the PIM parameter itself is overridden. std is
    the PIM standard instance, resolved the (org_dict, timing_dict) of its base
    class's resolve()."""
    org_dict, timing_dict = resolved
    cls = type(std)
    defaults = cls.pim_descriptor.defaults
    host = {k: v for k, v in timing_dict.items() if k not in defaults}
    for name in PIM_PARAMS:
        if name not in std._overrides:
            timing_dict[name] = cls._eval_expr(defaults[name], host)
    return org_dict, timing_dict


def apply_pim(cls, base, desc):
    """Give the subclass cls of base its PIM commands, parameters and rows."""
    if set(desc.defaults) != set(PIM_PARAMS):
        raise ValueError(f"{base.name}: PIM defaults must name exactly {PIM_PARAMS}")
    row_cmds = desc.actab_phases + desc.actsb_phases
    cls.pim_descriptor = desc
    cls.commands = list(base.commands) + row_cmds + COLUMN_COMMANDS
    cls.timing_params = list(base.timing_params) + PIM_PARAMS
    cls.timing_constraints = derive_pim(base, desc)
    cycles = dict(base.command_cycles)
    for c in SB_RD + SB_RDA + AB_RD + NB_RD:
        if desc.host_rd in base.command_cycles:
            cycles[c] = base.command_cycles[desc.host_rd]
    for c in SB_WR + SB_WRA + AB_WR + NB_WR:
        if desc.host_wr in base.command_cycles:
            cycles[c] = base.command_cycles[desc.host_wr]
    for phase, ab, sbp in zip(desc.act_phases, desc.actab_phases, desc.actsb_phases + [None] * 2):
        if phase in base.command_cycles:
            cycles[ab] = base.command_cycles[phase]
            if sbp:
                cycles[sbp] = base.command_cycles[phase]
    cls.command_cycles = cycles
    if base.row_commands:
        cls.row_commands = list(base.row_commands) + row_cmds
        cls.column_commands = list(base.column_commands) + COLUMN_COMMANDS
    cls.org_presets = base.org_presets
    cls.timing_presets = base.timing_presets
