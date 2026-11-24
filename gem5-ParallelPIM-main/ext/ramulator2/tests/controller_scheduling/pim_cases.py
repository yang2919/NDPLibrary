"""Controller tests of gem5's PIM command sequencer.

Added for gem5. Each case takes a standard name and is run by test_pim_<std>.py. The sequencer
gets its jobs through its test_jobs parameter (no memory system), and the
host requests come from the harness.
"""

import ramulator
import tests.controller_scheduling.harness as cs
import tests.device_timings.harness as device_timings
from tests.device_timings.pim_cases import PRESETS

CONTROLLERS = {"DDR5": "GenericDDR", "LPDDR6": "LPDDR6", "GDDR7": "GDDR7", "HBM4": "HBM34"}
PIM_COLUMNS = {
    "PIM_SB_RD", "PIM_SB_RD2", "PIM_SB_WR", "PIM_SB_RDA", "PIM_SB_RD2A", "PIM_SB_WRA",
    "PIM_AB_EW", "PIM_AB_MAC", "PIM_AB_BKREG", "PIM_AB_REGBK",
    "PIM_GB_RD", "PIM_GB_WR", "PIM_REG_RD", "PIM_REG_WR",
}


def make_dram(std, **overrides):
    org, timing, kw = PRESETS[std]
    return getattr(ramulator.dram, f"{std}_PIM")(
        org_preset=org, timing_preset=timing, **{**kw, **overrides}
    )


def refresh_manager(refresh):
    if refresh:
        return ramulator.refresh_manager.AllBank()
    return ramulator.refresh_manager.NoRefresh()


def make(std, jobs=(), drain=(), refresh=False, row_policy=None, seq_kw=None, dram_kw=None,
         **ctrl_kw):
    seq = ramulator.controller_plugin.PimSequencer(
        test_jobs=list(jobs), test_write_drain=list(drain), **(seq_kw or {})
    )
    controller = getattr(ramulator.controller, CONTROLLERS[std])(
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=refresh_manager(refresh),
        row_policy=row_policy or ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
        dram=make_dram(std, **(dram_kw or {})),
        controller_plugins=[seq],
        **ctrl_kw,
    )
    return cs.ControllerUnderTest(controller)


def job(at, cls, vec, ncols):
    return f"{at}:{cls}:{'/'.join(str(v) for v in vec)}:{ncols}"


def run(dut, cycles, host=()):
    """Tick; send host requests (clk, type, addr_vec) from clk on, retrying."""
    waiting = sorted(host, key=lambda h: h[0])
    trace = []
    for clk in range(cycles):
        while waiting and waiting[0][0] <= clk:
            _, kind, vec = waiting[0]
            try:
                dut.send_request(kind, vec)
            except RuntimeError:
                break  # buffer full: try again next cycle
            waiting.pop(0)
        trace += dut.tick()
    return trace


def stats(dut):
    """The controller's statistics with the PimSequencer's merged in."""
    s = dict(dut.stats())
    for plugin in s.get("controller_plugin", []):
        if plugin.get("impl") == "PimSequencer":
            s.update(plugin)
    return s


def addr(dut, **levels):
    vec = dut.addr_vec(**levels)
    vec[0] = 0
    return vec


def unit_level(dut):
    for name in ("Rank", "PseudoChannel"):
        if name in dut.level_names:
            return name
    return "Channel"


def unit_addr(dut, row=0, col=0, unit=0):
    """-1 below the unit, as the wrapper decodes an all-bank or non-bank address."""
    names = dut.level_names
    u = names.index(unit_level(dut))
    vec = [0] * len(names)
    for i in range(u + 1, names.index("Row")):
        vec[i] = -1
    if u > 0:
        vec[u] = unit
    vec[names.index("Row")] = row
    vec[names.index("Column")] = col
    return vec


def bank_addr(dut, row=7, col=0, **levels):
    names = dut.level_names
    vec = [0] * len(names)
    for k, v in levels.items():
        vec[names.index(k)] = v
    vec[names.index("Row")] = row
    vec[names.index("Column")] = col
    return vec


def pim_cmds(trace):
    return [c for c in trace if c.type_id == 2]


def assert_earliest(std, trace, arrive=0, dram_kw=None):
    """Replay the issued commands on a device: each PIM command issued at
    the first cycle its timing allows after the previous command (and after
    the jobs arrived)."""
    dev = device_timings.DeviceUnderTest(make_dram(std, **(dram_kw or {})))
    prev = -1
    for c in trace:
        if c.type_id == 2:
            first = dev.get_first_ready_clk(c.command, c.addr_vec, max(prev + 1, arrive))
            if std == "HBM4" and c.command not in ("PREpb", "PREab") and first % 2 == 0:
                first += 1  # HBM34 issues these on the rising edge only (hbm34_controller.cpp)
            assert first == c.clk, (c.command, c.addr_vec, c.clk, first)
        dev.issue(c.command, c.addr_vec, c.clk)
        prev = c.clk


def case(fn):
    CASES.append(fn)
    return fn


CASES = []


@case
def sb_job_through_plugin(std):
    """T2/T3 through the plugin: the SB job opens its bank(s), issues its columns
    at the earliest legal cycles, and completes nSB_RD_DONE after the last."""
    probe = make(std)
    a = bank_addr(probe, row=7, col=2)
    pair = bank_addr(probe, row=9, col=0, Bank=2)
    if "BankGroup" in probe.level_names:
        pair[probe.level_names.index("BankGroup")] = 1
    dut = make(std, jobs=[job(5, "SbRd", a, 4), job(5, "SbRd2", pair, 3), job(5, "SbWr", a, 2)])
    trace = run(dut, 3000)
    cmds = pim_cmds(trace)
    names = [c.command for c in cmds]
    assert names.count("PIM_SB_RD") == 4 and names.count("PIM_SB_RD2") == 3
    assert names.count("PIM_SB_WR") == 2
    assert_earliest(std, trace, arrive=5)
    # The pair opened both banks before its first column.
    first_rd2 = next(c for c in cmds if c.command == "PIM_SB_RD2")
    acts = [c for c in cmds if c.command.startswith("ACT") and c.clk < first_rd2.clk]
    banks = {tuple(c.addr_vec[1:-2]) for c in acts}
    assert len(banks) >= 3  # a, and both banks of the pair
    s = stats(dut)
    last = [c.clk for c in cmds if c.command == "PIM_SB_RD"][-1]
    assert s["pim_test_last_cmd_0"] == last
    assert s["pim_test_done_0"] == last + dut.timing("nSB_RD_DONE")
    assert s["pim_jobs"] == 3 and s["pim_col_cmds_SbRd"] == 4


@case
def k1_clock_sync_only_before_non_bank(std):
    """K1 (LPDDR6): the WCK sync CAS comes before a non-bank command, not before SB."""
    if std != "LPDDR6":
        return
    probe = make(std)
    jobs = [job(5, "SbRd", bank_addr(probe), 2), job(200, "GbRd", unit_addr(probe), 2)]
    dut = make(std, jobs=jobs)
    trace = run(dut, 1000)
    names = [c.command for c in trace]
    first_sb = names.index("PIM_SB_RD")
    assert "CAS" not in names[:first_sb + 1]
    gb = names.index("PIM_GB_RD")
    assert names[gb - 1] == "CAS"
    assert stats(dut)["pim_sync_cmds"] >= 1


@case
def k3_read_clock_before_gb_read(std):
    """K3 (GDDR7): with RCK stopped, RCKSTRT comes before PIM_GB_RD."""
    if std != "GDDR7":
        return
    dut = make(std, jobs=[job(5, "GbRd", unit_addr(make(std)), 2)], rck_mode="start_with_rckstrt")
    trace = run(dut, 500)
    names = [c.command for c in trace]
    assert "PIM_GB_RD" in names, names
    assert "RCKSTRT" in names[:names.index("PIM_GB_RD")]


@case
def k4_gate_per_unit(std):
    """K4: during an FR-FCFS job on unit 0, no host command to unit 0; the
    other unit's host traffic continues (DDR5 ranks, HBM4 pseudo channels)."""
    if std not in ("DDR5", "HBM4"):
        return
    probe = make(std)
    u = unit_level(probe)
    host = []
    for i in range(200):
        for unit in (0, 1):
            vec = bank_addr(probe, row=3 + (i % 2), col=(i * 8) % 64, **{u: unit, "Bank": 1})
            host.append((i * 3, "Read", vec))
    dut = make(std, jobs=[job(40, "SbRd", bank_addr(probe, row=7, **{u: 0, "Bank": 0}), 8)])
    trace = run(dut, 4000, host)
    lvl = probe.level_names.index(u)
    cmds = pim_cmds(trace)
    start, end = cmds[0].clk, cmds[-1].clk
    host_in = [c for c in trace if c.type_id in (0, 1) and start < c.clk < end]
    assert not [c for c in host_in if c.addr_vec[lvl] == 0]
    assert [c for c in host_in if c.addr_vec[lvl] == 1]


@case
def k5_write_drain(std):
    """K5: a write-drain request starts write mode while reads keep coming."""
    probe = make(std)
    reads = [(i, "Read", bank_addr(probe, row=3, col=(i * 4) % 32, Bank=1))
             for i in range(0, 3000, 2)]
    writes = [(10 + i, "Write", bank_addr(probe, row=4, col=i, Bank=2)) for i in range(4)]

    def first_wr(trace):
        return next((c.clk for c in trace if c.type_id == 1 and c.command.startswith("WR")), None)

    stock = first_wr(run(make(std), 2000, reads + writes))
    on = 400
    drained = first_wr(run(make(std, drain=[on, 1500]), 2000, reads + writes))
    assert drained is not None and drained >= on
    assert stock is None or stock > drained
    assert drained - on < 200


@case
def k8_maintenance_counters(std):
    """K8: PIM commands do not count as maintenance requests."""
    probe = make(std)
    dut = make(std, jobs=[job(5, "SbRd", bank_addr(probe), 4), job(5, "GbWr", unit_addr(probe), 2)])
    run(dut, 1000)
    s = stats(dut)
    assert s["num_maintenance_reqs"] == 0 and s["num_maintenance_reqs_served"] == 0
    assert s["pim_col_cmds_SbRd"] == 4 and s["pim_col_cmds_GbWr"] == 2


MODES = ("ideal", "aim", "baseline", "aespa")


def pim_jobs_done(dut, n):
    s = stats(dut)
    return all(s[f"pim_test_done_{i}"] > 0 for i in range(n))
AB_CLASSES = ("AbMac", "AbEw", "AbBkreg", "AbRegbk")


def unit_banks(dut):
    names = dut.level_names
    u = names.index(unit_level(dut))
    n = 1
    for name in names[u + 1:names.index("Row")]:
        n *= dut.org[name.lower()]
    return n


@case
def ab_modes_and_closing(std):
    """Each mode's commands for every AB class, with close_rows; every command
    at the earliest legal cycle (as tests/device_timings T12, T13)."""
    probe = make(std)
    banks = unit_banks(probe)
    sb = {"AbMac": "PIM_SB_RD", "AbEw": "PIM_SB_RD2", "AbBkreg": "PIM_SB_RD",
          "AbRegbk": "PIM_SB_WR"}
    ab = {"AbMac": "PIM_AB_MAC", "AbEw": "PIM_AB_EW", "AbBkreg": "PIM_AB_BKREG",
          "AbRegbk": "PIM_AB_REGBK"}
    act = "ACTsb2" if std == "LPDDR6" else "ACT"
    actab = "ACTab2" if std == "LPDDR6" else "ACTab"
    for mode in MODES:
        for cls in AB_CLASSES:
            dut = make(std, jobs=[job(5, cls, unit_addr(probe, row=6), 3)],
                       seq_kw={"ab_mode": mode, "close_rows": True})
            trace = run(dut, 9000)
            names = [c.command for c in pim_cmds(trace)]
            n = {k: names.count(k) for k in set(names)}
            if mode in ("ideal", "aim"):
                assert n.get(actab) == 1 and n.get(ab[cls]) == 3, (mode, cls, n)
                closes = {"ideal": ("PREab", 1), "aim": ("PREpb", banks)}[mode]
                assert n.get(closes[0]) == closes[1], (mode, cls, n)
            elif mode == "baseline":
                assert n.get(act) == banks and n.get(ab[cls]) == 3, (mode, cls, n)
                assert n.get("PREpb") == banks, (mode, cls, n)
            else:
                cols = 3 * (banks // 2 if cls == "AbEw" else banks)
                assert n.get(act) == banks and n.get(sb[cls]) == cols, (mode, cls, n)
                assert n.get("PREpb") == banks, (mode, cls, n)
            assert_earliest(std, trace, arrive=5)
            s = stats(dut)
            assert s["pim_test_done_0"] > 0, (mode, cls)


@case
def ab_rows_stay_open_for_the_same_row(std):
    """No closing PREs without close_rows, nor before a job of the unit that
    needs the same row; and every job completes in every mode, AESPA
    included, also when its row is already open, with and without
    close_rows."""
    probe = make(std)
    for mode in MODES:
        dut = make(std, jobs=[job(5, "AbMac", unit_addr(probe, row=6), 2)],
                   seq_kw={"ab_mode": mode})
        names = [c.command for c in pim_cmds(run(dut, 6000))]
        assert "PREpb" not in names and "PREab" not in names, (mode, names)
        assert pim_jobs_done(dut, 1), mode
        jobs = [job(5, "AbMac", unit_addr(probe, row=6), 2),
                job(5, "AbMac", unit_addr(probe, row=6, col=4), 2)]
        for close in (True, False):
            dut = make(std, jobs=jobs, seq_kw={"ab_mode": mode, "close_rows": close})
            names = [c.command for c in pim_cmds(run(dut, 9000))]
            assert pim_jobs_done(dut, 2), (mode, close, names)
            if close and mode != "aespa":
                cols = [i for i, c in enumerate(names) if c.startswith("PIM_")]
                between = names[cols[1]:cols[2]]
                assert not [c for c in between if c.startswith("PRE")], (mode, names)


@case
def k1_no_clock_sync_before_ab(std):
    """K1 (LPDDR6): no WCK sync CAS before SB or AB columns."""
    if std != "LPDDR6":
        return
    probe = make(std)
    dut = make(std, jobs=[job(5, "AbMac", unit_addr(probe, row=6), 4)], seq_kw={"ab_mode": "ideal"})
    trace = run(dut, 2000)
    assert "CAS" not in [c.command for c in trace]


@case
def k2_actab_rising_edge(std):
    """K2 (HBM4): ACTab issues on a rising edge and pairs as an all-bank row
    command: no column command of its pseudo channel in the same cycle and no
    PRE of it on the next falling edge."""
    if std != "HBM4":
        return
    probe = make(std)
    host = [(i * 4, "Read", bank_addr(probe, row=3, col=(i * 2) % 32, PseudoChannel=0, Bank=1))
            for i in range(60)]
    dut = make(std, jobs=[job(60, "AbMac", unit_addr(probe, row=6, unit=0), 4)],
               seq_kw={"ab_mode": "ideal"})
    trace = run(dut, 3000, host)
    pc = probe.level_names.index("PseudoChannel")
    actabs = [c for c in trace if c.command == "ACTab"]
    assert actabs
    for a in actabs:
        assert a.clk % 2 == 1
        same = [c for c in trace if c.clk in (a.clk, a.clk + 3) and c is not a
                and c.addr_vec[pc] == a.addr_vec[pc]]
        assert not [c for c in same if c.command.startswith(("PRE", "RD", "WR", "PIM_"))], same


@case
def k7_refresh_inside_ab_jobs(std):
    """K7: refresh due inside long AB jobs, every mode: rows reopened, no state
    failure, refresh at its interval. The refresh interval is tried with a few
    offsets, since one fixed interval can fall between the jobs' columns."""
    probe = make(std)
    tick = 2 if std == "HBM4" else 1
    row_cols = probe.org["column"] // type(probe.dram).internal_prefetch_size
    nrfc = probe.timing("nRFCab" if "nRFCab" in probe.timings else "nRFC")
    units = 2 if std in ("DDR5", "HBM4") else 1
    for mode in MODES:
        reopens = 0
        for offset in (0, 97, 211):
            nrefi = (nrfc + 1500) // tick + offset  # clocks: past the refresh time
            ncols = 4 if mode == "aespa" else row_cols
            n = 40
            jobs = [job(5, "AbMac", unit_addr(probe, row=6 + i), ncols) for i in range(n)]
            dut = make(std, jobs=jobs, refresh=True, seq_kw={"ab_mode": mode},
                       dram_kw={"nREFI": nrefi})
            trace, cycles = [], 0
            while cycles < 400000:
                trace += run(dut, 10000)
                cycles += 10000
                if pim_jobs_done(dut, n):
                    break
            assert pim_jobs_done(dut, n), (mode, offset)
            reopens += stats(dut)["pim_reopens"]
            refs = [c for c in trace if c.command == "REFab"]
            expected = units * cycles // dut.timing("nREFI")
            assert abs(len(refs) - expected) <= units, (mode, len(refs), expected)
        assert reopens > 0, mode


@case
def k9_closedcap_ignores_pim_columns(std):
    """K9 (DDR5): ClosedCAP (cap 4) counts no PIM column, and injects no PREpb
    for another rank's bank, during an AB job on rank 1."""
    if std != "DDR5":
        return
    probe = make(std)
    host = [(i * 6, "Read", bank_addr(probe, row=3, col=i * 8 % 64, Rank=0, Bank=3))
            for i in range(3)]
    dut = make(std, jobs=[job(200, "AbMac", unit_addr(probe, row=6, unit=1), 16)],
               row_policy=ramulator.row_policy.ClosedCAP(cap=4), seq_kw={"ab_mode": "ideal"})
    trace = run(dut, 3000, host)
    cmds = pim_cmds(trace)
    start, end = cmds[0].clk, cmds[-1].clk
    injected = [c for c in trace
                if c.type_id == -1 and c.command == "PREpb" and start <= c.clk <= end]
    assert not injected, injected
    assert [c.command for c in cmds].count("PIM_AB_MAC") == 16


@case
def k10_aespa_pipelines(std):
    """K10 (HBM4, GDDR7): AESPA issues columns before the last ACT; one PIM
    command per cycle."""
    if std not in ("HBM4", "GDDR7"):
        return
    probe = make(std)
    dut = make(std, jobs=[job(5, "AbMac", unit_addr(probe, row=6), 2)], seq_kw={"ab_mode": "aespa"})
    cmds = pim_cmds(run(dut, 6000))
    last_act = max(c.clk for c in cmds if c.command == "ACT")
    first_col = min(c.clk for c in cmds if c.command == "PIM_SB_RD")
    assert first_col < last_act
    clks = [c.clk for c in cmds]
    assert len(clks) == len(set(clks))


@case
def p4r8_chopim_host_first(std):
    """Chopim: no PIM command while host requests are queued."""
    probe = make(std)
    host = [(i, "Read", bank_addr(probe, row=3, col=(i * 4) % 32, Bank=1)) for i in range(0, 600)]
    dut = make(std, jobs=[job(5, "SbRd", bank_addr(probe, row=7), 4)],
               seq_kw={"arb_policy": "chopim"})
    trace = run(dut, 12000, host)
    host_cols = [c.clk for c in trace if c.type_id == 0 and c.command.startswith("RD")]
    pim = pim_cmds(trace)
    assert len([c for c in pim if c.command == "PIM_SB_RD"]) == 4
    # The host queue empties when its last read issues; PIM waits for it.
    assert min(c.clk for c in pim) > max(host_cols)


@case
def p4r8_asyncdimm_switches_after_pre(std):
    """AsyncDIMM: the side changes only after a PRE of the current side, or
    when that side has no work."""
    probe = make(std)
    host = [(i * 2, "Read", bank_addr(probe, row=3 + (i // 8) % 2, col=(i * 4) % 32, Bank=1))
            for i in range(0, 400)]
    dut = make(std, jobs=[job(5, "SbRd", bank_addr(probe, row=7), 8),
                          job(5, "SbRd", bank_addr(probe, row=8), 8)],
               row_policy=ramulator.row_policy.ClosedCAP(cap=2), seq_kw={"arb_policy": "asyncdimm"})
    trace = run(dut, 6000, host)
    pim = pim_cmds(trace)
    assert len([c for c in pim if c.command == "PIM_SB_RD"]) == 16
    # Every switch from host to PIM follows a host-side PRE, or an empty host queue.
    side, last = "host", None
    for c in trace:
        now = "pim" if c.type_id == 2 else "host"
        if now != side:
            if side == "host" and last is not None:
                closing = last.command.startswith("PRE") or last.command.startswith(("RDA", "WRA"))
                assert closing or not [h for h in host if h[0] > c.clk], last
            side = now
        last = c


@case
def p4r8_cosm_is_pending(std):
    """COSM is pending: choosing it stops at init."""
    import pytest

    with pytest.raises(RuntimeError, match="cosm is pending"):
        make(std, seq_kw={"arb_policy": "cosm"})


@case
def r4_closing_under_refresh(std):
    """With close_rows, a refresh's PREab can close an AB job's banks between
    its last column and its closing PREs; the job must then finish, not wait
    for a PRE it no longer needs. 40 AB MAC jobs, refresh at nRFC + 1500
    (+ offsets), every mode."""
    probe = make(std)
    tick = 2 if std == "HBM4" else 1
    row_cols = probe.org["column"] // type(probe.dram).internal_prefetch_size
    nrfc = probe.timing("nRFCab" if "nRFCab" in probe.timings else "nRFC")
    for mode in MODES:
        for offset in (0, 97, 211):
            nrefi = (nrfc + 1500) // tick + offset
            ncols = 4 if mode == "aespa" else row_cols
            n = 40
            jobs = [job(5, "AbMac", unit_addr(probe, row=6 + i), ncols) for i in range(n)]
            dut = make(std, jobs=jobs, refresh=True,
                       seq_kw={"ab_mode": mode, "close_rows": True},
                       dram_kw={"nREFI": nrefi})
            cycles = 0
            while cycles < 400000 and not pim_jobs_done(dut, n):
                run(dut, 10000)
                cycles += 10000
            done = sum(stats(dut)[f"pim_test_done_{i}"] > 0 for i in range(n))
            assert done == n, (mode, offset, done)


@case
def r6_lpddr6_act_pairs_under_refresh(std):
    """Under frequent all-bank refresh (K7's interval, nRFC + 1500, and
    offsets), every PIM ACTsb1 (ACTab1) is followed by its ACTsb2 (ACTab2)
    within nAAD, with no PREab of its unit in between, as a host ACT-2 is
    protected; and every job completes."""
    if std != "LPDDR6":
        return
    probe = make(std)
    row_cols = probe.org["column"] // type(probe.dram).internal_prefetch_size
    nrfc = probe.timing("nRFCab" if "nRFCab" in probe.timings else "nRFC")
    naad = probe.timing("nAAD")
    for mode in ("baseline", "ideal", "aim"):
        for offset in (0, 97, 211):
            n = 40
            jobs = [job(5, "AbMac", unit_addr(probe, row=6 + i), row_cols) for i in range(n)]
            dut = make(std, jobs=jobs, refresh=True, seq_kw={"ab_mode": mode},
                       dram_kw={"nREFI": nrfc + 1500 + offset})
            trace, cycles = [], 0
            while cycles < 300000 and not pim_jobs_done(dut, n):
                trace += run(dut, 10000)
                cycles += 10000
            assert pim_jobs_done(dut, n), (mode, offset)
            bank = probe.level_names.index("Bank")
            for i, c in enumerate(trace):
                if c.type_id != 2 or c.command not in ("ACTsb1", "ACTab1"):
                    continue
                two = "ACTsb2" if c.command == "ACTsb1" else "ACTab2"
                rest = trace[i + 1:]
                k = next(j for j, d in enumerate(rest)
                         if d.command == two
                         and (two == "ACTab2" or d.addr_vec[bank] == c.addr_vec[bank]))
                assert rest[k].clk - c.clk <= naad, (mode, c.clk, rest[k].clk)
                assert not [d for d in rest[:k] if d.command == "PREab"], (mode, c.clk)


@case
def r7_asyncdimm_refresh_keeps_pim_side(std):
    """Only host requests (types 0 and 1) are AsyncDIMM's host side.
    AB jobs on one row (no PIM PRE without close_rows) start just before a
    refresh, and host reads arrive after they started: the refresh must not
    hand the unit to the host, so no host command reaches the unit while a
    job is started."""
    probe = make(std)
    tick = 2 if std == "HBM4" else 1
    row_cols = probe.org["column"] // type(probe.dram).internal_prefetch_size
    nrfc = probe.timing("nRFCab" if "nRFCab" in probe.timings else "nRFC")
    nrefi = (nrfc + 1500) // tick
    n = 8
    at = tick * (nrefi - 300)  # just before the first refresh, in controller ticks
    jobs = [job(at, "AbMac", unit_addr(probe, row=6, col=i * (row_cols // n)), row_cols // n)
            for i in range(n)]
    host = [(at + 50 + i * 4, "Read", bank_addr(probe, row=3, col=(i * 4) % 32, Bank=1))
            for i in range(600)]
    dut = make(std, jobs=jobs, refresh=True,
               seq_kw={"ab_mode": "baseline", "arb_policy": "asyncdimm"},
               dram_kw={"nREFI": nrefi})
    trace = run(dut, tick * (3 * nrefi + 5000), host)
    s = stats(dut)
    assert pim_jobs_done(dut, n)
    pim = pim_cmds(trace)
    start, end = pim[0].clk, pim[-1].clk
    refs = [c for c in trace if c.command.startswith("REF") and start < c.clk < end]
    assert refs, "no refresh inside the jobs"
    assert s["pim_interrupts"] == 0, s["pim_interrupts"]


@case
def r14_gddr7_rck_counts_only_pim_reads(std):
    """README.gem5 change 10 counts only the two non-bank PIM reads of the
    priority buffer as reads waiting for RCK. A stock RD sent to the priority
    buffer (as Ramulator 2.1's own tests and plugins do) starts no RCKSTRT,
    as without that change; a PIM_GB_RD does (K3)."""
    if std != "GDDR7":
        return
    dut = make(std, rck_mode="start_with_rckstrt")
    bank = bank_addr(dut, row=7, Bank=0)
    dut.priority_send("ACT", bank)
    dut.priority_send("RD", bank)
    names = [c.command for c in run(dut, 400)]
    assert "RCKSTRT" not in names, names
    assert dut.stats()["num_rckstrt_issued"] == 0
