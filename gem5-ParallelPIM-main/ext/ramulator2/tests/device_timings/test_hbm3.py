import pytest

import ramulator
import tests.device_timings.harness as device_timings


pytestmark = pytest.mark.device_timings


def make_dut(*, channel_id=0, **overrides):
    dram = ramulator.dram.HBM3(**{
        "org_preset": "HBM3_16Gb_8hi",
        "timing_preset": "HBM3_6400Mbps",
        **overrides,
    })
    return device_timings.DeviceUnderTest(dram, channel_id=channel_id)


def _addr(dut, *, sid, bankgroup, bank, row):
    return dut.addr_vec(
        PseudoChannel=0,
        Sid=sid,
        BankGroup=bankgroup,
        Bank=bank,
        Row=row,
        Column=0,
    )


def _refresh_addr(dut, command, *, bank, sid=0, bankgroup=0):
    if command in {"REFab", "RFMab"}:
        return dut.addr_vec(
            PseudoChannel=0,
            Sid=dut.ALL,
            BankGroup=dut.ALL,
            Bank=dut.ALL,
            Row=dut.ALL,
            Column=dut.ALL,
        )
    return _addr(dut, sid=sid, bankgroup=bankgroup, bank=bank, row=0)


def _issue_two_acts_then_col(dut, command, a0, a1, *, act_gap_timing):
    dut.issue("ACT", a0, clk=0)
    dut.issue("ACT", a1, clk=dut.timings[act_gap_timing])
    rcd_timing = "nRCDRD" if command == "RD" else "nRCDWR"
    col_clk = max(
        dut.get_first_ready_clk(command, a0, dut.timings[rcd_timing]),
        dut.get_first_ready_clk(command, a1, dut.timings[rcd_timing]),
    )
    dut.issue(command, a0, clk=col_clk)
    return col_clk


def test_hbm3_same_sid_diff_bankgroup_column_spacing_uses_nccds():
    dut = make_dut()
    a0 = _addr(dut, sid=0, bankgroup=0, bank=0, row=0)
    a1 = _addr(dut, sid=0, bankgroup=1, bank=0, row=1)

    rd_clk = _issue_two_acts_then_col(dut, "RD", a0, a1, act_gap_timing="nRRDS")
    nccd = dut.timings["nCCDS"]

    dut.assert_earliest_ready_at("RD", a1, rd_clk + nccd)


def test_hbm3_diff_sid_column_spacing_uses_nccdr():
    dut = make_dut()
    a0 = _addr(dut, sid=0, bankgroup=0, bank=0, row=0)
    a1 = _addr(dut, sid=1, bankgroup=0, bank=0, row=1)

    rd_clk = _issue_two_acts_then_col(dut, "RD", a0, a1, act_gap_timing="nRRDS")
    nccd = dut.timings["nCCDR"]

    dut.assert_earliest_ready_at("RD", a1, rd_clk + nccd)


def test_hbm3_same_sid_same_bankgroup_column_spacing_uses_nccdl():
    dut = make_dut()
    a0 = _addr(dut, sid=0, bankgroup=0, bank=0, row=0)
    a1 = _addr(dut, sid=0, bankgroup=0, bank=1, row=1)

    rd_clk = _issue_two_acts_then_col(dut, "RD", a0, a1, act_gap_timing="nRRDL")
    nccd = dut.timings["nCCDL"]

    dut.assert_earliest_ready_at("RD", a1, rd_clk + nccd)


def test_hbm3_diff_sid_write_column_spacing_uses_nccds():
    dut = make_dut()
    a0 = _addr(dut, sid=0, bankgroup=0, bank=0, row=0)
    a1 = _addr(dut, sid=1, bankgroup=1, bank=0, row=1)

    wr_clk = _issue_two_acts_then_col(dut, "WR", a0, a1, act_gap_timing="nRRDS")
    nccd = dut.timings["nCCDS"]

    dut.assert_earliest_ready_at("WR", a1, wr_clk + nccd)


@pytest.mark.parametrize(
    "preceding,following,same_bank,timing",
    [
        ("REFab", "RFMab", False, "nRFC"),
        ("REFab", "RFMpb", False, "nRFC"),
        ("RFMab", "REFab", False, "nRFMab"),
        ("RFMab", "REFpb", False, "nRFMab"),
        ("REFpb", "RFMab", False, "nRFCpb"),
        ("RFMpb", "REFab", False, "nRFMpb"),
        ("RFMpb", "RFMab", False, "nRFMpb"),
        ("REFpb", "RFMpb", False, "nRREFD"),
        ("RFMpb", "REFpb", False, "nRREFD"),
        ("REFpb", "REFpb", True, "nRFCpb"),
        ("RFMpb", "REFpb", True, "nRFMpb"),
        ("RFMpb", "RFMpb", True, "nRFMpb"),
        ("RFMpb", "RFMpb", False, "nRREFD"),
    ],
)
def test_hbm3_refresh_rfm_command_spacing(preceding, following, same_bank, timing):
    dut = make_dut()
    preceding_addr = _refresh_addr(dut, preceding, bank=0)
    following_addr = _refresh_addr(dut, following, bank=0 if same_bank else 1)

    dut.issue(preceding, preceding_addr, clk=0)

    dut.assert_earliest_ready_at(following, following_addr, dut.timings[timing])


@pytest.mark.parametrize("following_command", ["ACT", "REFpb"])
def test_hbm3_mixed_activate_refresh_commands_share_the_four_activate_window(
    following_command,
):
    dut = make_dut(
        nFAW=10,
        nRRDS=1,
        nRRDL=1,
        nRREFD=1,
        nRFCpb=1,
        nRFMpb=1,
    )
    sequence = [
        ("ACT", _addr(dut, sid=0, bankgroup=0, bank=0, row=0)),
        ("REFpb", _refresh_addr(dut, "REFpb", bank=1)),
        ("RFMpb", _refresh_addr(dut, "RFMpb", bank=2)),
        ("REFpb", _refresh_addr(dut, "REFpb", bank=3)),
    ]
    next_clk = 0
    for command, address in sequence:
        clk = dut.get_first_ready_clk(command, address, start=next_clk)
        dut.issue(command, address, clk=clk)
        next_clk = clk + 1

    following = _refresh_addr(
        dut, following_command, bank=0, bankgroup=1
    )
    first_cycles = int(type(dut.dram).command_cycles["ACT"] * dut.tick_multiplier)
    following_cycles = int(
        type(dut.dram).command_cycles.get(following_command, 1)
        * dut.tick_multiplier
    )
    earliest = (
        first_cycles - 1
        + dut.timings["nFAW"]
        - (following_cycles - 1)
    )

    dut.assert_earliest_ready_at(following_command, following, earliest)
