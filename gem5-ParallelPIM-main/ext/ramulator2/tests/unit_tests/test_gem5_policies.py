"""Decode test of gem5's address-mapping policies ported to Ramulator 2.1.

Added for gem5. Each address goes through
Gem5PortInterleave, which takes the channel from the port as gem5's
interleaved port ranges select it, and then through the controller's
RoRaBaChCo, RoRaBaCoCh or RoCoRaBaCh mapper. The reference transcribes gem5:
the channel bits of the port ranges (gem5's ChanneledMemory: one rank-wide row
above the column for RoRaBaChCo, the interleave otherwise) removed as
AddrRange::getOffset does, DRAMInterface::decodePacket
(src/mem/dram_interface.cc:856-895) with burstsPerStripe of
src/mem/mem_interface.cc:64-65, and the flat bank split as bank group =
bank % bank groups (src/mem/dram_interface.cc:1140-1151), then bank and stack
ID. gem5's rank is the rank or the pseudo channel.
"""

import random

import pytest

import ramulator
import tests.controller_scheduling.harness as cs
from ramulator._ramulator_test import _ChannelMapperUnderTest

# Large organizations of the PIM standards (earlier PIM defaults; kept, as two
# DDR5 ranks and four HBM4 stack IDs exercise more address fields than the
# current default devices, whose decode gem5's PimRamulator2 checks at
# start-up) and DDR4 for the cross-check with native gem5.
STANDARDS = {
    "DDR4": ("DDR4_8Gb_x8", "DDR4_2400R", {"rank": 2}, "make_generic_ddr"),
    "DDR5": ("DDR5_32Gb_x4", "DDR5_4800AN", {"rank": 2}, "make_generic_ddr"),
    "LPDDR6": ("LPDDR6_16Gb_x12", "LPDDR6_10667_BL24", {}, None),
    # GDDR7 64 Gb has no refresh timings (Ramulator 2.1 gives them for 16 Gb
    # only); they do not affect decoding, so placeholders.
    "GDDR7": (
        "GDDR7_64Gb_x8",
        "GDDR7_28000_PAM3",
        {"nRFCab": 100, "nRFCpb": 20, "nRFMab": 100, "nRFMpb": 20},
        "make_gddr7",
    ),
    "HBM4": ("HBM4_32Gb_16Hi", "HBM4_8000Mbps_folded_banks_baseline", {}, "make_hbm34"),
}

POLICIES = ("RoRaBaChCo", "RoRaBaCoCh", "RoCoRaBaCh")


def log2(x):
    assert x > 0 and x & (x - 1) == 0, x
    return x.bit_length() - 1


def make_dut(standard, mapper, preset=None):
    """preset: (organization, timing, overrides) replacing STANDARDS's."""
    org, timing, kw, factory = STANDARDS[standard]
    if preset is not None:
        org, timing, kw = preset
    dram = getattr(ramulator.dram, standard)(org_preset=org, timing_preset=timing, **kw)
    if factory is None:
        controller = ramulator.controller.LPDDR6(
            scheduler=ramulator.scheduler.FRFCFS(),
            refresh_manager=ramulator.refresh_manager.NoRefresh(),
            row_policy=ramulator.row_policy.Open(),
            addr_mapper=mapper,
            dram=dram,
            controller_plugins=[],
        )
        return cs.ControllerUnderTest(controller)
    return getattr(cs.ControllerUnderTest, factory)(dram, addr_mapper=mapper)


class Gem5Reference:
    """gem5's view of one channel of the organization."""

    def __init__(self, dut):
        dram = dut.dram
        org = dut.org
        cls = type(dram)
        self.levels = list(dut.level_names)
        self.tx = cls.data_payload_bytes or cls.internal_prefetch_size * org["channel_width"] // 8
        self.cols = org["column"] // cls.internal_prefetch_size  # transactions per row
        self.row_bytes = self.cols * self.tx
        self.ranks = org.get("rank") or org.get("pseudochannel") or 1
        self.bgs = org.get("bankgroup", 1)
        self.banks = org["bank"]
        self.sids = org.get("sid", 1)
        self.rows = org["row"]
        self.capacity = self.row_bytes * self.rows * self.banks * self.bgs * self.sids * self.ranks

    def channel_low_bit(self, policy, interleave):
        return log2(self.row_bytes if policy == "RoRaBaChCo" else interleave)

    def decode(self, policy, channels, interleave, addr):
        # Channel bits and AddrRange::getOffset (src/base/addr_range.hh).
        low = self.channel_low_bit(policy, interleave)
        width = log2(channels)
        channel = (addr >> low) & (channels - 1)
        offset = ((addr >> (low + width)) << low) | (addr & ((1 << low) - 1))

        a = offset // self.tx  # burst address
        bursts_per_row = self.cols
        banks_per_rank = self.bgs * self.banks * self.sids
        if policy in ("RoRaBaChCo", "RoRaBaCoCh"):
            col = a % bursts_per_row
            a //= bursts_per_row
            bank = a % banks_per_rank
            a //= banks_per_rank
            rank = a % self.ranks
            a //= self.ranks
            row = a % self.rows
        else:
            stripe = interleave // self.tx if channels > 1 else 1
            if stripe > bursts_per_row:
                col = a % bursts_per_row
                a //= bursts_per_row
            else:
                col = a % stripe
                a //= stripe
            bank = a % banks_per_rank
            a //= banks_per_rank
            rank = a % self.ranks
            a //= self.ranks
            if stripe < bursts_per_row:
                col += (a % (bursts_per_row // stripe)) * stripe
                a //= bursts_per_row // stripe
            row = a % self.rows
        coords = {
            "Channel": channel,
            "Rank": rank,
            "PseudoChannel": rank,
            "BankGroup": bank % self.bgs,
            "Bank": (bank // self.bgs) % self.banks,
            "Sid": bank // (self.bgs * self.banks),
            "Row": row,
            "Column": col,
        }
        return [coords[name] for name in self.levels]


def make_mapper(policy, channels, interleave):
    if policy == "RoCoRaBaCh":
        # gem5's stripe is the interleave only when the range is interleaved.
        return ramulator.addr_mapper.RoCoRaBaCh(stripe_size=interleave if channels > 1 else 0)
    return getattr(ramulator.addr_mapper, policy)()


def interleaves(ref, policy):
    if policy == "RoRaBaChCo":
        return [None]  # fixed: one rank-wide row
    # One cache line up to beyond one row (RoCoRaBaCh then strips a whole row).
    return [64, 256, ref.row_bytes, 2 * ref.row_bytes]


def check(standard, policy, channels, samples=400, seed=1):
    dut = make_dut(standard, make_mapper(policy, 1, 64))
    ref = Gem5Reference(dut)
    rng = random.Random(seed)
    total = ref.capacity * channels
    checked = 0
    for interleave in interleaves(ref, policy):
        intlv = interleave or ref.row_bytes
        dut = make_dut(standard, make_mapper(policy, channels, intlv))
        port = _ChannelMapperUnderTest(
            {
                "impl": "Gem5PortInterleave",
                "interleave_size": 1 << ref.channel_low_bit(policy, intlv),
            },
            channels,
            log2(ref.tx),
        )
        # Random addresses plus the first and last transaction of the memory.
        addrs = [0, total - 1] + [rng.randrange(total) for _ in range(samples)]
        for addr in addrs:
            expected = ref.decode(policy, channels, intlv, addr)
            ch = expected[0]
            mapped = port.apply(addr, ingress_id=ch, source_id=0)
            got = list(dut._cpp.map_address(mapped["intra_channel_addr"]))
            got[0] = mapped["channel"]
            assert got == expected, (
                f"{standard} {policy} channels={channels} interleave={intlv} addr={addr:#x}: "
                f"got {dict(zip(ref.levels, got))}, gem5 {dict(zip(ref.levels, expected))}"
            )
            checked += 1
    return checked


@pytest.mark.parametrize("standard", list(STANDARDS))
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("channels", [1, 2, 8])
def test_gem5_policy_decode(standard, policy, channels):
    assert check(standard, policy, channels) > 0


@pytest.mark.parametrize("standard", list(STANDARDS))
@pytest.mark.parametrize("policy", POLICIES)
def test_gem5_policy_round_trip_is_a_bijection(standard, policy):
    """Small sweep: distinct addresses of one channel give distinct coordinates."""
    dut = make_dut(standard, make_mapper(policy, 1, 64))
    ref = Gem5Reference(dut)
    seen = set()
    n = ref.cols * ref.bgs * ref.banks * ref.sids * ref.ranks * 4  # four rows of every bank
    for t in range(n):
        vec = tuple(dut._cpp.map_address(t * ref.tx))
        assert vec not in seen
        seen.add(vec)
    assert len(seen) == n


def test_hbm4_order_under_gem5_policies():
    """HBM4 order, least significant first: column, bank group, bank, stack
    ID, pseudo channel, row."""
    dut = make_dut("HBM4", ramulator.addr_mapper.RoRaBaChCo())
    ref = Gem5Reference(dut)
    names = list(ref.levels)
    step = ref.row_bytes
    order = []
    levels = (("BankGroup", ref.bgs), ("Bank", ref.banks), ("Sid", ref.sids),
              ("PseudoChannel", ref.ranks))
    for name, count in levels:
        vec = list(dut._cpp.map_address(step))
        assert vec[names.index(name)] == 1, (name, vec)
        order.append(name)
        step *= count
    assert order == ["BankGroup", "Bank", "Sid", "PseudoChannel"]
    vec = list(dut._cpp.map_address(step))
    assert vec[names.index("Row")] == 1


@pytest.mark.parametrize("stripe_size", [16, 48, 96])
def test_ro_co_ra_ba_ch_rejects_bad_stripe(stripe_size):
    with pytest.raises(RuntimeError, match="stripe_size"):
        make_dut("DDR5", ramulator.addr_mapper.RoCoRaBaCh(stripe_size=stripe_size))
