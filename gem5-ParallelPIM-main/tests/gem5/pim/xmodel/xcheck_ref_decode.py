"""Cross-check of gem5's three address-mapping policies, as ported to
Ramulator 2.1, against libpim/tests/ref_decode.py vectors, on the PIM
descriptors' devices, as the vectors' geometries
(libpim/gen_pim_layout.py). Run in the scratch ramulator2 copy:
  PYTHONPATH=python:. python xcheck_ref_decode.py <vectors file>
"""

import sys

from ramulator._ramulator_test import _ChannelMapperUnderTest
from ramulator.dram.pim import DESCRIPTORS

from tests.unit_tests.test_gem5_policies import (
    make_dut,
    make_mapper,
)

POL = ("RoRaBaChCo", "RoRaBaCoCh", "RoCoRaBaCh")
STD = {"ddr5": "DDR5", "lpddr6": "LPDDR6", "gddr7": "GDDR7", "hbm4": "HBM4"}
FIELDS = ("ch", "unit", "sid", "bg", "bank", "row", "col")
LEVEL = {
    "Channel": "ch",
    "Rank": "unit",
    "PseudoChannel": "unit",
    "Sid": "sid",
    "BankGroup": "bg",
    "Bank": "bank",
    "Row": "row",
    "Column": "col",
}


def preset(std):
    d = DESCRIPTORS[std]
    p = d.presets
    kw = dict(p["org_overrides"])
    kw.update({k: v for k, (v, _) in d.host_overrides.items()})
    return p["org"], p["timing"], kw


cfg = None
n = bad = configs = 0
for line in open(sys.argv[1]):
    if line.startswith("#"):
        continue
    t = line.split()
    if t[0] == "geom":
        std, pol, ch, intlv = t[1], int(t[2]), int(t[3]), int(t[4])
        cols, colb = int(t[10]), int(t[11])
        if pol > 2:
            cfg = None
            continue
        policy = POL[pol]
        dut = make_dut(
            STD[std], make_mapper(policy, ch, intlv), preset(STD[std])
        )
        low = cols * colb if policy == "RoRaBaChCo" else intlv
        port = _ChannelMapperUnderTest(
            {"impl": "Gem5PortInterleave", "interleave_size": low},
            ch,
            colb.bit_length() - 1,
        )
        cfg = (std, policy, ch, intlv, dut, port)
        configs += 1
        continue
    if cfg is None:
        continue
    std, policy, ch, intlv, dut, port = cfg
    off = int(t[0], 16)
    exp = dict(zip(FIELDS + ("byte",), map(int, t[1:])))
    m = port.apply(off, ingress_id=exp["ch"], source_id=0)
    vec = list(dut._cpp.map_address(m["intra_channel_addr"]))
    vec[0] = m["channel"]
    got = {f: 0 for f in FIELDS}
    for name, v in zip(dut.level_names, vec):
        got[LEVEL[name]] = v
    n += 1
    if any(got[f] != exp[f] for f in FIELDS):
        bad += 1
        if bad <= 10:
            print(
                f"MISMATCH {std} {policy} ch={ch} intlv={intlv} off={off:#x}: got {got} ref {exp}"
            )
print(f"{configs} gem5-policy configurations, {n} vectors, {bad} mismatches")
sys.exit(1 if bad else 0)
