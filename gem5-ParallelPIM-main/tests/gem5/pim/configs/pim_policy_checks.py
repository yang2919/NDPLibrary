"""Checks C30 and C31 of the PIM memory, as configuration checks (no
simulation).

C30: a PimMemory whose size is not its capacity stops.
C31: the one-policy rule. A PimSystemMemory whose operating-system (OS)
memory has another policy or another interleave stops; with equal
settings, each policy with its default and with an overridden interleave
(refused where the policy fixes it), both memories decode an address
alike: their controllers carry the same address mapper and the same
organization, the channel mappers the same interleave, and gem5's port
ranges the same channel bits (the PIM memory's own decode is checked
against pim_addr.h at init). Also the four factories' defaults: every PIM
memory 32 GiB, with an OS memory of the same channel count and size.
Prints one line per check and exits with status 1 on a failure.

Run: build/RISCV_RAMULATOR2/gem5.opt tests/gem5/pim/configs/pim_policy_checks.py
"""

import json
import sys

from m5.objects import AddrRange

from gem5.components.memory import pim
from gem5.components.memory.ramulator2 import ramulator2_memory

fails = 0


def check(ok, what):
    global fails
    print(("PASS: " if ok else "FAIL: ") + what)
    fails += 0 if ok else 1


def raises(fn, text):
    try:
        fn()
    except ValueError as e:
        return text in str(e), str(e)
    return False, "no error"


class Line:
    def get_cache_line_size(self):
        return 64


# C30 (every default PIM memory is 32 GiB; only that size is accepted)
for std, size in (
    ("LPDDR6", "8GiB"),
    ("LPDDR6", "64GiB"),
    ("DDR5", "64GiB"),
    ("HBM4", "32GiB"),
):
    ok, msg = raises(
        lambda: pim.PimMemory(standard=std, size=size), "must be the capacity"
    )
    want = not (std == "HBM4" and size == "32GiB")
    check(
        ok == want,
        f"C30 {std} size {size}: "
        + ("refused" if ok else "accepted")
        + (f" ({msg[:70]})" if ok else ""),
    )


def ctrl_view(mem):
    cfg = json.loads(mem.ramulator2.ramulator_config)
    ms = cfg["memory_system"]
    out = []
    for c in ms["controllers"]:
        dram = dict(c["dram"])
        dram.pop("impl", None)  # LPDDR6 vs LPDDR6_PIM
        for k in list(dram):
            if (
                k.startswith("n")
                and k.endswith(("_AB", "_SB", "_DONE"))
                or "PIM" in k
            ):
                dram.pop(k)
        out.append(
            (
                json.dumps(c["addr_mapper"], sort_keys=True),
                json.dumps(
                    {k: dram[k] for k in ("org_preset", "rank") if k in dram},
                    sort_keys=True,
                ),
            )
        )
    return out, ms["channel_mapper"]


def build(policy, intlv, std="LPDDR6"):
    m = pim.pim_memory_system(std, policy=policy, channel_interleave=intlv)
    m.incorporate_memory(Line())
    m.set_memory_range([AddrRange(0x80000000, size=m.get_size())])
    return m


# C31: equal settings, each policy with its default and an override.
for policy in pim.POLICIES:
    for intlv in (None, 256):
        fixed = policy in ("RoRaBaChCo", "ChRaBaRoCo")
        if intlv is not None and fixed:
            ok, msg = raises(
                lambda: build(policy, intlv), "fixes the channel interleave"
            )
            check(ok, f"C31 {policy} interleave {intlv}: refused ({msg[:60]})")
            continue
        m = build(policy, intlv)
        os_c, os_ch = ctrl_view(m.os_memory)
        pim_c, pim_ch = ctrl_view(m.pim_memory)
        same_mapper = os_c == pim_c
        same_ch = os_ch == pim_ch
        osr = [r for r, _ in m.os_memory.get_mem_ports()]
        pr = [r for r, _ in m.pim_memory.get_mem_ports()]
        same_ports = (
            [(list(r.masks), int(r.intlvMatch)) for r in osr]
            == [(list(r.masks), int(r.intlvMatch)) for r in pr]
            if len(osr) > 1
            else True
        )
        base = int(m.pim_memory.get_range().start)
        check(
            same_mapper
            and same_ch
            and same_ports
            and base % m.pim_memory.get_size() == 0,
            f"C31 {policy} interleave {m.pim_memory.get_interleave()}: mapper "
            f"{os_c[0][0]}, channel mapper {pim_ch}, ports alike: "
            f"{same_mapper}/{same_ch}/{same_ports}, PIM base {base:#x}",
        )


# C31: different policy or interleave in the two memories.
def mixed(os_policy, os_intlv, pim_policy, pim_intlv):
    p = pim.PimMemory(policy=pim_policy, channel_interleave=pim_intlv)
    o = ramulator2_memory(
        "LPDDR6",
        p.get_channels(),
        p.get_size(),
        org_preset="LPDDR6_16Gb_x12",
        addr_mapper=os_policy,
        interleaving_size=os_intlv,
    )
    s = pim.PimSystemMemory(p, os_memory=o)
    s.incorporate_memory(Line())


for os_p, os_i, p_p, p_i, text in (
    ("RoBaRaCoCh", 64, "RoRaBaCoCh", 64, "both must use one policy"),
    ("MOP4CLXOR", 64, "RoBaRaCoCh", 64, "both must use one policy"),
    ("RoBaRaCoCh", 128, "RoBaRaCoCh", 64, "differs from the PIM memory"),
    ("MOP4CLXOR", 64, "MOP4CLXOR", 256, "differs from the PIM memory"),
):
    ok, msg = raises(lambda: mixed(os_p, os_i, p_p, p_i), text)
    check(
        ok, f"C31 OS {os_p}/{os_i} with PIM {p_p}/{p_i}: refused ({msg[:70]})"
    )
ok, msg = raises(lambda: mixed("RoBaRaCoCh", 64, "RoBaRaCoCh", 64), "zzz")
check(
    msg == "no error", "C31 OS RoBaRaCoCh/64 with PIM RoBaRaCoCh/64: accepted"
)

# The factory defaults: the OS memory alike the PIM memory.
for f, std, ch, size in (
    (pim.pim_lpddr6_memory, "LPDDR6", 16, 32),
    (pim.pim_ddr5_memory, "DDR5", 4, 32),
    (pim.pim_gddr7_memory, "GDDR7", 32, 32),
    (pim.pim_hbm4_memory, "HBM4", 32, 32),
):
    m = f()
    check(
        m.pim_memory.get_channels() == ch
        and m.pim_memory.get_size() == size << 30
        and m.get_size() == size << 30
        and m.os_memory._channels == ch,
        f"factory {std}: {ch} channels, PIM and OS memory {size} GiB",
    )
print(f"pim_policy_checks: {fails} failures")
sys.exit(1 if fails else 0)
