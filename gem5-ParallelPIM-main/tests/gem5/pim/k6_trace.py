"""K6: issued-command trace of stock standards with an empty host filter
and the write drain off, to compare before and after the PIM sequencer's
hooks in ``ControllerBase`` (``ext/ramulator2/README.gem5``, item 6).
Run in a scratch ramulator2 copy:
  PYTHONPATH=python:. python k6_trace.py <out.json>

``run_p4r.py``'s ``r2suites`` group runs it and compares the SHA-256 of
its output with ``K6_SHA256``, the trace taken before those hooks were
added.
"""

import json
import random
import sys

import ramulator

import tests.controller_scheduling.harness as cs

# The stdlib component's presets (src/python/gem5/components/memory/ramulator2.py).
PRESETS = {
    "DDR4": (
        "DDR4_8Gb_x8",
        "DDR4_2400R",
        "GenericDDR",
        dict(rank=2, nCL=17, nRCD=17, nRP=17),
    ),
    "DDR5": ("DDR5_16Gb_x8", "DDR5_4800AN", "GenericDDR", {}),
    "LPDDR5": ("LPDDR5_8Gb_x16", "LPDDR5_6400", "LPDDR5", {}),
    "LPDDR6": ("LPDDR6_16Gb_x12", "LPDDR6_10667_BL24", "LPDDR6", {}),
    "GDDR6": ("GDDR6_8Gb_x16", "GDDR6_14000_1250mV_double", "GenericDDR", {}),
    "GDDR7": ("GDDR7_16Gb_x8", "GDDR7_28000_PAM3", "GDDR7", {}),
    "HBM2": ("HBM2_2Gb", "HBM2_2000Mbps", "HBM12", {}),
    "HBM4": ("HBM4_32Gb_8Hi", "HBM4_8000Mbps", "HBM34", {}),
}


def run(std, row_policy, seed, cycles=20000):
    org, tim, ctrl, kw = PRESETS[std]
    dram = getattr(ramulator.dram, std)(
        org_preset=org, timing_preset=tim, **kw
    )
    controller = getattr(ramulator.controller, ctrl)(
        dram=dram,
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.AllBank(),
        row_policy=row_policy,
        addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
        read_buffer_size=16,
        write_buffer_size=16,
        controller_plugins=[],
    )
    dut = cs.ControllerUnderTest(controller)
    rng = random.Random(seed)
    sizes = [
        dut.org.get(n.lower(), 1) if n != "Column" else dut.org["column"]
        for n in dut.level_names
    ]
    hot_rows = [
        rng.randrange(sizes[dut.level_names.index("Row")]) for _ in range(4)
    ]
    trace = []
    pending = None
    for clk in range(cycles):
        if pending is None and rng.random() < 0.3:
            vec = [0] + [rng.randrange(max(1, s)) for s in sizes[1:]]
            vec[dut.level_names.index("Row")] = rng.choice(hot_rows)
            vec[dut.level_names.index("Column")] = (
                rng.randrange(sizes[-1] // 64 + 1) * 64 % sizes[-1]
            )
            pending = ("Read" if rng.random() < 0.6 else "Write", vec)
        if pending is not None:
            try:
                dut.send_request(pending[0], pending[1])
                pending = None
            except RuntimeError:
                pass
        for c in dut.tick():
            trace.append([c.clk, c.command, list(c.addr_vec), c.type_id])
    return {"trace": trace, "stats": dut.stats()}


out = {}
for std in PRESETS:
    for name, rp in (
        ("Open", ramulator.row_policy.Open),
        ("ClosedCAP", ramulator.row_policy.ClosedCAP),
    ):
        for seed in (1, 2):
            out[f"{std}-{name}-{seed}"] = run(std, rp(), seed)
json.dump(out, open(sys.argv[1], "w"), sort_keys=True, default=str)
print({k: len(v["trace"]) for k, v in out.items()})
