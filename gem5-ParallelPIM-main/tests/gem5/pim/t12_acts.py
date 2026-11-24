#!/usr/bin/env python3
"""The per-bank ACT clocks of case T12 (Baseline all-bank opening).

Test C28 compares the ACT commands of one Baseline all-bank job in gem5 with
this sequence. It recomputes ``t12_baseline_act_sequence`` of
``ext/ramulator2/tests/device_timings/pim_cases.py`` with that file's own
``Std`` helpers: every bank of unit 0 opened in flat bank order on an idle
device from clock 0, each at its earliest legal clock, and returns, per
standard, the clock of each ACT command (every phase) and the device's tick
multiplier.

It needs Ramulator 2.1's Python bindings, so it runs in a scratch copy of
``ext/ramulator2`` built with them (CMake, ``RAMULATOR_PYTHON_BINDINGS``),
from that copy's top directory::

    cd <scratch>/ramulator2 && PYTHONPATH=python python3 \\
        <gem5>/tests/gem5/pim/t12_acts.py > t12.json
"""

import json
import os
import sys

sys.path.insert(0, os.getcwd())

import tests.device_timings.pim_cases as cases  # noqa: E402


def acts(std):
    s = cases.Std(std, cases.make_dut(std))
    phases = s.desc.act_phases if not s.two_phase else s.actsb
    out = []
    clk = 0
    for bank in s.banks():
        for phase in phases:
            clk = s.issue(phase, bank, clk)
            out.append([phase, clk])
    return {"acts": out, "tick_multiplier": s.dut.tick_multiplier}


if __name__ == "__main__":
    print(json.dumps({std: acts(std) for std in cases.PRESETS}))
