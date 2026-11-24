import math

from ramulator.dram.ddr4 import DDR4
from ramulator.dram.spec import TimingConstraint


class DDR4_VRR(DDR4):
    name = "DDR4_VRR"

    commands = DDR4.commands + ["VRR"]

    timing_params = DDR4.timing_params + ["nVRR"]
    timing_constraints = DDR4.timing_constraints + [
        TimingConstraint(level="Bank", preceding=["VRR"], following=["ACT", "VRR"], latency="nVRR"),
        TimingConstraint(level="Bank", preceding=["ACT"], following=["VRR"], latency="nRC"),
        TimingConstraint(level="Rank", preceding=["PREpb", "PREab"], following=["VRR"], latency="nRP"),
    ]

    @classmethod
    def resolve_secondary_timings(cls, timing_dict, org_dict):
        super().resolve_secondary_timings(timing_dict, org_dict)
        timing_dict["nVRR"] = cls._resolve_nVRR(timing_dict["tCK_ps"])

    @staticmethod
    def _resolve_nVRR(tCK_ps):
        return math.ceil(280_000 / tCK_ps)  # Ramulator guesstimate


# Inherit all DDR4 presets; nVRR is resolved from tCK_ps.
DDR4_VRR.org_presets = DDR4.org_presets
DDR4_VRR.timing_presets = DDR4.timing_presets
