"""LPDDR6 with gem5's processing-in-memory (PIM) commands.

Added for gem5: a subclass of LPDDR6, as DDR5_VRR is of DDR5. The PIM commands,
parameters and rows come from ramulator.dram.pim; the stock LPDDR6 is unchanged.
"""

from ramulator.dram.lpddr6 import LPDDR6
from ramulator.dram.pim import DESCRIPTORS, apply_pim, resolve_pim_defaults, resolve_pim_timings


class LPDDR6_PIM(LPDDR6):
    name = "LPDDR6_PIM"

    @classmethod
    def resolve_secondary_timings(cls, timing_dict, org_dict):
        super().resolve_secondary_timings(timing_dict, org_dict)
        resolve_pim_timings(cls, timing_dict)

    def resolve(self):
        # PIM defaults follow the host values after the overrides.
        return resolve_pim_defaults(self, super().resolve())


apply_pim(LPDDR6_PIM, LPDDR6, DESCRIPTORS["LPDDR6"])
