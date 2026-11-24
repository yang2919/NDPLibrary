"""HBM4 with gem5's processing-in-memory (PIM) commands.

Added for gem5: a subclass of HBM4, as DDR5_VRR is of DDR5. The PIM commands,
parameters and rows come from ramulator.dram.pim; the stock HBM4 is unchanged.
"""

from ramulator.dram.hbm4 import HBM4
from ramulator.dram.pim import DESCRIPTORS, apply_pim, resolve_pim_defaults, resolve_pim_timings


class HBM4_PIM(HBM4):
    name = "HBM4_PIM"

    @classmethod
    def resolve_secondary_timings(cls, timing_dict, org_dict):
        super().resolve_secondary_timings(timing_dict, org_dict)
        resolve_pim_timings(cls, timing_dict)

    def resolve(self):
        # PIM defaults follow the host values after the overrides.
        return resolve_pim_defaults(self, super().resolve())


apply_pim(HBM4_PIM, HBM4, DESCRIPTORS["HBM4"])
