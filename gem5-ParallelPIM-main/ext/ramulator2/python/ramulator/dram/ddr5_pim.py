"""DDR5 with gem5's processing-in-memory (PIM) commands.

Added for gem5: a subclass of DDR5, as DDR5_VRR is of DDR5. The PIM commands,
parameters and rows come from ramulator.dram.pim; the stock DDR5 is unchanged.
"""

from ramulator.dram.ddr5 import DDR5
from ramulator.dram.pim import DESCRIPTORS, apply_pim, resolve_pim_defaults, resolve_pim_timings


class DDR5_PIM(DDR5):
    name = "DDR5_PIM"

    @classmethod
    def resolve_secondary_timings(cls, timing_dict, org_dict):
        super().resolve_secondary_timings(timing_dict, org_dict)
        resolve_pim_timings(cls, timing_dict)

    def resolve(self):
        # PIM defaults follow the host values after the overrides.
        return resolve_pim_defaults(self, super().resolve())


apply_pim(DDR5_PIM, DDR5, DESCRIPTORS["DDR5"])
