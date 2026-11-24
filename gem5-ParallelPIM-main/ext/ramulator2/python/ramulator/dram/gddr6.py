"""
 GDDR6 DRAM module

"""

import math

from ramulator.dram.spec import DRAMStandard, TimingConstraint


class GDDR6(DRAMStandard):
    """ GDDR6 DRAM module """

    name = "GDDR6"
    internal_prefetch_size = 16
    read_latency = "nCL + nBL"


    # Hierarchy
    levels = {
        "Channel" : "N_A",
        "BankGroup" : "N_A",
        "Bank" : "Closed",
        "Row" : "Closed",
        "Column" : "N_A"
    }


    # Commands
    commands = [
        "ACT", "PREab", "PREpb",
        "RD", "WR", "RDA", "WRA",
        "REFab", "REFpb",
    ]

    #Taken from Table 30 - Trueth Table Commands
    command_cycles = {"ACT": 1, "RD": 1, "RDA": 1, "WR": 1, "WRA": 1}


    # States
    states = ["Opened", "Closed", "N_A"]


    # Timing Parameters
    timing_params = [
        "rate", "nBL",
        "nCL", "nRCDRD", "nRCDWR", "nRP", "nRAS", "nRC",
        "nWR", "nRTP", "nCWL",
        "nCCDS", "nCCDL",
        "nRRDS", "nRRDL",
        "nWTRS", "nWTRL", "nRTW",
        "nFAW",
        "nRFCpb", "nRREFD", "nREFI", "nREFIpb",  # Refresh
        "tCK_ps",
        "nRFCab",
        "nPPD",
    ]


    supported_requests = {"Read": "RD", "Write": "WR"}


    timing_constraints = [
        # Channel
        # JESD250C Section 4.3, Figure 9, and Section 6.4 define the burst and
        # column-command spacing. nCCDS equals nBL for the modeled presets.
        TimingConstraint(level="Channel", preceding=["RD", "RDA"], following=["RD", "RDA"], latency="nBL"),
        TimingConstraint(level="Channel", preceding=["WR", "WRA"], following=["WR", "WRA"], latency="nBL"),

        # Channel
        TimingConstraint(level="Channel", preceding=["RD", "RDA"], following=["RD", "RDA"], latency="nCCDS"),
        TimingConstraint(level="Channel", preceding=["WR", "WRA"], following=["WR", "WRA"], latency="nCCDS"),
        TimingConstraint(level="Channel", preceding=["RD", "RDA"], following=["WR", "WRA"], latency="nRTW"),
        TimingConstraint(
            level="Channel",
            preceding=["WR", "WRA"],
            following=["RD", "RDA"],
            latency="nCWL + nBL + nWTRS",
        ),
        TimingConstraint(level="Channel", preceding=["RD", "RDA"], following=["PREab"], latency="nRTP"),
        TimingConstraint(level="Channel", preceding=["WR", "WRA"], following=["PREab"], latency="nCWL + nBL + nWR"),
        TimingConstraint(level="Channel", preceding=["ACT"], following=["ACT"], latency="nRRDS"),
        TimingConstraint(level="Channel", preceding=["ACT"], following=["ACT", "REFpb"], latency="nFAW", window=4),
        TimingConstraint(level="Channel", preceding=["ACT"], following=["PREab"], latency="nRAS"),
        TimingConstraint(level="Channel", preceding=["PREab"], following=["ACT"], latency="nRP"),
        TimingConstraint(level="Channel", preceding=["PREab", "PREpb"], following=["PREab", "PREpb"], latency="nPPD"),

        # RAS <-> REFab
        TimingConstraint(level="Channel", preceding=["ACT"], following=["REFab"], latency="nRC"),
        TimingConstraint(level="Channel", preceding=["PREab"], following=["REFab"], latency="nRP"),
        TimingConstraint(level="Channel", preceding=["RDA"], following=["REFab"], latency="nRP + nRTP"),
        TimingConstraint(level="Channel", preceding=["WRA"], following=["REFab"], latency="nCWL + nBL + nWR + nRP"),
        TimingConstraint(level="Channel", preceding=["REFab"], following=[
            "ACT", "PREab", "PREpb", "RD", "WR", "RDA", "WRA", "REFab", "REFpb",
        ], latency="nRFCab"),

        # RAS <-> REFpb
        TimingConstraint(level="Channel", preceding=["ACT"], following=["REFpb"], latency="nRRDS"),
        TimingConstraint(level="Channel", preceding=["PREab"], following=["REFpb"], latency="nRP"),
        TimingConstraint(level="Channel", preceding=["RDA"], following=["REFpb"], latency="nRP + nRTP"),
        TimingConstraint(level="Channel", preceding=["WRA"], following=["REFpb"], latency="nCWL + nBL + nWR + nRP"),
        TimingConstraint(level="Channel", preceding=["REFpb"], following=["ACT"], latency="nRREFD"),
        TimingConstraint(level="Channel", preceding=["REFpb"], following=["REFpb"], latency="nRREFD"),
        TimingConstraint(level="Channel", preceding=["REFpb"], following=["REFab"], latency="nRFCpb"),

        # Same Bank Group
        TimingConstraint(level="BankGroup", preceding=["RD", "RDA"], following=["RD", "RDA"], latency="nCCDL"),
        TimingConstraint(level="BankGroup", preceding=["WR", "WRA"], following=["WR", "WRA"], latency="nCCDL"),
        TimingConstraint(
            level="BankGroup",
            preceding=["WR", "WRA"],
            following=["RD", "RDA"],
            latency="nCWL + nBL + nWTRL",
        ),
        TimingConstraint(level="BankGroup", preceding=["ACT"], following=["ACT", "REFpb"], latency="nRRDL"),

        # Bank
        TimingConstraint(level="Bank", preceding=["ACT"], following=["ACT"], latency="nRC"),
        TimingConstraint(level="Bank", preceding=["ACT"], following=["RD", "RDA"], latency="nRCDRD"),
        TimingConstraint(level="Bank", preceding=["ACT"], following=["WR", "WRA"], latency="nRCDWR"),
        TimingConstraint(level="Bank", preceding=["ACT"], following=["PREpb"], latency="nRAS"),
        TimingConstraint(level="Bank", preceding=["PREpb"], following=["ACT"], latency="nRP"),

        TimingConstraint(level="Bank", preceding=["RD"], following=["PREpb"], latency="nRTP"),
        TimingConstraint(level="Bank", preceding=["WR"], following=["PREpb"], latency="nCWL + nBL + nWR"),
        TimingConstraint(level="Bank", preceding=["RDA"], following=["ACT"], latency="nRTP + nRP"),
        TimingConstraint(level="Bank", preceding=["WRA"], following=["ACT"], latency="nCWL + nBL + nWR + nRP"),

        # Bank RAS <-> REFpb
        TimingConstraint(level="Bank", preceding=["ACT"], following=["REFpb"], latency="nRC"),
        TimingConstraint(level="Bank", preceding=["PREpb"], following=["REFpb"], latency="nRP"),
        TimingConstraint(level="Bank", preceding=["RDA"], following=["REFpb"], latency="nRP + nRTP"),
        TimingConstraint(level="Bank", preceding=["WRA"], following=["REFpb"], latency="nCWL + nBL + nWR + nRP"),
        TimingConstraint(level="Bank", preceding=["REFpb"], following=["ACT", "REFpb"], latency="nRFCpb"),

    ]

    @classmethod
    def resolve_secondary_timings(cls, timing_dict, org_dict):
        tCK_ps = timing_dict["tCK_ps"]
        device_density = org_dict["device_density"]
        timing_dict["nRTW"] = cls._resolve_nRTW(
            timing_dict["nCL"], timing_dict["nBL"], timing_dict["nCWL"]
        )
        timing_dict["nRFCpb"] = cls._resolve_nRFCpb(
            device_density, tCK_ps
        )
        timing_dict["nREFI"] = cls._resolve_nREFI(tCK_ps)
        timing_dict["nREFIpb"] = cls._resolve_nREFIpb(
            org_dict["bankgroup"], org_dict["bank"], tCK_ps
        )
        timing_dict["nRFCab"] = cls._resolve_nRFCab(
            device_density, tCK_ps
        )

    @staticmethod
    def _resolve_nRTW(nCL, nBL, nCWL):
        # Samsung K4Z80325BC Rev. 1.3 Tables 92 and 93 define tRTW;
        # JESD250C Table 73 Note 38 leaves only bus turnaround to the system.
        bus_turnaround = 0  # Ramulator guesstimate
        return nCL + 2 * nBL - nCWL + 3 + bus_turnaround

    @staticmethod
    def _resolve_nRFCpb(device_density, tCK_ps):
        # Samsung K4Z80325BC Rev. 1.3 Tables 92 and 93 (8 Gb device).
        tRFCpb_ps = {8192: 60_000}.get(device_density)
        if tRFCpb_ps is None:
            return -1
        return math.ceil(tRFCpb_ps / tCK_ps)

    @staticmethod
    def _resolve_nREFI(tCK_ps):
        return math.floor(1_900_000 / tCK_ps)

    @staticmethod
    def _resolve_nREFIpb(bankgroup, bank, tCK_ps):
        # JESD250C Section 7.17 and Table 73: REFpb uses tREFI / 16.
        bank_count = bankgroup * bank
        if bank_count != 16:
            return -1
        return math.floor(1_900_000 / (bank_count * tCK_ps))

    @staticmethod
    def _resolve_nRFCab(device_density, tCK_ps):
        # Samsung K4Z80325BC Rev. 1.3 Tables 92 and 93 (8 Gb device).
        tRFCab_ps = {8192: 120_000}.get(device_density)
        if tRFCab_ps is None:
            return -1
        return math.ceil(tRFCab_ps / tCK_ps)

# JESD250C Table 19 defines two channels per GDDR6 device. One Ramulator
# GDDR6 instance models one channel; channels_per_device describes the device.
GDDR6.org_presets = {
    "GDDR6_8Gb_x8": {
        "device_density": 8192,
        "channel_density": 4096,
        "channels_per_device": 2,
        "dq": 8,
        "channel_width": 16,
        "bankgroup": 4,
        "bank": 4,
        "row": 1<<14,
        "column": 1<<11,
    },
    "GDDR6_8Gb_x16": {
        "device_density": 8192,
        "channel_density": 4096,
        "channels_per_device": 2,
        "dq": 16,
        "channel_width": 16,
        "bankgroup": 4,
        "bank": 4,
        "row": 1<<14,
        "column": 1<<10,
    },
    "GDDR6_16Gb_x8": {
        "device_density": 16384,
        "channel_density": 8192,
        "channels_per_device": 2,
        "dq": 8,
        "channel_width": 16,
        "bankgroup": 4,
        "bank": 4,
        "row": 1<<15,
        "column": 1<<11,
    },
    "GDDR6_16Gb_x16": {
        "device_density": 16384,
        "channel_density": 8192,
        "channels_per_device": 2,
        "dq": 16,
        "channel_width": 16,
        "bankgroup": 4,
        "bank": 4,
        "row": 1<<14,
        "column": 1<<11,
    },
    "GDDR6_32Gb_x8": {
        "device_density": 32768,
        "channel_density": 16384,
        "channels_per_device": 2,
        "dq": 8,
        "channel_width": 16,
        "bankgroup": 4,
        "bank": 4,
        "row": 1<<16,
        "column": 1<<11,
    },
    "GDDR6_32Gb_x16": {
        "device_density": 32768,
        "channel_density": 16384,
        "channels_per_device": 2,
        "dq": 16,
        "channel_width": 16,
        "bankgroup": 4,
        "bank": 4,
        "row": 1<<15,
        "column": 1<<11,
    },
}
GDDR6.timing_presets = {
    # Samsung K4Z80325BC Rev. 1.3, Tables 27, 91, and 92 (8 Gb, DDR WCK).
    "GDDR6_14000_1350mV_double": {
        "rate": 14000, "nBL": 2, "nCL": 24,
        "nRCDRD": 27, "nRCDWR": 16,
        "nRP": 27, "nRAS": 53, "nRC": 79,
        "nWR": 27, "nRTP": 4, "nCWL": 6,
        "nCCDS": 2, "nCCDL": 4,
        "nRRDS": 8, "nRRDL": 8,
        "nWTRS": 10, "nWTRL": 12,
        "nFAW": 29,
        "nRREFD": 15,
        "tCK_ps": 572,
        "nPPD": 1,
    },
    # Samsung K4Z80325BC Rev. 1.3, Tables 27, 91, and 93 (8 Gb, DDR WCK).
    "GDDR6_14000_1250mV_double": {  # Used in Tests
        "rate": 14000, "nBL": 2, "nCL": 24,
        "nRCDRD": 30, "nRCDWR": 20,
        "nRP": 30, "nRAS": 60, "nRC": 90,
        "nWR": 30, "nRTP": 4, "nCWL": 6,
        "nCCDS": 2, "nCCDL": 4,
        "nRRDS": 11, "nRRDL": 11,
        "nWTRS": 10, "nWTRL": 12,
        "nFAW": 43,
        "nRREFD": 22,
        "tCK_ps": 572,
        "nPPD": 2,
    },
}
