import math

from ramulator.dram.spec import DRAMStandard, TimingConstraint


class DDR4(DRAMStandard):
    name = "DDR4"
    internal_prefetch_size = 8
    read_latency = "nCL + nBL"

    # ---- Hierarchy (level name → init state) ----
    levels = {
        "Channel":      "N_A",
        "Rank":         "N_A",
        "BankGroup":    "N_A",
        "Bank":         "Closed",
        "Row":          "Closed",
        "Column":       "N_A",
    }

    # ---- Commands ----
    commands = [
        "ACT", "PREpb", "PREab",
        "RD", "WR", "RDA", "WRA",
        "REFab",
    ]

    # ---- States ----
    states = ["Opened", "Closed", "N_A"]

    # ---- Timing parameters (C++ Timing enum order) ----
    timing_params = [
        "rate", "nBL",
        "nCL", "nRCD", "nRP", "nRAS", "nRC",
        "nWR", "nRTP", "nCWL",
        "nCCDS", "nCCDL",
        "nRRDS", "nRRDL",
        "nWTRS", "nWTRL",
        "nFAW", "nRFC", "nREFI",
        "nCS", "tCK_ps",
    ]

    # ---- External request types ----
    supported_requests = {"Read": "RD", "Write": "WR"}

    # ---- Timing constraints ----
    timing_constraints = [
        # Channel — data bus occupancy
        TimingConstraint(level="Channel", preceding=["RD", "RDA"], following=["RD", "RDA"], latency="nBL"),
        TimingConstraint(level="Channel", preceding=["WR", "WRA"], following=["WR", "WRA"], latency="nBL"),

        # Rank — CAS timing
        TimingConstraint(level="Rank", preceding=["RD", "RDA"], following=["RD", "RDA"], latency="nCCDS"),
        TimingConstraint(level="Rank", preceding=["WR", "WRA"], following=["WR", "WRA"], latency="nCCDS"),
        TimingConstraint(level="Rank", preceding=["RD", "RDA"], following=["WR", "WRA"], latency="nCL + nBL + 2 - nCWL"),
        TimingConstraint(level="Rank", preceding=["WR", "WRA"], following=["RD", "RDA"], latency="nCWL + nBL + nWTRS"),
        # Rank — sibling (rank switching)
        TimingConstraint(level="Rank", preceding=["RD", "RDA"], following=["RD", "RDA", "WR", "WRA"], latency="nBL + nCS", window=1, sibling=True),
        TimingConstraint(level="Rank", preceding=["WR", "WRA"], following=["RD", "RDA"], latency="nCWL + nBL + nCS - nCL", window=1, sibling=True),
        # Rank — CAS to PREab
        TimingConstraint(level="Rank", preceding=["RD", "RDA"], following=["PREab"], latency="nRTP"),
        TimingConstraint(level="Rank", preceding=["WR", "WRA"], following=["PREab"], latency="nCWL + nBL + nWR"),
        # Rank — RAS timing
        TimingConstraint(level="Rank", preceding=["ACT"], following=["ACT"], latency="nRRDS"),
        TimingConstraint(level="Rank", preceding=["ACT"], following=["ACT"], latency="nFAW", window=4),
        TimingConstraint(level="Rank", preceding=["ACT"], following=["PREab"], latency="nRAS"),
        TimingConstraint(level="Rank", preceding=["PREab"], following=["ACT"], latency="nRP"),
        # Rank — RAS to REF
        TimingConstraint(level="Rank", preceding=["ACT"], following=["REFab"], latency="nRC"),
        TimingConstraint(level="Rank", preceding=["PREpb", "PREab"], following=["REFab"], latency="nRP"),
        TimingConstraint(level="Rank", preceding=["RDA"], following=["REFab"], latency="nRP + nRTP"),
        TimingConstraint(level="Rank", preceding=["WRA"], following=["REFab"], latency="nCWL + nBL + nWR + nRP"),
        TimingConstraint(level="Rank", preceding=["REFab"], following=["ACT", "PREab"], latency="nRFC"),

        # BankGroup — same-group CAS timing
        TimingConstraint(level="BankGroup", preceding=["RD", "RDA"], following=["RD", "RDA"], latency="nCCDL"),
        TimingConstraint(level="BankGroup", preceding=["WR", "WRA"], following=["WR", "WRA"], latency="nCCDL"),
        TimingConstraint(level="BankGroup", preceding=["WR", "WRA"], following=["RD", "RDA"], latency="nCWL + nBL + nWTRL"),
        # BankGroup — same-group RAS timing
        TimingConstraint(level="BankGroup", preceding=["ACT"], following=["ACT"], latency="nRRDL"),

        # Bank — single-bank timing
        TimingConstraint(level="Bank", preceding=["ACT"], following=["ACT"], latency="nRC"),
        TimingConstraint(level="Bank", preceding=["ACT"], following=["RD", "RDA", "WR", "WRA"], latency="nRCD"),
        TimingConstraint(level="Bank", preceding=["ACT"], following=["PREpb"], latency="nRAS"),
        TimingConstraint(level="Bank", preceding=["PREpb"], following=["ACT"], latency="nRP"),
        TimingConstraint(level="Bank", preceding=["RD"], following=["PREpb"], latency="nRTP"),
        TimingConstraint(level="Bank", preceding=["WR"], following=["PREpb"], latency="nCWL + nBL + nWR"),
        TimingConstraint(level="Bank", preceding=["RDA"], following=["ACT"], latency="nRTP + nRP"),
        TimingConstraint(level="Bank", preceding=["WRA"], following=["ACT"], latency="nCWL + nBL + nWR + nRP"),
    ]

    # ---- Secondary timing resolution ----
    @classmethod
    def resolve_secondary_timings(cls, timing_dict, org_dict):
        dq = org_dict["dq"]
        if dq in (4, 8, 16):
            page_size_bits = org_dict["column"] * dq
            page_size_bytes = page_size_bits // 8 if page_size_bits % 8 == 0 else -1
        else:
            page_size_bytes = -1
        timing_dict["nRRDS"] = cls._resolve_nRRDS(page_size_bytes, timing_dict["rate"])
        timing_dict["nRRDL"] = cls._resolve_nRRDL(page_size_bytes, timing_dict["rate"])
        timing_dict["nFAW"] = cls._resolve_nFAW(page_size_bytes, timing_dict["rate"])
        timing_dict["nRFC"] = cls._resolve_nRFC(org_dict["density"], timing_dict["tCK_ps"])
        timing_dict["nREFI"] = cls._resolve_nREFI(timing_dict["tCK_ps"])

    @staticmethod
    def _resolve_nRRDS(page_size_bytes, rate):
        _table = {
            (512, 1600):  4,
            (512, 1866):  4,
            (512, 2133):  4,
            (512, 2400):  4,
            (512, 2666):  4,
            (512, 2933):  4,
            (512, 3200):  4,
            (1024, 1600): 4,
            (1024, 1866): 4,
            (1024, 2133): 4,
            (1024, 2400): 4,
            (1024, 2666): 4,
            (1024, 2933): 4,
            (1024, 3200): 4,
            (2048, 1600): 5,
            (2048, 1866): 5,
            (2048, 2133): 6,
            (2048, 2400): 7,
            (2048, 2666): 8,
            (2048, 2933): 8,
            (2048, 3200): 9,
        }
        return _table.get((page_size_bytes, rate), -1)

    @staticmethod
    def _resolve_nRRDL(page_size_bytes, rate):
        _table = {
            (512, 1600):  5,
            (512, 1866):  5,
            (512, 2133):  6,
            (512, 2400):  6,
            (512, 2666):  7,
            (512, 2933):  8,
            (512, 3200):  8,
            (1024, 1600): 5,
            (1024, 1866): 5,
            (1024, 2133): 6,
            (1024, 2400): 6,
            (1024, 2666): 7,
            (1024, 2933): 8,
            (1024, 3200): 8,
            (2048, 1600): 6,
            (2048, 1866): 6,
            (2048, 2133): 7,
            (2048, 2400): 8,
            (2048, 2666): 9,
            (2048, 2933): 10,
            (2048, 3200): 11,
        }
        return _table.get((page_size_bytes, rate), -1)

    @staticmethod
    def _resolve_nFAW(page_size_bytes, rate):
        _table = {
            (512, 1600):  16,
            (512, 1866):  16,
            (512, 2133):  16,
            (512, 2400):  16,
            (512, 2666):  16,
            (512, 2933):  16,
            (512, 3200):  16,
            (1024, 1600): 20,
            (1024, 1866): 22,
            (1024, 2133): 23,
            (1024, 2400): 26,
            (1024, 2666): 28,
            (1024, 2933): 31,
            (1024, 3200): 34,
            (2048, 1600): 28,
            (2048, 1866): 28,
            (2048, 2133): 32,
            (2048, 2400): 36,
            (2048, 2666): 40,
            (2048, 2933): 44,
            (2048, 3200): 48,
        }
        return _table.get((page_size_bytes, rate), -1)

    @staticmethod
    def _resolve_nRFC(density, tCK_ps):
        tRFC_ns = {
            2048: 160,
            4096: 260,
            8192: 350,
            16384: 550,
        }.get(density)
        if tRFC_ns is None:
            return -1
        return math.ceil(tRFC_ns * 1000 / tCK_ps)

    @staticmethod
    def _resolve_nREFI(tCK_ps):
        return 7_800_000 // tCK_ps


# ---- DDR4 JEDEC preset data ----

DDR4.org_presets = {
    "DDR4_2Gb_x4":  {"density": 2048,  "dq": 4,  "channel_width": 64, "rank": 1, "bankgroup": 4, "bank": 4, "row": 1<<15, "column": 1<<10},
    "DDR4_2Gb_x8":  {"density": 2048,  "dq": 8,  "channel_width": 64, "rank": 1, "bankgroup": 4, "bank": 4, "row": 1<<14, "column": 1<<10},
    "DDR4_2Gb_x16": {"density": 2048,  "dq": 16, "channel_width": 64, "rank": 1, "bankgroup": 2, "bank": 4, "row": 1<<14, "column": 1<<10},
    "DDR4_4Gb_x4":  {"density": 4096,  "dq": 4,  "channel_width": 64, "rank": 1, "bankgroup": 4, "bank": 4, "row": 1<<16, "column": 1<<10},
    "DDR4_4Gb_x8":  {"density": 4096,  "dq": 8,  "channel_width": 64, "rank": 1, "bankgroup": 4, "bank": 4, "row": 1<<15, "column": 1<<10},
    "DDR4_4Gb_x16": {"density": 4096,  "dq": 16, "channel_width": 64, "rank": 1, "bankgroup": 2, "bank": 4, "row": 1<<15, "column": 1<<10},
    "DDR4_8Gb_x4":  {"density": 8192,  "dq": 4,  "channel_width": 64, "rank": 1, "bankgroup": 4, "bank": 4, "row": 1<<17, "column": 1<<10},
    "DDR4_8Gb_x8":  {"density": 8192,  "dq": 8,  "channel_width": 64, "rank": 1, "bankgroup": 4, "bank": 4, "row": 1<<16, "column": 1<<10},
    "DDR4_8Gb_x16": {"density": 8192,  "dq": 16, "channel_width": 64, "rank": 1, "bankgroup": 2, "bank": 4, "row": 1<<16, "column": 1<<10},
    "DDR4_16Gb_x4": {"density": 16384, "dq": 4,  "channel_width": 64, "rank": 1, "bankgroup": 4, "bank": 4, "row": 1<<18, "column": 1<<10},
    "DDR4_16Gb_x8": {"density": 16384, "dq": 8,  "channel_width": 64, "rank": 1, "bankgroup": 4, "bank": 4, "row": 1<<17, "column": 1<<10},
    "DDR4_16Gb_x16":{"density": 16384, "dq": 16, "channel_width": 64, "rank": 1, "bankgroup": 2, "bank": 4, "row": 1<<17, "column": 1<<10},
}

# Primary timings only — secondary timings (nRRDS, nRRDL, nFAW, nRFC, nREFI)
# are resolved from JEDEC tables in resolve_secondary_timings().
DDR4.timing_presets = {
    "DDR4_1600J": {
        "rate": 1600, "nBL": 4,
        "nCL": 10, "nRCD": 10, "nRP": 10, "nRAS": 28, "nRC": 38,
        "nWR": 12, "nRTP": 6, "nCWL": 9,
        "nCCDS": 4, "nCCDL": 5,
        "nWTRS": 2, "nWTRL": 6,
        "tCK_ps": 1250,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_1600K": {
        "rate": 1600, "nBL": 4,
        "nCL": 11, "nRCD": 11, "nRP": 11, "nRAS": 28, "nRC": 39,
        "nWR": 12, "nRTP": 6, "nCWL": 9,
        "nCCDS": 4, "nCCDL": 5,
        "nWTRS": 2, "nWTRL": 6,
        "tCK_ps": 1250,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_1600L": {
        "rate": 1600, "nBL": 4,
        "nCL": 12, "nRCD": 12, "nRP": 12, "nRAS": 28, "nRC": 40,
        "nWR": 12, "nRTP": 6, "nCWL": 9,
        "nCCDS": 4, "nCCDL": 5,
        "nWTRS": 2, "nWTRL": 6,
        "tCK_ps": 1250,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_1866L": {
        "rate": 1866, "nBL": 4,
        "nCL": 12, "nRCD": 12, "nRP": 12, "nRAS": 32, "nRC": 44,
        "nWR": 14, "nRTP": 7, "nCWL": 10,
        "nCCDS": 4, "nCCDL": 5,
        "nWTRS": 3, "nWTRL": 7,
        "tCK_ps": 1071,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_1866M": {
        "rate": 1866, "nBL": 4,
        "nCL": 13, "nRCD": 13, "nRP": 13, "nRAS": 32, "nRC": 45,
        "nWR": 14, "nRTP": 7, "nCWL": 10,
        "nCCDS": 4, "nCCDL": 5,
        "nWTRS": 3, "nWTRL": 7,
        "tCK_ps": 1071,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_1866N": {
        "rate": 1866, "nBL": 4,
        "nCL": 14, "nRCD": 14, "nRP": 14, "nRAS": 32, "nRC": 46,
        "nWR": 14, "nRTP": 7, "nCWL": 10,
        "nCCDS": 4, "nCCDL": 5,
        "nWTRS": 3, "nWTRL": 7,
        "tCK_ps": 1071,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_2133N": {
        "rate": 2133, "nBL": 4,
        "nCL": 14, "nRCD": 14, "nRP": 14, "nRAS": 36, "nRC": 50,
        "nWR": 16, "nRTP": 8, "nCWL": 11,
        "nCCDS": 4, "nCCDL": 6,
        "nWTRS": 3, "nWTRL": 8,
        "tCK_ps": 937,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_2133P": {
        "rate": 2133, "nBL": 4,
        "nCL": 15, "nRCD": 15, "nRP": 15, "nRAS": 36, "nRC": 51,
        "nWR": 16, "nRTP": 8, "nCWL": 11,
        "nCCDS": 4, "nCCDL": 6,
        "nWTRS": 3, "nWTRL": 8,
        "tCK_ps": 937,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_2133R": {
        "rate": 2133, "nBL": 4,
        "nCL": 16, "nRCD": 16, "nRP": 16, "nRAS": 36, "nRC": 52,
        "nWR": 16, "nRTP": 8, "nCWL": 11,
        "nCCDS": 4, "nCCDL": 6,
        "nWTRS": 3, "nWTRL": 8,
        "tCK_ps": 937,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_2400P": {
        "rate": 2400, "nBL": 4,
        "nCL": 15, "nRCD": 15, "nRP": 15, "nRAS": 39, "nRC": 54,
        "nWR": 18, "nRTP": 9, "nCWL": 12,
        "nCCDS": 4, "nCCDL": 6,
        "nWTRS": 3, "nWTRL": 9,
        "tCK_ps": 833,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_2400R": {
        "rate": 2400, "nBL": 4,
        "nCL": 16, "nRCD": 16, "nRP": 16, "nRAS": 39, "nRC": 55,
        "nWR": 18, "nRTP": 9, "nCWL": 12,
        "nCCDS": 4, "nCCDL": 6,
        "nWTRS": 3, "nWTRL": 9,
        "tCK_ps": 833,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_2400T": {
        "rate": 2400, "nBL": 4,
        "nCL": 17, "nRCD": 17, "nRP": 17, "nRAS": 39, "nRC": 56,
        "nWR": 18, "nRTP": 9, "nCWL": 12,
        "nCCDS": 4, "nCCDL": 6,
        "nWTRS": 3, "nWTRL": 9,
        "tCK_ps": 833,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_2400U": {
        "rate": 2400, "nBL": 4,
        "nCL": 18, "nRCD": 18, "nRP": 18, "nRAS": 39, "nRC": 57,
        "nWR": 18, "nRTP": 9, "nCWL": 12,
        "nCCDS": 4, "nCCDL": 6,
        "nWTRS": 3, "nWTRL": 9,
        "tCK_ps": 833,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_2666T": {
        "rate": 2666, "nBL": 4,
        "nCL": 17, "nRCD": 17, "nRP": 17, "nRAS": 43, "nRC": 60,
        "nWR": 20, "nRTP": 10, "nCWL": 14,
        "nCCDS": 4, "nCCDL": 7,
        "nWTRS": 4, "nWTRL": 10,
        "tCK_ps": 750,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_2666U": {
        "rate": 2666, "nBL": 4,
        "nCL": 18, "nRCD": 18, "nRP": 18, "nRAS": 43, "nRC": 61,
        "nWR": 20, "nRTP": 10, "nCWL": 14,
        "nCCDS": 4, "nCCDL": 7,
        "nWTRS": 4, "nWTRL": 10,
        "tCK_ps": 750,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_2666V": {
        "rate": 2666, "nBL": 4,
        "nCL": 19, "nRCD": 19, "nRP": 19, "nRAS": 43, "nRC": 62,
        "nWR": 20, "nRTP": 10, "nCWL": 14,
        "nCCDS": 4, "nCCDL": 7,
        "nWTRS": 4, "nWTRL": 10,
        "tCK_ps": 750,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_2666W": {
        "rate": 2666, "nBL": 4,
        "nCL": 20, "nRCD": 20, "nRP": 20, "nRAS": 43, "nRC": 63,
        "nWR": 20, "nRTP": 10, "nCWL": 14,
        "nCCDS": 4, "nCCDL": 7,
        "nWTRS": 4, "nWTRL": 10,
        "tCK_ps": 750,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_2933V": {
        "rate": 2933, "nBL": 4,
        "nCL": 19, "nRCD": 19, "nRP": 19, "nRAS": 47, "nRC": 66,
        "nWR": 22, "nRTP": 11, "nCWL": 16,
        "nCCDS": 4, "nCCDL": 8,
        "nWTRS": 4, "nWTRL": 11,
        "tCK_ps": 682,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_2933W": {
        "rate": 2933, "nBL": 4,
        "nCL": 20, "nRCD": 20, "nRP": 20, "nRAS": 47, "nRC": 67,
        "nWR": 22, "nRTP": 11, "nCWL": 16,
        "nCCDS": 4, "nCCDL": 8,
        "nWTRS": 4, "nWTRL": 11,
        "tCK_ps": 682,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_2933Y": {
        "rate": 2933, "nBL": 4,
        "nCL": 21, "nRCD": 21, "nRP": 21, "nRAS": 47, "nRC": 68,
        "nWR": 22, "nRTP": 11, "nCWL": 16,
        "nCCDS": 4, "nCCDL": 8,
        "nWTRS": 4, "nWTRL": 11,
        "tCK_ps": 682,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_2933AA": {
        "rate": 2933, "nBL": 4,
        "nCL": 22, "nRCD": 22, "nRP": 22, "nRAS": 47, "nRC": 69,
        "nWR": 22, "nRTP": 11, "nCWL": 16,
        "nCCDS": 4, "nCCDL": 8,
        "nWTRS": 4, "nWTRL": 11,
        "tCK_ps": 682,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_3200W": {
        "rate": 3200, "nBL": 4,
        "nCL": 20, "nRCD": 20, "nRP": 20, "nRAS": 52, "nRC": 72,
        "nWR": 24, "nRTP": 12, "nCWL": 16,
        "nCCDS": 4, "nCCDL": 8,
        "nWTRS": 4, "nWTRL": 12,
        "tCK_ps": 625,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_3200AA": {
        "rate": 3200, "nBL": 4,
        "nCL": 22, "nRCD": 22, "nRP": 22, "nRAS": 52, "nRC": 74,
        "nWR": 24, "nRTP": 12, "nCWL": 16,
        "nCCDS": 4, "nCCDL": 8,
        "nWTRS": 4, "nWTRL": 12,
        "tCK_ps": 625,
        "nCS": 2,  # Ramulator guesstimate
    },
    "DDR4_3200AC": {
        "rate": 3200, "nBL": 4,
        "nCL": 24, "nRCD": 24, "nRP": 24, "nRAS": 52, "nRC": 76,
        "nWR": 24, "nRTP": 12, "nCWL": 16,
        "nCCDS": 4, "nCCDL": 8,
        "nWTRS": 4, "nWTRL": 12,
        "tCK_ps": 625,
        "nCS": 2,  # Ramulator guesstimate
    },
}
