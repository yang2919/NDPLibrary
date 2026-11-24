import math

from ramulator.dram.spec import DRAMStandard, TimingConstraint


class LPDDR6(DRAMStandard):
    name = "LPDDR6"
    # LPDDR6 BL24 physically transfers 12 DQ x 24 beats, with 32 B payload plus
    # metadata. Keep the existing column granularity model but expose the real
    # payload size through data_payload_bytes.
    internal_prefetch_size = 16
    data_payload_bytes = 32
    read_latency = "nRL + nBL_min"

    levels = {
        "Channel":      "N_A",
        "Rank":         "N_A",
        "BankGroup":    "N_A",
        "Bank":         "Closed",
        "Row":          "Closed",
        "Column":       "N_A",
    }

    commands = [
        "ACT1", "ACT2", "PREpb", "PREab",
        "CAS",
        "RD_S", "WR_S", "RDA_S", "WRA_S",
        "RD_L", "WR_L", "RDA_L", "WRA_L",
        "REFab",
    ]

    # LPDDR6 commands use an every-other-CK command protocol.
    command_cycles = {
        "ACT1": 2, "ACT2": 2,
        "PREpb": 2, "PREab": 2,
        "CAS": 2,
        "RD_S": 2, "WR_S": 2, "RDA_S": 2, "WRA_S": 2,
        "RD_L": 2, "WR_L": 2, "RDA_L": 2, "WRA_L": 2,
        "REFab": 2,
    }

    states = ["Opened", "Closed", "Activating", "N_A"]

    # JESD209-6 Table 381: BL/n_min and BL/n_max definitions.
    timing_params = [
        "rate", "nBL_min", "nBL_max", "nBL_min_L", "nBL_max_L", "nRL", "nWL",
        "nACU",
        "nRCDr", "nRCDw", "nRP", "nRPab", "nRAS", "nRC",
        "nWTP", "nRTP", "nRTP_L", "nPPD",
        "nCCDS", "nCCDL", "nCCDL_L", "nCCDS_WR", "nCCDL_WR", "nCCDL_WR_L",
        "nRRD",
        "nWTRS", "nWTRL",
        "nRTW_S", "nRTW_L", "nRTW_S_L", "nRTW_L_L",
        "nWCK2DQO", "nRPST", "nODTLon", "nODTon_min",
        "nFAW", "nRFC", "nREFI",
        "nWCKPST", "nCAS", "nAAD", "nCS", "tCK_ps",
    ]

    supported_requests = {"Read": "RD_S", "Write": "WR_S"}

    timing_constraints = [
        # Channel — DQ-bus occupancy (JESD209-6 Table 381 BL/n_min).
        # BL48 is modeled as two transferred BL24 segments.
        TimingConstraint(level="Channel", preceding=["RD_S", "RDA_S"], following=["RD_S", "RDA_S", "RD_L", "RDA_L"], latency="nBL_min"),
        TimingConstraint(level="Channel", preceding=["WR_S", "WRA_S"], following=["WR_S", "WRA_S", "WR_L", "WRA_L"], latency="nBL_min"),
        TimingConstraint(level="Channel", preceding=["RD_L", "RDA_L"], following=["RD_S", "RDA_S", "RD_L", "RDA_L"], latency="nBL_min + nBL_min"),
        TimingConstraint(level="Channel", preceding=["WR_L", "WRA_L"], following=["WR_S", "WRA_S", "WR_L", "WRA_L"], latency="nBL_min + nBL_min"),

        # Bank — CAS access gap (JESD209-6 Table 394).
        TimingConstraint(level="Bank", preceding=["CAS"], following=["RD_S", "RDA_S", "RD_L", "RDA_L", "WR_S", "WRA_S", "WR_L", "WRA_L"], latency="nCAS"),

        # Rank — different-BG column timing and turnarounds
        # (JESD209-6 Tables 385 and 390).
        TimingConstraint(level="Rank", preceding=["RD_S", "RDA_S"], following=["RD_S", "RDA_S", "RD_L", "RDA_L"], latency="nCCDS"),
        TimingConstraint(level="Rank", preceding=["RD_L", "RDA_L"], following=["RD_S", "RDA_S", "RD_L", "RDA_L"], latency="nCCDS"),
        TimingConstraint(level="Rank", preceding=["WR_S", "WRA_S"], following=["WR_S", "WRA_S", "WR_L", "WRA_L"], latency="nCCDS_WR"),
        TimingConstraint(level="Rank", preceding=["WR_L", "WRA_L"], following=["WR_S", "WRA_S", "WR_L", "WRA_L"], latency="nCCDS_WR"),
        # RD->WR diff-BG = tRTW with BL/n_min (JESD209-6 Table 390).
        TimingConstraint(level="Rank", preceding=["RD_S", "RDA_S"], following=["WR_S", "WRA_S", "WR_L", "WRA_L"], latency="nRTW_S"),
        TimingConstraint(level="Rank", preceding=["RD_L", "RDA_L"], following=["WR_S", "WRA_S", "WR_L", "WRA_L"], latency="nRTW_S_L"),
        # WR->RD diff-BG = WL + BL/n_min + tWTR_S (JESD209-6 Table 385).
        TimingConstraint(level="Rank", preceding=["WR_S", "WRA_S"], following=["RD_S", "RDA_S", "RD_L", "RDA_L"], latency="nWL + nBL_min + nWTRS"),
        TimingConstraint(level="Rank", preceding=["WR_L", "WRA_L"], following=["RD_S", "RDA_S", "RD_L", "RDA_L"], latency="nWL + nBL_min_L + nWTRS"),

        # Rank switching (sibling) — local bus-clear model: transferred beats + nCS.
        TimingConstraint(level="Rank", preceding=["RD_S", "RDA_S"], following=["RD_S", "RDA_S", "RD_L", "RDA_L", "WR_S", "WRA_S", "WR_L", "WRA_L"], latency="nBL_min + nCS", window=1, sibling=True),
        TimingConstraint(level="Rank", preceding=["RD_L", "RDA_L"], following=["RD_S", "RDA_S", "RD_L", "RDA_L", "WR_S", "WRA_S", "WR_L", "WRA_L"], latency="nBL_min + nBL_min + nCS", window=1, sibling=True),
        TimingConstraint(level="Rank", preceding=["WR_S", "WRA_S"], following=["RD_S", "RDA_S", "RD_L", "RDA_L"], latency="nRL + nBL_min + nCS - nWL", window=1, sibling=True),
        TimingConstraint(level="Rank", preceding=["WR_L", "WRA_L"], following=["RD_S", "RDA_S", "RD_L", "RDA_L"], latency="nRL + nBL_min + nBL_min + nCS - nWL", window=1, sibling=True),

        # Direct all-bank precharge after column commands
        # (JESD209-6 Tables 383 and 391).
        TimingConstraint(level="Rank", preceding=["RD_S"], following=["PREab"], latency="nRTP"),
        TimingConstraint(level="Rank", preceding=["RD_L"], following=["PREab"], latency="nRTP_L"),
        TimingConstraint(level="Rank", preceding=["WR_S"], following=["PREab"], latency="nWL + nBL_max + nWTP"),
        TimingConstraint(level="Rank", preceding=["WR_L"], following=["PREab"], latency="nWL + nBL_max_L + nWTP"),

        # RAS and activation constraints (JESD209-6 Figures 67-72 and Tables 383-385, 414).
        TimingConstraint(level="Rank", preceding=["ACT2"], following=["ACT2"], latency="nRRD"),
        TimingConstraint(level="Rank", preceding=["ACT2"], following=["ACT2"], latency="nFAW", window=4),
        TimingConstraint(level="Rank", preceding=["ACT2"], following=["PREab"], latency="nRAS"),
        TimingConstraint(level="Rank", preceding=["PREab"], following=["ACT2"], latency="nRPab"),
        TimingConstraint(level="Rank", preceding=["PREpb", "PREab"], following=["PREpb", "PREab"], latency="nPPD"),

        # Rank — all-bank refresh timing (JESD209-6 Figure 111 and Tables 300-302, 414).
        TimingConstraint(level="Rank", preceding=["ACT2"], following=["REFab"], latency="nRAS + nRPab"),
        TimingConstraint(level="Rank", preceding=["PREpb"], following=["REFab"], latency="nRP"),
        TimingConstraint(level="Rank", preceding=["PREab"], following=["REFab"], latency="nRPab"),
        TimingConstraint(level="Rank", preceding=["RDA_S"], following=["REFab"], latency="nRP + nRTP"),
        TimingConstraint(level="Rank", preceding=["RDA_L"], following=["REFab"], latency="nRP + nRTP_L"),
        TimingConstraint(level="Rank", preceding=["WRA_S"], following=["REFab"], latency="nWL + nBL_max + nWTP + nRP"),
        TimingConstraint(level="Rank", preceding=["WRA_L"], following=["REFab"], latency="nWL + nBL_max_L + nWTP + nRP"),
        TimingConstraint(level="Rank", preceding=["REFab"], following=["ACT2"], latency="nRFC"),
        TimingConstraint(level="Rank", preceding=["REFab"], following=["PREab"], latency="nRFC"),

        # BankGroup — same-BG column timing (JESD209-6 Tables 382-384).
        TimingConstraint(level="BankGroup", preceding=["RD_S", "RDA_S"], following=["RD_S", "RDA_S", "RD_L", "RDA_L"], latency="nCCDL"),
        TimingConstraint(level="BankGroup", preceding=["RD_L", "RDA_L"], following=["RD_S", "RDA_S", "RD_L", "RDA_L"], latency="nCCDL_L"),
        TimingConstraint(level="BankGroup", preceding=["WR_S", "WRA_S"], following=["WR_S", "WRA_S", "WR_L", "WRA_L"], latency="nCCDL_WR"),
        TimingConstraint(level="BankGroup", preceding=["WR_L", "WRA_L"], following=["WR_S", "WRA_S", "WR_L", "WRA_L"], latency="nCCDL_WR_L"),

        # BankGroup — same-BG read-to-write (JESD209-6 Table 389).
        TimingConstraint(level="BankGroup", preceding=["RD_S", "RDA_S"], following=["WR_S", "WRA_S", "WR_L", "WRA_L"], latency="nRTW_L"),
        TimingConstraint(level="BankGroup", preceding=["RD_L", "RDA_L"], following=["WR_S", "WRA_S", "WR_L", "WRA_L"], latency="nRTW_L_L"),

        # BankGroup — same-BG write-to-read (JESD209-6 Tables 383-384).
        TimingConstraint(level="BankGroup", preceding=["WR_S", "WRA_S"], following=["RD_S", "RDA_S", "RD_L", "RDA_L"], latency="nWL + nBL_max + nWTRL"),
        TimingConstraint(level="BankGroup", preceding=["WR_L", "WRA_L"], following=["RD_S", "RDA_S", "RD_L", "RDA_L"], latency="nWL + nBL_max_L + nWTRL"),

        # BankGroup — ACT-2 to ACT-2 timing (JESD209-6 Figure 71).
        TimingConstraint(level="BankGroup", preceding=["ACT2"], following=["ACT2"], latency="nRRD"),

        # Bank — single-bank timing (JESD209-6 Figures 67-70 and Tables 383, 391, 414).
        TimingConstraint(level="Bank", preceding=["ACT2"], following=["ACT2"], latency="nRC"),
        TimingConstraint(level="Bank", preceding=["ACT2"], following=["RD_S", "RDA_S", "RD_L", "RDA_L"], latency="nRCDr"),
        TimingConstraint(level="Bank", preceding=["ACT2"], following=["WR_S", "WRA_S", "WR_L", "WRA_L"], latency="nRCDw"),
        TimingConstraint(level="Bank", preceding=["ACT2"], following=["PREpb"], latency="nRAS"),
        TimingConstraint(level="Bank", preceding=["PREpb"], following=["ACT2"], latency="nRP"),
        TimingConstraint(level="Bank", preceding=["RD_S"], following=["PREpb"], latency="nRTP"),
        TimingConstraint(level="Bank", preceding=["RD_L"], following=["PREpb"], latency="nRTP_L"),
        TimingConstraint(level="Bank", preceding=["WR_S"], following=["PREpb"], latency="nWL + nBL_max + nWTP"),
        TimingConstraint(level="Bank", preceding=["WR_L"], following=["PREpb"], latency="nWL + nBL_max_L + nWTP"),
        TimingConstraint(level="Bank", preceding=["RDA_S"], following=["ACT2"], latency="nRTP + nRP"),
        TimingConstraint(level="Bank", preceding=["RDA_L"], following=["ACT2"], latency="nRTP_L + nRP"),
        TimingConstraint(level="Bank", preceding=["WRA_S"], following=["ACT2"], latency="nWL + nBL_max + nWTP + nRP"),
        TimingConstraint(level="Bank", preceding=["WRA_L"], following=["ACT2"], latency="nWL + nBL_max_L + nWTP + nRP"),
    ]

    @classmethod
    def resolve_secondary_timings(cls, timing_dict, org_dict):
        rate = timing_dict["rate"]
        tCK_ps = timing_dict["tCK_ps"]
        timing_dict["nBL_max"] = cls._resolve_nBL_max(rate)
        timing_dict["nBL_min_L"] = cls._resolve_nBL_min_L(rate)
        timing_dict["nBL_max_L"] = cls._resolve_nBL_max_L(rate)
        timing_dict["nCCDL_L"] = cls._resolve_nCCDL_L(
            rate, timing_dict["nCCDL"]
        )
        timing_dict["nCCDL_WR_L"] = cls._resolve_nCCDL_WR_L(
            rate, timing_dict["nCCDL_WR"]
        )
        timing_dict["nACU"] = cls._resolve_nACU(rate)
        timing_dict["nRAS"] = cls._resolve_nRAS(tCK_ps)
        timing_dict["nRP"] = cls._resolve_nRP(timing_dict["nACU"], tCK_ps)
        timing_dict["nRPab"] = cls._resolve_nRPab(timing_dict["nACU"], tCK_ps)
        timing_dict["nRC"] = cls._resolve_nRC(
            timing_dict["nRAS"], timing_dict["nRP"]
        )

        nRL = timing_dict["nRL"]
        nWCK2DQO = timing_dict["nWCK2DQO"]
        nRPST = timing_dict["nRPST"]
        nODTLon = timing_dict["nODTLon"]
        nODTon_min = timing_dict["nODTon_min"]
        nWL = timing_dict["nWL"]
        odt_enabled = org_dict["odt_enabled"]
        timing_dict["nRTW_S"] = cls._resolve_nRTW_S(
            timing_dict["nBL_min"],
            nRL, nWCK2DQO, nRPST, nODTLon, nODTon_min, nWL, odt_enabled,
        )
        timing_dict["nRTW_L"] = cls._resolve_nRTW_L(
            timing_dict["nBL_max"],
            nRL, nWCK2DQO, nRPST, nODTLon, nODTon_min, nWL, odt_enabled,
        )
        timing_dict["nRTW_S_L"] = cls._resolve_nRTW_S_L(
            timing_dict["nBL_min_L"],
            nRL, nWCK2DQO, nRPST, nODTLon, nODTon_min, nWL, odt_enabled,
        )
        timing_dict["nRTW_L_L"] = cls._resolve_nRTW_L_L(
            timing_dict["nBL_max_L"],
            nRL, nWCK2DQO, nRPST, nODTLon, nODTon_min, nWL, odt_enabled,
        )
        timing_dict["nRFC"] = cls._resolve_nRFC(
            org_dict["die_density"], tCK_ps
        )
        timing_dict["nREFI"] = cls._resolve_nREFI(tCK_ps)

    @staticmethod
    def _lookup_speed_grade(rate, table):
        # LPDDR6 speed-grade rows use lower-exclusive, upper-inclusive bounds.
        for (lower_rate, upper_rate), value in table.items():
            if lower_rate < rate <= upper_rate:
                return value
        return -1

    @classmethod
    def _resolve_nACU(cls, rate):
        # JESD209-6 Tables 414-415: nACU for each defined speed-grade range.
        return cls._lookup_speed_grade(rate, {
            (80, 1067): 6,
            (1067, 1600): 9,
            (1600, 2133): 12,
            (2133, 2750): 16,
            (2750, 3200): 18,
            (3200, 3750): 21,
            (3750, 4267): 24,
            (4267, 4800): 27,
            (4800, 5500): 31,
            (5500, 6400): 36,
            (6400, 7500): 42,
            (7500, 8533): 47,
            (8533, 9600): 53,
            (9600, 10667): 59,
        })

    @staticmethod
    def _resolve_nRAS(tCK_ps):
        return max(math.ceil(20_000 / tCK_ps), 4)

    @staticmethod
    def _resolve_nRP(nACU, tCK_ps):
        if nACU == -1:
            return -1
        return nACU + max(math.ceil(18_000 / tCK_ps), 4)

    @staticmethod
    def _resolve_nRPab(nACU, tCK_ps):
        if nACU == -1:
            return -1
        return nACU + max(math.ceil(21_000 / tCK_ps), 4)

    @staticmethod
    def _resolve_nRC(nRAS, nRP):
        if nRAS == -1 or nRP == -1:
            return -1
        return nRAS + nRP

    # JESD209-6 Tables 381-382: remaining BL24 and BL48 quantities.
    @classmethod
    def _resolve_nBL_max(cls, rate):
        return cls._lookup_speed_grade(rate, {
            (80, 3200): 6,
            (3200, 10667): 12,
        })

    @classmethod
    def _resolve_nBL_min_L(cls, rate):
        return cls._lookup_speed_grade(rate, {
            (80, 3200): 12,
            (3200, 10667): 18,
        })

    @classmethod
    def _resolve_nBL_max_L(cls, rate):
        return cls._lookup_speed_grade(rate, {
            (80, 3200): 12,
            (3200, 10667): 24,
        })

    @classmethod
    def _resolve_nCCDL_L(cls, rate, nCCDL):
        return cls._lookup_speed_grade(rate, {
            (80, 3200): 12,
            (3200, 10667): 12 + nCCDL,
        })

    @classmethod
    def _resolve_nCCDL_WR_L(cls, rate, nCCDL_WR):
        return cls._lookup_speed_grade(rate, {
            (80, 3200): 12,
            (3200, 10667): 12 + nCCDL_WR,
        })

    # JESD209-6 Tables 389-390: read-to-write gaps are rounded to even nCK.
    @staticmethod
    def _resolve_nRTW_S(
        nBL_min, nRL, nWCK2DQO, nRPST, nODTLon, nODTon_min, nWL, odt_enabled
    ):
        if odt_enabled is True:
            value = nRL + nBL_min + nWCK2DQO + nRPST - nODTLon - nODTon_min + 1
        elif odt_enabled is False:
            value = nRL + nBL_min + nWCK2DQO - nWL
        else:
            return -1
        return value + value % 2

    @staticmethod
    def _resolve_nRTW_L(
        nBL_max, nRL, nWCK2DQO, nRPST, nODTLon, nODTon_min, nWL, odt_enabled
    ):
        if odt_enabled is True:
            value = nRL + nBL_max + nWCK2DQO + nRPST - nODTLon - nODTon_min + 1
        elif odt_enabled is False:
            value = nRL + nBL_max + nWCK2DQO - nWL
        else:
            return -1
        return value + value % 2

    @staticmethod
    def _resolve_nRTW_S_L(
        nBL_min_L, nRL, nWCK2DQO, nRPST, nODTLon, nODTon_min, nWL, odt_enabled
    ):
        if odt_enabled is True:
            value = nRL + nBL_min_L + nWCK2DQO + nRPST - nODTLon - nODTon_min + 1
        elif odt_enabled is False:
            value = nRL + nBL_min_L + nWCK2DQO - nWL
        else:
            return -1
        return value + value % 2

    @staticmethod
    def _resolve_nRTW_L_L(
        nBL_max_L, nRL, nWCK2DQO, nRPST, nODTLon, nODTon_min, nWL, odt_enabled
    ):
        if odt_enabled is True:
            value = nRL + nBL_max_L + nWCK2DQO + nRPST - nODTLon - nODTon_min + 1
        elif odt_enabled is False:
            value = nRL + nBL_max_L + nWCK2DQO - nWL
        else:
            return -1
        return value + value % 2

    @staticmethod
    def _resolve_nRFC(die_density, tCK_ps):
        # JESD209-6 Table 302 defines density per 2 sub-channels.
        tRFC_ns = {
            4096: None,
            6144: 210,
            8192: 210,
            12288: 280,
            16384: 280,
            24576: 380,
            32768: 380,
            49152: None,
            65536: None,
        }.get(die_density)
        if tRFC_ns is None:
            return -1
        return math.ceil(tRFC_ns * 1000 / tCK_ps)

    @staticmethod
    def _resolve_nREFI(tCK_ps):
        # JESD209-6 Table 302: tREFI = 3906 ns maximum average interval.
        return 3_906_000 // tCK_ps


LPDDR6.org_presets = {
    "LPDDR6_16Gb_x12": {
        # JESD209-6 Tables 2 and 302: one modeled x12 sub-channel is 16 Gb;
        # the corresponding two-sub-channel die is 32 Gb.
        "subchannel_density": 16384,
        "die_density": 32768,
        "dq": 12,
        "channel_width": 12,
        "odt_enabled": True,
        "rank": 1,
        "bankgroup": 4,
        "bank": 4,
        "row": 1 << 16,
        "column": 1 << 10,
    },
}


LPDDR6.timing_presets = {
    "LPDDR6_10667_BL24": {
        "rate": 10667,
        # JESD209-6 Tables 381-382: nBL_min = BL/n_min(BL24);
        # nBL_max / nBL_min_L / nBL_max_L / nCCDL_L / nCCDL_WR_L are derived per
        # the selected burst timing in resolve_secondary_timings.
        "nBL_min": 6,
        "nRL": 56, "nWL": 26,
        "nRCDr": 48, "nRCDw": 22,
        "nWTP": 32, "nRTP": 14,
        # JESD209-6 Tables 269-272: nRTP(BL48) column at the RL=56 row.
        "nRTP_L": 26,
        "nPPD": 4,
        "nCCDS": 6, "nCCDL": 10, "nCCDS_WR": 6, "nCCDL_WR": 10,
        "nRRD": 10, "nWTRS": 17, "nWTRL": 32,
        # JESD209-6 Tables 319-320 and 477: ODT-enabled, high-frequency mode.
        "nWCK2DQO": 5,
        "nODTLon": 16, "nODTon_min": 4,
        "nFAW": 40,
        "nWCKPST": 3, "nCAS": 2,
        # === Ramulator Guesstimate ===
        "nRPST": 0,
        "nAAD": 8, "nCS": 2,
        # =============================
        "tCK_ps": 375,
    },
}
