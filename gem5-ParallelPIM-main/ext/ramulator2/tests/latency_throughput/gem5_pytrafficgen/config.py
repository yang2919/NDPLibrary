"""gem5 PyTrafficGen config for Ramulator2."""

import argparse
import json

import ramulator
from ramulator.dram.spec import DRAMStandard


def _profile(dram_class, org_preset, timing_preset, controller_class):
    return {
        "dram_class": dram_class,
        "org_preset": org_preset,
        "timing_preset": timing_preset,
        "controller_class": controller_class,
        "scheduler_class": "FRFCFSRowHit",
        "row_policy": "Open",
    }


DRAM_PROFILES = {
    "DDR3": _profile("DDR3", "DDR3_2Gb_x8", "DDR3_1600H", "GenericDDR"),
    "DDR4": _profile("DDR4", "DDR4_8Gb_x8", "DDR4_2400R", "GenericDDR"),
    "DDR5": _profile("DDR5", "DDR5_16Gb_x8", "DDR5_4800AN", "GenericDDR"),
    "GDDR6": _profile("GDDR6", "GDDR6_8Gb_x16", "GDDR6_14000_1250mV_double", "GenericDDR"),
    "GDDR7": _profile("GDDR7", "GDDR7_16Gb_x8", "GDDR7_28000_PAM3", "GDDR7"),
    "HBM1": _profile("HBM1", "HBM1_2Gb", "HBM1_2Gbps", "HBM12"),
    "HBM2": _profile("HBM2", "HBM2_2Gb", "HBM2_2000Mbps", "HBM12"),
    "HBM3": _profile("HBM3", "HBM3_16Gb_8hi", "HBM3_6400Mbps", "HBM34"),
    "HBM4": _profile("HBM4", "HBM4_32Gb_8Hi", "HBM4_8000Mbps", "HBM34"),
    "LPDDR5": _profile("LPDDR5", "LPDDR5_8Gb_x16", "LPDDR5_6400", "LPDDR5"),
    "LPDDR6": _profile("LPDDR6", "LPDDR6_16Gb_x12", "LPDDR6_10667_BL24", "LPDDR6"),
}

def resolve_profile_spec(profile):
    """Return (bytes_per_req, peak_gbps_per_channel) for a DRAM profile."""
    std_cls = DRAMStandard._registry[profile["dram_class"]]
    dram = getattr(ramulator.dram, profile["dram_class"])(
        org_preset=profile["org_preset"],
        timing_preset=profile["timing_preset"],
    )
    org, timing = dram.resolve()
    channel_width = org["channel_width"]
    bytes_per_req = (
        std_cls.data_payload_bytes
        or channel_width * std_cls.internal_prefetch_size // 8
    )

    tck_ns = timing["tCK_ps"] / 1000.0
    burst_gap = timing.get("nBL_min", timing.get("nBL"))
    if std_cls.data_payload_bytes is not None and burst_gap is not None:
        peak_gbps = bytes_per_req / (burst_gap * tck_ns)
    else:
        peak_gbps = channel_width * timing["rate"] / 8 / 1000

    if "PseudoChannel" in std_cls.levels:
        peak_gbps *= org.get("pseudochannel", 2)

    return bytes_per_req, peak_gbps


def add_shared_args(parser):
    parser.add_argument("--duration", default="100us")
    parser.add_argument("--memory-per-channel", default="128MiB")
    parser.add_argument("--addr-mapper", default="MOP4CLXOR")
    parser.add_argument("--refresh-manager", default="NoRefresh")
    parser.add_argument("--read-buffer-size", type=int)
    parser.add_argument("--write-buffer-size", type=int)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dram", required=True, choices=sorted(DRAM_PROFILES))
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument(
        "--intensity",
        type=float,
        required=True,
        help="Fraction of aggregate peak bandwidth to request.",
    )
    parser.add_argument("--read-ratio", type=int, default=100)
    parser.add_argument("--traffic", choices=("stream", "random"), default="stream")
    add_shared_args(parser)
    return parser.parse_args()


def make_controller(profile, args):
    dram = getattr(ramulator.dram, profile["dram_class"])(
        org_preset=profile["org_preset"],
        timing_preset=profile["timing_preset"],
    )

    controller_kwargs = {}
    if args.read_buffer_size is not None:
        controller_kwargs["read_buffer_size"] = args.read_buffer_size
    if args.write_buffer_size is not None:
        controller_kwargs["write_buffer_size"] = args.write_buffer_size

    return getattr(ramulator.controller, profile["controller_class"])(
        dram=dram,
        scheduler=getattr(ramulator.scheduler, profile["scheduler_class"])(),
        refresh_manager=getattr(ramulator.refresh_manager, args.refresh_manager)(),
        row_policy=getattr(ramulator.row_policy, profile["row_policy"])(),
        addr_mapper=getattr(ramulator.addr_mapper, args.addr_mapper)(),
        **controller_kwargs,
    )


def main():
    from gem5.components.boards.abstract_board import AbstractBoard
    from gem5.components.boards.test_board import TestBoard
    from gem5.components.processors.abstract_generator import (
        AbstractGenerator,
        partition_range,
    )
    from gem5.components.processors.linear_generator_core import LinearGeneratorCore
    from gem5.components.processors.random_generator_core import RandomGeneratorCore
    from gem5.simulate.simulator import Simulator
    from gem5.utils.override import overrides
    from m5.util.convert import toMemorySize

    class PartitionedGenerator(AbstractGenerator):
        """Traffic generator where each core i cover address range slice i."""

        def __init__(self, core_cls, num_cores, rate_per_core, duration,
                     block_size, max_addr, rd_perc):
            super().__init__(
                cores=[
                    core_cls(
                        duration=duration,
                        rate=rate_per_core,
                        block_size=block_size,
                        min_addr=lo,
                        max_addr=hi,
                        rd_perc=rd_perc,
                        data_limit=0,
                    )
                    for lo, hi in partition_range(0, max_addr, num_cores)
                ]
            )

        @overrides(AbstractGenerator)
        def start_traffic(self) -> None:
            for core in self.cores:
                core.start_traffic()

    class DirectTestBoard(TestBoard):
        """TestBoard with generator core i wired directly to memory port i."""

        @overrides(TestBoard)
        def _connect_things(self) -> None:
            # TestBoard's own override supports only a single core and port.
            AbstractBoard._connect_things(self)

            cores = self.get_processor().get_cores()
            ports = self.get_memory().get_mem_ports()
            for core, (_range, port) in zip(cores, ports, strict=True):
                core.connect_dcache(port)

    args = parse_args()
    profile = DRAM_PROFILES[args.dram]
    block_size, peak_per_channel = resolve_profile_spec(profile)
    peak_gbps = peak_per_channel * args.channels
    rate_bps = int(peak_gbps * 1e9 * args.intensity)
    per_channel_bytes = toMemorySize(args.memory_per_channel)

    mem_sys = ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[make_controller(profile, args) for _ in range(args.channels)],
    )
    memory = ramulator.gem5.VectorPortMemory(
        mem_sys,
        size=f"{per_channel_bytes * args.channels}B",
        channels=args.channels,
        interleaving_size=per_channel_bytes,
    )

    core_cls = (
        LinearGeneratorCore if args.traffic == "stream" else RandomGeneratorCore
    )
    generator = PartitionedGenerator(
        core_cls=core_cls,
        num_cores=args.channels,
        rate_per_core=f"{rate_bps // args.channels}B/s",
        duration=args.duration,
        block_size=block_size,
        max_addr=memory.get_size(),
        rd_perc=args.read_ratio,
    )

    board = DirectTestBoard(
        clk_freq="3GHz",
        generator=generator,
        memory=memory,
        cache_hierarchy=None,
    )

    print("PyTrafficGen Ramulator2 Config:")
    config = {
        "dram": args.dram,
        **profile,
        "channels": args.channels,
        "traffic": args.traffic,
        "read_ratio": args.read_ratio,
        "intensity": args.intensity,
        "rate_bps": rate_bps,
        "peak_gbps": peak_gbps,
        "block_size": block_size,
        "duration": args.duration,
        "memory_per_channel": args.memory_per_channel,
        "addr_mapper": args.addr_mapper,
        "refresh_manager": args.refresh_manager,
        "read_buffer_size": args.read_buffer_size,
        "write_buffer_size": args.write_buffer_size,
        "port_ranges": [str(rng) for rng, _ in memory.get_mem_ports()],
    }
    print(json.dumps(config, sort_keys=True, indent=2))

    simulator = Simulator(board=board)
    simulator.run()
    print("Simulation complete")


if __name__ == "__m5_main__":
    main()
