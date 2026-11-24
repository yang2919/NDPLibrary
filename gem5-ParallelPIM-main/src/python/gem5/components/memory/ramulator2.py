"""Ramulator 2.1 as a gem5 standard library memory system.

Available only in builds with the Kconfig option USE_RAMULATOR2 (for example
build/RISCV_RAMULATOR2, from build_opts/RISCV_RAMULATOR2). Ramulator 2.1 and
its Python package are compiled into that gem5 binary from ext/ramulator2.

Two ways to build a memory:

* ``ramulator2_memory("DDR4", channels=2, size="2GiB")``: a preset per DRAM
  standard (DDR4, DDR5, LPDDR5, LPDDR6, GDDR6, GDDR7, HBM2, HBM4), with an
  FRFCFS scheduler, all-bank refresh, an open-row policy, the RoBaRaCoCh
  address mapping and read and write buffers of 64 and 128. Every preset
  value can be overridden.
* ``Ramulator2Memory(memory_system, size)``, from any Ramulator 2.1
  ``GenericDRAM`` memory system built with the ``ramulator`` package, without
  a channel mapper.

gem5 selects the channel: the memory has one port per controller, each with
an interleaved address range. Ramulator 2.1 statistics are written per
instance to ``<object name>.ramulator_stats.yaml`` (at exit) and
``<object name>.ramulator_stats.<tick>.yaml`` (at every statistics dump).
"""

import copy
import json
from math import log2
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import ramulator
from ramulator.components import Component
from ramulator.dram.pim import DESCRIPTORS

from m5.objects import (
    AddrRange,
    Port,
    Ramulator2VectorPorts,
)
from m5.util import warn
from m5.util.convert import toMemorySize

from ...utils.override import overrides
from ..boards.abstract_board import AbstractBoard
from .abstract_memory_system import AbstractMemorySystem


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def ramulator2_config(memory_system, interleaving_size: int) -> str:
    """The Ramulator 2.1 configuration of a vector-port memory, as JSON.

    The memory system must have no channel mapper: gem5 selects the channel,
    and Ramulator 2.1's ``Gem5PortInterleave`` mapper forwards each request
    to the controller of the port it arrived on. Same output as Ramulator
    2.1's ``ramulator.gem5`` helper.
    """
    if isinstance(memory_system, Component):
        ms = memory_system.to_config()
    else:
        ms = copy.deepcopy(memory_system)
    if "channel_mapper" in ms:
        raise ValueError(
            "Ramulator2Memory selects channels itself; build the Ramulator "
            "2.1 memory system without a channel_mapper"
        )
    if not isinstance(ms.get("controllers"), list):
        raise ValueError(
            "Ramulator2Memory needs a memory system with a controllers list"
        )
    ms["channel_mapper"] = {
        "impl": "Gem5PortInterleave",
        "interleave_size": interleaving_size,
    }
    config = {
        "frontend": {"impl": "External", "clock_ratio": 1},
        "memory_system": ms,
    }
    return json.dumps(config)


def interleaved_port_ranges(
    mem_range: AddrRange, channels: int, interleaving_size: int
) -> List[AddrRange]:
    """One address range per channel, interleaved every interleaving_size
    bytes; one channel gets the whole range."""
    if channels == 1:
        return [mem_range]
    intlv_low_bit = int(log2(interleaving_size))
    intlv_bits = int(log2(channels))
    return [
        AddrRange(
            start=mem_range.start,
            size=mem_range.size(),
            intlvHighBit=intlv_low_bit + intlv_bits - 1,
            xorHighBit=0,
            intlvBits=intlv_bits,
            intlvMatch=i,
        )
        for i in range(channels)
    ]


def closed_page_read_latency(dram) -> int:
    """Picoseconds of a read to a closed bank, nRP + nRCD + nCL, as gem5's
    DRAMInterface::accessLatency (src/mem/dram_interface.hh) computes it;
    the standards name the activate-to-read and read latencies variously."""
    _, timing = dram.resolve()

    def first(*names):
        for n in names:
            if timing.get(n) is not None:
                return timing[n]
        raise ValueError(
            "Ramulator2Memory: %s has none of %s"
            % (type(dram).__name__, ", ".join(names))
        )

    clocks = (
        first("nRP") + first("nRCD", "nRCDRD", "nRCDr") + first("nCL", "nRL")
    )
    return clocks * first("tCK_ps")


def channel_capacity(dram) -> int:
    """Bytes one channel of a Ramulator 2.1 DRAM device holds.

    Transaction bytes times the count of every level below the channel,
    with the column count divided by the prefetch size. The C++ wrapper
    cannot check this: Ramulator 2.1 exposes no organization through its
    memory-system interface.
    """
    cls = type(dram)
    org, _ = dram.resolve()
    tx = cls.data_payload_bytes or (
        cls.internal_prefetch_size * org["channel_width"] // 8
    )
    capacity = tx
    for level in list(cls.levels)[1:]:
        count = org[level.lower()]
        if level == "Column":
            count //= cls.internal_prefetch_size
        capacity *= count
    return capacity


class Ramulator2Memory(AbstractMemorySystem):
    """A Ramulator 2.1 memory system with one gem5 port per channel.

    :param memory_system: a Ramulator 2.1 ``GenericDRAM`` component (or its
        configuration dictionary) without a channel mapper. Its controllers
        are the channels; their count must be a power of two.
    :param size: the memory size. It must fit the channels' capacity; a
        smaller size only warns, as gem5's native DRAM interfaces do.
    :param interleaving_size: bytes per channel interleave. Default: the
        board's cache line size. At least one cache line, a power of two.
    :param static_frontend_latency: charged on every response.
    :param static_backend_latency: charged on reads.
    """

    def __init__(
        self,
        memory_system,
        size: Union[str, int],
        interleaving_size: Optional[Union[str, int]] = None,
        # Verification using a gem5 native model: these defaults equal
        # MemCtrl's static latencies (src/mem/MemCtrl.py), and were checked
        # against native DDR4_2400_8x8 on the same device.
        static_frontend_latency: str = "10ns",
        static_backend_latency: str = "10ns",
    ) -> None:
        super().__init__()
        self._memory_system = memory_system
        self._size = toMemorySize(size) if isinstance(size, str) else size

        if isinstance(memory_system, Component):
            controllers = memory_system.controllers
        else:
            controllers = memory_system.get("controllers")
        if not isinstance(controllers, list) or not controllers:
            raise ValueError(
                "Ramulator2Memory needs a memory system with at least one "
                "controller"
            )
        self._channels = len(controllers)
        if not _is_power_of_two(self._channels):
            raise ValueError(
                "Ramulator2Memory needs a power-of-two number of channels "
                f"(controllers), not {self._channels}"
            )
        self._check_capacity(controllers)

        if interleaving_size is None:
            self._interleaving_size = None
        elif isinstance(interleaving_size, str):
            self._interleaving_size = toMemorySize(interleaving_size)
        else:
            self._interleaving_size = int(interleaving_size)

        self.ramulator2 = Ramulator2VectorPorts(
            static_frontend_latency=static_frontend_latency,
            static_backend_latency=static_backend_latency,
        )
        if isinstance(memory_system, Component):
            self.ramulator2.atomic_read_latency = "%dps" % (
                closed_page_read_latency(controllers[0].dram)
            )
        self._port_ranges = []
        self._mem_range = None
        if self._interleaving_size is not None:
            self._set_interleaving(self._interleaving_size)

    def _check_capacity(self, controllers) -> None:
        drams = [
            getattr(c, "dram", None)
            for c in controllers
            if isinstance(c, Component)
        ]
        if len(drams) != len(controllers) or any(
            not hasattr(d, "resolve") for d in drams
        ):
            # A configuration dictionary: nothing to resolve.
            return
        capacity = sum(channel_capacity(d) for d in drams)
        if self._size > capacity:
            per_channel = channel_capacity(drams[0])
            needed = -(-self._size // per_channel)
            needed_pow2 = 1 << (needed - 1).bit_length()
            raise ValueError(
                f"Ramulator2Memory: {self._size} bytes do not fit "
                f"{self._channels} channel(s) of {per_channel // 2**20} MiB "
                f"({type(drams[0]).__name__}); use at least {needed_pow2} "
                "channels or a larger device"
            )
        if self._size < capacity:
            warn(
                f"Ramulator2Memory: the memory ({self._size} bytes) is "
                f"smaller than its DRAM devices ({capacity} bytes); the top "
                "rows are not used."
            )

    def _set_interleaving(self, interleaving_size: int) -> None:
        if not _is_power_of_two(interleaving_size):
            raise ValueError(
                "Ramulator2Memory: interleaving_size must be a power of two, "
                f"not {interleaving_size}"
            )
        self._interleaving_size = interleaving_size
        self.ramulator2.ramulator_config = ramulator2_config(
            self._memory_system, interleaving_size
        )
        self._update_port_ranges()

    def _update_port_ranges(self) -> None:
        # Boards set the range in their constructor, before
        # incorporate_memory() knows the default interleave.
        if self._interleaving_size is None or self._mem_range is None:
            return
        self._port_ranges = interleaved_port_ranges(
            self._mem_range, self._channels, self._interleaving_size
        )
        self.ramulator2.port_ranges = self._port_ranges

    @overrides(AbstractMemorySystem)
    def incorporate_memory(self, board: AbstractBoard) -> None:
        cache_line = int(board.get_cache_line_size())
        if self._interleaving_size is None:
            self._set_interleaving(cache_line)
        if self._interleaving_size < cache_line:
            raise ValueError(
                "Ramulator2Memory: interleaving_size "
                f"({self._interleaving_size}) cannot be smaller than the "
                f"board's cache line ({cache_line})"
            )

    @overrides(AbstractMemorySystem)
    def get_mem_ports(self) -> Sequence[Tuple[AddrRange, Port]]:
        return [
            (self._port_ranges[i], self.ramulator2.ports[i])
            for i in range(self._channels)
        ]

    @overrides(AbstractMemorySystem)
    def get_memory_controllers(self) -> List:
        return [self.ramulator2]

    @overrides(AbstractMemorySystem)
    def get_size(self) -> int:
        return self._size

    @overrides(AbstractMemorySystem)
    def set_memory_range(self, ranges: List[AddrRange]) -> None:
        if len(ranges) != 1 or ranges[0].size() != self._size:
            raise Exception(
                "Ramulator2Memory needs one address range of its size "
                f"({self._size} bytes)."
            )
        self._mem_range = ranges[0]
        self.ramulator2.range = ranges[0]
        self._update_port_ranges()

    @overrides(AbstractMemorySystem)
    def get_uninterleaved_range(self) -> List[AddrRange]:
        return [self.ramulator2.range]


def _descriptor_preset(standard: str) -> Tuple[str, str, str, Dict]:
    """The device of a PIM standard's descriptor
    (ext/ramulator2/python/ramulator/dram/pim.py, the one source):
    organization and timing presets, controller class, and the
    organization and host timing overrides as DRAM overrides."""
    desc = DESCRIPTORS[standard]
    p = desc.presets
    kw = dict(p["org_overrides"])
    kw.update({k: v for k, (v, _) in desc.host_overrides.items()})
    return (p["org"], p["timing"], p["controller"], kw)


# Organization preset, timing preset, controller class and DRAM overrides
# per standard. The four PIM standards take the PIM memory's device from
# the descriptors, so that a plain preset and the PIM memory of a standard
# are the same device: GDDR7 is GDDR7_32Gb_x8 with the descriptor's refresh
# times, and HBM4 uses the timing preset HBM4_8000Mbps_folded_banks_baseline.
_PRESETS: Dict[str, Tuple[str, str, str, Dict]] = {
    "DDR4": (
        "DDR4_8Gb_x8",
        "DDR4_2400R",
        "GenericDDR",
        dict(rank=2, nCL=17, nRCD=17, nRP=17),
    ),
    "DDR5": _descriptor_preset("DDR5"),
    "LPDDR5": ("LPDDR5_8Gb_x16", "LPDDR5_6400", "LPDDR5", {}),
    "LPDDR6": _descriptor_preset("LPDDR6"),
    "GDDR6": ("GDDR6_8Gb_x16", "GDDR6_14000_1250mV_double", "GenericDDR", {}),
    "GDDR7": _descriptor_preset("GDDR7"),
    "HBM2": ("HBM2_2Gb", "HBM2_2000Mbps", "HBM12", {}),
    "HBM4": _descriptor_preset("HBM4"),
}

STANDARDS = tuple(_PRESETS)


def ramulator2_memory(
    standard: str,
    channels: int,
    size: Union[str, int],
    org_preset: Optional[str] = None,
    timing_preset: Optional[str] = None,
    controller: Optional[str] = None,
    scheduler: str = "FRFCFS",
    refresh_manager: str = "AllBank",
    row_policy: str = "Open",
    addr_mapper: str = "RoBaRaCoCh",
    read_buffer_size: int = 64,
    write_buffer_size: int = 128,
    dram_overrides: Optional[Dict] = None,
    **memory_kwargs,
) -> Ramulator2Memory:
    """A Ramulator2Memory from a per-standard preset.

    :param standard: one of ``STANDARDS``.
    :param channels: the number of channels (controllers), a power of two.
    :param size: the memory size.
    :param dram_overrides: DRAM organization or timing values, replacing the
        preset's (for DDR4 the preset sets rank=2 and nCL, nRCD and nRP to
        17).
    :param memory_kwargs: passed on to ``Ramulator2Memory`` (for example
        ``interleaving_size`` or the static latencies).
    """
    if standard not in _PRESETS:
        raise ValueError(
            f"Unknown standard {standard}; choose one of {', '.join(STANDARDS)}"
        )
    org, tim, ctrl, kw = _PRESETS[standard]
    kw = dict(kw)
    if dram_overrides:
        kw.update(dram_overrides)

    def make_controller():
        dram = getattr(ramulator.dram, standard)(
            org_preset=org_preset or org,
            timing_preset=timing_preset or tim,
            **kw,
        )
        return getattr(ramulator.controller, controller or ctrl)(
            dram=dram,
            scheduler=getattr(ramulator.scheduler, scheduler)(),
            refresh_manager=getattr(
                ramulator.refresh_manager, refresh_manager
            )(),
            row_policy=getattr(ramulator.row_policy, row_policy)(),
            addr_mapper=getattr(ramulator.addr_mapper, addr_mapper)(),
            read_buffer_size=read_buffer_size,
            write_buffer_size=write_buffer_size,
        )

    memory_system = ramulator.memory_system.GenericDRAM(
        clock_ratio=1, controllers=[make_controller() for _ in range(channels)]
    )
    return Ramulator2Memory(memory_system, size, **memory_kwargs)
