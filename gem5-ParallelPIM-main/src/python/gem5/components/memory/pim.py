"""Processing-in-memory (PIM) memories on Ramulator 2.1.

Available only in builds with the Kconfig option USE_RAMULATOR2 (for example
build/RISCV_RAMULATOR2), as ``gem5.components.memory.ramulator2``.

* ``PimMemory``: a Ramulator 2.1 memory whose whole range is PIM memory:
  the derived PIM standard, one controller per channel, each with a
  ``PimSequencer``, ``PimDRAM`` as the memory system and one
  ``PimRamulator2`` wrapper with one port per channel. Its size is the
  devices' capacity, its base a multiple of its size, and it is not
  reported to the system-call emulation (SE) frame pool (Way 1).
* ``PimSystemMemory``: what the board sees. An operating-system (OS) memory,
  a plain ``Ramulator2Memory`` with the same address-mapping policy (by
  default the same standard, organization and channel count), and a
  ``PimMemory`` placed at the first multiple of its size above the OS
  memory. ``get_size()`` is the OS memory's, so the board's memory ranges
  and the full-system (FS) device tree describe the OS memory only. Its
  ``_pre_instantiate`` hook gives the caches the dirty-read knob, the
  prefetchers the PIM memory as an excluded range and every SE process the
  emulated ``/dev/mem`` (``PimMemDriver``) over the PIM memory.
  With ``way=2`` (FS only) the PIM memory is instead a non-uniform memory
  access (NUMA) node of the OS:
  ``get_numa_nodes()`` gives the OS memory as node 0 and the PIM memory
  without its register rows as node 1, and only the register rows are
  reserved. It needs a policy with the row on top.
* ``pim_lpddr6_memory()``, ``pim_ddr5_memory()``, ``pim_gddr7_memory()``,
  ``pim_hbm4_memory()``: a ``PimSystemMemory`` from each standard's preset.

The address-mapping policies are gem5's ``RoRaBaChCo``, ``RoRaBaCoCh`` and
``RoCoRaBaCh`` (ported to Ramulator 2.1) and Ramulator 2.1's ``RoBaRaCoCh``,
``ChRaBaRoCo`` and ``MOP4CLXOR``; one policy holds for both memories. The
channel interleave is one rank-wide row under ``RoRaBaChCo``, one channel's
capacity under ``ChRaBaRoCo`` (each channel one contiguous range), and
``channel_interleave`` (default one cache line) under the others.

The PIM library's layout header (``include/gem5/pim/pim_layout_<std>.h``,
written by ``libpim/gen_pim_layout.py``) is checked at start-up when
``layout_file`` names it: every ``PIM_LAYOUT_<NAME>`` constant must equal
the memory's own value.
"""

import json
import re
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import ramulator
from ramulator.dram.pim import DESCRIPTORS

from m5.objects import (
    AddrRange,
    BaseCache,
    BasePrefetcher,
    PimMemDriver,
    PimRamulator2,
    Port,
    Process,
    RiscvISA,
)
from m5.util import warn
from m5.util.convert import toMemorySize

from ...utils.override import overrides
from ..boards.abstract_board import AbstractBoard
from .abstract_memory_system import AbstractMemorySystem
from .ramulator2 import (
    _PRESETS,
    Ramulator2Memory,
    closed_page_read_latency,
    interleaved_port_ranges,
    ramulator2_config,
)

# gem5's three policies in the order of its AddrMap enum, then Ramulator
# 2.1's three; the layout header's PIM_LAYOUT_POLICY is the index.
POLICIES = (
    "RoRaBaChCo",
    "RoRaBaCoCh",
    "RoCoRaBaCh",
    "RoBaRaCoCh",
    "ChRaBaRoCo",
    "MOP4CLXOR",
)
DEFAULT_POLICY = "RoRaBaChCo"
# Way 2: the policies with the row on top, so that a row index is a
# contiguous range of the PIM memory and its top rows (the registers) are
# its top.
ROW_ON_TOP = (
    "RoRaBaChCo",
    "RoRaBaCoCh",
    "RoCoRaBaCh",
    "RoBaRaCoCh",
    "MOP4CLXOR",
)

CACHE_LINE_BYTES = 64


# The PIM standards' presets: organization and timing presets, controller
# class, DRAM overrides (the organization's and the host timing overrides)
# and the default channel count. They are read from the per-standard
# descriptors in ext/ramulator2/python/ramulator/dram/pim.py, the one
# source that libpim/gen_pim_layout.py and libpim/timing/pim_timing.py read
# too.
def _pim_preset(desc) -> dict:
    p = desc.presets
    overrides = dict(p["org_overrides"])
    overrides.update({k: v for k, (v, _) in desc.host_overrides.items()})
    return dict(
        org=p["org"],
        timing=p["timing"],
        controller=p["controller"],
        overrides=overrides,
        channels=p["channels"],
    )


PIM_PRESETS: Dict[str, dict] = {
    std: _pim_preset(desc) for std, desc in DESCRIPTORS.items()
}
PIM_STANDARDS = tuple(PIM_PRESETS)

# The form of every constant gem5 reads from the layout header.
LAYOUT_LINE_RE = re.compile(
    r"^\s*#\s*define\s+PIM_LAYOUT_(\w+)\s+(0x[0-9a-fA-F]+|\d+)[uUlL]*\s*$"
)


def _pow2(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _bytes(value: Union[str, int, None]) -> Optional[int]:
    if value is None:
        return None
    return toMemorySize(value) if isinstance(value, str) else int(value)


def read_layout(path: str) -> List[str]:
    """The constants of a layout header as ``NAME=VALUE`` strings."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LAYOUT_LINE_RE.match(line)
            if m:
                out.append(f"{m.group(1)}={int(m.group(2), 0)}")
    if not out:
        raise ValueError(f"{path}: no PIM_LAYOUT_ constant")
    return out


class _Device:
    """One channel's DRAM device of a standard: how to build it and its
    geometry, as gem5's stdlib (``channel_capacity``) computes it."""

    def __init__(self, cls, org: str, timing: str, kw: dict) -> None:
        self.cls, self.org, self.timing, self.kw = cls, org, timing, kw
        dev = self.make()
        o, _ = dev.resolve()
        self.col_bytes = cls.data_payload_bytes or (
            cls.internal_prefetch_size * o["channel_width"] // 8
        )
        self.row_bytes = (
            o["column"] // cls.internal_prefetch_size * self.col_bytes
        )
        capacity = self.col_bytes
        for level in list(cls.levels)[1:]:
            count = o[level.lower()]
            if level == "Column":
                count //= cls.internal_prefetch_size
            capacity *= count
        self.capacity = capacity
        self.rows = o["row"]
        # The rank count (None for a standard without a rank level).
        self.ranks = o.get("rank") if "Rank" in cls.levels else None
        self.atomic_read_latency = closed_page_read_latency(dev)

    def make(self):
        return self.cls(
            org_preset=self.org, timing_preset=self.timing, **self.kw
        )


# The standards whose rank count PimMemory takes: DDR5, with 1 or 2 ranks
# per channel. LPDDR6 has one rank, GDDR7
# and HBM4 none.
RANKS = {"DDR5": (1, 2)}


def default_ranks(standard: str) -> Optional[int]:
    """The rank count of a standard's preset (None without ranks)."""
    p = PIM_PRESETS[standard]
    return _Device(
        getattr(ramulator.dram, standard),
        p["org"],
        p["timing"],
        dict(p["overrides"]),
    ).ranks


def policy_interleave(
    policy: str, device: _Device, interleave: Optional[int] = None
) -> int:
    """The channel interleave of a policy, checked as
    libpim/gen_pim_layout.py checks it."""
    if policy not in POLICIES:
        raise ValueError(
            f"unknown policy {policy}; choose one of {', '.join(POLICIES)}"
        )
    fixed = {
        "RoRaBaChCo": device.row_bytes,
        "ChRaBaRoCo": device.capacity,
    }.get(policy)
    if fixed is not None:
        if interleave is not None and interleave != fixed:
            raise ValueError(
                f"{policy} fixes the channel interleave at {fixed} bytes "
                f"(one {'row' if policy == 'RoRaBaChCo' else 'channel'}); "
                "do not set channel_interleave"
            )
        return fixed
    value = CACHE_LINE_BYTES if interleave is None else interleave
    if not _pow2(value):
        raise ValueError(f"channel interleave {value} is not a power of two")
    if value < CACHE_LINE_BYTES or value < device.col_bytes:
        raise ValueError(
            f"channel interleave {value} is below one cache line "
            f"({CACHE_LINE_BYTES} bytes) or one column ({device.col_bytes})"
        )
    if value > device.capacity:
        raise ValueError(
            f"channel interleave {value} is above one channel's capacity "
            f"({device.capacity} bytes)"
        )
    if policy in ("RoRaBaCoCh", "RoCoRaBaCh") and value > device.row_bytes:
        raise ValueError(
            f"{policy} needs a channel interleave of at most one row "
            f"({device.row_bytes} bytes), as gem5 does"
        )
    return value


def _mapper(policy: str, interleave: int, channels: int):
    if policy == "RoCoRaBaCh":
        # The column is split at the channel stripe; with one channel gem5
        # stripes by one burst (src/mem/mem_interface.cc:64-65).
        return ramulator.addr_mapper.RoCoRaBaCh(
            stripe_size=interleave if channels > 1 else 0
        )
    return getattr(ramulator.addr_mapper, policy)()


def _controllers(
    device: _Device,
    controller: str,
    channels: int,
    policy: str,
    interleave: int,
    row_policy: str,
    scheduler: str,
    refresh_manager: str,
    read_buffer_size: int,
    write_buffer_size: int,
    plugins=None,
):
    def one():
        kw = {}
        if plugins is not None:
            kw["controller_plugins"] = plugins()
        return getattr(ramulator.controller, controller)(
            dram=device.make(),
            scheduler=getattr(ramulator.scheduler, scheduler)(),
            refresh_manager=getattr(
                ramulator.refresh_manager, refresh_manager
            )(),
            row_policy=getattr(ramulator.row_policy, row_policy)(),
            addr_mapper=_mapper(policy, interleave, channels),
            read_buffer_size=read_buffer_size,
            write_buffer_size=write_buffer_size,
            **kw,
        )

    return [one() for _ in range(channels)]


class PimMemory(AbstractMemorySystem):
    """A Ramulator 2.1 memory whose whole range is PIM memory.

    :param standard: one of ``PIM_STANDARDS`` (default LPDDR6).
    :param channels: a power of two; default the standard's (DDR5 2, the
        others 8).
    :param size: must be the devices' capacity (the default).
    :param policy: one of ``POLICIES`` (default ``RoRaBaChCo``).
    :param channel_interleave: bytes, for the policies that leave it open.
    :param row_policy, scheduler, refresh_manager, read_buffer_size,
        write_buffer_size: Ramulator 2.1 controller settings.
    :param ab_mode, arb_policy, close_rows: the ``PimSequencer``'s;
        ``close_rows`` defaults to the row policy (``ClosedCAP`` closes
        rows), and ``False`` with ``ClosedCAP`` is refused.
    :param posted, values, num_fmt, acc_fmt, num_regs, pim_queue_size: the
        ``PimRamulator2`` wrapper's.
    :param dirty_read_fatal: set on every cache by ``PimSystemMemory``.
    :param layout_file: the PIM library's layout header to check at
        start-up (None: no check, with a warning).
    :param ranks: DDR5's rank count per channel, 1 or 2;
        None keeps the preset's (one rank). The OS memory follows it.
    :param overrides: DRAM organization or timing values (including the
        PIM timing parameters), replacing the preset's.
    :param cmd_trace: a path; record every DRAM command there
        (``CmdTraceRecorder``, a test tool).
    """

    def __init__(
        self,
        standard: str = "LPDDR6",
        channels: Optional[int] = None,
        size: Union[str, int, None] = None,
        ranks: Optional[int] = None,
        policy: str = DEFAULT_POLICY,
        channel_interleave: Union[str, int, None] = None,
        row_policy: str = "Open",
        scheduler: str = "FRFCFS",
        refresh_manager: str = "AllBank",
        read_buffer_size: int = 64,
        write_buffer_size: int = 128,
        ab_mode: str = "baseline",
        arb_policy: str = "frfcfs",
        close_rows: Optional[bool] = None,
        posted: bool = True,
        values: bool = True,
        num_fmt: str = "bf16",
        acc_fmt: str = "fp32",
        num_regs: int = 1,
        pim_queue_size: int = 32,
        dirty_read_fatal: bool = True,
        layout_file: Optional[str] = None,
        overrides: Optional[Dict] = None,
        cmd_trace: Optional[str] = None,
        # Verification using a gem5 native model: these defaults equal
        # MemCtrl's static latencies (src/mem/MemCtrl.py), as
        # Ramulator2Memory's.
        static_frontend_latency: str = "10ns",
        static_backend_latency: str = "10ns",
    ) -> None:
        super().__init__()
        # Under a closing row policy PIM rows close too
        # (SB columns with auto-precharge, AB jobs with their PREs).
        closing = row_policy == "ClosedCAP"
        if close_rows is None:
            close_rows = closing
        elif closing and not close_rows:
            raise ValueError(
                "PimMemory: row_policy ClosedCAP closes PIM rows too; "
                "close_rows=False contradicts it"
            )
        if standard not in PIM_PRESETS:
            raise ValueError(
                f"PimMemory: unknown standard {standard}; choose one of "
                f"{', '.join(PIM_STANDARDS)}"
            )
        preset = PIM_PRESETS[standard]
        self._standard = standard
        self._channels = preset["channels"] if channels is None else channels
        if not _pow2(self._channels):
            raise ValueError(
                f"PimMemory: {self._channels} channels is not a power of two"
            )
        kw = dict(preset["overrides"])
        if ranks is not None:
            if standard not in RANKS:
                raise ValueError(
                    f"PimMemory: ranks is a "
                    f"{', '.join(RANKS)} option; {standard} has no rank "
                    "count to choose"
                )
            if ranks not in RANKS[standard]:
                raise ValueError(
                    f"PimMemory: {standard} ranks must be one of "
                    f"{RANKS[standard]}, not {ranks}"
                )
            kw["rank"] = ranks
        self._ranks = ranks
        if overrides:
            kw.update(overrides)
        self._device = _Device(
            getattr(ramulator.dram, standard + "_PIM"),
            preset["org"],
            preset["timing"],
            kw,
        )
        capacity = self._channels * self._device.capacity
        self._size = capacity if size is None else _bytes(size)
        if self._size != capacity:
            raise ValueError(
                f"PimMemory: the size ({self._size} bytes) must be the "
                f"capacity of {self._channels} {standard} channels "
                f"({capacity} bytes, {capacity // 2**30} GiB): the whole "
                "memory is PIM memory"
            )
        self._policy = policy
        self._num_regs = num_regs
        self._interleave = policy_interleave(
            policy, self._device, _bytes(channel_interleave)
        )
        self._dirty_read_fatal = dirty_read_fatal
        self._layout_file = layout_file
        self._controller_settings = dict(
            row_policy=row_policy,
            scheduler=scheduler,
            refresh_manager=refresh_manager,
            read_buffer_size=read_buffer_size,
            write_buffer_size=write_buffer_size,
        )
        self._static_latencies = dict(
            static_frontend_latency=static_frontend_latency,
            static_backend_latency=static_backend_latency,
        )

        def plugins():
            out = [
                ramulator.controller_plugin.PimSequencer(
                    close_rows=close_rows,
                    ab_mode=ab_mode,
                    arb_policy=arb_policy,
                )
            ]
            if cmd_trace:
                out.append(
                    ramulator.controller_plugin.CmdTraceRecorder(
                        path=cmd_trace
                    )
                )
            return out

        memory_system = ramulator.memory_system.PimDRAM(
            clock_ratio=1,
            controllers=_controllers(
                self._device,
                preset["controller"],
                self._channels,
                policy,
                self._interleave,
                plugins=plugins,
                **self._controller_settings,
            ),
        )
        self.ramulator2 = PimRamulator2(
            ramulator_config=ramulator2_config(
                memory_system, self._interleave
            ),
            policy=policy,
            channel_interleave=self._interleave,
            num_regs=num_regs,
            num_fmt=num_fmt,
            acc_fmt=acc_fmt,
            values=values,
            posted=posted,
            pim_queue_size=pim_queue_size,
            atomic_read_latency="%dps" % self._device.atomic_read_latency,
            # Way 1: never in the SE frame pool.
            conf_table_reported=False,
            **self._static_latencies,
        )
        if layout_file:
            self.ramulator2.expected_layout = read_layout(layout_file)
        else:
            warn(
                "PimMemory: no layout_file, so the PIM library's layout "
                "header is not checked at start-up"
            )
        self._range = None
        self._port_ranges = []

    def get_range(self) -> Optional[AddrRange]:
        """The PIM memory's physical range, once set."""
        return self._range

    def get_register_range(self) -> AddrRange:
        """The register rows (the top ``num_regs`` + 1 rows of every bank,
        the layout header's ``REG_ROW0`` up) as one physical range,
        which they are under a policy with the row on top."""
        if self._policy not in ROW_ON_TOP:
            raise ValueError(
                f"PimMemory: under {self._policy} the register rows are not "
                "one range (the row is not on top)"
            )
        if self._range is None:
            raise ValueError("PimMemory: the range is not set yet")
        slice_bytes = self._size // self._device.rows
        size = (self._num_regs + 1) * slice_bytes
        return AddrRange(int(self._range.end) - size, size=size)

    def get_standard(self) -> str:
        return self._standard

    def get_channels(self) -> int:
        return self._channels

    def get_ranks(self) -> Optional[int]:
        """The rank count per channel (None for a standard without
        ranks)."""
        return self._device.ranks

    def get_policy(self) -> str:
        return self._policy

    def get_interleave(self) -> int:
        """The channel interleave in bytes."""
        return self._interleave

    def get_device(self) -> "_Device":
        """One channel's DRAM device and its geometry."""
        return self._device

    @overrides(AbstractMemorySystem)
    def incorporate_memory(self, board: AbstractBoard) -> None:
        line = int(board.get_cache_line_size())
        if line != CACHE_LINE_BYTES:
            raise ValueError(
                f"PimMemory: the board's cache line ({line} bytes) must be "
                f"{CACHE_LINE_BYTES} bytes, the line the layout assumes"
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
            raise ValueError(
                f"PimMemory needs one address range of its size "
                f"({self._size} bytes)"
            )
        if int(ranges[0].start) % self._size:
            raise ValueError(
                f"PimMemory: the base {int(ranges[0].start):#x} must be a "
                f"multiple of the size ({self._size:#x}), so that offset "
                "bits are physical address bits"
            )
        self._range = ranges[0]
        self.ramulator2.range = ranges[0]
        self._port_ranges = interleaved_port_ranges(
            ranges[0], self._channels, self._interleave
        )
        self.ramulator2.port_ranges = self._port_ranges

    @overrides(AbstractMemorySystem)
    def get_uninterleaved_range(self) -> List[AddrRange]:
        return [self._range]

    def get_reserved_ranges(self) -> List[AddrRange]:
        return [self._range] if self._range is not None else []


def _os_mapping(memory: Ramulator2Memory) -> Tuple[str, Optional[int]]:
    """The policy and the stripe of an OS memory's controllers, from its
    Ramulator 2.1 configuration."""
    config = json.loads(memory.ramulator2.ramulator_config)
    mappers = {
        json.dumps(c.get("addr_mapper"), sort_keys=True)
        for c in config["memory_system"]["controllers"]
    }
    if len(mappers) != 1:
        raise ValueError(
            "PimSystemMemory: the OS memory's controllers use different "
            "address mappers"
        )
    mapper = json.loads(mappers.pop()) or {}
    return mapper.get("impl"), mapper.get("stripe_size")


def pim_os_memory(
    pim: PimMemory,
    standard: Optional[str] = None,
    channels: Optional[int] = None,
    size: Union[str, int, None] = None,
) -> Ramulator2Memory:
    """The OS memory of a ``PimSystemMemory``: a plain
    ``Ramulator2Memory`` with the PIM memory's policy and controller
    settings; by default its standard, organization and channel count, and
    the size of its capacity; its rank count follows the PIM memory's
    ``ranks``."""
    standard = standard or pim._standard
    channels = pim._channels if channels is None else channels
    if standard == pim._standard:
        preset = PIM_PRESETS[standard]
        kw = dict(preset["overrides"])
        if pim._ranks is not None:
            kw["rank"] = pim._ranks
        device = _Device(
            getattr(ramulator.dram, standard),
            preset["org"],
            preset["timing"],
            kw,
        )
        controller = preset["controller"]
    else:
        if standard not in _PRESETS:
            raise ValueError(
                f"PimSystemMemory: unknown OS standard {standard}"
            )
        org, tim, controller, kw = _PRESETS[standard]
        device = _Device(getattr(ramulator.dram, standard), org, tim, dict(kw))
    # The same policy; a fixed interleave is the OS device's own row or
    # channel, an open one the PIM memory's value.
    fixed = pim._policy in ("RoRaBaChCo", "ChRaBaRoCo")
    interleave = policy_interleave(
        pim._policy, device, None if fixed else pim._interleave
    )
    memory_system = ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=_controllers(
            device,
            controller,
            channels,
            pim._policy,
            interleave,
            **pim._controller_settings,
        ),
    )
    size = channels * device.capacity if size is None else _bytes(size)
    return Ramulator2Memory(
        memory_system,
        size,
        interleaving_size=interleave,
        **pim._static_latencies,
    )


class _PimConfig:
    """The ``_pre_instantiate`` hook: what the rest of the board needs to
    know about the PIM memory."""

    def __init__(self, pim: PimMemory) -> None:
        self.pim = pim

    def apply(self, root) -> None:
        board = root.board
        window = self.pim.get_range()
        vlen_warned = False
        for obj in board.descendants():
            if isinstance(obj, BaseCache):
                obj.pim_dirty_read_fatal = self.pim._dirty_read_fatal
            elif isinstance(obj, BasePrefetcher):
                obj.excluded_ranges = list(obj.excluded_ranges) + [window]
            elif isinstance(obj, Process):
                # SE: /dev/mem maps the whole PIM memory linearly at a
                # virtual base aligned to its size (Way 1).
                obj.drivers = list(obj.drivers) + [
                    PimMemDriver(window=window, alignment=window.size())
                ]
            elif isinstance(obj, RiscvISA) and not vlen_warned:
                # A vector PIM instruction moves at most VLENB
                # bytes, so below one column it covers partial columns.
                vlenb = int(obj.vlen) // 8
                if vlenb < self.pim._device.col_bytes:
                    warn(
                        f"PimSystemMemory: VLENB ({vlenb} bytes) is below "
                        f"the {self.pim._standard} column "
                        f"({self.pim._device.col_bytes} bytes); vector PIM "
                        "instructions then cover partial columns"
                    )
                    vlen_warned = True


class PimSystemMemory(AbstractMemorySystem):
    """An OS memory and a PIM memory.

    :param pim_memory: the ``PimMemory``.
    :param os_memory: the OS memory; default ``pim_os_memory(pim_memory,
        os_standard, os_channels, os_size)``. Its policy must be the PIM
        memory's, and so must its interleave where the policy leaves it
        open.
    :param way: 1 (default), the PIM memory hidden from the OS and mapped
        whole; 2 (FS only), the PIM memory a NUMA node without its
        register rows (a policy with the row on top).
    """

    def __init__(
        self,
        pim_memory: PimMemory,
        os_memory: Optional[Ramulator2Memory] = None,
        os_standard: Optional[str] = None,
        os_channels: Optional[int] = None,
        os_size: Union[str, int, None] = None,
        way: int = 1,
    ) -> None:
        super().__init__()
        if way not in (1, 2):
            raise ValueError(f"PimSystemMemory: way {way}: use 1 or 2")
        if way == 2 and pim_memory._policy not in ROW_ON_TOP:
            raise ValueError(
                f"PimSystemMemory: Way 2 needs a policy with the row on top "
                f"({', '.join(ROW_ON_TOP)}), not {pim_memory._policy}"
            )
        self._way = way
        self._os_ranges = []
        if os_memory is None:
            os_memory = pim_os_memory(
                pim_memory, os_standard, os_channels, os_size
            )
        elif (os_standard, os_channels, os_size) != (None, None, None):
            raise ValueError(
                "PimSystemMemory: give os_memory or os_standard, "
                "os_channels and os_size, not both"
            )
        self.os_memory = os_memory
        self.pim_memory = pim_memory
        self._checked = False

    def check_one_policy(self) -> None:
        """Both memories use one policy and, where the policy leaves it
        open, one channel interleave."""
        policy, stripe = _os_mapping(self.os_memory)
        pim = self.pim_memory
        if policy != pim._policy:
            raise ValueError(
                f"PimSystemMemory: the OS memory uses {policy}, the PIM "
                f"memory {pim._policy}; both must use one policy"
            )
        if pim._policy not in ("RoRaBaChCo", "ChRaBaRoCo"):
            os_intlv = self.os_memory._interleaving_size
            if os_intlv != pim._interleave:
                raise ValueError(
                    f"PimSystemMemory: the OS memory's channel interleave "
                    f"({os_intlv} bytes) differs from the PIM memory's "
                    f"({pim._interleave} bytes) under {pim._policy}"
                )
            want = os_intlv if self.os_memory._channels > 1 else 0
            if pim._policy == "RoCoRaBaCh" and stripe != want:
                raise ValueError(
                    "PimSystemMemory: the OS memory's RoCoRaBaCh stripe "
                    f"({stripe}) is not its channel interleave ({want})"
                )
        self._checked = True

    @overrides(AbstractMemorySystem)
    def incorporate_memory(self, board: AbstractBoard) -> None:
        self.os_memory.incorporate_memory(board)
        self.pim_memory.incorporate_memory(board)
        self.check_one_policy()

    @overrides(AbstractMemorySystem)
    def get_mem_ports(self) -> Sequence[Tuple[AddrRange, Port]]:
        return list(self.os_memory.get_mem_ports()) + list(
            self.pim_memory.get_mem_ports()
        )

    @overrides(AbstractMemorySystem)
    def get_memory_controllers(self) -> List:
        return (
            self.os_memory.get_memory_controllers()
            + self.pim_memory.get_memory_controllers()
        )

    @overrides(AbstractMemorySystem)
    def get_size(self) -> int:
        """The OS memory's size: the board describes it alone."""
        return self.os_memory.get_size()

    @overrides(AbstractMemorySystem)
    def set_memory_range(self, ranges: List[AddrRange]) -> None:
        """The OS memory gets the board's range; the PIM memory starts at
        the first multiple of its size at or above its end."""
        self.os_memory.set_memory_range(ranges)
        self._os_ranges = list(ranges)
        end = max(int(r.end) for r in ranges)
        size = self.pim_memory.get_size()
        base = -(-end // size) * size
        self.pim_memory.set_memory_range([AddrRange(base, size=size)])

    @overrides(AbstractMemorySystem)
    def get_uninterleaved_range(self) -> List[AddrRange]:
        return self.os_memory.get_uninterleaved_range()

    def get_way(self) -> int:
        return self._way

    def get_reserved_ranges(self) -> List[AddrRange]:
        """Way 1: the PIM memory, not for the OS; Way 2: its register
        rows."""
        if self._way == 2:
            return [self.pim_memory.get_register_range()]
        return self.pim_memory.get_reserved_ranges()

    def get_numa_nodes(self) -> List[Tuple[int, List[AddrRange]]]:
        """Way 2: the OS memory as node 0, the PIM memory without its
        register rows as node 1. Way 1: none."""
        if self._way != 2:
            return []
        regs = self.pim_memory.get_register_range()
        return [
            (0, list(self._os_ranges)),
            (1, self.pim_memory.get_range().exclude([regs])),
        ]

    def _pre_instantiate(self, root) -> None:
        if not self._checked:
            self.check_one_policy()
        _PimConfig(self.pim_memory).apply(root)


def pim_memory_system(
    standard: str,
    os_standard: Optional[str] = None,
    os_channels: Optional[int] = None,
    os_size: Union[str, int, None] = None,
    way: int = 1,
    **pim_kwargs,
) -> PimSystemMemory:
    """A ``PimSystemMemory`` with a ``PimMemory`` of a standard's preset.

    :param way: 1 or 2 (``PimSystemMemory``).
    :param pim_kwargs: ``PimMemory``'s arguments (channels, policy,
        channel_interleave, ab_mode, layout_file, ...).
    """
    pim = PimMemory(standard=standard, **pim_kwargs)
    return PimSystemMemory(
        pim,
        os_standard=os_standard,
        os_channels=os_channels,
        os_size=os_size,
        way=way,
    )


def pim_lpddr6_memory(**kwargs) -> PimSystemMemory:
    """LPDDR6_16Gb_x12, LPDDR6_10667_BL24, 16 channels (32 GiB)."""
    return pim_memory_system("LPDDR6", **kwargs)


def pim_ddr5_memory(**kwargs) -> PimSystemMemory:
    """DDR5_16Gb_x8 with 1 rank, DDR5_4800AN, 4 channels (32 GiB);
    ``ranks=2`` gives 2 ranks per channel (64 GiB)."""
    return pim_memory_system("DDR5", **kwargs)


def pim_gddr7_memory(**kwargs) -> PimSystemMemory:
    """GDDR7_32Gb_x8, GDDR7_28000_PAM3, 32 channels (32 GiB)."""
    return pim_memory_system("GDDR7", **kwargs)


def pim_hbm4_memory(**kwargs) -> PimSystemMemory:
    """HBM4_32Gb_8Hi, HBM4_8000Mbps_folded_banks_baseline, 32 channels
    (32 GiB)."""
    return pim_memory_system("HBM4", **kwargs)
