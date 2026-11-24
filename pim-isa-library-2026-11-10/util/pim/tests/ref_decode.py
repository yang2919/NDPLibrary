#!/usr/bin/env python3
"""Reference decoders for the PIM address tests (P-LIB step L1).

Test aids, not the simulator's mapper files: each decoder transcribes the
simulator code it names, independently of gen_pim_layout.py's bit map.

- gem5's RoRaBaChCo, RoRaBaCoCh and RoCoRaBaCh follow
  DRAMInterface::decodePacket (src/mem/dram_interface.cc:856-895), with the
  channel ranges of src/python/gem5/components/memory/memory.py:146-176
  and AddrRange::getOffset / removeIntlvBits (src/base/addr_range.hh).
  The flat bank is split as bank group = bank % bank groups
  (dram_interface.cc:1140-1151), then bank, then stack ID; the rank is the
  PIM unit (P-LIB section 14, DP-A (a)).
- Ramulator 2.1's RoBaRaCoCh, ChRaBaRoCo and MOP4CLXOR follow
  ext/ramulator2/src/ramulator/controller/addr_mapper/impl/*.cpp and
  memory_system/channel_mapper/impl/gem5_port_interleave.cpp.

Usage:
  ref_decode.py vectors OUT    write test vectors for test_addr.c
  ref_decode.py selftest       check CONTIG_BYTES and the slice property
                               of every configuration by brute force
"""

import os
import random
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gen_pim_layout as g  # noqa: E402


def log2(x):
    return x.bit_length() - 1


class Geometry:
    """Counts of one configuration, as Ramulator 2.1 and gem5 see them."""

    def __init__(self, lay):
        c = lay.counts
        self.lay = lay
        self.policy = lay.policy
        self.channels = lay.channels
        self.intlv = lay.interleave
        self.units = lay.units
        self.sids = c["sid"] or 1
        self.bgs = c["bg"] or 1
        self.banks = c["bank"]
        self.rows = c["row"]
        self.cols = c["col"]
        self.col_bytes = lay.col_bytes
        self.capacity = lay.channel_capacity
        # Ramulator 2.1's mapped levels below the channel, in level order
        # (addr_mapper_base.cpp:10-18), with the column last.
        self.levels = []
        for field in g.HIERARCHY:
            if field in lay.org["present"]:
                self.levels.append((field, lay.counts[field]))
        self.levels.append(("col", self.cols))

    # gem5 --------------------------------------------------------------

    def gem5_channel_and_offset(self, addr):
        """memory.py:146-176 ranges and AddrRange::getOffset."""
        if self.policy == "RoRaBaChCo":
            low = log2(self.cols * self.col_bytes)
        else:
            low = log2(self.intlv)
        bits = log2(self.channels)
        if bits == 0:
            return 0, addr
        ch = (addr >> low) & (self.channels - 1)
        # removeIntlvBits: remove bits [low, low + bits), one at a time.
        a = addr
        for i in range(bits):
            bit = low  # the next bit to remove has moved down to `low`
            a = ((a >> (bit + 1)) << bit) | (a & ((1 << bit) - 1))
        return ch, a

    def gem5_decode(self, addr):
        ch, a = self.gem5_channel_and_offset(addr)
        burst = self.col_bytes
        byte = a % burst
        a //= burst
        bursts_per_row = self.cols
        banks_per_rank = self.sids * self.bgs * self.banks
        ranks = self.units
        if self.policy in ("RoRaBaChCo", "RoRaBaCoCh"):
            col = a % bursts_per_row
            a //= bursts_per_row
            bank = a % banks_per_rank
            a //= banks_per_rank
            rank = a % ranks
            a //= ranks
            row = a % self.rows
        else:  # RoCoRaBaCh
            stripe = self.intlv // burst if self.channels > 1 else 1
            assert stripe <= bursts_per_row
            col = a % stripe
            a //= stripe
            bank = a % banks_per_rank
            a //= banks_per_rank
            rank = a % ranks
            a //= ranks
            if stripe < bursts_per_row:
                col += (a % (bursts_per_row // stripe)) * stripe
                a //= bursts_per_row // stripe
            row = a % self.rows
        bg = bank % self.bgs
        rest = bank // self.bgs
        return dict(
            ch=ch,
            unit=rank,
            sid=rest // self.banks,
            bg=bg,
            bank=rest % self.banks,
            row=row,
            col=col,
            byte=byte,
        )

    # Ramulator 2.1 -----------------------------------------------------

    def port_interleave(self, addr):
        """gem5_port_interleave.cpp:50-60, with gem5's port ranges."""
        if self.policy == "ChRaBaRoCo":
            size = self.capacity  # one uninterleaved range per channel
        else:
            size = self.intlv
        shift = log2(size)
        width = log2(self.channels)
        ch = (addr >> shift) & (self.channels - 1)
        if self.channels == 1:
            return ch, addr
        low = addr & ((1 << shift) - 1)
        high = addr >> (shift + width)
        return ch, (high << shift) | low

    def ramulator_decode(self, addr):
        ch, intra = self.port_interleave(addr)
        tx = log2(self.col_bytes)
        a = intra >> tx
        names = [n for n, _ in self.levels]
        bits = [log2(c) for _, c in self.levels]
        n = len(bits)
        col_idx = n - 1
        row_idx = names.index("row")
        vec = [0] * n

        def take(w):
            nonlocal a
            v = a & ((1 << w) - 1)
            a >>= w
            return v

        if self.policy == "RoBaRaCoCh":
            vec[col_idx] = take(bits[col_idx])
            for i in range(row_idx + 1):
                vec[i] = take(bits[i])
        elif self.policy == "ChRaBaRoCo":
            for i in range(n - 1, -1, -1):
                vec[i] = take(bits[i])
        else:  # MOP4CLXOR
            vec[col_idx] = take(2)
            for i in range(row_idx):
                vec[i] = take(bits[i])
            vec[col_idx] += take(bits[col_idx] - 2) << 2
            vec[row_idx] = a
            x = 0
            for i in range(col_idx):
                if bits[i] > 0:
                    vec[i] ^= (vec[col_idx] >> x) & ((1 << bits[i]) - 1)
                    x += bits[i]
        out = dict(ch=ch, unit=0, sid=0, bg=0, bank=0, row=0, col=0)
        for name, v in zip(names, vec):
            out[name] = v
        out["byte"] = intra & (self.col_bytes - 1)
        return out

    def decode(self, addr):
        if self.policy in ("RoRaBaChCo", "RoRaBaCoCh", "RoCoRaBaCh"):
            return self.gem5_decode(addr)
        return self.ramulator_decode(addr)


FIELDS = ("ch", "unit", "sid", "bg", "bank", "row", "col", "byte")


def configurations():
    """Every standard x policy x interleave x channel count tested."""
    orgs = {s: g.resolve_organization(s) for s in g.STANDARDS}
    for std in g.STANDARDS:
        for policy in g.POLICIES:
            for channels in (1, 2, 8):
                seen = set()
                for intlv in (None, 64, 128, 256, 1024, 2048, 8192, 1 << 20):
                    try:
                        lay = g.Layout(
                            std,
                            channels=channels,
                            policy=policy,
                            interleave=intlv,
                            organization=orgs[std],
                        )
                    except g.LayoutError:
                        continue
                    if lay.interleave in seen:
                        continue
                    seen.add(lay.interleave)
                    yield lay


def geometry_line(lay):
    geo = Geometry(lay)
    return " ".join(
        str(v)
        for v in (
            lay.standard,
            g.POLICIES.index(lay.policy),
            geo.channels,
            geo.intlv,
            geo.units,
            geo.sids,
            geo.bgs,
            geo.banks,
            geo.rows,
            geo.cols,
            geo.col_bytes,
            lay.contig_bytes,
            lay.constants["ROW_SHIFT"],
        )
    )


def write_vectors(path, per_config=600, seed=1):
    rng = random.Random(seed)
    count = 0
    with open(path, "w") as f:
        f.write(
            "# geom std policy channels intlv units sids bank_groups "
            "banks rows cols col_bytes contig row_shift\n"
            "# off ch unit sid bg bank row col byte\n"
        )
        for lay in configurations():
            geo = Geometry(lay)
            size = lay.window_size
            f.write("geom " + geometry_line(lay) + "\n")
            offs = [0, size - 1, size - geo.col_bytes]
            offs += [1 << k for k in range(log2(size))]
            offs += [rng.randrange(size) for _ in range(per_config)]
            for off in offs:
                d = geo.decode(off)
                f.write(
                    "{:x} {}\n".format(
                        off, " ".join(str(d[k]) for k in FIELDS)
                    )
                )
                count += 1
    return count


def selftest():
    """CONTIG_BYTES is exact, and the slice property, by brute force."""
    rng = random.Random(2)
    failures = 0
    configs = 0
    for lay in configurations():
        configs += 1
        geo = Geometry(lay)
        c = lay.contig_bytes
        row = lay.row_bytes
        cb = lay.col_bytes

        def same_run(base, length):
            d0 = geo.decode(base)
            for k in range(1, length // cb):
                d = geo.decode(base + k * cb)
                for f in ("ch", "unit", "sid", "bg", "bank", "row"):
                    if d[f] != d0[f]:
                        return False
                if d["col"] != d0["col"] + k:
                    return False
            return True

        for _ in range(48):
            base = rng.randrange(lay.window_size // c) * c
            if not same_run(base, c):
                print(
                    "FAIL contig {}: block at {:#x}".format(
                        geometry_line(lay), base
                    )
                )
                failures += 1
                break
        # Maximality: some 2c-aligned block of 2c bytes is not one run.
        if 2 * c <= row:
            ok = any(
                not same_run(
                    rng.randrange(lay.window_size // (2 * c)) * 2 * c, 2 * c
                )
                for _ in range(64)
            )
            if not ok:
                print("FAIL contig not maximal: %s" % geometry_line(lay))
                failures += 1
        # Row on top: when the row is the top field and no row bit is
        # XORed, offset >> ROW_SHIFT is the row.
        shift = lay.constants["ROW_SHIFT"]
        xored = any(f == "row" for f, _ in lay.xor_sources()) and any(
            low < lay.width("col")
            for f, low in lay.xor_sources()
            if f == "row"
        )
        on_top = shift + log2(geo.rows) == log2(lay.window_size)
        if on_top and not xored:
            for _ in range(16):
                off = rng.randrange(lay.window_size)
                if geo.decode(off)["row"] != off >> shift:
                    print(
                        "FAIL row shift {} at {:#x}".format(
                            geometry_line(lay), off
                        )
                    )
                    failures += 1
                    break
    print("%d configurations, %d failures" % (configs, failures))
    return 1 if failures else 0


def main(argv):
    if len(argv) == 3 and argv[1] == "vectors":
        n = write_vectors(argv[2])
        print("wrote %d vectors to %s" % (n, argv[2]))
        return 0
    if len(argv) == 2 and argv[1] == "selftest":
        return selftest()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
