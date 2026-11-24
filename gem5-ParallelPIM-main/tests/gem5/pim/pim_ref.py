#!/usr/bin/env python3
"""Python reference of the Processing-in-Memory (PIM) commands.

The simulator side's reference:

- number formats as gem5's functional model
  (src/mem/ramulator2/pim/pim_numfmt.hh): fp32 arithmetic rounded to
  nearest even (RNE), SoftFloat's RISC-V NaN rules (an arithmetic NaN
  result is the canonical 0x7fc00000; fp16 NaNs are 0x7e00), bf16 by a
  16-bit shift, integers wrapping; MAC products summed sequentially in
  ascending element order in acc_fmt, one accumulator update per column;
- the commands' semantics and software errors, on a sparse byte image of
  the PIM memory, with the address decoder and encoder of
  include/gem5/pim/pim_addr.h's pim_geom_*() functions;
- the runner of the shared vectors in libpim/tests/vectors (format:
  libpim/tests/vectors/README.md); with --dump its output equals the C
  reference model's ``test_model --dump`` bit for bit;
- writers of PimTester scripts (src/cpu/testers/pim_tester/PimTester.py)
  with their expected values.

Usage:
  pim_ref.py --dump VECTORS...    print every expect line's actual values
  pim_ref.py VECTORS...           check every expect line
  pim_ref.py rdwr OUT --base B --size S [--seed N] [--count N]
                                  a normal read/write PimTester script
"""

import argparse
import math
import random
import struct
import sys

# Number formats ---------------------------------------------------------

FORMATS = ("bf16", "fp16", "fp32", "int8", "int16", "int32")
BF16, FP16, FP32, INT8, INT16, INT32 = range(6)
_BYTES = (2, 2, 4, 1, 2, 4)
F32_NAN = 0x7FC00000  # SoftFloat's RISC-V default NaN
F16_NAN = 0x7E00


def fmt_bytes(fmt):
    return _BYTES[fmt]


def fmt_is_float(fmt):
    return fmt <= FP32


def check_formats(num, acc, col_bytes):
    """None if num_fmt/acc_fmt suit a column of col_bytes, else the rule."""
    if fmt_is_float(num) != fmt_is_float(acc):
        return "acc_fmt and num_fmt must both be integer or both floating"
    if fmt_bytes(acc) < fmt_bytes(num):
        return "acc_fmt must be at least as wide as num_fmt"
    if col_bytes == 0 or col_bytes % fmt_bytes(num):
        return "the size of num_fmt must divide the burst size"
    return None


def _u2f(u):
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def _f2u(x):
    """fp32 bits of a Python float, rounded to nearest even."""
    if math.isnan(x):
        return F32_NAN
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        # The rounding went past the largest finite value.
        return 0xFF800000 if x < 0 else 0x7F800000


def _f16_to_f32(h):
    sign = (h & 0x8000) << 16
    exp = (h >> 10) & 0x1F
    man = h & 0x3FF
    if exp == 0x1F:
        return F32_NAN if man else sign | 0x7F800000
    if exp == 0:
        if man == 0:
            return sign
        exp = 1
        while not man & 0x400:
            man <<= 1
            exp -= 1
        man &= 0x3FF
    return sign | ((exp + 112) << 23) | (man << 13)


def _f32_to_f16(u):
    sign = (u >> 16) & 0x8000
    a = u & 0x7FFFFFFF
    if a > 0x7F800000:
        return F16_NAN
    if a >= 0x477FF000:
        return sign | 0x7C00
    if a < 0x38800000:
        if a < 0x33000000:
            return sign
        exp = a >> 23
        man = (a & 0x7FFFFF) | 0x800000
        shift = 126 - exp
        half = 1 << (shift - 1)
        rest = man & ((1 << shift) - 1)
        r = man >> shift
        if rest > half or (rest == half and r & 1):
            r += 1
        return sign | r
    r = (a - 0x38000000) >> 13
    rest = a & 0x1FFF
    if rest > 0x1000 or (rest == 0x1000 and r & 1):
        r += 1
    return sign | r


def to_f32(fmt, bits):
    """fp32 bits of a floating element's bits."""
    if fmt == BF16:
        return (bits << 16) & 0xFFFFFFFF
    if fmt == FP16:
        return _f16_to_f32(bits)
    return bits


def from_f32(fmt, u):
    """A floating format's bits of fp32 bits (RNE)."""
    if fmt == BF16:
        if (u & 0x7FFFFFFF) > 0x7F800000:
            return (u >> 16) | 0x0040
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF
    if fmt == FP16:
        return _f32_to_f16(u)
    return u


def f32_add(a, b):
    # A double holds the exact product and rounds sums once, so one
    # rounding to fp32 gives the correctly rounded result.
    return _f2u(_u2f(a) + _u2f(b))


def f32_mul(a, b):
    return _f2u(_u2f(a) * _u2f(b))


def load_bits(fmt, buf, i=0):
    return int.from_bytes(buf[i : i + fmt_bytes(fmt)], "little")


def store_bits(fmt, buf, i, bits):
    n = fmt_bytes(fmt)
    buf[i : i + n] = (bits & ((1 << (8 * n)) - 1)).to_bytes(n, "little")


def load_int(fmt, buf, i=0):
    v = load_bits(fmt, buf, i)
    sign = 1 << (8 * fmt_bytes(fmt) - 1)
    return (v ^ sign) - sign


def ew_element(fmt, mul, a, b):
    """One element-wise result from element bits a and b, as bits."""
    if fmt_is_float(fmt):
        x, y = to_f32(fmt, a), to_f32(fmt, b)
        return from_f32(fmt, f32_mul(x, y) if mul else f32_add(x, y))
    nbits = 8 * fmt_bytes(fmt)
    sign = 1 << (nbits - 1)
    x, y = (a ^ sign) - sign, (b ^ sign) - sign
    return (x * y if mul else x + y) & ((1 << nbits) - 1)


def mac_column(num, acc_fmt, xs, ys, acc):
    """acc (bits) + the sum of xs[i] * ys[i] (bits), one column update."""
    if not xs:
        return acc
    if not fmt_is_float(num):
        nbits = 8 * fmt_bytes(num)
        sign = 1 << (nbits - 1)
        part = sum(
            ((x ^ sign) - sign) * ((y ^ sign) - sign) for x, y in zip(xs, ys)
        )
        abits = 8 * fmt_bytes(acc_fmt)
        asign = 1 << (abits - 1)
        return ((acc ^ asign) - asign + part) & ((1 << abits) - 1)

    def in_acc(v):
        return to_f32(acc_fmt, from_f32(acc_fmt, v))

    partial = None
    for x, y in zip(xs, ys):
        a = in_acc(to_f32(num, x))
        b = in_acc(to_f32(num, y))
        p = in_acc(f32_mul(a, b))
        partial = p if partial is None else in_acc(f32_add(partial, p))
    return from_f32(acc_fmt, f32_add(to_f32(acc_fmt, acc), partial))


# Address decode and encode (pim_addr.h's pim_geom_*()) -------------------

POLICIES = (
    "RoRaBaChCo",
    "RoRaBaCoCh",
    "RoCoRaBaCh",
    "RoBaRaCoCh",
    "ChRaBaRoCo",
    "MOP4CLXOR",
)


def _log2(x):
    return (x & -x).bit_length() - 1 if x else 0


class Coord:
    __slots__ = ("ch", "unit", "sid", "bg", "bank", "row", "col")

    def __init__(self, ch=0, unit=0, sid=0, bg=0, bank=0, row=0, col=0):
        self.ch, self.unit, self.sid, self.bg = ch, unit, sid, bg
        self.bank, self.row, self.col = bank, row, col

    def copy(self, **kw):
        c = Coord(
            self.ch,
            self.unit,
            self.sid,
            self.bg,
            self.bank,
            self.row,
            self.col,
        )
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def key(self):
        return (
            self.ch,
            self.unit,
            self.sid,
            self.bg,
            self.bank,
            self.row,
            self.col,
        )


class Geometry:
    """pim_geom_t: every count a power of two."""

    def __init__(
        self,
        policy,
        channels,
        intlv,
        units,
        sids,
        bgs,
        banks,
        rows,
        cols,
        col_bytes,
    ):
        self.policy, self.channels, self.intlv = policy, channels, intlv
        self.units, self.sids, self.bgs, self.banks = units, sids, bgs, banks
        self.rows, self.cols, self.col_bytes = rows, cols, col_bytes
        b = self.b = {}
        b["tx"] = _log2(col_bytes)
        b["col"] = _log2(cols)
        b["ch"] = _log2(channels)
        b["s"] = _log2(intlv)
        b["unit"] = _log2(units)
        b["sid"] = _log2(sids)
        b["bg"] = _log2(bgs)
        b["bank"] = _log2(banks)
        b["row"] = _log2(rows)
        b["stripe"] = 0
        if channels > 1 and intlv > col_bytes:
            b["stripe"] = min(_log2(intlv // col_bytes), b["col"])

    def row_bytes(self):
        return self.cols * self.col_bytes

    def banks_per_unit(self):
        return self.sids * self.bgs * self.banks

    def size(self):
        return (
            self.channels
            * self.units
            * self.banks_per_unit()
            * self.row_bytes()
            * self.rows
        )

    def _fields(self):
        """(field, width) from the lowest bit, the column split as needed."""
        b = self.b
        p = self.policy
        if p in (0, 1):
            return [
                ("col", b["col"]),
                ("bg", b["bg"]),
                ("bank", b["bank"]),
                ("sid", b["sid"]),
                ("unit", b["unit"]),
                ("row", b["row"]),
            ]
        if p == 2:
            return [
                ("col", b["stripe"]),
                ("bg", b["bg"]),
                ("bank", b["bank"]),
                ("sid", b["sid"]),
                ("unit", b["unit"]),
                ("colhi", b["col"] - b["stripe"]),
                ("row", b["row"]),
            ]
        if p == 3:
            return [
                ("col", b["col"]),
                ("unit", b["unit"]),
                ("sid", b["sid"]),
                ("bg", b["bg"]),
                ("bank", b["bank"]),
                ("row", b["row"]),
            ]
        if p == 4:
            return [
                ("col", b["col"]),
                ("row", b["row"]),
                ("bank", b["bank"]),
                ("bg", b["bg"]),
                ("sid", b["sid"]),
                ("unit", b["unit"]),
            ]
        if p == 5:
            return [
                ("col", 2),
                ("unit", b["unit"]),
                ("sid", b["sid"]),
                ("bg", b["bg"]),
                ("bank", b["bank"]),
                ("colhi", b["col"] - 2),
                ("row", b["row"]),
            ]
        raise ValueError("unknown policy %d" % p)

    def _mop_masks(self, col):
        b = self.b
        masks = {}
        x = 0
        for f in ("unit", "sid", "bg", "bank", "row"):
            w = b[f]
            masks[f] = 0
            if w > 0:
                masks[f] = (col >> x) & ((1 << w) - 1) if x < 32 else 0
                x += w
        return masks

    def encode(self, c):
        """Offset of byte 0 of the coordinates' column."""
        b = self.b
        vals = {f: getattr(c, f) for f in Coord.__slots__}
        lo_col = {2: b["stripe"], 5: 2}.get(self.policy)
        if self.policy == 5:
            m = self._mop_masks(c.col)
            for f in m:
                vals[f] ^= m[f]
        a = 0
        pos = 0
        for f, w in self._fields():
            v = vals["col"] >> lo_col if f == "colhi" else vals[f]
            a |= (v & ((1 << w) - 1)) << pos
            pos += w
        byte_addr = a << b["tx"]
        low = byte_addr & ((1 << b["s"]) - 1)
        return (
            ((byte_addr >> b["s"]) << (b["s"] + b["ch"]))
            | ((c.ch & ((1 << b["ch"]) - 1)) << b["s"])
            | low
        )

    def decode(self, off):
        """(coordinates, byte within the column) of an offset."""
        b = self.b
        low = off & ((1 << b["s"]) - 1)
        byte_addr = ((off >> (b["s"] + b["ch"])) << b["s"]) | low
        a = byte_addr >> b["tx"]
        c = Coord(ch=(off >> b["s"]) & ((1 << b["ch"]) - 1))
        pos = 0
        lo_col = {2: b["stripe"], 5: 2}.get(self.policy)
        for f, w in self._fields():
            v = (a >> pos) & ((1 << w) - 1)
            pos += w
            if f == "colhi":
                c.col |= v << lo_col
            else:
                setattr(c, f, v)
        if self.policy == 5:
            m = self._mop_masks(c.col)
            for f in m:
                setattr(c, f, getattr(c, f) ^ m[f])
        return c, byte_addr & ((1 << b["tx"]) - 1)


# The model ---------------------------------------------------------------

OPS = (
    "none",
    "pim.sb.ewadd",
    "pim.sb.ewmul",
    "pim.sb.ewadd.gbuf",
    "pim.sb.ewmul.gbuf",
    "pim.sb.mac",
    "pim.sb.mov.bkgbuf",
    "pim.sb.mov.gbufbk",
    "pim.ab.ewadd",
    "pim.ab.ewmul",
    "pim.ab.ewadd.gbuf",
    "pim.ab.ewmul.gbuf",
    "pim.ab.mac",
    "pim.ab.mov.regbk",
    "pim.ab.mov.bkreg",
    "pim.rd.gb",
    "pim.wr.gb",
    "pim.rd.reg",
    "pim.wr.reg",
)
# gem5's PimOp names (src/mem/pim_op.hh), in the same order.
PIM_OPS = (
    "None",
    "SbEwadd",
    "SbEwmul",
    "SbEwaddGbuf",
    "SbEwmulGbuf",
    "SbMac",
    "SbMovBkgbuf",
    "SbMovGbufbk",
    "AbEwadd",
    "AbEwmul",
    "AbEwaddGbuf",
    "AbEwmulGbuf",
    "AbMac",
    "AbMovRegbk",
    "AbMovBkreg",
    "RdGb",
    "WrGb",
    "RdReg",
    "WrReg",
)
(
    SB_EWADD,
    SB_EWMUL,
    SB_EWADD_GBUF,
    SB_EWMUL_GBUF,
    SB_MAC,
    SB_MOV_BKGBUF,
    SB_MOV_GBUFBK,
    AB_EWADD,
    AB_EWMUL,
    AB_EWADD_GBUF,
    AB_EWMUL_GBUF,
    AB_MAC,
    AB_MOV_REGBK,
    AB_MOV_BKREG,
    RD_GB,
    WR_GB,
    RD_REG,
    WR_REG,
) = range(1, 19)


def op_is_load(op):
    return op in (RD_GB, RD_REG)


class SoftwareError(Exception):
    pass


class Memory:
    """A sparse byte image."""

    PAGE = 4096

    def __init__(self):
        self.pages = {}

    def page(self, addr):
        p = self.pages.get(addr // self.PAGE)
        if p is None:
            p = self.pages[addr // self.PAGE] = bytearray(self.PAGE)
        return p

    def read(self, addr, n):
        out = bytearray()
        while n:
            p = self.pages.get(addr // self.PAGE)
            i = addr % self.PAGE
            k = min(n, self.PAGE - i)
            out += p[i : i + k] if p is not None else bytes(k)
            addr += k
            n -= k
        return out

    def write(self, addr, data):
        j = 0
        while j < len(data):
            p = self.page(addr)
            i = addr % self.PAGE
            k = min(len(data) - j, self.PAGE - i)
            p[i : i + k] = data[j : j + k]
            addr += k
            j += k


class Model:
    """The 18 commands on a PIM memory of geometry geom.

    State per unit (channel x rank or pseudo channel): the global buffer
    (GB), one row; per processing unit (PU, one per bank): num_regs
    registers of one row and an accumulator in acc_fmt. Register j of a
    bank's PU is row rows - num_regs - 1 + j of the register window; the
    next row is the accumulator row.
    """

    def __init__(
        self,
        geom,
        num_regs=1,
        num_fmt=BF16,
        acc_fmt=FP32,
        vlenb=None,
        errors_fatal=True,
        mem=None,
    ):
        why = check_formats(num_fmt, acc_fmt, geom.col_bytes)
        if why or num_regs < 1:
            raise ValueError(why or "num_regs < 1")
        self.g = geom
        self.num_regs, self.num_fmt, self.acc_fmt = num_regs, num_fmt, acc_fmt
        self.vlenb = vlenb or geom.col_bytes
        self.errors_fatal = errors_fatal
        self.mem = mem if mem is not None else Memory()
        self.row_bytes = geom.row_bytes()
        self.elem = fmt_bytes(num_fmt)
        self.acc_bytes = fmt_bytes(acc_fmt)
        self.size = geom.size()
        self.gb = {}
        self.regs = {}
        self.accs = {}
        self.errors = 0
        self.count = [0] * 19

    # State ----------------------------------------------------------------

    def unit_of(self, c):
        return c.ch * self.g.units + c.unit

    def pu_of(self, c):
        g = self.g
        return (self.unit_of(c), (c.sid * g.bgs + c.bg) * g.banks + c.bank)

    def gb_of(self, unit):
        b = self.gb.get(unit)
        if b is None:
            b = self.gb[unit] = bytearray(self.row_bytes)
        return b

    def reg_of(self, c, j):
        key = self.pu_of(c) + (j,)
        b = self.regs.get(key)
        if b is None:
            b = self.regs[key] = bytearray(self.row_bytes)
        return b

    def acc_of(self, c):
        key = self.pu_of(c)
        b = self.accs.get(key)
        if b is None:
            b = self.accs[key] = bytearray(self.acc_bytes)
        return b

    def reg_row0(self):
        return self.g.rows - self.num_regs - 1

    def row_addr(self, c, i):
        """Offset of byte i of the row of bank c."""
        return (
            self.g.encode(c.copy(col=i // self.g.col_bytes))
            + i % self.g.col_bytes
        )

    def row_read(self, c, lo, n):
        return bytearray(
            self.mem.read(self.row_addr(c, i), 1)[0] for i in range(lo, lo + n)
        )

    def row_write(self, c, lo, data):
        cb = self.g.col_bytes
        i = 0
        while i < len(data):
            k = min(len(data) - i, cb - (lo + i) % cb)
            self.mem.write(self.row_addr(c, lo + i), data[i : i + k])
            i += k

    def _row_bytes(self, c, lo, hi):
        """Bytes [lo, hi) of the row of bank c, read column by column."""
        cb = self.g.col_bytes
        out = bytearray()
        i = lo
        while i < hi:
            k = min(hi - i, cb - i % cb)
            out += self.mem.read(self.row_addr(c, i), k)
            i += k
        return out

    # Errors ---------------------------------------------------------------

    def sw_error(self, op, off, msg):
        self.errors += 1
        if self.errors_fatal:
            raise SoftwareError(
                f"pim_ref: {OPS[op]} at offset {off:#x}: {msg}"
            )
        return True

    # Commands -------------------------------------------------------------

    def _bank_command(self, op, b, lo, hi, off):
        g = self.g
        ab = AB_EWADD <= op <= AB_MOV_BKREG
        is_ew = op in (SB_EWADD, SB_EWMUL, AB_EWADD, AB_EWMUL)
        writes_reg = (
            SB_EWADD <= op <= SB_EWMUL_GBUF or AB_EWADD <= op <= AB_MOV_BKREG
        )
        computes = SB_EWADD <= op <= SB_MAC or AB_EWADD <= op <= AB_MAC
        if b.row >= self.reg_row0():
            return self.sw_error(
                op, off, "bank operation on window row %u" % b.row
            )
        if is_ew and b.bank & 1:
            return self.sw_error(
                op, off, "odd bank %u of an element-wise pair" % b.bank
            )
        if writes_reg and self.num_regs > 1:
            return self.sw_error(
                op, off, "needs one register per PU, not %u" % self.num_regs
            )
        if computes and (lo % self.elem or hi % self.elem):
            return self.sw_error(
                op, off, "range [%u, %u) is not element-aligned" % (lo, hi)
            )
        if ab:
            banks = [
                b.copy(sid=s, bg=q, bank=k)
                for s in range(g.sids)
                for q in range(g.bgs)
                for k in range(g.banks)
            ]
        else:
            banks = [b]
        gb = self.gb_of(self.unit_of(b))
        nf, e, cb = self.num_fmt, self.elem, g.col_bytes
        for k in banks:
            if op in (SB_EWADD, SB_EWMUL, AB_EWADD, AB_EWMUL):
                if k.bank & 1:
                    continue
                mul = op in (SB_EWMUL, AB_EWMUL)
                x = self._row_bytes(k, lo, hi)
                y = self._row_bytes(k.copy(bank=k.bank + 1), lo, hi)
                reg = self.reg_of(k, 0)
                for i in range(0, hi - lo, e):
                    store_bits(
                        nf,
                        reg,
                        lo + i,
                        ew_element(
                            nf, mul, load_bits(nf, x, i), load_bits(nf, y, i)
                        ),
                    )
            elif op in (
                SB_EWADD_GBUF,
                SB_EWMUL_GBUF,
                AB_EWADD_GBUF,
                AB_EWMUL_GBUF,
            ):
                mul = op in (SB_EWMUL_GBUF, AB_EWMUL_GBUF)
                y = self._row_bytes(k, lo, hi)
                reg = self.reg_of(k, 0)
                for i in range(0, hi - lo, e):
                    store_bits(
                        nf,
                        reg,
                        lo + i,
                        ew_element(
                            nf,
                            mul,
                            load_bits(nf, gb, lo + i),
                            load_bits(nf, y, i),
                        ),
                    )
            elif op in (SB_MAC, AB_MAC):
                acc = self.acc_of(k)
                i = lo
                while i < hi:
                    end = min((i // cb + 1) * cb, hi)
                    y = self._row_bytes(k, i, end)
                    xs = [load_bits(nf, gb, j) for j in range(i, end, e)]
                    ys = [load_bits(nf, y, j) for j in range(0, end - i, e)]
                    a = mac_column(
                        nf, self.acc_fmt, xs, ys, load_bits(self.acc_fmt, acc)
                    )
                    store_bits(self.acc_fmt, acc, 0, a)
                    i = end
            elif op == SB_MOV_BKGBUF:
                gb[lo:hi] = self._row_bytes(k, lo, hi)
            elif op == SB_MOV_GBUFBK:
                self.row_write(k, lo, gb[lo:hi])
            elif op == AB_MOV_REGBK:
                self.row_write(k, lo, self.reg_of(k, 0)[lo:hi])
            elif op == AB_MOV_BKREG:
                self.reg_of(k, 0)[lo:hi] = self._row_bytes(k, lo, hi)
        return False

    def scalar(self, op, off, desc):
        """A scalar form (SbEwadd ... SbMac) at offset off."""
        if off >= self.size:
            return self.sw_error(op, off, "address outside the PIM memory")
        self.count[op] += 1
        if off & 7:
            return self.sw_error(op, off, "address not 8-byte aligned")
        if desc >> 16:
            return self.sw_error(
                op, off, "reserved descriptor bits in %#x" % desc
            )
        c, _ = self.g.decode(off)
        cols = self.g.cols
        n = (desc & 0xFFFF) or cols - c.col
        if c.col + n > cols:
            return self.sw_error(
                op, off, "columns %u + %u beyond the row" % (c.col, n)
            )
        cb = self.g.col_bytes
        return self._bank_command(op, c, c.col * cb, (c.col + n) * cb, off)

    def vlmax(self, nbytes):
        return min(nbytes, 8 * self.vlenb)

    def _window(self, off):
        """(buffer, index) of a window byte, (None, None) for a byte that
        reads as zero, or None outside the window."""
        c, byte = self.g.decode(off)
        i = c.col * self.g.col_bytes + byte
        r0 = self.reg_row0()
        if c.row < r0:
            return None
        if c.row < r0 + self.num_regs:
            return self.reg_of(c, c.row - r0), i
        if i < self.acc_bytes:
            return self.acc_of(c), i
        return None, None

    def vector(self, op, off, nbytes, src=None):
        """A vector form; returns the bytes a load reads (zeros on error)."""
        vl = self.vlmax(nbytes)
        if vl == 0:
            return bytearray()
        out = bytearray(vl)
        if off + vl > self.size:
            self.sw_error(op, off, "address outside the PIM memory")
            return out
        self.count[op] += 1
        if op in (RD_REG, WR_REG):
            places = []
            for i in range(vl):
                w = self._window(off + i)
                if w is None:
                    self.sw_error(op, off + i, "outside the register window")
                    return out
                places.append(w)
            for i, (buf, j) in enumerate(places):
                if op == RD_REG:
                    out[i] = buf[j] if buf is not None else 0
                elif buf is not None:
                    buf[j] = src[i]
            return out
        cb = self.g.col_bytes
        c, byte = self.g.decode(off)
        lo = c.col * cb + byte
        hi = lo + vl
        if hi > self.row_bytes:
            self.sw_error(op, off, "%u bytes cross the end of the row" % vl)
            return out
        if op <= AB_MOV_BKREG:
            i = cb - byte
            while i < vl:
                d, _ = self.g.decode(off + i)
                same = (d.ch, d.unit, d.sid, d.bg, d.bank, d.row) == (
                    c.ch,
                    c.unit,
                    c.sid,
                    c.bg,
                    c.bank,
                    c.row,
                )
                if not same or d.col * cb != lo + i:
                    self.sw_error(
                        op,
                        off,
                        "byte %u is not the next byte of the bank row "
                        "(the range crosses a row, bank or unit)" % i,
                    )
                    return out
                i += cb
            self._bank_command(op, c, lo, hi, off)
        elif op == RD_GB:
            out[:] = self.gb_of(self.unit_of(c))[lo:hi]
        else:
            self.gb_of(self.unit_of(c))[lo:hi] = src[:vl]
        return out


# The shared-vector runner (libpim/tests/test_model.c) -----------------


class VectorError(Exception):
    pass


class Runner:
    def __init__(self, dump, out=sys.stdout):
        self.dump = dump
        self.out = out
        self.model = None
        self.last_read = bytearray()
        self.checks = 0
        self.failures = 0
        self.where = ""

    def die(self, msg, tok=None):
        raise VectorError(
            "{}: {}{}".format(self.where, msg, ": " + tok if tok else "")
        )

    def coord(self, tok, first, j=False, count=False):
        c = Coord()
        res = {"byte": 0, "j": 0, "count": 0, "cols": False}
        i = first
        while i < len(tok):
            key, eq, v = tok[i].partition("=")
            if not eq:
                break
            if key in ("ch", "unit", "sid", "bg", "bank", "col"):
                setattr(c, key, int(v, 0))
            elif key == "row":
                r0 = self.model.reg_row0()
                if v.startswith("reg"):
                    c.row = r0 + int(v[3:], 0)
                elif v == "acc":
                    c.row = r0 + self.model.num_regs
                else:
                    c.row = int(v, 0)
            elif key == "byte":
                res["byte"] = int(v, 0)
            elif j and key == "j":
                res["j"] = int(v, 0)
            elif count and key == "bytes":
                res["count"] = int(v, 0)
            elif count and key == "ncols":
                res["count"] = int(v, 0)
                res["cols"] = True
            else:
                break
            i += 1
        return c, res, i

    def values(self, tok, first):
        if first >= len(tok):
            return bytearray(), None
        if tok[first] not in FORMATS:
            self.die("unknown format", tok[first])
        fmt = FORMATS.index(tok[first])
        e = fmt_bytes(fmt)
        out = bytearray()
        for t in tok[first + 1 :]:
            b = bytearray(e)
            if t.startswith("0x"):
                store_bits(fmt, b, 0, int(t, 16))
            elif fmt_is_float(fmt):
                store_bits(fmt, b, 0, from_f32(fmt, _f2u(float(t))))
            else:
                store_bits(fmt, b, 0, int(t, 0))
            out += b
        return out, fmt

    def state(self, kind, c, res, n):
        """The bytes a state statement addresses: a list of
        (getter, setter) pairs, one per byte."""
        m = self.model
        start = c.col * m.g.col_bytes + res["byte"]
        if kind == "row":
            row = c.copy(col=0)
            return [
                (
                    lambda i=i: m.row_read(row, start + i, 1)[0],
                    lambda v, i=i: m.row_write(row, start + i, bytes([v])),
                )
                for i in range(n)
            ]
        if kind in ("gb", "reg"):
            buf = (
                m.gb_of(m.unit_of(c))
                if kind == "gb"
                else m.reg_of(c, res["j"])
            )

            def get(i):
                return buf[start + i]

            def put(v, i):
                buf[start + i] = v

            return [
                (lambda i=i: get(i), lambda v, i=i: put(v, i))
                for i in range(n)
            ]
        if kind == "acc":
            buf = m.acc_of(c)
            pairs = []
            for i in range(min(n, 8)):
                if i < len(buf):
                    pairs.append(
                        (
                            lambda i=i: buf[i],
                            lambda v, i=i: buf.__setitem__(i, v),
                        )
                    )
                else:
                    pairs.append((lambda: 0, lambda v: None))
            return pairs
        return None

    def statement(self, tok, text):
        if tok[0] == "geom":
            kw = {"units": 1, "sids": 1, "bgs": 1}
            cfg = {"num_regs": 1, "num_fmt": BF16, "acc_fmt": FP32, "vlenb": 0}
            names = {
                "policy": "policy",
                "channels": "channels",
                "intlv": "intlv",
                "units": "units",
                "sids": "sids",
                "bgs": "bgs",
                "banks": "banks",
                "rows": "rows",
                "cols": "cols",
                "col_bytes": "col_bytes",
            }
            for t in tok[1:]:
                key, _, v = t.partition("=")
                if key in names:
                    kw[names[key]] = int(v, 0)
                elif key in ("num_fmt", "acc_fmt"):
                    cfg[key] = FORMATS.index(v)
                elif key in ("num_regs", "vlenb"):
                    cfg[key] = int(v, 0)
                else:
                    self.die("unknown geom key", t)
            try:
                geom = Geometry(**kw)
                self.model = Model(
                    geom,
                    cfg["num_regs"],
                    cfg["num_fmt"],
                    cfg["acc_fmt"],
                    cfg["vlenb"] or geom.col_bytes,
                )
            except (ValueError, TypeError) as e:
                self.die("cannot start the model for this geometry: %s" % e)
            self.last_read = bytearray()
            return
        if self.model is None:
            self.die("no geom line before", tok[0])
        m = self.model
        if tok[0] == "errors" and len(tok) == 2:
            m.errors_fatal = tok[1] == "fatal"
            return
        if tok[0] in ("row", "gb", "reg", "acc"):
            c, res, nxt = self.coord(tok, 1, j=True)
            data, _ = self.values(tok, nxt)
            for (get, put), v in zip(
                self.state(tok[0], c, res, len(data)), data
            ):
                put(v)
            return
        if tok[0] == "cmd" and len(tok) >= 2:
            if tok[1] not in OPS[1:]:
                self.die("unknown command", tok[1])
            op = OPS.index(tok[1])
            c, res, nxt = self.coord(tok, 2, count=True)
            data, _ = self.values(tok, nxt)
            off = m.g.encode(c) + res["byte"]
            if op <= SB_MAC:
                m.scalar(op, off, res["count"])
            elif op_is_load(op):
                self.last_read = m.vector(op, off, res["count"])
            else:
                m.vector(
                    op, off, len(data) or res["count"], data if data else None
                )
            return
        if tok[0] == "expect" and len(tok) >= 2:
            if tok[1] == "errors" and len(tok) == 3:
                self.report(text, str(m.errors), m.errors == int(tok[2], 0))
                return
            if tok[1] == "count" and len(tok) == 4:
                if tok[2] not in OPS[1:]:
                    self.die("unknown command", tok[2])
                got = m.count[OPS.index(tok[2])]
                self.report(text, str(got), got == int(tok[3], 0))
                return
            if tok[1] == "read":
                want, fmt = self.values(tok, 2)
                got = bytearray(
                    self.last_read[i] if i < len(self.last_read) else 0
                    for i in range(len(want))
                )
            else:
                c, res, nxt = self.coord(tok, 2, j=True)
                want, fmt = self.values(tok, nxt)
                pairs = self.state(tok[1], c, res, len(want))
                if pairs is None:
                    self.die("unknown expect kind", tok[1])
                got = bytearray(get() for get, _ in pairs)
                got += bytes(len(want) - len(got))
            if fmt is None:
                self.die("expect needs values")
            e = fmt_bytes(fmt)
            shown = " ".join(
                [FORMATS[fmt]]
                + [
                    "0x%0*x" % (2 * e, load_bits(fmt, got, i))
                    for i in range(0, len(got) - e + 1, e)
                ]
            )
            self.report(text, shown, got == want)
            return
        self.die("unknown statement", tok[0])

    def report(self, text, shown, ok):
        if self.dump:
            self.out.write(f"{text} => {shown}\n")
            return
        self.checks += 1
        if not ok:
            self.failures += 1
            sys.stderr.write(f"{self.where}: {text}\n    got: {shown}\n")

    def run_file(self, path):
        with open(path) as f:
            for n, line in enumerate(f, 1):
                self.where = "%s:%d" % (path, n)
                line = line.split("#", 1)[0].rstrip("\r\n")
                tok = line.split()
                if tok:
                    self.statement(tok, line.strip(" \t"))


# PimTester scripts -------------------------------------------------------


class Script:
    """Lines of a PimTester script (src/cpu/testers/pim_tester)."""

    def __init__(self):
        self.lines = []

    def add(self, delay, kind, addr=None, size=None, data=None, expect=None):
        parts = [str(delay), kind]
        if kind != "drain":
            parts += ["%#x" % addr, str(size)]
            if data is not None or expect is not None:
                parts.append(bytes(data).hex() if data is not None else "-")
            if expect is not None:
                parts.append(bytes(expect).hex())
        self.lines.append(" ".join(parts))

    def pim(self, delay, op, addr, size, data=None, expect=None):
        self.add(delay, "pim:" + PIM_OPS[op], addr, size, data, expect)

    def write(self, path, header=None):
        with open(path, "w") as f:
            if header:
                for h in header.splitlines():
                    f.write("# %s\n" % h)
            f.write("\n".join(self.lines) + "\n")


def rdwr_script(base, size, seed=1, count=200, line=64):
    """Normal timing writes and reads with their expected data.

    Random cache-line writes over [base, base + size), each read back after
    a drain, then read again in another order, plus functional reads of the
    same lines and functional writes read back by timing reads.
    """
    rng = random.Random(seed)
    s = Script()
    image = {}
    lines = rng.sample(range(size // line), count)
    for i, n in enumerate(lines):
        data = bytes(rng.getrandbits(8) for _ in range(line))
        image[n] = data
        s.add(rng.choice((0, 1000, 5000)), "wr", base + n * line, line, data)
    s.add(0, "drain")
    for n in lines:
        s.add(
            rng.choice((0, 1000)), "rd", base + n * line, line, None, image[n]
        )
    s.add(0, "drain")
    for n in rng.sample(lines, len(lines)):
        s.add(0, "frd", base + n * line, line, None, image[n])
    for n in rng.sample(lines, count // 4):
        data = bytes(rng.getrandbits(8) for _ in range(line))
        image[n] = data
        s.add(0, "fwr", base + n * line, line, data)
        s.add(500, "rd", base + n * line, line, None, data)
    # Partial reads inside a line.
    for n in rng.sample(lines, count // 4):
        k = rng.choice((8, 16, 32))
        at = rng.randrange(0, line, k)
        s.add(100, "rd", base + n * line + at, k, None, image[n][at : at + k])
    s.add(0, "drain")
    return s


# Scripts on a PIM memory -------------------------------------------------


def random_value(rng, fmt):
    """Bits of a random element: no NaN or infinity, and moderate
    exponents, so that sums and products of a few stay finite."""
    if not fmt_is_float(fmt):
        return rng.getrandbits(8 * fmt_bytes(fmt))
    sign = rng.getrandbits(1)
    if fmt == BF16:
        e = 0 if rng.random() < 0.05 else rng.randint(127 - 20, 127 + 10)
        return sign << 15 | e << 7 | rng.getrandbits(7)
    if fmt == FP16:
        e = 0 if rng.random() < 0.05 else rng.randint(1, 19)
        return sign << 15 | e << 10 | rng.getrandbits(10)
    e = 0 if rng.random() < 0.05 else rng.randint(127 - 30, 127 + 10)
    return sign << 31 | e << 23 | rng.getrandbits(23)


def random_elems(rng, fmt, nbytes):
    out = bytearray(nbytes)
    for i in range(0, nbytes - fmt_bytes(fmt) + 1, fmt_bytes(fmt)):
        store_bits(fmt, out, i, random_value(rng, fmt))
    return out


class PimScript:
    """A PimTester script on a PIM memory, with its expected values.

    The model runs what the memory runs: functional writes into the bank
    image or the register window, PIM requests (one tester request is one
    request: no vector-length cap), normal reads and writes. Software
    errors are counted (the memory's ``sw_error_fatal`` off): the request
    is a no-op and a load returns zeros. With ``values`` off PIM requests
    change nothing, loads and window reads return zeros and window writes
    are dropped.
    """

    def __init__(
        self,
        geom,
        num_regs=1,
        num_fmt=BF16,
        acc_fmt=FP32,
        values=True,
        base=0,
    ):
        self.g = geom
        self.m = Model(
            geom,
            num_regs,
            num_fmt,
            acc_fmt,
            vlenb=1 << 40,
            errors_fatal=False,
        )
        self.values = values
        self.base = base
        self.s = Script()
        self.pim_lines = 0
        # Start after tick 0: gem5 dumps no statistics at the tick of the
        # previous dump, which starts at 0 (atomic runs take no time).
        self.s.add(1000, "drain")

    # Addresses ----------------------------------------------------------

    def coord_of(self, unit, bank, row=0, col=0):
        """Coordinates of bank bank (numbered within its unit, as gem5's
        PimLayout) of unit unit (over the whole memory)."""
        g = self.g
        return Coord(
            ch=unit // g.units,
            unit=unit % g.units,
            sid=bank // (g.banks * g.bgs),
            bg=(bank // g.banks) % g.bgs,
            bank=bank % g.banks,
            row=row,
            col=col,
        )

    def addr(self, unit=0, bank=0, row=0, col=0, byte=0):
        return self.g.encode(self.coord_of(unit, bank, row, col)) + byte

    def units(self):
        return self.g.channels * self.g.units

    def banks(self):
        return self.g.banks_per_unit()

    def reg_row(self, j=0):
        return self.m.reg_row0() + j

    def acc_row(self):
        return self.m.reg_row0() + self.m.num_regs

    def contig(self):
        """Bytes from the start of a row that are consecutive addresses."""
        g = self.g
        c = self.coord_of(0, 0, 1, 0)
        a0 = g.encode(c)
        n = 1
        while n < g.cols and g.encode(c.copy(col=n)) == a0 + n * g.col_bytes:
            n += 1
        return n * g.col_bytes

    # Lines --------------------------------------------------------------

    def _functional(self, off, data=None, n=0):
        out = bytearray()
        count = len(data) if data is not None else n
        r0 = self.m.reg_row0()
        for i in range(count):
            c, _ = self.g.decode(off + i)
            if c.row >= r0:
                buf, j = (None, None)
                if self.values:
                    buf, j = self.m._window(off + i)
                if data is not None:
                    if buf is not None:
                        buf[j] = data[i]
                else:
                    out.append(buf[j] if buf is not None else 0)
            elif data is not None:
                self.m.mem.write(off + i, bytes([data[i]]))
            else:
                out += self.m.mem.read(off + i, 1)
        return out

    def fwr(self, off, data, delay=0):
        self._functional(off, data=data)
        self.s.add(delay, "fwr", self.base + off, len(data), data)

    def frd(self, off, n, delay=0):
        self.s.add(
            delay, "frd", self.base + off, n, None, self._functional(off, n=n)
        )

    def fwr_row(self, unit, bank, row, data, lo=0):
        """Functional writes of row bytes [lo, lo + len(data)) of a bank,
        one line per run of consecutive addresses."""
        c = self.coord_of(unit, bank, row)
        cb = self.g.col_bytes
        i = lo
        while i < lo + len(data):
            a = self.m.row_addr(c, i)
            k = min(lo + len(data) - i, cb - i % cb)
            while (
                i + k < lo + len(data) and self.m.row_addr(c, i + k) == a + k
            ):
                k += min(lo + len(data) - i - k, cb)
            self.fwr(a, data[i - lo : i - lo + k])
            i += k

    def frd_row(self, unit, bank, row, lo=0, hi=None):
        c = self.coord_of(unit, bank, row)
        cb = self.g.col_bytes
        hi = self.g.row_bytes() if hi is None else hi
        i = lo
        while i < hi:
            a = self.m.row_addr(c, i)
            k = min(hi - i, cb - i % cb)
            while i + k < hi and self.m.row_addr(c, i + k) == a + k:
                k += min(hi - i - k, cb)
            self.frd(a, k)
            i += k

    def pim(self, op, off, nbytes=None, data=None, desc=0, delay=0):
        """A PIM request; returns what a load reads."""
        self.pim_lines += 1
        if SB_EWADD <= op <= SB_MAC:
            payload = int(desc).to_bytes(8, "little")
            if self.values:
                self.m.scalar(op, off, desc)
            self.s.pim(delay, op, self.base + off, 8, payload)
            return None
        if op_is_load(op):
            out = self.m.vector(op, off, nbytes)
            if not self.values:
                out = bytearray(len(out))
            self.s.pim(delay, op, self.base + off, nbytes, None, out)
            return out
        if data is not None:
            nbytes = len(data)
        if self.values:
            self.m.vector(op, off, nbytes, data)
        self.s.pim(delay, op, self.base + off, nbytes, data)
        return None

    def wr(self, off, data, delay=0):
        self.m.mem.write(off, data)
        self.s.add(delay, "wr", self.base + off, len(data), data)

    def rd(self, off, n, delay=0):
        self.s.add(
            delay, "rd", self.base + off, n, None, self.m.mem.read(off, n)
        )

    def drain(self):
        self.s.add(0, "drain")

    def write(self, path, header=None):
        self.s.write(path, header)


def c1_script(geom, num_fmt=BF16, acc_fmt=FP32, values=True, seed=1):
    """C1: each of the 18 operations, scalar and vector
    forms, on units 0 and 1, with RdReg, RdGb and functional reads of the
    rows and register rows they change. Also C17's split MAC and C20's
    accumulator-row reads.

    Vector requests stay inside the first ``contig()`` bytes of a row,
    which are consecutive addresses of one channel, as the library's
    requests do (``PIM_CONTIG_BYTES``)."""
    rng = random.Random(seed)
    s = PimScript(geom, num_fmt=num_fmt, acc_fmt=acc_fmt, values=values)
    nf = num_fmt
    rb = geom.row_bytes()
    cb = geom.col_bytes
    e = fmt_bytes(nf)
    vec = min(rb, s.contig())  # one vector request stays in these bytes

    def at(unit, bank, row, lo):
        """The address of row byte lo (< vec) of a bank."""
        return s.addr(unit, bank, row, lo // cb, lo % cb)

    def elems(n):
        return max(e, n - n % e)

    units = [0, 1] if s.units() > 1 else [0]
    banks = range(s.banks())
    for u in units:
        for b in banks:
            for row in (1, 2, 3):
                s.fwr_row(u, b, row, random_elems(rng, nf, rb))
    # The GBs.
    s.pim(WR_GB, at(0, 0, 1, 0), vec, random_elems(rng, nf, vec))
    if len(units) > 1:
        s.pim(
            WR_GB,
            at(1, 3, 2, 2 * e),
            data=random_elems(rng, nf, elems(vec // 2)),
        )
    s.pim(RD_GB, at(0, 0, 0, e), elems(vec - 2 * e))
    # A WrGb address keeps its bank bytes: the payload never reaches the
    # backing store.
    s.frd(at(0, 0, 1, 0), min(vec, 64))
    # Scalar forms.
    s.pim(SB_EWADD, s.addr(0, 0, 1, 0), desc=0)
    s.pim(SB_EWMUL, s.addr(0, 2, 2, 3), desc=min(5, geom.cols - 3))
    s.pim(SB_EWADD_GBUF, s.addr(0, 1, 1, 2), desc=0)
    s.pim(SB_EWMUL_GBUF, s.addr(0, 3, 3, 0), desc=min(4, geom.cols))
    s.pim(SB_MAC, s.addr(0, 5, 2, 0), desc=0)
    s.pim(SB_MAC, s.addr(0, 5, 3, 1), desc=min(3, geom.cols - 1))
    if len(units) > 1:
        s.pim(SB_MAC, s.addr(1, 4, 1, 0), desc=0)
    for b in (0, 1, 2, 3, 5):
        s.pim(RD_REG, at(0, b, s.reg_row(), 0), vec)
        s.pim(RD_REG, at(0, b, s.acc_row(), 0), cb)
    # Vector bank forms.
    lo = cb if vec > cb else 0
    s.pim(SB_MOV_BKGBUF, at(0, 4, 3, lo), vec - lo)
    s.pim(RD_GB, at(0, 0, 0, 0), vec)
    s.pim(SB_MOV_GBUFBK, at(0, 6, 4, 0), vec)
    s.frd_row(0, 6, 4, 0, vec)
    s.pim(AB_EWADD, at(0, 0, 1, 0), vec)
    s.pim(AB_EWMUL, at(0, 2, 2, 2 * e), elems(vec - 4 * e))
    s.pim(AB_EWADD_GBUF, at(0, 1, 3, 0), vec)
    s.pim(AB_EWMUL_GBUF, at(0, 0, 3, elems(vec // 2)), vec - elems(vec // 2))
    for b in banks:
        s.pim(RD_REG, at(0, b, s.reg_row(), 0), vec)
    s.pim(AB_MAC, at(0, 0, 1, 0), vec)
    # A MAC split at a point inside a column (C17): each product once.
    split = cb + 3 * e if cb + 3 * e < vec else elems(vec // 2 + e)
    if split >= vec:
        split = e
    s.pim(AB_MAC, at(0, 0, 2, 0), split)
    s.pim(AB_MAC, at(0, 0, 2, split), vec - split)
    for b in banks:
        s.pim(RD_REG, at(0, b, s.acc_row(), 0), cb)
    s.pim(AB_MOV_REGBK, at(0, 0, 5, 0), vec)
    for b in banks:
        s.frd_row(0, b, 5, 0, vec)
    s.pim(AB_MOV_BKREG, at(0, 0, 3, 0), vec)
    for b in banks:
        s.pim(RD_REG, at(0, b, s.reg_row(), 0), vec)
    # Register writes, including the accumulator row's ignored bytes.
    w = bytes(rng.getrandbits(8) for _ in range(min(100, vec - 6)))
    s.pim(WR_REG, at(0, 7, s.reg_row(), 6), data=w)
    s.pim(RD_REG, at(0, 7, s.reg_row(), 0), vec)
    s.pim(WR_REG, at(0, 7, s.acc_row(), 0), data=bytes(range(1, 9)))
    s.pim(RD_REG, at(0, 7, s.acc_row(), 0), cb)
    # Functional reads of register rows see the registers.
    s.frd(at(0, 7, s.reg_row(), 0), min(vec, 128))
    s.frd(at(0, 5, s.acc_row(), 0), cb)
    s.drain()
    return s


def c14_script(geom, values=True, seed=2):
    """C14: functional reads and writes of register rows."""
    rng = random.Random(seed)
    s = PimScript(geom, values=values)
    cb = geom.col_bytes
    for b in (0, 1):
        d = bytes(rng.getrandbits(8) for _ in range(2 * cb))
        s.fwr(s.addr(0, b, s.reg_row(), 1), d)
        s.frd(s.addr(0, b, s.reg_row(), 1), 2 * cb)
        s.pim(RD_REG, s.addr(0, b, s.reg_row(), 1), 2 * cb)
    s.fwr(s.addr(0, 2, s.acc_row(), 0), bytes(range(10, 10 + cb)))
    s.frd(s.addr(0, 2, s.acc_row(), 0), cb)
    s.pim(RD_REG, s.addr(0, 2, s.acc_row(), 0), cb)
    # A packet across a bank row and the next row is not possible; a
    # register write by PIM, read functionally.
    s.pim(WR_REG, s.addr(0, 3, s.reg_row(), 0), data=bytes(range(cb)))
    s.frd(s.addr(0, 3, s.reg_row(), 0), cb)
    s.drain()
    return s


# The software errors: (name, function of a PimScript).
def _err_cases(s):
    cb = s.g.col_bytes
    last = s.g.cols - 1
    return [
        (
            "descriptor",
            lambda: s.pim(SB_MAC, s.addr(0, 0, 1, 0), desc=1 << 16),
        ),
        ("ncols", lambda: s.pim(SB_MAC, s.addr(0, 0, 1, last - 1), desc=3)),
        ("alignment", lambda: s.pim(AB_MAC, s.addr(0, 0, 1, 0, 1), cb)),
        ("odd-bank-sb", lambda: s.pim(SB_EWADD, s.addr(0, 1, 1, 0), desc=1)),
        ("odd-bank-ab", lambda: s.pim(AB_EWADD, s.addr(0, 1, 1, 0), cb)),
        (
            "window-bank",
            lambda: s.pim(SB_MOV_BKGBUF, s.addr(0, 0, s.reg_row(), 0), cb),
        ),
        ("reg-outside", lambda: s.pim(RD_REG, s.addr(0, 0, 1, 0), cb)),
    ]


def crossing_cases(s):
    """Requests whose bytes leave the bank row inside one channel: on a
    one-channel memory, past the end of the row, and (under a policy that
    puts bank bits below the top column bits) into another bank."""
    cb = s.g.col_bytes
    last = s.g.cols - 1
    s.pim(AB_MAC, s.addr(0, 0, 1, last), 2 * cb)
    s.pim(SB_MOV_GBUFBK, s.addr(0, 0, 1, last, cb // 2), cb)
    n = s.contig()
    if n < s.g.row_bytes():
        s.pim(AB_EWADD, s.addr(0, 0, 1, 0), 2 * n)
    s.pim(RD_REG, s.addr(0, 0, s.acc_row(), 0), cb)
    s.drain()


def error_cases(geom):
    """The software-error cases on geom: [(name, PimScript)] with one PIM
    request each (expected values with errors counted)."""
    out = []
    for i, (name, _) in enumerate(_err_cases(PimScript(geom))):
        s = PimScript(geom)
        _err_cases(s)[i][1]()
        s.drain()
        out.append((name, s))
    return out


def error_script(geom):
    """Every software-error case in one script, then a valid request whose
    result shows that the errors changed nothing."""
    s = PimScript(geom)
    rng = random.Random(3)
    s.fwr_row(0, 0, 1, random_elems(rng, s.m.num_fmt, geom.row_bytes()))
    for _, f in _err_cases(s):
        f()
    s.pim(RD_GB, s.addr(0, 0, 0, 0), geom.col_bytes)
    s.pim(RD_REG, s.addr(0, 0, s.acc_row(), 0), geom.col_bytes)
    s.drain()
    return s


# Timing-path runs ------------------------------------------------------


def _fill(s, rng, units, banks, rows):
    for u in units:
        for b in banks:
            for row in rows:
                s.fwr_row(
                    u, b, row, random_elems(rng, s.m.num_fmt, s.g.row_bytes())
                )


def c3_script(geom, seed=4):
    """C3: a normal write, then a PIM request on its row, which waits for
    the write (rule 1) and sees its data (run with host_reads() keeping
    the controller reading, so that the write would stay buffered)."""
    rng = random.Random(seed)
    s = PimScript(geom)
    cb = geom.col_bytes
    s.s.lines[0] = "20000 drain"  # let a host stream start first
    _fill(s, rng, [0], [1], [3])
    a = s.addr(0, 1, 3, 0)
    s.wr(a, random_elems(rng, s.m.num_fmt, min(2 * cb, s.contig())))
    s.pim(SB_MAC, a, desc=4)
    s.pim(RD_REG, s.addr(0, 1, s.acc_row(), 0), cb)
    s.drain()
    return s


def c4_script(geom, seed=5):
    """C4: an older write, a posted PIM request on its row, then younger
    reads of that row: held until the request is Done, answered with the
    older write's data, not forwarded."""
    rng = random.Random(seed)
    s = PimScript(geom)
    cb = geom.col_bytes
    n = min(2 * cb, s.contig())
    _fill(s, rng, [0], range(s.banks()), [2])
    a = s.addr(0, 3, 2, 0)
    s.wr(a, random_elems(rng, s.m.num_fmt, n))
    s.pim(AB_MAC, s.addr(0, 0, 2, 0), min(geom.row_bytes(), s.contig()))
    s.rd(a, n)
    s.rd(s.addr(0, 5, 2, 0), n)
    s.pim(RD_REG, s.addr(0, 3, s.acc_row(), 0), cb)
    s.drain()
    return s


def c5_script(geom, seed=6):
    """C5: a write, a PIM request reading its row, then a younger write to
    the same address: held, not coalesced, both written in order."""
    rng = random.Random(seed)
    s = PimScript(geom)
    cb = geom.col_bytes
    n = min(2 * cb, s.contig())
    vec = min(geom.row_bytes(), s.contig())
    _fill(s, rng, [0], range(s.banks()), [4])
    a = s.addr(0, 2, 4, 0)
    s.wr(a, random_elems(rng, s.m.num_fmt, n))
    s.pim(AB_MOV_BKREG, s.addr(0, 0, 4, 0), vec)
    s.wr(a, random_elems(rng, s.m.num_fmt, n))
    s.drain()
    s.rd(a, n)
    s.pim(RD_REG, s.addr(0, 2, s.reg_row(), 0), vec)
    s.drain()
    return s


def c6_script(geom, seed=7):
    """C6: one write, then a PIM request on its row (the host script of
    host_reads() keeps the controller reading)."""
    rng = random.Random(seed)
    s = PimScript(geom)
    cb = geom.col_bytes
    s.s.lines[0] = "20000 drain"  # let the host stream start first
    _fill(s, rng, [0], [0], [6])
    a = s.addr(0, 0, 6, 0)
    s.wr(a, random_elems(rng, s.m.num_fmt, min(2 * cb, s.contig())))
    s.pim(SB_MAC, a, desc=2)
    s.pim(RD_REG, s.addr(0, 0, s.acc_row(), 0), cb)
    s.drain()
    return s


def host_reads(geom, count, unit=0, rows=range(40, 60), seed=8, delay=0):
    """A host stream of reads of rows other requests do not use, with their
    expected (zero) data."""
    rng = random.Random(seed)
    s = PimScript(geom)
    n = min(64, s.contig())
    for _ in range(count):
        b = rng.randrange(s.banks())
        row = rng.choice(list(rows))
        col = rng.randrange(0, geom.cols, max(1, n // geom.col_bytes))
        s.rd(s.addr(unit, b, row, col), n, delay)
    s.drain()
    return s


def c7_script(geom, seed=16):
    """C7: store-like requests 200 ns apart, loads only at the end: posted,
    each is acknowledged after the static latency; not posted, at Done."""
    rng = random.Random(seed)
    s = PimScript(geom)
    vec = min(geom.row_bytes(), s.contig())
    nf = s.m.num_fmt
    _fill(s, rng, [0], range(s.banks()), [1, 2])
    d = 200000
    s.pim(WR_GB, s.addr(0, 0, 0, 0), data=random_elems(rng, nf, vec), delay=d)
    s.pim(SB_MAC, s.addr(0, 2, 1, 0), desc=0, delay=d)
    s.pim(SB_EWADD, s.addr(0, 0, 2, 0), desc=0, delay=d)
    s.pim(AB_MAC, s.addr(0, 0, 1, 0), vec, delay=d)
    s.pim(AB_EWADD_GBUF, s.addr(0, 0, 2, 0), vec, delay=d)
    s.pim(SB_MOV_GBUFBK, s.addr(0, 5, 3, 0), vec, delay=d)
    s.pim(AB_MOV_REGBK, s.addr(0, 0, 4, 0), vec, delay=d)
    s.pim(WR_REG, s.addr(0, 6, s.reg_row(), 0), data=bytes(range(32)), delay=d)
    s.drain()
    s.pim(RD_REG, s.addr(0, 2, s.acc_row(), 0), geom.col_bytes, delay=d)
    s.pim(RD_REG, s.addr(0, 0, s.reg_row(), 0), vec)
    s.frd_row(0, 5, 3, 0, vec)
    s.drain()
    return s


def c26_script(geom, seed=17):
    """C26: one isolated request per class and size (1, 2 and a row of
    columns), each after the previous one is Done (run non-posted) and
    2 us more for its closing commands, for comparing U with timing mode."""
    rng = random.Random(seed)
    s = PimScript(geom)
    cb = geom.col_bytes
    nf = s.m.num_fmt
    vec = min(geom.row_bytes(), s.contig())
    sizes = sorted({1, 2, vec // cb})
    d = 2000000
    row = [10]

    def isolated(op, addr, nbytes=None, data=None, desc=0):
        s.drain()
        s.pim(op, addr, nbytes, data, desc, delay=d)

    def fresh():
        row[0] += 1
        return row[0]

    isolated(WR_GB, s.addr(0, 0, 0, 0), data=random_elems(rng, nf, vec))
    for n in sizes:
        for op in (SB_MAC, SB_EWADD, SB_EWADD_GBUF):
            isolated(op, s.addr(0, 2, fresh(), 0), desc=n)
        for op in (
            SB_MOV_BKGBUF,
            SB_MOV_GBUFBK,
            AB_EWADD,
            AB_EWADD_GBUF,
            AB_MAC,
            AB_MOV_BKREG,
            AB_MOV_REGBK,
        ):
            isolated(op, s.addr(0, 0, fresh(), 0), n * cb)
        isolated(WR_GB, s.addr(0, 0, 0, 0), data=random_elems(rng, nf, n * cb))
        isolated(RD_GB, s.addr(0, 0, 0, 0), n * cb)
        isolated(WR_REG, s.addr(0, 1, s.reg_row(), 0), data=bytes(n * cb))
        isolated(RD_REG, s.addr(0, 1, s.reg_row(), 0), n * cb)
    s.drain()
    return s


def _readback(s, units, banks, rows):
    """Script B of C13 and C25: every GB, register and accumulator of the
    units, and the given rows of their banks."""
    g = s.g
    vec = min(g.row_bytes(), s.contig())
    s.s = Script()
    s.s.add(1000, "drain")
    for u in units:
        s.pim(RD_GB, s.addr(u, 0, 0, 0), vec)
        for b in banks:
            s.pim(RD_REG, s.addr(u, b, s.reg_row(), 0), vec)
            s.pim(RD_REG, s.addr(u, b, s.acc_row(), 0), g.col_bytes)
            for r in rows:
                s.frd_row(u, b, r, 0, vec)
    s.drain()
    return s.s


def c13_scripts(geom, seed=18):
    """C13: script A sets up PIM state and ends with posted requests still
    queued (the run then checkpoints: the drain waits for their Done);
    script B, after the restore, reads the state back."""
    rng = random.Random(seed)
    s = PimScript(geom)
    nf = s.m.num_fmt
    vec = min(geom.row_bytes(), s.contig())
    units = [0, 1] if s.units() > 1 else [0]
    banks = list(range(min(8, s.banks())))
    _fill(s, rng, units, banks, [1, 2, 3])
    for u in units:
        s.pim(WR_GB, s.addr(u, 0, 0, 0), data=random_elems(rng, nf, vec))
    s.pim(
        WR_REG,
        s.addr(0, 3, s.reg_row(), 0),
        data=bytes(rng.getrandbits(8) for _ in range(vec)),
    )
    for r in (1, 2):
        s.pim(AB_MAC, s.addr(0, 0, r, 0), vec)
        s.pim(AB_EWADD_GBUF, s.addr(0, 0, r, 0), vec)
    s.pim(SB_MOV_GBUFBK, s.addr(units[-1], 2, 3, 0), vec)
    s.pim(AB_MOV_REGBK, s.addr(0, 0, 3, 0), vec)
    s.pim(SB_MAC, s.addr(units[-1], 4, 1, 0), desc=0)
    s.drain()
    a = s.s
    return a, _readback(s, units, banks, [1, 2, 3])


def c25_scripts(geom, seed=19):
    """C25: script A sends PIM requests, normal reads and writes that they
    hold (split: two transactions each on 32-byte columns), then waits 50
    us; a checkpoint taken in that wait drains them all. Script B, after
    the restore, reads the state back."""
    rng = random.Random(seed)
    s = PimScript(geom)
    nf = s.m.num_fmt
    vec = min(geom.row_bytes(), s.contig())
    n = min(64, s.contig())
    banks = list(range(min(8, s.banks())))
    _fill(s, rng, [0], banks, [1, 2])
    s.pim(WR_GB, s.addr(0, 0, 0, 0), data=random_elems(rng, nf, vec))
    for r in (1, 2):
        s.pim(AB_MAC, s.addr(0, 0, r, 0), vec)
        s.rd(s.addr(0, 5, r, 0), n)
        s.wr(s.addr(0, 6, r, 0), random_elems(rng, nf, n))
        s.pim(AB_EWADD_GBUF, s.addr(0, 0, r, 0), vec)
        s.rd(s.addr(0, 6, r, 0), n)
    s.s.add(50000000, "drain")
    a = s.s
    return a, _readback(s, [0], banks, [1, 2])


def c8_script(geom):
    """C8: one AB MAC over a whole row piece, then a register read that the
    unit's order puts after it."""
    s = PimScript(geom)
    s.pim(AB_MAC, s.addr(0, 0, 1, 0), min(geom.row_bytes(), s.contig()))
    s.pim(RD_REG, s.addr(0, 0, s.acc_row(), 0), geom.col_bytes)
    s.drain()
    return s


def c9_script(geom, jobs=12, seed=9):
    """C9: AB MACs on rows 1 .. jobs of unit 0, then their accumulators."""
    rng = random.Random(seed)
    s = PimScript(geom)
    vec = min(geom.row_bytes(), s.contig())
    _fill(s, rng, [0], range(s.banks()), range(1, jobs + 1))
    s.pim(WR_GB, s.addr(0, 0, 0, 0), data=random_elems(rng, s.m.num_fmt, vec))
    for r in range(1, jobs + 1):
        s.pim(AB_MAC, s.addr(0, 0, r, 0), vec)
    for b in range(s.banks()):
        s.pim(RD_REG, s.addr(0, b, s.acc_row(), 0), geom.col_bytes)
    s.drain()
    return s


def c16_script(geom, per_unit=10, seed=10):
    """C16: SB MACs alternating between units 0 and 1 of channel 0."""
    rng = random.Random(seed)
    s = PimScript(geom)
    _fill(s, rng, [0, 1], [0, 2], [1])
    for i in range(per_unit):
        for u in (0, 1):
            s.pim(
                SB_MAC,
                s.addr(u, 2 * (i % 2), 1, 0),
                desc=geom.cols // (1 + i % 3),
            )
    for u in (0, 1):
        for b in (0, 2):
            s.pim(RD_REG, s.addr(u, b, s.acc_row(), 0), geom.col_bytes)
    s.drain()
    return s


def c18_script(geom, seed=11):
    """C18: a register row written in one request, read in several."""
    rng = random.Random(seed)
    s = PimScript(geom)
    vec = min(geom.row_bytes(), s.contig())
    s.pim(
        WR_REG,
        s.addr(0, 3, s.reg_row(), 0),
        data=bytes(rng.getrandbits(8) for _ in range(vec)),
    )
    step = max(geom.col_bytes, vec // 4)
    for lo in range(0, vec, step):
        s.pim(
            RD_REG,
            s.addr(
                0, 3, s.reg_row(), lo // geom.col_bytes, lo % geom.col_bytes
            ),
            step,
        )
    s.pim(RD_REG, s.addr(0, 3, s.reg_row(), 0), vec)
    s.drain()
    return s


def c23_pim_script(geom, jobs=3, seed=12):
    """C23: AB MACs on row 1 (the host script reads row 1: held)."""
    rng = random.Random(seed)
    s = PimScript(geom)
    vec = min(geom.row_bytes(), s.contig())
    _fill(s, rng, [0], range(s.banks()), [1])
    for _ in range(jobs):
        s.pim(AB_MAC, s.addr(0, 0, 1, 0), vec)
    s.pim(RD_REG, s.addr(0, 0, s.acc_row(), 0), geom.col_bytes)
    s.drain()
    return s


def c23_order_script(geom, reads=8, seed=13):
    """C23 (second part): older reads of a row, then a PIM request on it,
    which may start once their columns issued."""
    rng = random.Random(seed)
    s = PimScript(geom)
    n = min(64, s.contig())
    _fill(s, rng, [0], [4], [2])
    for i in range(reads):
        s.rd(s.addr(0, 4, 2, (i * n // geom.col_bytes) % geom.cols), n)
    s.pim(SB_MAC, s.addr(0, 4, 2, 0), desc=geom.cols // 2)
    s.pim(RD_REG, s.addr(0, 4, s.acc_row(), 0), geom.col_bytes)
    s.drain()
    return s


def c24_pim_script(geom, jobs=10, seed=14):
    """C24: long AB jobs on channel 0 only."""
    rng = random.Random(seed)
    s = PimScript(geom)
    vec = min(geom.row_bytes(), s.contig())
    _fill(s, rng, [0], range(s.banks()), range(1, jobs + 1))
    for r in range(1, jobs + 1):
        s.pim(AB_MAC, s.addr(0, 0, r, 0), vec)
    s.pim(RD_REG, s.addr(0, 0, s.acc_row(), 0), geom.col_bytes)
    s.drain()
    return s


TIMING_SCRIPTS = (
    "c3",
    "c4",
    "c5",
    "c6",
    "c7",
    "c8",
    "c9",
    "c16",
    "c18",
    "c23-pim",
    "c23-order",
    "c24-pim",
    "c26",
    "c13",
    "c25",
    "host",
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("rdwr", help="a normal read/write PimTester script")
    r.add_argument("out")
    r.add_argument("--base", type=lambda v: int(v, 0), default=0)
    r.add_argument("--size", type=lambda v: int(v, 0), required=True)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--count", type=int, default=200)
    for name in ("c1", "c14", "errors") + TIMING_SCRIPTS:
        c = sub.add_parser(name, help="a %s PimTester script" % name)
        c.add_argument("out")
        c.add_argument(
            "--geometry", required=True, help="JSON of Geometry's arguments"
        )
        c.add_argument("--num-fmt", default="bf16", choices=FORMATS)
        c.add_argument("--acc-fmt", default="fp32", choices=FORMATS)
        c.add_argument("--no-values", action="store_true")
        c.add_argument("--seed", type=int, default=1)
        c.add_argument("--case", help="errors: write only this case")
        c.add_argument("--count", type=int, default=400, help="host: reads")
        c.add_argument("--unit", type=int, default=0, help="host: unit")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("c1", "c14", "errors") + TIMING_SCRIPTS:
        import json

        a = p.parse_args(argv)
        geom = Geometry(**json.loads(a.geometry))
        nf, af = FORMATS.index(a.num_fmt), FORMATS.index(a.acc_fmt)
        if a.cmd == "c1":
            s = c1_script(geom, nf, af, not a.no_values, a.seed)
        elif a.cmd == "c14":
            s = c14_script(geom, not a.no_values, a.seed)
        elif a.cmd in ("c13", "c25"):
            sa, sb = globals()[a.cmd + "_scripts"](geom)
            sa.write(a.out, "pim_ref.py %s (script A)" % " ".join(argv))
            sb.write(
                a.out + ".restore", "pim_ref.py %s (script B)" % " ".join(argv)
            )
            print(
                "%s: %d lines; %s.restore: %d lines"
                % (a.out, len(sa.lines), a.out, len(sb.lines))
            )
            return 0
        elif a.cmd == "host":
            s = host_reads(geom, a.count, a.unit, seed=a.seed)
        elif a.cmd in TIMING_SCRIPTS:
            s = globals()[a.cmd.replace("-", "_") + "_script"](geom)
        elif a.case:
            s = dict(error_cases(geom))[a.case]
        else:
            s = error_script(geom)
        s.write(a.out, "pim_ref.py %s" % " ".join(argv))
        print(
            "%s: %d lines, %d PIM requests, %d software errors"
            % (a.out, len(s.s.lines), s.pim_lines, s.m.errors)
        )
        return 0
    if argv and argv[0] == "rdwr":
        a = p.parse_args(argv)
        rdwr_script(a.base, a.size, a.seed, a.count).write(
            a.out,
            "pim_ref.py rdwr --base %#x --size %#x --seed %d --count %d"
            % (a.base, a.size, a.seed, a.count),
        )
        return 0
    dump = bool(argv) and argv[0] == "--dump"
    files = argv[1:] if dump else argv
    if not files:
        p.print_help()
        return 2
    run = Runner(dump)
    try:
        for f in files:
            run.run_file(f)
    except (VectorError, SoftwareError) as e:
        sys.stderr.write("%s\n" % e)
        return 2
    if not dump:
        print("pim_ref: %d checks, %d failures" % (run.checks, run.failures))
        return 1 if run.failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
