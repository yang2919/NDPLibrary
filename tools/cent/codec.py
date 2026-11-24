"""Decode the existing NDP1 wire schema strictly; no Tensor planner is implemented here."""
import struct


class Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def read(self, fmt):
        size = struct.calcsize('<' + fmt)
        if size > len(self.data) - self.pos:
            raise ValueError('Truncated wire field')
        value = struct.unpack_from('<' + fmt, self.data, self.pos)
        self.pos += size
        return value[0] if len(value) == 1 else value

    def address(self):
        space, offset = self.read('BQ')
        if space not in (1, 2, 3, 4):
            raise ValueError('Invalid address space')
        return (space, offset)

    def operand(self):
        return (self.address(), self.read('Q'))

    def count(self, item_bytes):
        count = self.read('I')
        if count > (len(self.data) - self.pos) // item_bytes:
            raise ValueError('Invalid wire count')
        return count

    def boolean(self):
        value = self.read('B')
        if value not in (0, 1):
            raise ValueError('Invalid boolean')
        return bool(value)


def decode(data):
    if len(data) > 8 * 1024 * 1024 - 1:
        raise ValueError('Batch exceeds execution frame limit')
    r = Reader(data)
    magic, version, profile, length, count = r.read('4sHHII')
    if (magic, version, profile, length) != (b'NDP1', 2, 1, len(data)) or count > 65536:
        raise ValueError('Invalid batch header')
    if count > (len(data) - 16) // 8:
        raise ValueError('Invalid record count')
    commands = []
    for _ in range(count):
        op, reserved, size = r.read('HHI')
        if reserved or size > len(data) - r.pos:
            raise ValueError('Invalid record header')
        b = Reader(data[r.pos:r.pos + size])
        r.pos += size
        c = {'op': op}
        if op in (1, 9):
            c.update(src=b.address(), dst=b.address(), size=b.read('Q'))
        elif op == 2:
            c['channels'], c['banks'], c['offset'] = b.read('QQQ')
        elif op == 4:
            c['channels'], c['row'], c['column'], c['bursts'] = b.read('QQQQ')
        elif op == 5:
            c['channels'], c['offset'] = b.read('QQ')
        elif op == 7:
            c['kernel'] = b.read('H')
            if not 1 <= c['kernel'] <= 10:
                raise ValueError('Invalid PNM kernel')
            c['inputs'] = [b.operand() for _ in range(b.count(17))]
            c['output'] = b.operand()
            c['rows'], c['width'], c['inner'] = b.read('QQQ')
            c['transpose'] = b.boolean()
            c['factor'], c['epsilon'] = b.read('ff')
            c['indices'] = [b.read('Q') for _ in range(b.count(8))]
        elif op == 8:
            c['src'], c['dst'] = b.address(), b.address()
            c['rows'], c['cols'], c['banks'], c['tile'], c['row_bytes'] = b.read('QQQQQ')
            c['pack'] = b.boolean()
        elif op == 10:
            c['dst'], c['size'], c['value'] = b.address(), b.read('Q'), b.read('B')
        else:
            raise ValueError(f'Unsupported replay opcode {op}')
        if b.pos != len(b.data):
            raise ValueError('Trailing command bytes')
        commands.append(c)
    if r.pos != len(data):
        raise ValueError('Trailing batch bytes')
    return commands
