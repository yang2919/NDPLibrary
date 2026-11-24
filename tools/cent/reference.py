"""Independent scalar arithmetic and address rules for CENT conformance checks."""
import ctypes
import ctypes.util
import math
import struct


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def half_bytes(values):
    return b''.join(struct.pack('<e', x) for x in values)


def halves(data):
    if len(data) % 2:
        raise ValueError('Odd FP16 byte count')
    return [x[0] for x in struct.iter_unpack('<e', data)]


def tree16(a, b):
    if len(a) != 16 or len(b) != 16:
        raise ValueError('MAC burst requires 16 lanes')
    lanes = [f32(x * y) for x, y in zip(a, b)]
    while len(lanes) > 1:
        lanes = [f32(lanes[i] + lanes[i + 1]) for i in range(0, len(lanes), 2)]
    return lanes[0]


_libm = ctypes.CDLL(ctypes.util.find_library('m') or None)
_fma = _libm.fmaf
_fma.argtypes = [ctypes.c_float] * 3
_fma.restype = ctypes.c_float


def matmul(a, b, rows, width, inner, transpose, factor, bias=None):
    output = []
    for r in range(rows):
        for c in range(width):
            value = 0.0
            for k in range(inner):
                value = _fma(a[r * inner + k], b[c * inner + k if transpose else k * width + c], value)
            value = f32(value * factor)
            if bias is not None:
                value = f32(value + bias[c])
            output.append(value)
    return half_bytes(output)


def packed_image(dense, n, k, banks, tile, row_bytes):
    groups = (n + banks - 1) // banks
    tiles = (k + tile - 1) // tile
    expected = bytearray(groups * tiles * banks * row_bytes)
    for i in range(n):
        for j in range(k):
            row = (i // banks) * tiles + j // tile
            offset = (row * banks + i % banks) * row_bytes + 2 * (j % tile)
            expected[offset:offset + 2] = dense[2 * (i * k + j):2 * (i * k + j + 1)]
    return bytes(expected)


def bound32(products):
    n = 4 * len(products) + 32
    u = 2 ** -24
    if n * u >= 1 or not all(math.isfinite(x) for x in products):
        raise ValueError('Outside finite arithmetic-bound domain')
    return (n * u / (1 - n * u)) * math.fsum(abs(x) for x in products)
