"""Check independent decoder rejection and hand-calculated layout/numeric examples."""
import struct
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools' / 'cent'))
from codec import decode
from reference import halves, half_bytes, packed_image, tree16


class CodecTests(unittest.TestCase):
    def test_transfer(self):
        body = struct.pack('<BQBQQ', 1, 256, 2, 0, 32)
        record = struct.pack('<HHI', 1, 0, len(body)) + body
        data = struct.pack('<4sHHII', b'NDP1', 2, 1, 16 + len(record), 1) + record
        self.assertEqual(decode(data), [{'op': 1, 'src': (1, 256), 'dst': (2, 0), 'size': 32}])
        for position in (0, 4, 6, 8, 12, 18, 20, 24):
            malformed = bytearray(data); malformed[position] ^= 255
            with self.subTest(position=position), self.assertRaises(ValueError):
                decode(malformed)
        for end in range(len(data)):
            with self.assertRaises(ValueError):
                decode(data[:end])
        with self.assertRaises(ValueError):
            decode(data + b'\0')

    def test_unknown_and_oversized_count(self):
        for opcode, count in [(0xffff, 1), (6, 1), (1, 65537)]:
            data = struct.pack('<4sHHIIHHI', b'NDP1', 2, 1, 24, count, opcode, 0, 0)
            with self.assertRaises(ValueError):
                decode(data)

    def test_irregular_packing(self):
        dense = half_bytes(list(range(175)))
        packed = packed_image(dense, 5, 35, 4, 32, 64)
        self.assertEqual(len(packed), 1024)
        # W[4,34] is row offset 3, bank 0, element 2 in the allocation.
        self.assertEqual(halves(packed[772:774]), [174])
        self.assertEqual(packed[774:], bytes(250))
        self.assertEqual(halves(packed[64:66]), [35])

    def test_tree(self):
        self.assertEqual(tree16([1]*16, [2]*16), 32)
        self.assertEqual(tree16([1,-1]*8, [1]*16), 0)
        with self.assertRaises(ValueError):
            tree16([1], [1])


if __name__ == '__main__':
    unittest.main()
