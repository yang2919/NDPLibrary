"""Exercise N0 fixtures over a live v3 process, including failed-request recovery."""
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from client import Client
from launch_fixture import main as write_golden


def static_elf(entry):
    # Minimal static RISC-V ET_EXEC with addi a0,zero,0; ret and an 8-byte BSS.
    ident = b"\x7fELF\x02\x01\x01".ljust(16, b"\0")
    header = struct.pack("<16sHHIQQQIHHHHHH", ident, 2, 243, 1, entry, 64, 0, 0,
                         64, 56, 1, 0, 0, 0)
    segment = struct.pack("<II6Q", 1, 5, 256, entry, entry, 8, 16, 256)
    return (header + segment).ljust(256, b"\0") + struct.pack("<II", 0x513, 0x8067)


class ServiceTest(unittest.TestCase):
    def test_session(self):
        with tempfile.TemporaryDirectory() as tmp, subprocess.Popen(
                [worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
            directory = Path(tmp)
            write_golden(directory)
            c = Client(process.stdout, process.stdin)
            c.allocate(4096, 3)  # Match N0's first operand address.
            for profile, name, dims, sizes in [(1, "add", (5, 0), (10, 10, 10)),
                                               (2, "gemv", (2, 3), (16, 6, 4))]:
                template = struct.pack("<IIIIQQQ", 1, 40, *dims, 0, 0, 0)
                slots, bindings = [], []
                for i, size in enumerate(sizes):
                    offset = 2 if name == "add" and i == 0 else 0
                    access = 2 if i == 2 else 1
                    alignment = 4 if name == "gemv" and i == 0 else 2
                    buffer = c.allocate(size + offset, access)
                    slots.append((16 + 8*i, access, alignment, size))
                    bindings.append((16 + 8*i, access, buffer, offset, size))
                entry = 0x10000000 + 4096 * profile
                image = static_elf(entry)
                for malformed, key in [(b"invalid", profile), (image, 99)]:
                    with self.assertRaises(RuntimeError):
                        c.load(malformed, key, template, slots)
                program = c.load(image, profile, template, slots)
                for _ in range(2):
                    pc, args = c.launch(program, template, bindings)
                    self.assertEqual(pc, entry)
                    self.assertEqual(args, (directory / f"{name}.args").read_bytes())
                for patch in [(16, 1, 999, 0, sizes[0]), (16, 1, bindings[0][2], 80, sizes[0]),
                              bindings[1], (16, 2, bindings[0][2], 0, sizes[0])]:
                    with self.assertRaises(RuntimeError):
                        c.launch(program, template, [patch, *bindings[1:]])
                c.unload(program)
                with self.assertRaises(RuntimeError):
                    c.launch(program, template, bindings)
            c.release(buffer)
            with self.assertRaises(RuntimeError):
                c.release(buffer)
            process.stdin.close()
            self.assertEqual(process.wait(timeout=5), 0)


if __name__ == "__main__":
    worker = sys.argv.pop(1)
    unittest.main()
