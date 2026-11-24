"""Detect real fixture corruption rather than merely comparing matching implementations."""
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools' / 'cent'))
from replay import load_cent, replay
from check_outputs import check_outputs

EXPORTER, SOURCE = Path(sys.argv.pop(1)).resolve(), Path(sys.argv.pop(1)).resolve()


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.fixture = cls.root / 'base'
        subprocess.run([str(EXPORTER), str(cls.fixture), '2','2','64','64','5','35','7','random','gemv','1'],
                       check=True, capture_output=True, timeout=60)
        cls.loaded = load_cent(SOURCE)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def copy(self):
        out = self.root / self.id().split('.')[-1]
        shutil.copytree(self.fixture, out)
        events = [json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
        return out, events

    def test_logical_output_oracle(self):
        out, events = self.copy()
        self.assertEqual(check_outputs(out), 11)
        # A matching producer/replayer bug could reorder valid rows. Check the API result independently.
        downloads = [event for event in events if event["kind"] == "download"]
        path = out / downloads[1]["file"]
        original = path.read_bytes()
        path.write_bytes(original[2:] + original[:2])
        with self.assertRaisesRegex(ValueError, "Logical output differs"):
            check_outputs(out)

    def test_both_modes(self):
        for mode in ('native','profile'):
            result = replay(self.fixture, SOURCE, mode, self.loaded)
            self.assertEqual(result['pim_jobs'],6)
            self.assertEqual(result['packed'],4)

    def test_padding_corruption(self):
        out, events = self.copy()
        file = out / next(e['file'] for e in events if e['kind']=='packed')
        data = bytearray(file.read_bytes()); data[-1] ^= 1; file.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'packed bytes/padding'):
            replay(out,SOURCE,loaded=self.loaded)

    def test_mac_address_corruption(self):
        out, events = self.copy()
        for event in events:
            if event['kind'] != 'submit': continue
            file = out/event['file']; data = bytearray(file.read_bytes()); pos=16
            while pos < len(data):
                op, reserved, size = struct.unpack_from('<HHI',data,pos)
                if op == 4:
                    struct.pack_into('<Q',data,pos+8+16,1) # MAC burst-column, not weight metadata.
                    file.write_bytes(data)
                    with self.assertRaisesRegex(ValueError, 'schedule differs'):
                        replay(out,SOURCE,loaded=self.loaded)
                    return
                pos += 8+size
        self.fail('No MAC found')

    def test_output_corruption(self):
        out, events = self.copy()
        file = out / next(e['file'] for e in reversed(events) if e['kind']=='download')
        data = bytearray(file.read_bytes()); data[0] ^= 1; file.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'Downloaded FP16'):
            replay(out,SOURCE,loaded=self.loaded)

    def test_wrong_profile(self):
        out, _ = self.copy()
        path=out/'manifest.json'; data=json.loads(path.read_text()); data['R']=63
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'Invalid profile'):
            replay(out,SOURCE,loaded=self.loaded)

    def test_rejected_export_dimensions(self):
        for n,k in [('0','35'),('5','0'),('18446744073709551615','35')]:
            result=subprocess.run([str(EXPORTER),str(self.root/'rejected'),'2','2','64','64',n,k,'7','exact','gemv','1'],
                                  capture_output=True,timeout=60)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((self.root/'rejected'/'manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
