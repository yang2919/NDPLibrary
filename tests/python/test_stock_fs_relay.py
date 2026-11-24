"""Exercise the stock FS mailbox and guest worker with native exit/resume events."""
import os
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.extend([str(ROOT / "tools/gem5/fs"), str(ROOT / "tools/gem5/pim_kernel")])
import k4_relay


@unittest.skipUnless(os.getenv("NDP_GEM5_PIM_WORKER"), "requires native FP16 worker")
class StockRelayTest(unittest.TestCase):
    def test_projection(self):
        worker = os.environ["NDP_GEM5_PIM_WORKER"]
        pauses = queue.Queue()
        with tempfile.TemporaryDirectory() as tmp, socket.socket() as listener:
            folder = Path(tmp)
            request = folder / "readfile"
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(10)

            m5 = folder / "m5"
            os.mkfifo(folder / "pause")
            os.mkfifo(folder / "resume")
            m5.write_text("""#!/bin/sh
set -eu
case "$1" in
  exit) echo paused > "$MAILBOX/pause"; read token < "$MAILBOX/resume" ;;
  readfile) cat "$MAILBOX/readfile" ;;
  writefile) cp "$2" "$MAILBOX/$2" ;;
  *) exit 1 ;;
esac
""")
            m5.chmod(0o700)
            errors = []

            def guest():
                try:
                    subprocess.run(["sh", str(ROOT / "tools/gem5/fs/k4_guest.sh"), worker, str(m5)],
                                   env=dict(os.environ, MAILBOX=str(folder)), check=True, timeout=30)
                except Exception as error:
                    errors.append(error)
                finally:
                    pauses.put(None)

            def simulator():
                try:
                    (folder / "pause").read_text()
                    assert (folder / "k4-ready.txt").is_file()
                    client, _ = listener.accept()
                    with client, client.makefile("rb") as stream:
                        for done in k4_relay.exchange_events(folder, request, client, stream):
                            if done:
                                break
                            (folder / "resume").write_text("resume\n")
                            if request.read_bytes()[4:] != b"Q":
                                (folder / "pause").read_text()
                            else:
                                pauses.get(timeout=10)
                except Exception as error:
                    errors.append(error)

            threads = [threading.Thread(target=f) for f in (guest, simulator)]
            for thread in threads:
                thread.start()
            try:
                client_run = subprocess.run(
                    [sys.executable, str(ROOT / "tools/gem5/pim_kernel/verify_remote.py"),
                     "--port", str(listener.getsockname()[1])],
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(client_run.returncode, 0, client_run.stderr)
                self.assertIn("K4_CLIENT_PASS", client_run.stdout)
            finally:
                for thread in threads:
                    thread.join(15)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])

    def test_truncated_stream(self):
        import io
        with self.assertRaises(EOFError):
            k4_relay.read_frame(io.BytesIO(b"\x03\0\0\0H"))
