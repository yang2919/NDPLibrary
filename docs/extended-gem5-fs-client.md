# External Python client on extended gem5 FS

For a new Linux server, follow the [workspace handoff](linux-server-handoff.md) first.
Run from the repository root with Docker running. Use a fresh run for
each row below. Terminal A boots the existing patched Linux image on Atomic,
starts the FP16 worker with PIM_MODE=fs, and keeps startup and pim_init on Atomic.
At the first worker input read, hypercall 10 switches to Timing before the TCP
listener announces readiness. The FSSTREAM wall-clock timeout defaults to three hours.
An exit status of 124 in result.txt indicates the outer timeout expired.

```sh
STAGE=FSSTREAM FS_CASE=gemv ./tools/gem5/pim_kernel/run.sh
```

Wait for **NDP_FS_HOST_READY** in terminal A. In terminal B run:

```sh
export PYTHONPATH="$PWD/build-python/python"
python3 tools/gem5/pim_kernel/verify_remote.py --label NDP_FS
```

Expect GEMV outputs [6, 1] and [-1, -4] and NDP_FS_CLIENT_PASS. Terminal A
must also print **NDP_FS_EXTERNAL_PASS case=gemv**. For subsequent runs replace
FS_CASE in terminal A and use the indicated Python client in terminal B.

| FS_CASE | Terminal B command (after READY) | Acceptance |
|---|---|---|
| chain | `python3 tools/gem5/pim_kernel/verify_fs_remote.py --case chain` | Reuse the same input, intermediate and output allocations for two inputs. The second GEMV consumes the first result before either download. Final values 11 and 2, four successful launches. |
| partial | `python3 tools/gem5/pim_kernel/verify_fs_remote.py --case partial` | Disconnect one byte before the end of GEMV. Require NDP_FS_PARTIAL_GEMV received=12 expected=13, zero GEMVs dispatched and zero completed all-bank MACs. |
| lost | `python3 tools/gem5/pim_kernel/verify_fs_remote.py --case lost` | Send a complete GEMV and close without reading its reply. Exactly one successful GEMV, positive controller completions, guest shutdown. |
| decoder | `python3 tools/gem5/pim_kernel/verify_decoder_remote.py` | Production single-device adapter passes prefill/decode, tokenwise and reset comparisons. The shared client prints NDP_SE_DECODER_CLIENT_PASS. |

For every case require **NDP_FS_EXTERNAL_PASS case=...** in A. For disconnect
cases B only reports injection. A checks the guest completion history, worker
shutdown, guest script exit and timing trace. The simulator wall-clock limit
includes boot and the wait for B. Set BOOT_TIMEOUT in seconds for a longer run.
The run directory printed by A contains simulation.log, result.txt,
sim/board.platform.terminal and sim/pim.trace.

The TCP connection terminates at the simulator process. Guest hypercalls pause
execution between frames while the host exchanges one request or reply through
gem5 readfile/writefile. Only complete requests enter the guest. On disconnect
the guest sees EOF at the next request boundary, after the preceding synchronous
dispatch. Its session destructs and libpim finalizes before the close marker.
This tests ordered cleanup of a disconnected client on a dedicated instance.

After synchronous dispatch and session-buffer cleanup, the relay delivers the
close reply and prints NDP_FS_SESSION_DRAINED. Terminal B may finish while
Linux unmaps the worker region. New runs switch back to the boot (Atomic) CPU
after the close reply and print NDP_FS_CLEANUP_ATOMIC. All client operations,
including the dependent GEMVs, execute on Timing. Terminal A checks finalization, guest exit and
the controller trace before reporting NDP_FS_EXTERNAL_PASS.

## Save both terminal verdicts

Use the absolute directory printed after `Results:` in terminal A. For a custom
port, set `PORT` in A and pass the same number with `--port` in B. For example,
with a GEMV run listening on port 45242, capture the client output in B:

```sh
RUN_DIR="/absolute/path/printed/by/terminal/A"
set -o pipefail
PYTHONPATH="$PWD/build-python/python" python3 tools/gem5/pim_kernel/verify_remote.py --label NDP_FS --port 45242 2>&1 | tee "$RUN_DIR/client.log"
```

Pair the client output with the final verdict in that directory's `result.txt`.
