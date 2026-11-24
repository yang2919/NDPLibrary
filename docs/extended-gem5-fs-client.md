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

## Repeat chain on the Mac and retain both verdicts

In terminal A, from the project root:

```sh
PORT=45242 STAGE=FSSTREAM FS_CASE=chain ./tools/gem5/pim_kernel/run.sh
```

After this run prints `NDP_FS_HOST_READY port=45242 (host loopback)`, run
the following in terminal B from the same root. Use the absolute directory
printed after `Results:` in A for RUN_DIR.

```sh
RUN_DIR="/absolute/path/printed/by/terminal/A"
(set -o pipefail; python3 tools/gem5/pim_kernel/verify_fs_remote.py --case chain --port 45242 2>&1 | tee "$RUN_DIR/client.log")
```

Expect `NDP_FS_CLIENT_PASS chain` in client.log. Leave A running through
`NDP_FS_CLEANUP_ATOMIC` until it exits. Its result.txt must contain
`NDP_FS_EXTERNAL_PASS case=chain`. Together these establish the numerical
checks (11 and 2), four completed GEMVs, guest exit and controller completions
for the same run.

## Repeat standalone GEMV on the Mac

Start terminal A from the project root:

```sh
PORT=45242 STAGE=FSSTREAM FS_CASE=gemv ./tools/gem5/pim_kernel/run.sh
```

After this run reports READY on port 45242, run the following from the same
project root in terminal B. The runner writes latest-run.txt at startup. Use
this shortcut when this is the most recently started run.

```sh
RUN_DIR=$(cat build/extended-gem5/latest-run.txt)
(set -o pipefail; PYTHONPATH="$PWD/build-python/python" python3 tools/gem5/pim_kernel/verify_remote.py --label NDP_FS --port 45242 2>&1 | tee "$RUN_DIR/client.log")
```

Require outputs [6, 1] and [-1, -4] and NDP_FS_CLIENT_PASS in client.log,
then NDP_FS_EXTERNAL_PASS case=gemv in result.txt after guest cleanup.
