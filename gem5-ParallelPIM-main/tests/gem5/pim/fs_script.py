#!/usr/bin/env python3
"""Write a guest run script that carries a static RISC-V program.

A full-system (FS) run of ``configs/example/gem5_library/riscv-pim.py``
gives its guest one script through gem5's ``readfile``: the Ubuntu
image's ``after_boot.sh`` reads it with ``gem5-bridge readfile``, runs it
and then exits the simulation with hypercall 3. This tool embeds a static
program in that script as base64, so a program reaches the guest without
any change to the disk image.

The script prints ``PIM-FS begin`` with ``uname -a``, runs the setup
commands as root (for example reserving huge pages), writes the program
to ``/tmp``, runs it (as root through ``sudo -S`` when the script does not
run as root: the image logs in as the ``gem5`` user, whose password is
gem5-resources' documented ``12345``, and ``/dev/mem`` needs root),
brackets it with hypercalls 4 and 5 (``riscv-pim.py`` resets the
statistics at 4 and dumps them at 5) and prints ``PIM-FS exit <status>``.

As a module: ``make_script(program, args, ...)`` returns the text. As a
tool:

    fs_script.py --program <static binary> [--root] [--env NAME=VALUE]
        -o <script> -- <args>
    fs_script.py --command 'uname -a' -o <script>
"""

import argparse
import base64
import shlex
import sys
from pathlib import Path

GUEST_PATH = "/tmp/pim-program"
# gem5-resources' documented password of the image's gem5 user.
PASSWORD = "12345"


def make_script(
    program=None,
    args=(),
    root=True,
    commands=None,
    work_markers=True,
    password=PASSWORD,
    env=(),
    setup=(),
    wrap=(),
    console=False,
):
    """The guest script: run the shell commands `setup` as root, then
    `program` with `args` and the environment `env` ("NAME=VALUE"
    strings), started by the command `wrap` when given (for example
    ``numactl --membind=1``), and/or the shell `commands`, with statistics
    markers around the program. With `console`, the program writes to
    ``/dev/console`` directly, not through ``sudo``'s pseudo-terminal
    (``use_pty``), so that ``tcdrain()`` waits for the serial console: a
    program whose marker ends the run loses no output."""
    lines = [
        "#!/bin/bash",
        "# Written by tests/gem5/pim/fs_script.py.",
        'echo "PIM-FS begin $(uname -a)"',
    ]
    for cmd in setup:
        lines += [
            'if [ "$(id -u)" -eq 0 ]; then',
            f"    sh -c {shlex.quote(cmd)}",
            "else",
            f"    echo {shlex.quote(password)} | sudo -S -p '' sh -c "
            f"{shlex.quote(cmd)}",
            "fi",
        ]
    if program is not None:
        data = base64.encodebytes(Path(program).read_bytes()).decode()
        lines += [
            f"base64 -d > {GUEST_PATH} <<'PIM_FS_EOF'",
            data.rstrip("\n"),
            "PIM_FS_EOF",
            f"chmod 755 {GUEST_PATH}",
        ]
        run = shlex.join(
            (["env", *env] if env else [])
            + list(wrap)
            + [GUEST_PATH, *map(str, args)]
        )
        if console:
            run = shlex.join(["sh", "-c", f"exec {run} >/dev/console 2>&1"])
        if root:
            lines += [
                'if [ "$(id -u)" -eq 0 ]; then',
                f"    run() {{ {run}; }}",
                "else",
                f"    run() {{ echo {shlex.quote(password)} | "
                f"sudo -S -p '' {run}; }}",
                "fi",
            ]
        else:
            lines.append(f"run() {{ {run}; }}")
        if work_markers:
            lines.append("gem5-bridge hypercall 4")
        lines += ["run", "rc=$?"]
        if work_markers:
            lines.append("gem5-bridge hypercall 5")
    else:
        lines.append("rc=0")
    if commands:
        lines += [commands, "rc=$((rc | $?))"]
    lines.append('echo "PIM-FS exit $rc"')
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--program", help="static RISC-V program to embed")
    ap.add_argument(
        "--root",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the program as root (default)",
    )
    ap.add_argument("--command", help="shell commands to run after it")
    ap.add_argument(
        "--wrap",
        default="",
        help="a command that starts the program, for example "
        "'numactl --membind=1'",
    )
    ap.add_argument(
        "--setup",
        action="append",
        default=[],
        help="a shell command to run as root first (repeatable)",
    )
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="an environment variable of the program (repeatable)",
    )
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("args", nargs="*", help="the program's arguments")
    a = ap.parse_args()
    if a.program is None and a.command is None:
        ap.error("give --program or --command")
    Path(a.output).write_text(
        make_script(
            a.program,
            a.args,
            a.root,
            a.command,
            env=a.env,
            setup=a.setup,
            wrap=shlex.split(a.wrap),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
