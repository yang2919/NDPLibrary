"""TestLib suites that run the Processing-in-Memory (PIM) test drivers.

The PIM tests keep their checks in two drivers, ``run_p4r.py`` (groups of
cases) and ``run_lib_se.py`` (the PIM library's SE runs). A suite made
here runs one driver over one group, or over the part of a group that
belongs to one test length, and passes when the driver reports no failed
and no skipped case. TestLib only selects, builds, runs and reports; the
driver's ``summary.txt`` (one line per case) is printed into the suite's
log, and its whole output stays in
``<result path>/pim/<suite name>/``.

Build tags: ``RISCV_RAMULATOR2`` names ``build/RISCV_RAMULATOR2``, the
only build with Ramulator 2.1. Cases that need no Ramulator 2.1 run on
``build/ALL`` (tag ``ALL``), which the driver takes as its native build
(``--native``). No suite is tagged ``RISCV``, so TestLib never
reconfigures ``build/RISCV``: the cases that must run on exactly that
build (the ``r6`` build checks) stay with the drivers.

The drivers expect the gem5 binaries under ``<gem5>/build``, TestLib's
default build directory. They build test programs and the PIM library
(``libpim``, whose object directory for generated layouts is shared) in
the source tree, so one PIM suite runs its driver at a time (a file lock);
each driver runs its own cases in parallel, as many at a time as
TestLib's ``-j`` (its SCons thread count, unused with ``--skip-build``).
"""

import contextlib
import fcntl
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from testlib.configuration import (
    config,
    constants,
)
from testlib.suite import TestSuite
from testlib.test_util import TestFunction

from gem5.fixture import Gem5Fixture

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R2 = constants.riscv_ramulator2_tag
ALL = constants.all_compiled_tag
SUMMARY = re.compile(r"^run_p4r: (\d+) PASS, (\d+) FAIL, (\d+) SKIP$", re.M)


def _jobs():
    return config.threads or os.cpu_count()


def _binary(tag, variant):
    return Path(config.build_dir) / tag / f"gem5.{variant}"


def _check_build_dir():
    if Path(config.build_dir).resolve() != (ROOT / "build").resolve():
        raise RuntimeError(
            "the PIM drivers use the binaries under "
            f"{ROOT / 'build'}, not {config.build_dir}"
        )


@contextlib.contextmanager
def _driver_lock():
    """One PIM driver at a time, across TestLib's parallel suites."""
    lock = Path(config.result_path) / "pim" / ".driver.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _out(name):
    out = Path(config.result_path) / "pim" / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    return out


def _suite(name, tag, variant, length, builds, run):
    """A TestSuite of one test, with a build fixture for each of
    `builds` ((tag, variant) pairs, the suite's own first)."""
    fixtures = [Gem5Fixture(t, v) for t, v in builds]
    return TestSuite(
        name=f"{name}-{tag}-{constants.host_x86_64_tag}-{variant}",
        fixtures=fixtures,
        tags=[tag, variant, length, constants.host_x86_64_tag],
        tests=[TestFunction(run, name=name)],
    )


def p4r_suite(
    name,
    groups,
    length,
    tag=R2,
    variant=constants.opt_tag,
    only=None,
    extra_builds=(),
):
    """A suite that runs ``run_p4r.py --groups <groups>`` (and ``--only
    <only>``); `extra_builds` are further (tag, variant) builds its cases
    use. Cases that the driver runs on its native build get
    ``build/ALL/gem5.opt``."""
    builds = [(tag, variant), *extra_builds]

    def run(params):
        _check_build_dir()
        out = _out(name)
        cmd = [
            sys.executable,
            str(HERE / "run_p4r.py"),
            "--out",
            str(out),
            "--groups",
            ",".join(groups),
            "-j",
            str(_jobs()),
        ]
        if only:
            cmd += ["--only", only]
        if (ALL, constants.opt_tag) in builds:
            cmd += ["--native", str(_binary(ALL, constants.opt_tag))]
        with _driver_lock():
            rc = subprocess.run(
                cmd, cwd=ROOT, stdout=subprocess.DEVNULL
            ).returncode
        summary = out / "summary.txt"
        text = summary.read_text() if summary.exists() else ""
        sys.stdout.write(text)
        m = SUMMARY.search(text)
        if not m:
            raise RuntimeError(f"run_p4r.py wrote no summary (rc {rc})")
        npass, nfail, nskip = map(int, m.groups())
        if rc or nfail or nskip or not npass:
            raise RuntimeError(
                f"run_p4r.py: {npass} PASS, {nfail} FAIL, {nskip} SKIP "
                f"(rc {rc}); details in {out}"
            )

    return _suite(name, tag, variant, length, builds, run)


def lib_se_suite(name, length):
    """A suite that runs every case of ``run_lib_se.py``."""

    def run(params):
        _check_build_dir()
        out = _out(name)
        log = out / "run_lib_se.log"
        with open(log, "w") as f, _driver_lock():
            rc = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "run_lib_se.py"),
                    "--gem5",
                    str(_binary(R2, constants.opt_tag)),
                    "--work",
                    str(out),
                    "-j",
                    str(_jobs()),
                ],
                cwd=ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
            ).returncode
        text = log.read_text(errors="replace")
        sys.stdout.write(text)
        if rc:
            raise RuntimeError(f"run_lib_se.py failed (rc {rc}); see {log}")

    return _suite(
        name, R2, constants.opt_tag, length, [(R2, constants.opt_tag)], run
    )
