#!/usr/bin/env python3
"""Run gem5 PyTrafficGen latency-throughput sweep and write result into CSV."""

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import config as gem5_config

CONFIG_PATH = SCRIPT_DIR / "config.py"
DEFAULT_READ_RATIOS = [100, 90, 80, 70, 60, 50]
DEFAULT_INTENSITIES = [i / 100.0 for i in range(5, 101, 5)]
STAT_FLOAT_RE = re.compile(r"^([A-Za-z0-9_:.]+)\s+([-+A-Za-z0-9_.]+)")

CSV_FIELDS = [
    "dram",
    "dram_class",
    "org_preset",
    "timing_preset",
    "controller_class",
    "scheduler_class",
    "row_policy",
    "channels",
    "traffic",
    "read_ratio",
    "intensity",
    "rate_bps",
    "peak_gbps",
    "block_size",
    "duration",
    "memory_per_channel",
    "addr_mapper",
    "refresh_manager",
    "read_buffer_size",
    "write_buffer_size",
    "observed_gbps",
    "read_bw_Bps",
    "write_bw_Bps",
    "avg_read_latency_ns",
    "avg_write_latency_ns",
    "total_reads",
    "total_writes",
    "run_dir",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one-channel-count gem5 PyTrafficGen sweeps."
    )
    parser.add_argument(
        "--dram", required=True, choices=sorted(gem5_config.DRAM_PROFILES)
    )
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument(
        "--traffic",
        nargs="+",
        choices=("stream", "random"),
        default=["stream", "random"],
    )
    parser.add_argument(
        "--read-ratios", type=int, nargs="+", default=DEFAULT_READ_RATIOS
    )
    parser.add_argument(
        "--intensities", type=float, nargs="+", default=DEFAULT_INTENSITIES
    )
    gem5_config.add_shared_args(parser)
    parser.add_argument(
        "--gem5-bin",
        default=os.environ.get("GEM5_BIN", "build/X86/gem5.opt"),
        help="gem5 binary path. Defaults to GEM5_BIN or build/X86/gem5.opt.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory. Default: /tmp/gem5-pytrafficgen/<DRAM>_<MAPPER>_ch<N>/.",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    args.gem5_bin = Path(args.gem5_bin).resolve()
    if args.out_dir is None:
        args.out_dir = Path("/tmp/gem5-pytrafficgen") / (
            f"{args.dram}_{args.addr_mapper}_ch{args.channels}"
        )
    else:
        args.out_dir = args.out_dir.resolve()
    return args


def gem5_env():
    env = os.environ.copy()
    for key, prefix in (
        ("PYTHONPATH", f"{REPO_ROOT}/python{os.pathsep}{REPO_ROOT}"),
        ("LD_LIBRARY_PATH", str(REPO_ROOT)),
    ):
        env[key] = prefix + (os.pathsep + env[key] if key in env else "")
    return env


def generator_stats(stats, suffix):
    matches = [
        value
        for name, value in stats.items()
        if name.endswith(f".generator.{suffix}")
    ]
    if not matches:
        raise KeyError(f"missing gem5 generator stat {suffix}")
    return matches


def weighted_latency(stats, latency_suffix, count_suffix):
    latencies = generator_stats(stats, latency_suffix)
    counts = generator_stats(stats, count_suffix)

    weighted_sum = 0.0
    total_count = 0.0
    for latency, count in zip(latencies, counts, strict=True):
        if math.isnan(latency) or count == 0:
            continue
        weighted_sum += latency * count
        total_count += count
    return math.nan if total_count == 0 else weighted_sum / total_count


def parse_run_outputs(run_dir):
    stats = {}
    for line in (run_dir / "stats.txt").read_text().splitlines():
        match = STAT_FLOAT_RE.match(line.strip())
        if match:
            stats[match.group(1)] = float(match.group(2))

    if "simFreq" not in stats:
        raise KeyError("missing gem5 stat simFreq")
    sim_freq = stats["simFreq"]
    read_bw = sum(generator_stats(stats, "readBW"))
    write_bw = sum(generator_stats(stats, "writeBW"))
    avg_read_latency_ticks = weighted_latency(
        stats, "avgReadLatency", "totalReads"
    )
    avg_write_latency_ticks = weighted_latency(
        stats, "avgWriteLatency", "totalWrites"
    )
    total_reads = sum(generator_stats(stats, "totalReads"))
    total_writes = sum(generator_stats(stats, "totalWrites"))
    return {
        "read_bw_Bps": read_bw,
        "write_bw_Bps": write_bw,
        "observed_gbps": (read_bw + write_bw) / 1e9,
        "avg_read_latency_ns": avg_read_latency_ticks / sim_freq * 1e9,
        "avg_write_latency_ns": avg_write_latency_ticks / sim_freq * 1e9,
        "total_reads": int(total_reads),
        "total_writes": int(total_writes),
    }


def base_row(args, profile, block_size, peak_gbps):
    return {
        "dram": args.dram,
        **profile,
        "channels": args.channels,
        "peak_gbps": peak_gbps,
        "block_size": block_size,
        "duration": args.duration,
        "memory_per_channel": args.memory_per_channel,
        "addr_mapper": args.addr_mapper,
        "refresh_manager": args.refresh_manager,
        "read_buffer_size": args.read_buffer_size,
        "write_buffer_size": args.write_buffer_size,
    }


def make_command(args, row):
    cmd = [
        str(args.gem5_bin),
        "-d",
        row["run_dir"],
        str(CONFIG_PATH),
        "--dram",
        args.dram,
        "--channels",
        str(args.channels),
        "--intensity",
        str(row["intensity"]),
        "--read-ratio",
        str(row["read_ratio"]),
        "--traffic",
        row["traffic"],
        "--duration",
        args.duration,
        "--memory-per-channel",
        args.memory_per_channel,
        "--addr-mapper",
        args.addr_mapper,
        "--refresh-manager",
        args.refresh_manager,
    ]
    for key in ("read_buffer_size", "write_buffer_size"):
        if getattr(args, key) is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(getattr(args, key))])
    return cmd


def run_gem5_point(args, row):
    run_dir = Path(row["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = make_command(args, row)
    (run_dir / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    (run_dir / "resolved_config.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n"
    )

    print("Run config:")
    print(json.dumps(row, indent=2, sort_keys=True))
    print("Command:")
    print(" ".join(cmd))

    try:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=gem5_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        (run_dir / "gem5.log").write_text(exc.stdout or "")
        raise TimeoutError(f"gem5 timed out after {args.timeout}s: {run_dir}") from exc

    (run_dir / "gem5.log").write_text(completed.stdout)
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise RuntimeError(
            f"gem5 failed with exit code {completed.returncode}: {run_dir}\n{tail}"
        )
    return parse_run_outputs(run_dir)


def rows_for_sweep(args, profile, block_size, peak_gbps):
    rows = []
    common = base_row(args, profile, block_size, peak_gbps)
    total = len(args.traffic) * len(args.read_ratios) * len(args.intensities)
    index = 0
    for traffic in args.traffic:
        for read_ratio in args.read_ratios:
            for intensity in args.intensities:
                index += 1
                run_dir = (
                    args.out_dir
                    / "runs"
                    / traffic
                    / f"rr{read_ratio}"
                    / f"i{intensity:g}".replace(".", "p")
                )
                row = {
                    **common,
                    "traffic": traffic,
                    "read_ratio": read_ratio,
                    "intensity": intensity,
                    "rate_bps": int(peak_gbps * 1e9 * intensity),
                    "run_dir": str(run_dir),
                }
                print(
                    f"[run {index}/{total}] "
                    f"{traffic} ch={args.channels} rr={read_ratio} "
                    f"intensity={intensity:g}"
                )
                rows.append({**row, **run_gem5_point(args, row)})
    return rows


def main():
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        if not args.force:
            raise FileExistsError(
                f"output directory is not empty: {args.out_dir}; use --force"
            )
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    profile = gem5_config.DRAM_PROFILES[args.dram]
    block_size, peak_per_channel = gem5_config.resolve_profile_spec(profile)
    peak_gbps = peak_per_channel * args.channels

    sweep_config = {
        **base_row(args, profile, block_size, peak_gbps),
        "traffic": args.traffic,
        "read_ratios": args.read_ratios,
        "intensities": args.intensities,
        "peak_gbps_per_channel": peak_per_channel,
        "gem5_bin": str(args.gem5_bin),
        "out_dir": str(args.out_dir),
    }
    print(f"Output directory: {args.out_dir}")
    print("Resolved sweep config:")
    print(json.dumps(sweep_config, indent=2, sort_keys=True))
    (args.out_dir / "resolved_sweep_config.json").write_text(
        json.dumps(sweep_config, indent=2, sort_keys=True) + "\n"
    )

    rows = rows_for_sweep(args, profile, block_size, peak_gbps)
    results_csv = args.out_dir / "results.csv"
    with results_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {results_csv}")


if __name__ == "__main__":
    main()
