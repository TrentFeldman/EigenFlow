#!/usr/bin/env python3
"""Compare HDF5 storage copies without changing the production snapshot writer.

Uses the C++ fixture tool plus installed h5repack/h5diff; Python stdlib only.
Writes a fresh run folder with deterministic fixtures, all trials, metadata,
per-field storage sizes and a Markdown/CSV summary. Whole-file timings include
process startup and cached input reads; writes include close plus fsync. These
are single-process repack measurements, not sustained CFD-writer benchmarks.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import time

PROFILES = ["gradient", "taylor_green", "multiscale", "noise"]
VARIANTS = {
    "contiguous": ["-l", "CONTI"],
    "chunk64": ["-l", "CHUNK=64x64"],
    "gzip1_chunk64": ["-l", "CHUNK=64x64", "-f", "GZIP=1"],
    "shuffle_gzip1_chunk64": ["-l", "CHUNK=64x64", "-f", "SHUF", "-f", "GZIP=1"],
    "shuffle_gzip4_chunk64": ["-l", "CHUNK=64x64", "-f", "SHUF", "-f", "GZIP=4"],
    "shuffle_gzip1_chunk256": ["-l", "CHUNK=256x256", "-f", "SHUF", "-f", "GZIP=1"],
}
SEED = 20261020


def call(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def flush_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path("build/release/fluid_compression_fixture"))
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.size < 256 or args.repeats < 1:
        parser.error("Require size >=256 and repeats >=1.")
    fixture = args.fixture.resolve(strict=True)
    for tool in ("h5repack", "h5diff"):
        if not shutil.which(tool):
            parser.error(f"Required installed tool is missing: {tool}")
    date = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    directory = args.output or Path("runs") / f"{date}-compression-baseline-01"
    directory.mkdir(parents=True, exist_ok=False)
    status = directory / "status.txt"
    status.write_text("running\n")
    try:
        environment = {
            "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "platform": platform.platform(), "python": platform.python_version(),
            "cpu": call(["lscpu"]), "filesystem": call(["df", "-T", str(directory)]),
            "h5repack": call(["h5repack", "--version"]),
            "fixture_binary": str(fixture), "fixture_sha256": fingerprint(fixture),
            "fixture_build": (fixture.parent / "generated" / "BuildInfo.h").read_text(),
            "seed": SEED, "size": args.size, "repeats": args.repeats,
            "profiles": PROFILES, "variants": VARIANTS,
            "timing": "cached repack process + output file/directory fsync; warm whole-file reads",
            "warning": "Synthetic snapshots; no time-evolving CFD data or cold/sustained bandwidth claim.",
        }
        (directory / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
        originals = {}
        for profile in PROFILES:
            path = directory / f"input-{profile}.h5"
            call([str(fixture), "--generate", str(path), str(args.size), profile])
            flush_file(path)
            originals[profile] = path
        trials = [(profile, variant, repeat) for profile in PROFILES for variant in VARIANTS
                  for repeat in range(args.repeats)]
        random.Random(SEED).shuffle(trials)
        (directory / "plan.json").write_text(json.dumps(trials, indent=2) + "\n")
        results = []
        fields = {}
        with (directory / "trials.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["profile", "variant", "repeat", "bytes",
                "write_fsync_seconds", "warm_read_seconds", "bitwise_equal"])
            writer.writeheader()
            for index, (profile, variant, repeat) in enumerate(trials, 1):
                output = directory / f"{profile}-{variant}-{repeat}.h5"
                started = time.perf_counter()
                call(["h5repack", *VARIANTS[variant], str(originals[profile]), str(output)])
                flush_file(output)
                written = time.perf_counter() - started
                started = time.perf_counter()
                call([str(fixture), "--read", str(output)])
                read = time.perf_counter() - started
                call([str(fixture), "--verify", str(originals[profile]), str(output)])
                call(["h5diff", "-q", str(originals[profile]), str(output)])
                inspection = json.loads(call([str(fixture), "--inspect", str(output)]))
                expected_filters = (2 if variant.startswith("shuffle") else
                                    1 if variant.startswith("gzip") else 0)
                if any(field["filter_count"] != expected_filters for field in inspection.values()):
                    raise RuntimeError("Requested compression filters were not actually installed.")
                fields[f"{profile}/{variant}"] = inspection
                row = dict(profile=profile, variant=variant, repeat=repeat, bytes=output.stat().st_size,
                           write_fsync_seconds=written, warm_read_seconds=read, bitwise_equal=True)
                results.append(row)
                writer.writerow(row)
                stream.flush()
                print(f"{index}/{len(trials)} {profile} {variant}: "
                      f"{output.stat().st_size / 2**20:.2f} MiB, write+fsync {written:.3f}s", flush=True)
        (directory / "field_storage.json").write_text(json.dumps(fields, indent=2) + "\n")
        summary = []
        for profile in PROFILES:
            for variant in VARIANTS:
                rows = [r for r in results if r["profile"] == profile and r["variant"] == variant]
                summary.append(dict(profile=profile, variant=variant,
                    bytes=statistics.median(r["bytes"] for r in rows),
                    write_seconds=statistics.median(r["write_fsync_seconds"] for r in rows),
                    read_seconds=statistics.median(r["warm_read_seconds"] for r in rows),
                    write_min=min(r["write_fsync_seconds"] for r in rows),
                    write_max=max(r["write_fsync_seconds"] for r in rows)))
        for row in summary:
            baseline = next(r for r in summary if r["profile"] == row["profile"] and r["variant"] == "contiguous")
            row["size_ratio"] = baseline["bytes"] / row["bytes"]
            row["write_time_ratio"] = row["write_seconds"] / baseline["write_seconds"]
        with (directory / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader(); writer.writerows(summary)
        lines = ["# Compression measurement", "",
            "Question: how do lossless HDF5 filters change size and cached repack/flush cost?", "",
            f"Configuration: {args.size}×{args.size}, seven float64 fields, {args.repeats} repetitions; seed {SEED}.",
            "Inputs: deterministic gradient, exact Taylor–Green t=0.3, synthetic 5-mode field, and",
            "seeded random velocity/pressure. Material and zero forces remain constant in all profiles.",
            "Multiscale/noise pressures are synthetic, not validated Navier–Stokes states.", "",
            "All trials passed bitwise field readback and h5diff (values, shapes and attributes).", "",
            "| Input | Layout/filter | MiB | Size ratio | Write+fsync ms | vs plain | Warm read ms |",
            "|---|---|---:|---:|---:|---:|---:|"]
        for r in summary:
            lines.append(f"| {r['profile']} | {r['variant']} | {r['bytes']/2**20:.3f} | "
                f"{r['size_ratio']:.2f}× | {1000*r['write_seconds']:.1f} | "
                f"{r['write_time_ratio']:.2f}× | {1000*r['read_seconds']:.1f} |")
        lines += ["", "Times are medians; raw trials/min/max are retained in CSV. Trial order is seeded/shuffled.",
            "Write includes repack input reading, process startup, HDF5 close, file fsync and directory fsync.",
            "Read includes process startup, decompression, materialization and checksum of all fields.",
            "Inputs/outputs are page-cache warm; no cache dropping, GPU transfer or concurrent CFD was measured.",
            "A single snapshot is not a sustained-write benchmark; Btrfs/filesystem caching may affect results.",
            "Read-time checksums prevent unused reads. Per-field stored bytes are in field_storage.json.", "",
            "Four of seven fields are uniform: density, viscosity and two forces. These favor compression.",
            "Use velocity/pressure field sizes as a separate indicator; future turbulent frames may compress differently.",
            "No guarantee of 16× compression or reduction in required generation bandwidth follows from this run.", "",
            "Failures: none in completed trials. If interrupted, running/failed status and partial CSV remain.",
            "Decision: investigate on actual CFD trajectories. Keep production output contiguous/uncompressed",
            "until an end-to-end writer benchmark establishes a throughput benefit for the chosen workload.", "",
            "Reproduce in a NEW folder with:", "```bash",
            f"python3 tools/benchmark_compression.py --fixture {args.fixture} --size {args.size} "
            f"--repeats {args.repeats} --output runs/{date}-compression-baseline-02", "```", "",
            "Reference: https://portal.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html", ""]
        (directory / "README.md").write_text("\n".join(lines))
        status.write_text("complete\ndecision=investigate_on_actual_trajectories\n")
        print(f"Results: {directory}/README.md", flush=True)
    except BaseException as error:
        status.write_text(f"failed\n{type(error).__name__}: {error}\n")
        raise


if __name__ == "__main__":
    main()
