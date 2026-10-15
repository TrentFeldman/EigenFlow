"""Plan or explicitly generate a reproducible higher-order ODE dataset.

Default execution is a dry-run: validate settings and print counts, with no
sampling, solving or writes. --generate is required to create data. Workers
solve independent jobs; only the coordinator writes archives and the ledger.
"""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform

import numpy as np

from .config import FAMILIES, SPLITS, GeneratorConfig
from .equations import LinearODE, stable_coefficients
from .reference import SolveJob, solve_job


def fingerprint(values) -> str:
    """Content ID for JSON-compatible parameters or initial conditions."""
    return sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


def iter_jobs(config: GeneratorConfig):
    """Lazily sample requests; this function does not solve trajectories.

    Each job's RNG comes from its logical indices, so worker scheduling never
    changes the sampled data. Parameter groups stay entirely inside one split.
    RNG keys for parameters and initial states occupy separate namespaces.
    """
    for split_index, split in enumerate(SPLITS):
        for order in config.orders:
            for family in config.families:
                family_index = FAMILIES.index(family)
                for group in range(config.groups_per_split[split_index]):
                    key = [config.seed, split_index, order, family_index, group]
                    rng = np.random.default_rng(np.random.SeedSequence([*key, 0]))
                    coefficients = stable_coefficients(
                        order, rng, config.decay_range, config.frequency_range)
                    amplitude, frequency, phase = 0.0, 0.0, 0.0
                    if family == "linear_sinusoidal":
                        amplitude = float(rng.uniform(*config.amplitude_range))
                        frequency = float(rng.uniform(*config.frequency_range))
                        phase = float(rng.uniform(-np.pi, np.pi))
                    parameter_id = fingerprint([coefficients, amplitude, frequency, phase])
                    for initial_index in range(config.initial_conditions_per_group):
                        initial_rng = np.random.default_rng(
                            np.random.SeedSequence([*key, 1, initial_index]))
                        initial = tuple(float(x) for x in initial_rng.uniform(
                            *config.initial_state_range, size=order))
                        trajectory_id = f"{split}-o{order}-{family}-g{group}-i{initial_index}"
                        yield SolveJob(trajectory_id, parameter_id, split, LinearODE(
                            coefficients, initial, amplitude, frequency, phase))


def iter_results(jobs, config: GeneratorConfig):
    """Yield deterministic-order results with at most 2*workers in flight.

    Threads satisfy concurrent solving without sharing mutable solver state.
    Processes are available for CPU-heavy Python callbacks, where the GIL can
    limit thread speed. Throughput must be benchmarked later; no speedup is
    assumed. Spawn keeps process behavior explicit across supported platforms.
    """
    if config.backend == "serial":
        for job in jobs:
            yield solve_job(job, config)
        return
    executor_class = ThreadPoolExecutor if config.backend == "thread" else ProcessPoolExecutor
    kwargs = {"max_workers": config.workers}
    if config.backend == "process":
        kwargs["mp_context"] = multiprocessing.get_context("spawn")
    jobs = iter(jobs)
    with executor_class(**kwargs) as executor:
        pending = deque()
        for _ in range(2 * config.workers):
            job = next(jobs, None)
            if job is None:
                break
            pending.append(executor.submit(solve_job, job, config))
        while pending:
            yield pending.popleft().result()
            job = next(jobs, None)
            if job is not None:
                pending.append(executor.submit(solve_job, job, config))


def generate(config: GeneratorConfig, output: Path) -> dict:
    """Create a new dataset directory; refuse to overwrite existing paths.

    records.jsonl contains every success/failure in request order. manifest.json
    records counts, resolved settings, dependency versions and source hashes.
    An interrupted run stays incomplete and must not be consumed by training.
    This function is intentionally never called during a dry-run.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1, "status": "incomplete", "config": asdict(config),
        "planned": config.trajectory_count, "succeeded": 0, "failed": 0,
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "scipy")},
        "source_sha256": {path.name: sha256(path.read_bytes()).hexdigest()
                          for path in sorted(Path(__file__).parent.glob("*.py"))},
        "reference_validation": "per-trajectory tolerance refinement; analytic benchmark pending",
        "split_policy": "joint holdout: equation parameters and initial states disjoint",
    }
    parameter_splits, initial_splits = {}, {}
    manifest_path = output / "manifest.json"

    def write_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    write_manifest()
    try:
        with (output / "records.jsonl").open("x") as ledger:
            for result in iter_results(iter_jobs(config), config):
                job = result.job
                # Check numerical values, not just labels, for cross-split leakage.
                for seen, key in ((parameter_splits, job.parameter_id),
                                  (initial_splits, fingerprint(job.equation.initial_state))):
                    if key in seen and seen[key] != job.split:
                        raise ValueError("Cross-split parameter or initial-state collision")
                    seen[key] = job.split
                record = {
                    "trajectory_id": job.trajectory_id, "parameter_id": job.parameter_id,
                    "split": job.split, "family": job.equation.family,
                    "order": job.equation.order, "equation": asdict(job.equation),
                    "nfev": result.nfev, "refinement_error_ratio": result.refinement_error,
                    "status": "failed" if result.error else "success", "error": result.error,
                }
                if result.error:
                    manifest["failed"] += 1
                else:
                    relative_path = Path(job.split) / f"{job.trajectory_id}.npz"
                    path = output / relative_path
                    path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(path, times=result.times, states=result.states)
                    record.update(path=relative_path.as_posix(), sha256=sha256(path.read_bytes()).hexdigest())
                    manifest["succeeded"] += 1
                ledger.write(json.dumps(record, allow_nan=False) + "\n")
                ledger.flush()
        if manifest["succeeded"] + manifest["failed"] != manifest["planned"]:
            raise RuntimeError("Result count does not match planned count")
        manifest["status"] = "complete" if not manifest["failed"] else "complete_with_failures"
        manifest["records_sha256"] = sha256((output / "records.jsonl").read_bytes()).hexdigest()
    finally:
        write_manifest()
    return manifest


def main(argv=None) -> int:
    """CLI entry point; no --generate means no sampling, solving or writes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/generator.json")
    parser.add_argument("--output", type=Path, help="New dataset directory; required for --generate")
    parser.add_argument("--generate", action="store_true", help="Explicitly enable numerical generation")
    args = parser.parse_args(argv)
    config = GeneratorConfig.load(args.config)
    if not args.generate:
        print(json.dumps(config.summary(), indent=2))
        return 0
    if args.output is None:
        parser.error("--generate requires --output pointing to a new directory")
    manifest = generate(config, args.output)
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
