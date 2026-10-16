"""Plan or generate a reproducible dataset from validated ODE settings.

The pipeline samples equations, asks reference workers to solve them, then
writes accepted arrays and a ledger of every result. Closed-form validation
runs before output creation. The CLI defaults to printing the generation plan.
"""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, replace
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import FAMILIES, SPLITS, GeneratorConfig, provenance, record_split
from .equations import LinearODE, fingerprint, stable_coefficients
from .reference import SolveJob, solve_job
from .validate_reference import validate_reference
from .resources import GENERATION_KEYS, load_resources


PARAMETER_STREAM = 0
INITIAL_STATE_STREAM = 1


def _group_jobs(config, split_index, order, family, group):
    """Sample one parameter group and its independently seeded initial states."""
    split = SPLITS[split_index]
    family_index = FAMILIES.index(family)
    # SeedSequence keys are [seed, split, order, family, group, stream, (IC)].
    # Keeping stream 0 for parameters and 1 for initial states makes each job
    # independent of scheduling; adding an IC cannot change the parameters.
    seed_key = [config.seed, split_index, order, family_index, group]
    rng = np.random.default_rng(np.random.SeedSequence([*seed_key, PARAMETER_STREAM]))
    coefficients = stable_coefficients(order, rng, config.decay_range, config.frequency_range)
    amplitude, frequency, phase = 0.0, 0.0, 0.0

    if family == "linear_sinusoidal":
        amplitude = float(rng.uniform(*config.amplitude_range))
        frequency = float(rng.uniform(*config.frequency_range))
        phase = float(rng.uniform(-np.pi, np.pi))

    for initial_index in range(config.initial_conditions_per_group):
        initial_rng = np.random.default_rng(
            np.random.SeedSequence([*seed_key, INITIAL_STATE_STREAM, initial_index])
        )
        initial = tuple(
            float(value)
            for value in initial_rng.uniform(*config.initial_state_range, size=order)
        )
        trajectory_id = f"{split}-o{order}-{family}-g{group}-i{initial_index}"
        equation = LinearODE(coefficients, initial, amplitude, frequency, phase)
        yield SolveJob(trajectory_id, equation.parameter_id, split, equation)


def iter_jobs(config: GeneratorConfig):
    """Sample reproducible requests in split/order/family/group/initial-state order.

    Args:
        config: Sampling bounds, group counts and root seed.
    Yields:
        SolveJob in a core joint holdout or an explicitly crossed diagnostic.
        Diagnostic parameters/ICs reuse training and validation pools only.
    """
    for split_index, split in enumerate(SPLITS):
        for order in config.orders:
            for family in config.families:
                for group in range(config.groups_per_split[split_index]):
                    yield from _group_jobs(config, split_index, order, family, group)

    # Cross existing train/validation pools without using any test identity.
    # Parameter diagnostic: validation equation + seen training IC.
    # IC diagnostic: seen training equation + validation IC.
    for order in config.orders:
        for family in config.families:
            for group in range(config.diagnostic_groups):
                train = list(_group_jobs(config, 0, order, family, group))
                validation = list(_group_jobs(config, 1, order, family, group))
                for index, (seen, unseen) in enumerate(zip(train, validation)):
                    for split, parameters, initial in (
                        ("validation_parameter", unseen, seen),
                        ("validation_initial", seen, unseen),
                    ):
                        equation = replace(parameters.equation, initial_state=initial.equation.initial_state)
                        identity = f"{split}-o{order}-{family}-g{group}-i{index}"
                        yield SolveJob(identity, equation.parameter_id, split, equation)


def iter_results(jobs, config: GeneratorConfig):
    """Solve jobs with bounded concurrency and yield results in request order.

    Args:
        jobs: Iterable of SolveJob requests.
        config: Solver settings, backend and worker count.
    Yields:
        SolveResult for each requested job, including failed solves.
    """
    if config.backend == "serial":
        for job in jobs:
            yield solve_job(job, config)

        return

    executor_class = ThreadPoolExecutor if config.backend == "thread" else ProcessPoolExecutor
    executor_options = {"max_workers": config.workers}

    if config.backend == "process":
        executor_options["mp_context"] = multiprocessing.get_context("spawn")

    jobs = iter(jobs)
    # Backpressure bounds queued arrays while ordered yielding preserves the ledger.
    max_in_flight = 2 * config.workers

    with executor_class(**executor_options) as executor:
        pending = deque()

        for _ in range(max_in_flight):
            job = next(jobs, None)

            if job is None:
                break

            pending.append(executor.submit(solve_job, job, config))

        while pending:
            yield pending.popleft().result()
            job = next(jobs, None)

            if job is not None:
                pending.append(executor.submit(solve_job, job, config))


def _build_manifest(config, validation):
    """Initialize counts, settings, provenance and reference-validation evidence."""
    return {
        "schema_version": 1,
        "status": "incomplete",
        "config": asdict(config),
        "planned": config.trajectory_count,
        "succeeded": 0,
        "failed": 0,
        **provenance(("numpy", "scipy")),
        "reference_validation": {
            "closed_form_preflight": validation,
            "per_trajectory": "tolerance refinement plus matrix exponential at selected times",
        },
        "family_counts": {
            family: {"succeeded": 0, "failed": 0}
            for family in config.families
        },
        "split_policy": "core joint holdout; diagnostics cross train and validation identities only",
    }


def _record_result(result, output, manifest) -> dict:
    """Save successful arrays, update overall/family counts and return a record."""
    job = result.job
    record = {
        "trajectory_id": job.trajectory_id,
        "parameter_id": job.parameter_id,
        "split": job.split,
        "family": job.equation.family,
        "order": job.equation.order,
        "equation": asdict(job.equation),
        "nfev": result.nfev,
        "stage_seconds": result.stage_seconds,
        "refinement_error_ratio": result.refinement_ratio,
        "independent_error_ratio": result.independent_ratio,
        "status": "failed" if result.error else "success",
        "error": result.error,
    }

    if result.error:
        outcome = "failed"
    else:
        relative_path = Path(job.split) / f"{job.trajectory_id}.npz"
        path = output / relative_path
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, times=result.times, states=result.states)
        record.update(path=relative_path.as_posix(), sha256=sha256(path.read_bytes()).hexdigest())
        outcome = "succeeded"

    manifest[outcome] += 1
    manifest["family_counts"][job.equation.family][outcome] += 1

    return record


def _write_ledger(config, output, manifest):
    """Check split isolation and flush each result so failures remain inspectable."""
    parameter_splits, initial_splits = {}, {}

    with (output / "records.jsonl").open("x") as ledger:
        for result in iter_results(iter_jobs(config), config):
            job = result.job
            collision_message = "Cross-split parameter or initial-state collision"
            if job.split in SPLITS:
                record_split(parameter_splits, job.parameter_id, job.split, collision_message)
                record_split(initial_splits, fingerprint(job.equation.initial_state),
                             job.split, collision_message)
            record = _record_result(result, output, manifest)
            ledger.write(json.dumps(record, allow_nan=False) + "\n")
            ledger.flush()


def generate(config: GeneratorConfig, output: Path) -> dict:
    """Create a new dataset after closed-form preflight succeeds.

    Args:
        config: Validated sampling, integration and worker settings.
        output: New directory for archives, records.jsonl and manifest.json.
    Returns:
        Manifest dict with counts, provenance, checksums and completion status.
    Raises:
        FileExistsError: The output path already exists.
        RuntimeError: Preflight fails or the result count differs from the plan.
        ValueError: A parameter or initial state leaks across splits.
    Note: Interrupted runs retain an incomplete manifest and cannot be loaded.
    """
    output = Path(output)

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")

    started = perf_counter()
    validation = validate_reference(config)

    if not validation["passed"]:
        failed = [case["name"] for case in validation["cases"] if not case["passed"]]
        raise RuntimeError(f"Reference preflight failed; no dataset created: {failed}")

    output.mkdir(parents=True, exist_ok=False)
    manifest = _build_manifest(config, validation)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    try:
        _write_ledger(config, output, manifest)

        if manifest["succeeded"] + manifest["failed"] != manifest["planned"]:
            raise RuntimeError("Result count does not match planned count")

        manifest["status"] = "complete" if not manifest["failed"] else "complete_with_failures"
        manifest["records_sha256"] = sha256((output / "records.jsonl").read_bytes()).hexdigest()
    finally:
        manifest["seconds"] = perf_counter() - started
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    return manifest


def main(argv=None) -> int:
    """Print the generation plan, or create data when --generate is supplied.

    Args:
        argv: CLI argument strings; None reads the process arguments.
    Returns:
        Exit code 0 for a dry run or no failed solves, otherwise 1.
    Raises:
        SystemExit: CLI arguments are invalid or --generate lacks --output.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs/generator.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New dataset directory; required for --generate",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Explicitly enable numerical generation",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Override workers with the CPUs allocated to this job",
    )
    parser.add_argument("--backend", choices=("thread", "process", "serial"))
    parser.add_argument("--resources", type=Path, help="Execution profile; CLI overrides take precedence")
    args = parser.parse_args(argv)
    config = GeneratorConfig.load(args.config)
    profile = load_resources(args.resources)
    config = replace(config, **{key: value for key, value in profile.items() if key in GENERATION_KEYS})
    overrides = {
        name: getattr(args, name)
        for name in ("workers", "backend")
        if getattr(args, name) is not None
    }

    if overrides:
        config = replace(config, **overrides)

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
