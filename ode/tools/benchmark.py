"""Measure generation and loaded-checkpoint inference on the current hardware.

Generation includes preflight, process startup, refinement, independent checks,
and archive I/O. Inference excludes load time, warms up first and synchronizes
GPU work. No result in this module predicts an untested cluster's performance.
"""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter

from ..config import CONFIG_DIR




def hardware():
    """Return host/CPU allocation evidence, including any scheduler environment.

    Scheduler variables are reported only if present; logical CPU count is not
    an HPC allocation. Linux affinity can further constrain usable CPUs.
    """
    cpu = platform.processor()
    path = Path('/proc/cpuinfo')
    if path.exists():
        cpu = next((line.split(':', 1)[1].strip() for line in path.read_text().splitlines()
                    if line.startswith('model name')), cpu)
    return {"host": platform.node(), "platform": platform.platform(), "cpu": cpu,
            "logical_cpus": os.cpu_count(),
            "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            "scheduler": {key: os.environ[key] for key in (
                'SLURM_JOB_ID', 'SLURM_CPUS_PER_TASK', 'SLURM_JOB_CPUS_PER_NODE',
                'SLURM_MEM_PER_NODE', 'SLURM_JOB_PARTITION', 'SLURM_JOB_ACCOUNT',
                'SLURM_JOB_GPUS', 'PBS_JOBID', 'PBS_NP') if key in os.environ},
            "native_threads": {key: os.environ.get(key) for key in (
                'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}


def generation(config, worker_counts, repeats):
    """Benchmark full configured generation repeatedly in temporary directories.

    Returns every timing and stage sum; stage sums are worker-seconds, not
    additive wall-time contributions under parallel execution. Failures remain
    in manifests and invalidate a throughput-based dataset-size decision.
    """
    from ..generation.generate_data import generate
    runs = []
    for workers in worker_counts:
        for repeat in range(repeats):
            resolved = replace(config, workers=workers, backend='serial' if workers == 1 else 'process')
            with TemporaryDirectory(prefix='eigenflow-generation-') as directory:
                root = Path(directory) / 'data'
                manifest = generate(resolved, root)
                records = [json.loads(line) for line in (root / 'records.jsonl').read_text().splitlines()]
                totals = {name: sum(r['stage_seconds'].get(name, 0) for r in records)
                          for name in ('coarse', 'refined', 'matrix_exponential')}
                runs.append({"workers": workers, "repeat": repeat, "manifest": manifest,
                             "trajectories_per_second": manifest['succeeded'] / manifest['seconds'],
                             "stage_worker_seconds": totals,
                             "archive_bytes": sum(p.stat().st_size for p in root.rglob('*.npz'))})
                print(f"workers={workers} repeat={repeat}: {manifest['seconds']:.3f}s", flush=True)
    return {"kind": "generation", "hardware": hardware(), "runs": runs}


def inference(checkpoint, device, batch_sizes, repeats, warmup, threads):
    """Measure end-to-end loaded prediction for B time points per call.

    Includes encoding, host/device transfers, forward pass and de-normalization;
    excludes checkpoint loading. Calls use one representative second-order
    equation, so throughput is not a numerical solver speedup claim.
    """
    import numpy as np
    import torch
    from ..config import provenance
    from ..evaluation.predict import Predictor
    from ..generation.equations import LinearODE
    torch.set_num_threads(threads)
    predictor = Predictor(checkpoint, device)
    equation = LinearODE((1.25, 1.), (1., 0.), 0.5, 1., 0.)
    rows = []
    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()
    for size in batch_sizes:
        times = np.linspace(0, 5, size)
        for _ in range(warmup):
            predictor.predict(equation, times, size)
        sync()
        durations = []
        for _ in range(repeats):
            sync()
            started = perf_counter()
            predictor.predict(equation, times, size)
            sync()
            durations.append(perf_counter() - started)
        rows.append({"batch_size_points": size, "repeats": repeats, "warmup": warmup,
                     "median_seconds": float(np.median(durations)),
                     "p95_seconds": float(np.percentile(durations, 95)),
                     "points_per_second": size / float(np.median(durations))})
    return {"kind": "inference", "hardware": hardware(), "device": device,
            "device_name": torch.cuda.get_device_name() if device == 'cuda' else hardware()['cpu'],
            "cuda_build": torch.version.cuda, "hip_build": torch.version.hip,
            "threads": threads, "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
            "scope": "loaded end-to-end prediction, excludes checkpoint load",
            **provenance(('numpy', 'scipy', 'torch')), "results": rows}


def main(argv=None):
    """Write a new benchmark report; arguments control work explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('generation', 'inference'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=CONFIG_DIR / 'generator.json')
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 6, 12])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 101, 4096])
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if min([args.repeats, args.warmup, args.threads, *args.workers, *args.batch_sizes]) < 1:
        parser.error('Counts must be positive')
    if args.kind == 'generation':
        from ..config import GeneratorConfig
        result = generation(GeneratorConfig.load(args.config), args.workers, args.repeats)
    else:
        if args.checkpoint is None:
            parser.error('inference requires --checkpoint')
        result = inference(args.checkpoint, args.device, args.batch_sizes, args.repeats, args.warmup, args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
