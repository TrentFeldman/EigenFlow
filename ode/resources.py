"""Load portable execution profiles independently of the scientific recipe.

Profiles tune workers, memory, batch size and devices without changing sampled
trajectories. Explicit CLI options take precedence. 'auto' CPU counts respect
process affinity and SLURM_CPUS_PER_TASK when available; no node size is guessed.
"""

import json
import os
from pathlib import Path


GENERATION_KEYS = {'workers', 'backend'}
TRAINING_KEYS = {'threads', 'loader_workers', 'device', 'preload', 'batch_size', 'preload_limit_gib'}


def available_cpus():
    """Return the positive CPU bound visible to this process/allocation.

    Affinity and SLURM task CPU allocation are upper bounds, not throughput
    recommendations. Without either, use os.cpu_count(); never total cluster cores.
    Malformed scheduler counts raise ValueError instead of silently oversubscribing.
    """
    counts = [os.cpu_count() or 1]
    if hasattr(os, 'sched_getaffinity'):
        counts.append(len(os.sched_getaffinity(0)))
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        count = int(os.environ['SLURM_CPUS_PER_TASK'])
        if count < 1:
            raise ValueError('SLURM_CPUS_PER_TASK must be positive')
        counts.append(count)
    return max(1, min(counts))


def load_resources(path):
    """Return validated execution defaults from a JSON object, or {} for None.

    workers/threads accept positive integers or 'auto'. Loader workers may be
    zero. preload_limit_gib bounds float32 preload array bytes (not peak RSS).
    Unknown settings, invalid counts or unsupported choices raise ValueError.
    """
    if path is None:
        return {}
    values = json.loads(Path(path).read_text())
    if not isinstance(values, dict) or set(values) - GENERATION_KEYS - TRAINING_KEYS:
        raise ValueError('Unknown resource profile fields or non-object profile')
    for name in ('workers', 'threads', 'batch_size', 'loader_workers'):
        if name not in values:
            continue
        if name in ('workers', 'threads') and values[name] == 'auto':
            values[name] = available_cpus()
        value = values[name]
        minimum = 0 if name == 'loader_workers' else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if 'backend' in values and values['backend'] not in ('serial', 'thread', 'process'):
        raise ValueError('Invalid generation backend')
    if 'device' in values and values['device'] not in ('cpu', 'cuda', 'auto'):
        raise ValueError('Invalid training device')
    if 'preload' in values and not isinstance(values['preload'], bool):
        raise ValueError('preload must be boolean')
    if 'preload_limit_gib' in values:
        value = values['preload_limit_gib']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float('inf'):
            raise ValueError('preload_limit_gib must be finite and positive')
    return values
