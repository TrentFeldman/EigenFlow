"""Strict, reviewable settings for future generation; loading never solves ODEs.

JSON is used so settings need no extra configuration library. Unknown keys are
errors: a misspelled tolerance must not silently change a scientific experiment.
"""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import numpy as np


SPLITS = ("train", "validation", "test")
FAMILIES = ("linear_homogeneous", "linear_sinusoidal")


@dataclass(frozen=True)
class GeneratorConfig:
    """Sampling, solver and parallelism contract; ranges are nondimensional.

    groups_per_split is ordered as train, validation, test. Each group shares
    equation parameters across several independently sampled initial states.
    Orders can be extended without editing solver code. max_order is also the
    model's padding limit, not a claim about accuracy at those orders.
    """

    seed: int = 20261015
    orders: tuple[int, ...] = (2, 3)
    max_order: int = 6
    families: tuple[str, ...] = FAMILIES
    groups_per_split: tuple[int, int, int] = (16, 4, 4)
    initial_conditions_per_group: int = 4
    decay_range: tuple[float, float] = (0.1, 1.0)
    frequency_range: tuple[float, float] = (0.5, 2.0)
    initial_state_range: tuple[float, float] = (-1.0, 1.0)
    amplitude_range: tuple[float, float] = (0.1, 1.0)
    t_start: float = 0.0
    t_stop: float = 5.0
    time_points: int = 101
    method: str = "DOP853"
    rtol: float = 1e-9
    atol: float = 1e-11
    refinement_factor: float = 10.0
    reference_rtol: float = 1e-6
    reference_atol: float = 1e-8
    independent_check_points: int = 5
    max_abs_state: float = 1e6
    max_rhs_calls: int = 100000
    backend: str = "thread"
    workers: int = 4

    def __post_init__(self):
        """Reject unsafe or ambiguous experiment settings before any sampling."""
        for name in ("seed", "max_order", "initial_conditions_per_group", "time_points",
                     "max_rhs_calls", "workers", "independent_check_points"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.time_points < 2:
            raise ValueError("time_points must include at least start and end")
        if self.independent_check_points < 2:
            raise ValueError("independent_check_points must include start and end")
        if self.seed >= 2**64:
            raise ValueError("seed must fit in an unsigned 64-bit integer for PyTorch")
        if not self.orders or len(set(self.orders)) != len(self.orders):
            raise ValueError("orders must be a nonempty list without duplicates")
        if any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= self.max_order
               for n in self.orders):
            raise ValueError("Every order must be between 1 and max_order")
        if not self.families or len(set(self.families)) != len(self.families):
            raise ValueError("families must be nonempty and unique")
        if any(family not in FAMILIES for family in self.families):
            raise ValueError(f"Supported families: {FAMILIES}")
        if len(self.groups_per_split) != 3 or any(
                isinstance(n, bool) or not isinstance(n, int) or n < 1
                for n in self.groups_per_split):
            raise ValueError("groups_per_split needs three positive integer counts")
        for name in ("decay_range", "frequency_range", "initial_state_range", "amplitude_range"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError(f"{name} must have two finite increasing endpoints")
        if min(self.decay_range[0], self.frequency_range[0], self.amplitude_range[0]) <= 0:
            raise ValueError("Decay, frequency and forced amplitude ranges must be positive")
        if not np.isfinite([self.t_start, self.t_stop]).all() or self.t_stop <= self.t_start:
            raise ValueError("Need finite t_start < t_stop")
        for name in ("rtol", "atol", "reference_rtol", "reference_atol", "max_abs_state"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.rtol < 1e-12 or not np.isfinite(self.refinement_factor) or self.refinement_factor <= 1:
            raise ValueError("Use rtol >= 1e-12 and refinement_factor > 1")
        if self.rtol / self.refinement_factor < 100 * np.finfo(float).eps:
            raise ValueError("Refined rtol is below the supported floating-point precision")
        if self.method not in ("DOP853", "RK45", "Radau", "BDF"):
            raise ValueError("Use DOP853, RK45, Radau or BDF; LSODA is excluded from threaded use")
        if self.backend not in ("thread", "process", "serial"):
            raise ValueError("backend must be thread, process or serial")

    @classmethod
    def load(cls, path: str | Path) -> "GeneratorConfig":
        """Read a JSON object, rejecting unknown fields and invalid settings."""
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        for name in ("orders", "families", "groups_per_split", "decay_range",
                     "frequency_range", "initial_state_range", "amplitude_range"):
            if name in data:
                data[name] = tuple(data[name])
        return cls(**data)

    @property
    def trajectory_count(self) -> int:
        """Total planned trajectories, not time-point training examples."""
        return (len(self.orders) * len(self.families) * sum(self.groups_per_split)
                * self.initial_conditions_per_group)

    def summary(self) -> dict:
        """Describe work and approximate raw state storage without sampling.

        Storage is a lower bound: archive metadata, solver workspace and future
        compression are excluded. This is suitable for a safe CLI dry-run.
        """
        states_per_group = self.time_points * sum(self.orders)
        return {
            "mode": "dry-run; no equations sampled or trajectories solved",
            "config": asdict(self),
            "planned_trajectories": self.trajectory_count,
            "raw_state_bytes": (states_per_group * len(self.families)
                                * sum(self.groups_per_split)
                                * self.initial_conditions_per_group * 8),
            "reference_solves_per_trajectory": 2,
            "independent_checks_per_trajectory": min(self.independent_check_points, self.time_points),
            "split_policy": "disjoint equation groups and initial states; joint holdout",
        }
