"""Validate the settings shared by ODE generation and training.

JSON configuration controls sampling, numerical checks and worker execution.
Shared helpers record software provenance and enforce split isolation when
writing or reading a dataset.
"""

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np

from .resources import available_cpus


SPLITS = ("train", "validation", "test")
DIAGNOSTIC_SPLITS = ("validation_parameter", "validation_initial")
ALL_SPLITS = SPLITS + DIAGNOSTIC_SPLITS
FAMILIES = ("linear_homogeneous", "linear_sinusoidal")
# Leave enough float64 precision for a second, tighter solve.
MIN_SOLVER_RTOL = 1e-12
# SciPy clips extremely small tolerances; stay above that rounding region.
MIN_REFINED_RTOL = 100 * np.finfo(float).eps
FLOAT64_BYTES = 8
# Presets and resource profiles; CLI defaults resolve here from any directory.
CONFIG_DIR = Path(__file__).parent / "configs"


def record_split(assignments, identity, split, error_message):
    """Register an identity in one split and reject cross-split reuse.

    Args:
        assignments: Mutable mapping of hashable identities to split names.
        identity: Parameter digest or initial-state identity to register.
        split: Requested train, validation or test split.
        error_message: Caller-specific explanation if a collision occurs.
    Raises:
        ValueError: The identity was already registered in another split.
    """
    if identity in assignments and assignments[identity] != split:
        raise ValueError(error_message)

    assignments[identity] = split


def provenance(dependencies, source_names=None, include_machine=False) -> dict:
    """Collect the software versions and source hashes for an artifact.

    Args:
        dependencies: Distribution names whose installed versions to record.
        source_names: Paths inside ode/; None selects every non-test module.
        include_machine: Include the machine architecture in the result.
    Returns:
        Dict with python, dependencies, source_sha256 and optional machine keys.
        Source keys are paths inside ode/, e.g. "training/train.py"; tests excluded.
    Raises:
        OSError: A source file cannot be read.
    """
    source_root = Path(__file__).parent
    paths = (
        sorted(path for path in source_root.rglob("*.py")
               if path.relative_to(source_root).parts[0] not in ("tests", "runs", "data"))
        if source_names is None
        else [source_root / name for name in source_names]
    )
    report = {"python": platform.python_version()}

    if include_machine:
        report["machine"] = platform.machine()

    report["dependencies"] = {name: version(name) for name in dependencies}
    report["source_sha256"] = {path.relative_to(source_root).as_posix(): sha256(path.read_bytes()).hexdigest() for path in paths}

    return report


@dataclass(frozen=True)
class GeneratorConfig:
    """Hold the sampling, solver and execution recipe used throughout the pipeline.

    Args:
        seed, orders, max_order, families: Reproducibility and equation scope.
        groups_per_split, initial_conditions_per_group: Core dataset population.
        train_order_weights: Optional per-order multipliers for the train split only.
        diagnostic_groups: Extra crossed train/validation groups per order/family.
        decay_range, frequency_range, initial_state_range, amplitude_range:
            Sampling bounds in the nondimensional units documented below.
        t_start, t_stop, time_points: Stored time grid.
        method, rtol, atol, refinement_factor: Adaptive integration controls.
        reference_rtol, reference_atol, independent_check_points: Check criteria.
        max_abs_state, max_rhs_calls: Per-solve resource limits.
        backend, workers: Worker execution controls.
    Raises:
        ValueError: Any setting violates the constraints documented below.
    """

    # Unsigned 64-bit seed; fixes parameter and initial-state draws.
    seed: int = 20261015

    # Positive derivative orders n (counts); each gets both configured families.
    orders: tuple[int, ...] = (2, 3)

    # Maximum equation order (count), also the model's padding width M.
    max_order: int = 6

    # Equation-family names; choose absent or sinusoidal forcing.
    families: tuple[str, ...] = FAMILIES

    # Train/validation/test group counts PER ORDER AND FAMILY, not total counts.
    groups_per_split: tuple[int, int, int] = (16, 4, 4)

    # Optional multiplier on the TRAIN group count for each entry of orders, e.g.
    # (0.5, 1, 1, 1.5) to spend more data on harder orders. Validation and test
    # stay balanced so every order is scored on equal footing.
    train_order_weights: tuple[float, ...] | None = None

    # Initial-state count for each shared parameter group; multiplies dataset size.
    initial_conditions_per_group: int = 4

    # Additional groups per order/family for EACH diagnostic, reusing train/validation pools.
    diagnostic_groups: int = 0

    # Positive decay bounds (inverse time); larger values give faster decay.
    decay_range: tuple[float, float] = (0.1, 1.0)

    # Angular frequency bounds (radians/time) for modes and sine forcing.
    frequency_range: tuple[float, float] = (0.5, 2.0)

    # Nondimensional bounds used independently for every initial derivative.
    initial_state_range: tuple[float, float] = (-1.0, 1.0)

    # Positive nondimensional forcing bounds; homogeneous amplitude stays zero.
    amplitude_range: tuple[float, float] = (0.1, 1.0)

    # Absolute nondimensional start time; forcing uses elapsed time from here.
    t_start: float = 0.0

    # Absolute nondimensional stop time; fixes the prediction horizon.
    t_stop: float = 5.0

    # Uniform stored time count T, including both endpoints.
    time_points: int = 101

    # SciPy solver name; controls the adaptive integration algorithm.
    method: str = "DOP853"

    # Relative local solver tolerance (unitless); controls adaptive step accuracy.
    rtol: float = 1e-9

    # Absolute local solver tolerance (state units); controls accuracy near zero.
    atol: float = 1e-11

    # Divisor > 1 (unitless) applied to rtol/atol for the second, tighter solve.
    refinement_factor: float = 10.0

    # Relative acceptance threshold (unitless) for checks, not solver step sizing.
    reference_rtol: float = 1e-6

    # Absolute acceptance threshold (state units) for checks near zero.
    reference_atol: float = 1e-8

    # Number of matrix-exponential check times, capped at T; includes endpoints.
    independent_check_points: int = 5

    # Largest permitted absolute component (state units), including solver stages.
    max_abs_state: float = 1e6

    # RHS evaluation budget (count) for EACH solve; also counts rejected steps.
    max_rhs_calls: int = 100000

    # Worker mode; serial, threads or spawned processes.
    backend: str = "thread"

    # Concurrent worker count; bounds the number of active trajectory solves.
    workers: int = 4

    def __post_init__(self):
        """Validate settings in groups before constructing a usable recipe."""
        # Integer counts define array sizes, populations and reproducible seeds.
        self._validate_counts()

        # Sampling ranges bound the nondimensional equation and initial state.
        self._validate_ranges()

        # Stored times must cover a finite, forward interval.
        if not np.isfinite([self.t_start, self.t_stop]).all():
            raise ValueError("t_start and t_stop must be finite")

        if self.t_stop <= self.t_start:
            raise ValueError("t_stop must be greater than t_start")

        # Solver tolerances control steps; reference thresholds accept solutions.
        self._validate_tolerances()

        # Choices restrict families and supported execution algorithms.
        self._validate_choices()

    def _validate_counts(self):
        """Check integer bounds and distinct orders before allocating arrays."""
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be an integer >= 0")

        for name in (
            "max_order",
            "initial_conditions_per_group",
            "time_points",
            "max_rhs_calls",
            "workers",
            "independent_check_points",
        ):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")

        if self.time_points < 2:
            raise ValueError("time_points must include at least start and end")

        if self.independent_check_points < 2:
            raise ValueError("independent_check_points must include start and end")

        if self.seed >= 2**64:
            raise ValueError("seed must fit in an unsigned 64-bit integer for PyTorch")

        if not self.orders:
            raise ValueError("orders must be nonempty")

        if len(set(self.orders)) != len(self.orders):
            raise ValueError("orders must not contain duplicates")

        if any(
            isinstance(order, bool)
            or not isinstance(order, int)
            or not 1 <= order <= self.max_order
            for order in self.orders
        ):
            raise ValueError("Every order must be between 1 and max_order")

        if (isinstance(self.diagnostic_groups, bool) or not isinstance(self.diagnostic_groups, int)
                or self.diagnostic_groups < 0):
            raise ValueError("diagnostic_groups must be a nonnegative integer")

        if len(self.groups_per_split) != len(SPLITS):
            raise ValueError("groups_per_split needs three counts")

        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in self.groups_per_split
        ):
            raise ValueError("groups_per_split counts must be positive integers")

        if self.diagnostic_groups > min(self.groups_per_split[:2]):
            raise ValueError("diagnostic_groups cannot exceed train or validation group count")

        if self.train_order_weights is not None:
            weights = self.train_order_weights
            if len(weights) != len(self.orders) or not all(
                    isinstance(w, (int, float)) and not isinstance(w, bool) and np.isfinite(w) and w > 0
                    for w in weights):
                raise ValueError("train_order_weights needs one positive number per order")
            if min(self.groups(0, order) for order in self.orders) < max(1, self.diagnostic_groups):
                raise ValueError("train_order_weights leaves an order with too few train groups")

    def groups(self, split_index: int, order: int) -> int:
        """Return groups per family for one split and order (train scaled by train_order_weights)."""
        count = self.groups_per_split[split_index]
        if split_index == 0 and self.train_order_weights is not None:
            count = round(count * self.train_order_weights[self.orders.index(order)])
        return count

    def _validate_ranges(self):
        """Check finite increasing endpoints and positive decay/forcing bounds."""
        for name in ("decay_range", "frequency_range", "initial_state_range", "amplitude_range"):
            bounds = getattr(self, name)

            if len(bounds) != 2:
                raise ValueError(f"{name} must have two endpoints")

            if not np.isfinite(bounds).all():
                raise ValueError(f"{name} endpoints must be finite")

            if bounds[0] >= bounds[1]:
                raise ValueError(f"{name} endpoints must be increasing")

        for name in ("decay_range", "frequency_range", "amplitude_range"):
            if getattr(self, name)[0] <= 0:
                raise ValueError(f"{name} must be positive")

    def _validate_tolerances(self):
        """Keep solver precision and acceptance thresholds finite and usable."""
        for name in ("rtol", "atol", "reference_rtol", "reference_atol", "max_abs_state"):
            value = getattr(self, name)

            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")

            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.rtol < MIN_SOLVER_RTOL:
            raise ValueError("Use rtol >= 1e-12")

        if not np.isfinite(self.refinement_factor):
            raise ValueError("refinement_factor must be finite")

        if self.refinement_factor <= 1:
            raise ValueError("refinement_factor must be > 1")

        if self.rtol / self.refinement_factor < MIN_REFINED_RTOL:
            raise ValueError("Refined rtol is below the supported floating-point precision")

    def _validate_choices(self):
        """Restrict families, solver methods and worker backends to supported names."""
        if not self.families:
            raise ValueError("families must be nonempty")

        if len(set(self.families)) != len(self.families):
            raise ValueError("families must be unique")

        if any(family not in FAMILIES for family in self.families):
            raise ValueError(f"Supported families: {FAMILIES}")

        if self.method not in ("DOP853", "RK45", "Radau", "BDF"):
            raise ValueError("Use DOP853, RK45, Radau or BDF; LSODA is excluded from threaded use")

        if self.backend not in ("thread", "process", "serial"):
            raise ValueError("backend must be thread, process or serial")

    @classmethod
    def from_dict(cls, data) -> "GeneratorConfig":
        """Construct validated settings from a decoded JSON object.

        Args:
            data: Mapping of setting names to scalar values or JSON lists.
        Returns:
            GeneratorConfig with list-valued settings converted to tuples.
        Raises:
            ValueError: The object, field names or setting values are invalid.
        """
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object")

        unknown = set(data) - {field.name for field in fields(cls)}

        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")

        settings = data.copy()

        # "auto" means every CPU this process/job may use (see resources.py).
        if settings.get("workers") == "auto":
            settings["workers"] = available_cpus()

        for name in (
            "orders",
            "families",
            "groups_per_split",
            "train_order_weights",
            "decay_range",
            "frequency_range",
            "initial_state_range",
            "amplitude_range",
        ):
            if settings.get(name) is not None:
                settings[name] = tuple(settings[name])

        return cls(**settings)

    @classmethod
    def load(cls, path: str | Path) -> "GeneratorConfig":
        """Read a JSON configuration file.

        Args:
            path: Path to the configuration object.
        Returns:
            Validated GeneratorConfig.
        Raises:
            OSError: The file cannot be read.
            ValueError: JSON, field names or setting values are invalid.
        """
        return cls.from_dict(json.loads(Path(path).read_text()))

    @property
    def trajectory_count(self) -> int:
        """Count planned trajectories across all orders, families and splits.

        Returns:
            Integer count, including every initial condition in each group.
        """
        return (
            len(self.families)
            * sum(sum(self.groups(split, order) for split in range(len(SPLITS))) + 2 * self.diagnostic_groups
                  for order in self.orders)
            * self.initial_conditions_per_group
        )

    def summary(self) -> dict:
        """Describe the planned work and uncompressed state-array size.

        Returns:
            JSON-compatible dict of settings, counts and raw_state_bytes.
        Note: Storage excludes archive metadata, solver workspace and compression.
        """
        # Two orders (2, 3) with 101 times store 505 floats per family/group/IC.
        total_stored_floats = (
            self.time_points
            * len(self.families)
            * sum(order * (sum(self.groups(split, order) for split in range(len(SPLITS))) + 2 * self.diagnostic_groups)
                  for order in self.orders)
            * self.initial_conditions_per_group
        )

        return {
            "mode": "dry-run; no equations sampled or trajectories solved",
            "config": asdict(self),
            "planned_trajectories": self.trajectory_count,
            "raw_state_bytes": total_stored_floats * FLOAT64_BYTES,
            "reference_solves_per_trajectory": 2,
            "independent_checks_per_trajectory": min(self.independent_check_points, self.time_points),
            "split_policy": "joint core holdout; optional crossed training/validation diagnostics",
        }
