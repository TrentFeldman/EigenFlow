"""Load generated trajectories and prepare mixed-order model examples.

The loader verifies manifests and archives, then encodes one time point per
example. Training fits normalization here and saves it for prediction. Masks
mark real derivatives so padding contributes neither to statistics nor loss.
"""

from collections import OrderedDict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .config import SPLITS, record_split
from .equations import LinearODE


TRAJECTORY_CACHE_SIZE = 8

# Nearly constant columns use scale 1 rather than amplifying rounding noise.
SCALE_FLOOR = 1e-8


def feature_width(max_order: int) -> int:
    """Count the columns in the linear-padded-v1 encoding.

    Args:
        max_order: Padding width M shared by equations, states and masks.
    Returns:
        Integer width: three M-wide blocks, four forcing values and one time.
    """
    return 3 * max_order + 5


def feature_slices(max_order: int) -> dict[str, slice]:
    """Name the five contiguous blocks in the model input.

    Args:
        max_order: Padding width M.
    Returns:
        Dict of slices named coefficients, initial_state, mask, forcing and time.
    """
    time_column = feature_width(max_order) - 1

    return {
        "coefficients": slice(0, max_order),
        "initial_state": slice(max_order, 2 * max_order),
        "mask": slice(2 * max_order, 3 * max_order),
        "forcing": slice(3 * max_order, time_column),
        "time": slice(time_column, time_column + 1),
    }


def encode(equation: LinearODE, elapsed_times, max_order: int):
    """Encode the equation at each requested elapsed time.

    Args:
        equation: Coefficients, initial state and forcing to represent.
        elapsed_times: Real array-like (T,), converted to float64.
        max_order: Padding width M, at least the equation order n.
    Returns:
        Float64 (features, mask) arrays of shapes (T, feature_width(M)), (T, M).
    Raises:
        ValueError: n exceeds M, or times are non-finite, negative or not a vector.
    Layout for M=6:
        [0:6) a0..a5 | [6:12) y(0)..y^(5)(0) | [12:18) mask |
        18 amplitude | 19 frequency | 20 sin(phase) | 21 cos(phase) | 22 elapsed time
    """
    if max_order < equation.order:
        raise ValueError("Equation order exceeds this model's configured max_order")

    times = np.asarray(elapsed_times, dtype=np.float64)

    if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("elapsed_times must be a finite nonnegative vector")

    layout = feature_slices(max_order)
    features = np.zeros((len(times), feature_width(max_order)), dtype=np.float64)
    mask = np.zeros((len(times), max_order), dtype=np.float64)
    mask[:, :equation.order] = 1

    coefficient_start = layout["coefficients"].start
    initial_start = layout["initial_state"].start
    features[:, coefficient_start:coefficient_start + equation.order] = equation.coefficients
    features[:, initial_start:initial_start + equation.order] = equation.initial_state
    features[:, layout["mask"]] = mask

    # sin/cos put phases -pi and pi at the same point (-0, -1)/(0, -1).
    # Homogeneous equations use zeros because their frequency/phase do not act.
    forcing = (
        (
            equation.amplitude,
            equation.frequency,
            np.sin(equation.phase),
            np.cos(equation.phase),
        )
        if equation.amplitude
        else (0, 0, 0, 0)
    )
    features[:, layout["forcing"]] = forcing
    features[:, layout["time"]] = times[:, None]

    return features, mask


class TrajectoryDataset(Dataset):
    """Provide time-point examples from a verified dataset split.

    Args:
        root: Directory containing manifest.json, records.jsonl and archives.
        split: Train, validation or test partition name.
        max_order: Model padding width M.
        allow_failures: Allow completed datasets containing recorded solve failures.
    Raises:
        ValueError: Metadata, split isolation or model-order checks fail.
        OSError: Required metadata files cannot be read.
    Note: Archive integrity is checked on first access; failed records are excluded.
    """

    def __init__(self, root, split: str, max_order: int, allow_failures: bool = False):
        self.root = Path(root).resolve()

        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")

        self.split, self.max_order = split, max_order
        self.normalizer = None
        self._cache = OrderedDict()
        self._preloaded = None

        self.manifest = self._load_manifest(allow_failures)
        self.records = self._select_records(self._load_ledger())
        self.time_points = self.manifest["config"]["time_points"]

    def _load_manifest(self, allow_failures):
        """Verify the schema, completion status and explicit failure opt-in."""
        manifest = json.loads((self.root / "manifest.json").read_text())

        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported dataset schema")

        status = manifest.get("status")

        if status not in ("complete", "complete_with_failures"):
            raise ValueError("Dataset is incomplete")

        if status == "complete_with_failures" and not allow_failures:
            raise ValueError("Dataset contains failed solves; inspect them before opting in")

        return manifest

    def _load_ledger(self):
        """Verify the ledger checksum, planned count and success/failure totals."""
        ledger = (self.root / "records.jsonl").read_bytes()

        if sha256(ledger).hexdigest() != self.manifest.get("records_sha256"):
            raise ValueError("Dataset ledger checksum mismatch")

        records = [json.loads(line) for line in ledger.splitlines()]

        if len(records) != self.manifest["planned"]:
            raise ValueError("Ledger count does not match manifest")

        succeeded = sum(record["status"] == "success" for record in records)
        failed = sum(record["status"] == "failed" for record in records)

        if (
            succeeded != self.manifest["succeeded"]
            or failed != self.manifest["failed"]
            or succeeded + failed != len(records)
        ):
            raise ValueError("Success/failure counts do not match manifest")

        return records

    def _select_records(self, records):
        """Verify IDs, equation metadata, split isolation, padding capacity and occupancy."""
        trajectory_ids, parameter_splits, initial_splits = set(), {}, {}
        selected = []

        for record in records:
            if record["trajectory_id"] in trajectory_ids or record["split"] not in SPLITS:
                raise ValueError("Duplicate trajectory ID or unknown split")

            trajectory_ids.add(record["trajectory_id"])
            equation = LinearODE(**record["equation"])

            if equation.parameter_id != record["parameter_id"] or equation.order != record["order"]:
                raise ValueError("Equation metadata does not match its identity/order")

            if equation.family != record["family"]:
                raise ValueError("Equation family mismatch")

            record_split(
                parameter_splits,
                equation.parameter_id,
                record["split"],
                "Detected cross-split leakage",
            )
            record_split(
                initial_splits,
                equation.initial_state,
                record["split"],
                "Detected cross-split leakage",
            )

            if record["split"] == self.split and record["status"] == "success":
                if equation.order > self.max_order:
                    raise ValueError("Dataset exceeds model max_order")

                selected.append(record)

        if not selected:
            raise ValueError(f"No successful trajectories in {self.split}")

        return selected

    def __len__(self):
        """Count time-point examples in this split.

        Returns:
            Number of successful trajectories multiplied by T.
        """
        return len(self.records) * self.time_points

    def preload(self):
        """Keep standardized float32 examples in host RAM for shuffled training.

        Raises:
            ValueError: No training-fitted normalizer has been assigned.
        Note: Use zero loader workers to avoid copying these arrays into workers.
        """
        if self.normalizer is None:
            raise ValueError("Assign training-fitted normalization before preloading")

        preloaded = {
            "features": np.empty((len(self), feature_width(self.max_order)), dtype=np.float32),
            "target": np.empty((len(self), self.max_order), dtype=np.float32),
            "mask": np.empty((len(self), self.max_order), dtype=np.float32),
        }

        for trajectory_index in range(len(self.records)):
            features, target, mask = self.raw_trajectory(trajectory_index)
            features, target = self.normalizer.transform(features, target, mask)
            rows = slice(trajectory_index * self.time_points, (trajectory_index + 1) * self.time_points)

            for name, array in (("features", features), ("target", target), ("mask", mask)):
                preloaded[name][rows] = array

        self._preloaded = preloaded
        self._cache.clear()

    def _read_archive(self, record, equation):
        """Verify path containment, checksum, time grid, state shape and initial state."""
        path = (self.root / record["path"]).resolve()

        if not path.is_relative_to(self.root):
            raise ValueError("Archive path escapes dataset directory")

        if sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Trajectory checksum mismatch")

        with np.load(path, allow_pickle=False) as archive:
            times, states = archive["times"], archive["states"]

        expected_times = np.linspace(
            self.manifest["config"]["t_start"],
            self.manifest["config"]["t_stop"],
            self.time_points,
        )

        if not np.array_equal(times, expected_times):
            raise ValueError("Invalid sample times")

        if states.shape != (self.time_points, equation.order) or not np.isfinite(states).all():
            raise ValueError("Invalid trajectory shape or non-finite state")

        if not np.allclose(
            states[0],
            equation.initial_state,
            rtol=0,
            atol=self.manifest["config"]["atol"],
        ):
            raise ValueError("Initial state mismatch")

        return times, states

    def raw_trajectory(self, index):
        """Read and cache one trajectory in the model's padded representation.

        Args:
            index: Trajectory index within this split's successful records.
        Returns:
            Float64 features (T, feature_width(M)), targets (T, M) and mask (T, M).
        Raises:
            ValueError: Archive integrity, time grid or initial-state checks fail.
            IndexError: The trajectory index is out of range.
        """
        if index in self._cache:
            self._cache.move_to_end(index)

            return self._cache[index]

        record = self.records[index]
        equation = LinearODE(**record["equation"])
        times, states = self._read_archive(record, equation)
        features, mask = encode(equation, times - times[0], self.max_order)
        target = np.zeros_like(mask)
        target[:, :equation.order] = states
        self._cache[index] = (features, target, mask)

        if len(self._cache) > TRAJECTORY_CACHE_SIZE:
            self._cache.popitem(last=False)

        return self._cache[index]

    def __getitem__(self, index):
        """Return one float32 example for PyTorch's default batch collator.

        Args:
            index: Time-point index in [0, len(self)).
        Returns:
            Dict of float32 arrays: features (feature_width(M),), target/mask (M,).
        Raises:
            IndexError: The time-point index is outside this split.
            ValueError: Loading the underlying archive fails validation.
        """
        if index < 0 or index >= len(self):
            raise IndexError(index)

        if self._preloaded is not None:
            return {name: array[index] for name, array in self._preloaded.items()}

        # With T=101, example 102 is trajectory 1, time index 1.
        trajectory_index, time_index = divmod(index, self.time_points)
        features, target, mask = self.raw_trajectory(trajectory_index)
        features, target, mask = features[time_index], target[time_index], mask[time_index]

        if self.normalizer is not None:
            features, target = self.normalizer.transform(features, target, mask)

        return {
            "features": features.astype(np.float32),
            "target": target.astype(np.float32),
            "mask": mask.astype(np.float32),
        }


def _merge_moments(values, active_mask, counts, means, squared_deviations):
    """Merge masked float64 (T, width) values using Chan's parallel variance rule.

    Return new counts, means and sums of squared deviations (width,), leaving
    every input array unchanged.
    Streaming per-trajectory moments bounds memory and avoids subtracting large
    near-equal raw squares for constant columns. Combining [1, 1] and [3, 3],
    for example, gives mean 2 and squared-deviation sum 4 (variance 1).
    """
    count = active_mask.sum(axis=0)
    mean = (values * active_mask).sum(axis=0) / np.maximum(count, 1)
    squared_deviation = (((values - mean) ** 2) * active_mask).sum(axis=0)

    total = counts + count
    delta = mean - means
    merged_deviations = squared_deviations + (
        squared_deviation + delta**2 * counts * count / np.maximum(total, 1)
    )
    merged_means = means + delta * count / np.maximum(total, 1)

    return total, merged_means, merged_deviations


class Normalizer:
    """Store per-component training statistics for training and inference.

    Args:
        input_mean: Float64-compatible vector (feature_width(M),).
        input_scale: Positive float64-compatible vector (feature_width(M),).
        target_mean: Float64-compatible vector (M,).
        target_scale: Positive float64-compatible vector (M,).
    Raises:
        ValueError: Vectors have invalid shapes, non-finite values or nonpositive scales.
    Note: Unobserved derivatives keep mean zero and scale one.
    """

    def __init__(self, input_mean, input_scale, target_mean, target_scale):
        self.input_mean = np.asarray(input_mean, dtype=np.float64)
        self.input_scale = np.asarray(input_scale, dtype=np.float64)
        self.target_mean = np.asarray(target_mean, dtype=np.float64)
        self.target_scale = np.asarray(target_scale, dtype=np.float64)
        max_order = self.target_mean.size

        if max_order < 1:
            raise ValueError("target_mean must contain at least one component")

        for name in ("input_mean", "input_scale", "target_mean", "target_scale"):
            array = getattr(self, name)

            if array.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional vector")

            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")

        if self.input_mean.shape != (feature_width(max_order),):
            raise ValueError(f"input_mean must have shape ({feature_width(max_order)},)")

        if self.input_scale.shape != self.input_mean.shape:
            raise ValueError("input_scale must have the same shape as input_mean")

        if self.target_scale.shape != self.target_mean.shape:
            raise ValueError("target_scale must have the same shape as target_mean")

        if np.any(self.input_scale <= 0):
            raise ValueError("input_scale must contain only positive values")

        if np.any(self.target_scale <= 0):
            raise ValueError("target_scale must contain only positive values")

    @classmethod
    def fit(cls, dataset: TrajectoryDataset):
        """Fit masked moments one trajectory at a time to bound memory use.

        Args:
            dataset: Training split with float64 raw_trajectory arrays.
        Returns:
            Normalizer whose vector shapes follow the dataset's padding width M.
        Raises:
            ValueError: The dataset is not the training split.
        """
        if dataset.split != "train":
            raise ValueError("Normalization may only be fitted on training data")

        layout = feature_slices(dataset.max_order)
        # Each tuple holds counts, means and sums of squared deviations.
        feature_stats = tuple(np.zeros(feature_width(dataset.max_order)) for _ in range(3))
        target_stats = tuple(np.zeros(dataset.max_order) for _ in range(3))

        for trajectory_index in range(len(dataset.records)):
            features, target, mask = dataset.raw_trajectory(trajectory_index)
            feature_mask = np.ones_like(features)
            feature_mask[:, layout["coefficients"]] = mask
            feature_mask[:, layout["initial_state"]] = mask

            feature_stats = _merge_moments(features, feature_mask, *feature_stats)
            target_stats = _merge_moments(target, mask, *target_stats)

        feature_count, feature_mean, feature_deviations = feature_stats
        target_count, target_mean, target_deviations = target_stats

        feature_scale = np.sqrt(feature_deviations / np.maximum(feature_count, 1))
        target_scale = np.sqrt(target_deviations / np.maximum(target_count, 1))

        for scale in (feature_scale, target_scale):
            scale[scale < SCALE_FLOOR] = 1.0

        # Mask columns are categorical; e.g. [1, 1, 0, 0] must stay binary.
        feature_mean[layout["mask"]] = 0
        feature_scale[layout["mask"]] = 1

        return cls(feature_mean, feature_scale, target_mean, target_scale)

    def transform_features(self, features, mask):
        """Standardize inputs while restoring zero padding in derivative blocks.

        Args:
            features: Float array (..., feature_width(M)); one example or a batch.
            mask: Binary numeric array (..., M) marking real derivatives.
        Returns:
            Float64 standardized features of shape (..., feature_width(M)).
        """
        normalized = (features - self.input_mean) / self.input_scale
        layout = feature_slices(len(self.target_mean))
        normalized[..., layout["coefficients"]] *= mask
        normalized[..., layout["initial_state"]] *= mask

        return normalized

    def transform(self, features, target, mask):
        """Standardize inputs and targets and zero their inactive derivatives.

        Args:
            features: Float array (..., feature_width(M)).
            target: Float array (..., M) in original state units.
            mask: Binary numeric array (..., M) marking real derivatives.
        Returns:
            Float64 features (..., feature_width(M)) and targets (..., M).
        """
        features = self.transform_features(features, mask)
        target = ((target - self.target_mean) / self.target_scale) * mask

        return features, target

    def inverse_target(self, values):
        """Restore model outputs to the original nondimensional state scale.

        Args:
            values: Standardized float array (..., M).
        Returns:
            Float64 states (..., M), including any padding slots.
        """
        return values * self.target_scale + self.target_mean

    def to_dict(self):
        """Serialize normalization vectors for checkpoint or JSON storage.

        Returns:
            Dict of float lists: input vectors (feature_width(M),), targets (M,).
        """
        return {
            name: getattr(self, name).tolist()
            for name in ("input_mean", "input_scale", "target_mean", "target_scale")
        }
