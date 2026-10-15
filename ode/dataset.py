"""Read trusted dataset manifests and encode mixed-order trajectory examples.

One training example is one time point, but partitions are assigned to whole
equation groups before generation. Padded derivative slots are marked by a mask
(1 = real value, 0 = padding). Padding must never contribute to loss or scaling.
"""

from bisect import bisect_right
from collections import OrderedDict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .config import SPLITS
from .equations import LinearODE


def encode(equation: LinearODE, elapsed_times, max_order: int):
    """Return (features, state_mask) for a vector of elapsed times.

    Features have shape (T, 3*max_order+5): padded coefficients, padded initial
    state, derivative mask, amplitude/frequency/sin(phase)/cos(phase), and time.
    The mask has shape (T, max_order). Phase is encoded on a circle so phases
    near -pi and pi are represented as neighbors instead of distant numbers.
    Zero-amplitude forcing is canonicalized so irrelevant phase/frequency do
    not create different descriptions of the same homogeneous equation.
    """
    if max_order < equation.order:
        raise ValueError("Equation order exceeds this model's configured max_order")
    times = np.asarray(elapsed_times, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("elapsed_times must be a finite nonnegative vector")
    features = np.zeros((len(times), 3 * max_order + 5), dtype=np.float64)
    mask = np.zeros((len(times), max_order), dtype=np.float64)
    mask[:, :equation.order] = 1
    features[:, :equation.order] = equation.coefficients
    features[:, max_order:max_order + equation.order] = equation.initial_state
    features[:, 2 * max_order:3 * max_order] = mask
    forcing = (equation.amplitude, equation.frequency,
               np.sin(equation.phase), np.cos(equation.phase)) if equation.amplitude else (0, 0, 0, 0)
    features[:, 3 * max_order:3 * max_order + 4] = forcing
    features[:, -1] = times
    return features, mask


class TrajectoryDataset(Dataset):
    """Lazy time-point view over one split, with a small archive cache.

    The ledger hash, counts, IDs and split isolation are checked at startup.
    An archive's checksum and arrays are checked when first loaded. Incomplete
    datasets are rejected. Failed solves require explicit allow_failures=True,
    and their failure rate must still be reported in scientific evaluation.
    """

    def __init__(self, root, split: str, max_order: int, allow_failures: bool = False):
        self.root = Path(root).resolve()
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.split, self.max_order = split, max_order
        self.normalizer = None
        self._cache = OrderedDict()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported dataset schema")
        status = self.manifest.get("status")
        if status not in ("complete", "complete_with_failures"):
            raise ValueError("Dataset is incomplete")
        if status == "complete_with_failures" and not allow_failures:
            raise ValueError("Dataset contains failed solves; inspect them before opting in")
        ledger = (self.root / "records.jsonl").read_bytes()
        if sha256(ledger).hexdigest() != self.manifest.get("records_sha256"):
            raise ValueError("Dataset ledger checksum mismatch")
        records = [json.loads(line) for line in ledger.splitlines()]
        if len(records) != self.manifest["planned"]:
            raise ValueError("Ledger count does not match manifest")
        succeeded = sum(record["status"] == "success" for record in records)
        failed = sum(record["status"] == "failed" for record in records)
        if (succeeded != self.manifest["succeeded"] or failed != self.manifest["failed"]
                or succeeded + failed != len(records)):
            raise ValueError("Success/failure counts do not match manifest")
        ids, parameter_splits, initial_splits = set(), {}, {}
        self.records = []
        for record in records:
            if record["trajectory_id"] in ids or record["split"] not in SPLITS:
                raise ValueError("Duplicate trajectory ID or unknown split")
            ids.add(record["trajectory_id"])
            equation = LinearODE(**record["equation"])
            actual_id = sha256(json.dumps([
                equation.coefficients, equation.amplitude, equation.frequency, equation.phase
            ], sort_keys=True, allow_nan=False).encode()).hexdigest()
            if actual_id != record["parameter_id"] or equation.order != record["order"]:
                raise ValueError("Equation metadata does not match its identity/order")
            if equation.family != record["family"]:
                raise ValueError("Equation family mismatch")
            for seen, key in ((parameter_splits, actual_id),
                              (initial_splits, tuple(equation.initial_state))):
                if key in seen and seen[key] != record["split"]:
                    raise ValueError("Detected cross-split leakage")
                seen[key] = record["split"]
            if record["split"] == split and record["status"] == "success":
                if equation.order > max_order:
                    raise ValueError("Dataset exceeds model max_order")
                self.records.append(record)
        if not self.records:
            raise ValueError(f"No successful trajectories in {split}")
        self.time_points = self.manifest["config"]["time_points"]
        self._ends = np.arange(1, len(self.records) + 1) * self.time_points

    def __len__(self):
        """Number of time-point examples in this split, not trajectories."""
        return len(self.records) * self.time_points

    def raw_trajectory(self, index):
        """Read one trajectory as (features, padded targets, mask), all float64.

        Targets and mask have shape (T, max_order). A cache holds at most eight
        decoded trajectories per dataset worker to keep memory use bounded.
        """
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        record = self.records[index]
        path = (self.root / record["path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Archive path escapes dataset directory")
        if sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Trajectory checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            times, states = archive["times"], archive["states"]
        equation = LinearODE(**record["equation"])
        expected_times = np.linspace(self.manifest["config"]["t_start"],
                                     self.manifest["config"]["t_stop"], self.time_points)
        if not np.array_equal(times, expected_times):
            raise ValueError("Invalid sample times")
        if states.shape != (self.time_points, equation.order) or not np.isfinite(states).all():
            raise ValueError("Invalid trajectory shape or non-finite state")
        if not np.allclose(states[0], equation.initial_state, rtol=0,
                           atol=self.manifest["config"]["atol"]):
            raise ValueError("Initial state mismatch")
        features, mask = encode(equation, times - times[0], self.max_order)
        target = np.zeros_like(mask)
        target[:, :equation.order] = states
        self._cache[index] = (features, target, mask)
        if len(self._cache) > 8:
            self._cache.popitem(last=False)
        return self._cache[index]

    def __getitem__(self, index):
        """Return a float32 dict suitable for PyTorch's default batch collator."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        trajectory_index = bisect_right(self._ends, index)
        time_index = index - trajectory_index * self.time_points
        features, target, mask = self.raw_trajectory(trajectory_index)
        x, y, valid = features[time_index], target[time_index], mask[time_index]
        if self.normalizer is not None:
            x, y = self.normalizer.transform(x, y, valid)
        return {"features": x.astype(np.float32), "target": y.astype(np.float32),
                "mask": valid.astype(np.float32)}


class Normalizer:
    """Training-only per-component scaling, saved alongside model weights.

    Padding is excluded from statistics and reset to zero after scaling.
    Components never seen in training keep neutral scaling; this makes unseen
    orders representable but gives no evidence of prediction accuracy there.
    """

    def __init__(self, input_mean, input_scale, target_mean, target_scale):
        """Load saved scaling vectors, rejecting corrupt dimensions or scales."""
        self.input_mean = np.asarray(input_mean, dtype=np.float64)
        self.input_scale = np.asarray(input_scale, dtype=np.float64)
        self.target_mean = np.asarray(target_mean, dtype=np.float64)
        self.target_scale = np.asarray(target_scale, dtype=np.float64)
        order = self.target_mean.size
        arrays = (self.input_mean, self.input_scale, self.target_mean, self.target_scale)
        if (order < 1 or any(array.ndim != 1 or not np.isfinite(array).all() for array in arrays)
                or self.input_mean.shape != (3 * order + 5,)
                or self.input_scale.shape != self.input_mean.shape
                or self.target_scale.shape != self.target_mean.shape
                or np.any(self.input_scale <= 0) or np.any(self.target_scale <= 0)):
            raise ValueError("Invalid normalization vectors")

    @classmethod
    def fit(cls, dataset: TrajectoryDataset):
        """Fit streaming statistics on the training split; reject other splits."""
        if dataset.split != "train":
            raise ValueError("Normalization may only be fitted on training data")
        order = dataset.max_order
        widths = (3 * order + 5, order)
        counts = [np.zeros(width) for width in widths]
        means = [np.zeros(width) for width in widths]
        m2s = [np.zeros(width) for width in widths]
        for i in range(len(dataset.records)):
            features, target, mask = dataset.raw_trajectory(i)
            feature_valid = np.ones_like(features)
            feature_valid[:, :order] = mask
            feature_valid[:, order:2 * order] = mask
            for j, (values, valid) in enumerate(((features, feature_valid), (target, mask))):
                # Merge per-trajectory variance statistics without keeping all data.
                count = valid.sum(axis=0)
                mean = (values * valid).sum(axis=0) / np.maximum(count, 1)
                m2 = (((values - mean) ** 2) * valid).sum(axis=0)
                total = counts[j] + count
                delta = mean - means[j]
                m2s[j] += m2 + delta**2 * counts[j] * count / np.maximum(total, 1)
                means[j] += delta * count / np.maximum(total, 1)
                counts[j] = total
        scales = [np.sqrt(m2 / np.maximum(count, 1)) for m2, count in zip(m2s, counts)]
        for scale in scales:
            scale[scale < 1e-8] = 1.0
        # The derivative mask is categorical, so preserve exact 0/1 values.
        means[0][2 * order:3 * order] = 0
        scales[0][2 * order:3 * order] = 1
        return cls(means[0], scales[0], means[1], scales[1])

    def transform(self, features, target, mask):
        """Standardize one example or batch and zero its inactive slots."""
        x = (features - self.input_mean) / self.input_scale
        order = len(self.target_mean)
        x[..., :order] *= mask
        x[..., order:2 * order] *= mask
        y = ((target - self.target_mean) / self.target_scale) * mask
        return x, y

    def to_dict(self):
        """Plain lists for JSON notes and safe PyTorch checkpoint loading."""
        return {name: getattr(self, name).tolist() for name in (
            "input_mean", "input_scale", "target_mean", "target_scale")}
