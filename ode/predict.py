"""Independent CPU inference using saved weights and preprocessing.

No dataset or training script is imported. This interface predicts the scalar
value and all derivatives below the supplied equation's order at elapsed times.
"""

import numpy as np
import torch

from .dataset import Normalizer, encode
from .equations import LinearODE
from .model import OrderConditionedMLP


def predict(checkpoint_path, equation: LinearODE, elapsed_times) -> np.ndarray:
    """Return physical/nondimensional states of shape (T, equation.order).

    The checkpoint must use the current encoding. Order is bounded by its
    max_order. Representation of an input does not imply training coverage;
    callers must check the saved config's orders, parameter bounds and horizon.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1 or checkpoint.get("encoding") != "linear-padded-v1":
        raise ValueError("Unsupported checkpoint/feature encoding")
    model = OrderConditionedMLP(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    normalizer = Normalizer(**checkpoint["normalizer"])
    features, mask = encode(equation, elapsed_times, model.max_order)
    features, _ = normalizer.transform(features, np.zeros_like(mask), mask)
    with torch.inference_mode():
        normalized = model(torch.as_tensor(features, dtype=torch.float32)).numpy()
    states = normalized * normalizer.target_scale + normalizer.target_mean
    if not np.isfinite(states).all():
        raise RuntimeError("Model produced a non-finite prediction")
    return states[:, :equation.order]
