"""Predict ODE states on CPU from a saved model and its normalization.

This is the inference end of the config-to-generation-to-training pipeline.
It uses dataset encoding and the model architecture to return every derivative
below the requested equation's order at the supplied elapsed times.
"""

import numpy as np
import torch

from .dataset import Normalizer, encode
from .equations import LinearODE
from .model import OrderConditionedMLP


def predict(checkpoint_path, equation: LinearODE, elapsed_times) -> np.ndarray:
    """Load a checkpoint and predict states in their original nondimensional scale.

    Args:
        checkpoint_path: Saved linear-padded-v1 checkpoint path.
        equation: Equation parameters and initial state, with n <= checkpoint M.
        elapsed_times: Real array-like (T,) of finite nonnegative elapsed times.
    Returns:
        Float64 state array (T, n), ordered [y, y', ..., y^(n-1)].
    Raises:
        ValueError: Encoding, normalization, order or times are invalid.
        RuntimeError: Weights cannot load or predictions are non-finite.
    Note: Check the saved training configuration for supported ranges and horizons.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if checkpoint.get("schema_version") != 1 or checkpoint.get("encoding") != "linear-padded-v1":
        raise ValueError("Unsupported checkpoint/feature encoding")

    model = OrderConditionedMLP(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    normalizer = Normalizer(**checkpoint["normalizer"])
    features, mask = encode(equation, elapsed_times, model.max_order)
    features = normalizer.transform_features(features, mask)

    with torch.inference_mode():
        normalized = model(torch.as_tensor(features, dtype=torch.float32)).numpy()

    states = normalizer.inverse_target(normalized)

    if not np.isfinite(states).all():
        raise RuntimeError("Model produced a non-finite prediction")

    return states[:, :equation.order]
