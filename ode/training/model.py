"""Define the neural trajectory baseline and its padding-aware loss.

Training supplies normalized features from dataset.py to this feed-forward
model. Prediction reloads the same architecture and weights, then converts
outputs back to the original state scale with the saved normalizer.
"""

import torch
from torch import nn

from .dataset import feature_width


class OrderConditionedMLP(nn.Module):
    """Predict padded state components from equation and time features.

    Args:
        max_order: Padding width M and output component count.
        width: Neuron count in each hidden layer.
        depth: Number of hidden Linear/SiLU layer pairs.
    Raises:
        ValueError: Any architecture size is not a positive integer.
    """

    def __init__(self, max_order: int, width: int = 128, depth: int = 3):
        super().__init__()

        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (max_order, width, depth)
        ):
            raise ValueError("max_order, width and depth must be positive integers")

        self.max_order = max_order
        self.architecture = {"max_order": max_order, "width": width, "depth": depth}

        layers = []
        input_width = feature_width(max_order)

        for _ in range(depth):
            layers.extend((nn.Linear(input_width, width), nn.SiLU()))
            input_width = width

        layers.append(nn.Linear(width, max_order))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        """Predict standardized derivatives at each input's elapsed time.

        Args:
            features: Float32 tensor (B, feature_width(M)), matching model dtype/device.
        Returns:
            Float32 tensor (B, M); callers mask or remove inactive derivatives.
        """
        return self.network(features)


def masked_mse(prediction, target, mask, weight=None, per_row=False):
    """Average squared error over active derivatives, then over examples.

    Args:
        prediction: Floating tensor (B, M) of standardized predictions.
        target: Floating tensor (B, M), matching prediction dtype/device.
        mask: Binary tensor (B, M) with at least one active slot per example.
        weight: Optional tensor (B, M) multiplying each squared error, e.g.
            train.relative_loss_weights to score error relative to each
            trajectory's size the way evaluation does.
        per_row: Return the (B,) per-example losses instead of their mean.
            Values are not re-checked here: inspecting GPU values forces a
            CPU/GPU sync every batch. dataset.encode builds valid masks and
            train.TensorLoader verifies them once per split.
    Returns:
        Scalar (or (B,) with per_row) tensor with an autograd connection.
    Raises:
        ValueError: Shapes differ.
    """
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have matching shapes")

    if mask.shape != target.shape:
        raise ValueError("Mask and target must have matching shapes")

    if mask.ndim != 2:
        raise ValueError("Mask must have shape (B, M)")

    if weight is not None and weight.shape != target.shape:
        raise ValueError("Weight and target must have matching shapes")

    # Errors [1,1,pad] and [2,2,2] give (1+4)/2=2.5: equal example weights.
    squared = (prediction - target).square()
    if weight is not None:
        squared = squared * weight

    rows = squared.masked_fill(mask == 0, 0).sum(-1) / mask.sum(-1)
    return rows if per_row else rows.mean()
