"""Small baseline conditioned on coefficients, initial state, time and order.

It predicts all derivatives below the equation order. A fixed max_order bounds
the tensor sizes; a mask distinguishes missing derivatives from real zeros.
This is a baseline for a specified linear equation family, not a universal ODE
solver. Wider mathematical coverage will need richer equation representations.
"""

import torch
from torch import nn


class OrderConditionedMLP(nn.Module):
    """Feed-forward trajectory predictor with an explicit maximum order.

    Inputs: float tensors (batch, 3*max_order+5), encoded by dataset.encode.
    Output: standardized state (batch, max_order). Inactive outputs must be
    masked in the loss and inference. width/depth control baseline capacity.
    """

    def __init__(self, max_order: int, width: int = 128, depth: int = 3):
        super().__init__()
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (max_order, width, depth)):
            raise ValueError("max_order, width and depth must be positive integers")
        self.max_order = max_order
        self.architecture = {"max_order": max_order, "width": width, "depth": depth}
        layers = []
        input_width = 3 * max_order + 5
        for _ in range(depth):
            layers.extend((nn.Linear(input_width, width), nn.SiLU()))
            input_width = width
        layers.append(nn.Linear(width, max_order))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        """Predict standardized derivatives; training chooses the valid slots."""
        return self.network(features)


def masked_mse(prediction, target, mask):
    """Mean squared error giving equal weight to each time-point example.

    First average over the real derivatives of each example, then across the
    batch. Otherwise a sixth-order example would count three times as much as
    a second-order example solely because it has more output components.
    """
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Prediction, target and mask must have matching shapes")
    if mask.ndim != 2 or torch.any((mask != 0) & (mask != 1)) or torch.any(mask.sum(-1) == 0):
        raise ValueError("Each example needs a binary mask with at least one active derivative")
    squared = (prediction - target).square()
    return (squared.masked_fill(mask == 0, 0).sum(-1) / mask.sum(-1)).mean()
