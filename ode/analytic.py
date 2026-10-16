"""Independent reference calculations for constant-coefficient linear ODEs.

The production solver takes adaptive time steps. This module instead uses a
matrix exponential, exp(A*t), to propagate a linear state directly. It is an
independent numerical algorithm, not symbolic exact arithmetic. It handles
repeated roots and resonance without dividing by differences between roots.
"""

import numpy as np
from scipy.linalg import expm

from .equations import LinearODE


def linear_reference(equation: LinearODE, elapsed_times) -> np.ndarray:
    """Return reference states of shape (T, order) at nonnegative elapsed times.

    For homogeneous equations, state' = A*state. For sine forcing, append two
    states, sin(omega*t + phase) and cos(omega*t + phase), whose derivatives
    are omega*cos and -omega*sin. The augmented system is then also linear.
    The exponential is computed separately at each requested time, avoiding
    accumulated stepping error. Floating-point conditioning still matters.
    """
    times = np.asarray(elapsed_times, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("Reference times must be a finite nonnegative vector")
    order = equation.order
    dimension = order + (2 if equation.amplitude else 0)
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    matrix[np.arange(order - 1), np.arange(1, order)] = 1
    matrix[order - 1, :order] = -np.asarray(equation.coefficients)
    initial = np.zeros(dimension)
    initial[:order] = equation.initial_state
    if equation.amplitude:
        matrix[order - 1, order] = equation.amplitude
        matrix[order, order + 1] = equation.frequency
        matrix[order + 1, order] = -equation.frequency
        initial[order:] = [np.sin(equation.phase), np.cos(equation.phase)]
    states = np.empty((len(times), order))
    for index, time in enumerate(times):
        states[index] = (expm(matrix * time) @ initial)[:order]
    if not np.isfinite(states).all():
        raise ValueError("Matrix-exponential reference produced non-finite states")
    return states
