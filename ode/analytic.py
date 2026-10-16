"""Compute independent linear reference states using a matrix exponential.

Reference solving compares adaptive integration with these states; closed-form
benchmarks validate both methods before generation. Repeated roots and resonant
forcing are handled by evolving an augmented linear system.
"""

import numpy as np
from scipy.linalg import expm

from .equations import LinearODE


def _linear_system(equation):
    """Build the float64 system matrix and initial vector, adding forcing states."""
    order = equation.order
    dimension = order + (2 if equation.amplitude else 0)
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    # The companion matrix for n=3 is [[0, 1, 0], [0, 0, 1], [-a0, -a1, -a2]].
    # For (a0, a1, a2)=(6, 11, 6), its last row is [-6, -11, -6].
    matrix[:order, :order] = np.eye(order, k=1)
    matrix[order - 1, :order] = -np.asarray(equation.coefficients)
    initial = np.zeros(dimension)
    initial[:order] = equation.initial_state

    if equation.amplitude:
        # Augmented (n+2)-square matrix, with e_last=[0, ..., 0, 1]:
        # [ companion   amplitude*e_last    0     ]
        # [     0               0         omega  ]  acts on [state, sin, cos].
        # [     0             -omega        0     ]
        matrix[order - 1, order] = equation.amplitude
        matrix[order, order + 1] = equation.frequency
        matrix[order + 1, order] = -equation.frequency
        initial[order:] = [np.sin(equation.phase), np.cos(equation.phase)]

    return matrix, initial


def linear_reference(equation: LinearODE, elapsed_times) -> np.ndarray:
    """Evolve each requested state directly as exp(matrix * time) @ initial.

    Args:
        equation: Constant-coefficient ODE and initial state.
        elapsed_times: Real array-like of shape (T,), converted to float64.
    Returns:
        Float64 state array of shape (T, n) in derivative order.
    Raises:
        ValueError: Times are negative/non-finite or states become non-finite.
    Note: Matrix exponentials remain subject to floating-point conditioning.
    """
    times = np.asarray(elapsed_times, dtype=np.float64)

    if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("Reference times must be a finite nonnegative vector")

    matrix, initial = _linear_system(equation)
    states = np.empty((len(times), equation.order))

    for time_index, time in enumerate(times):
        states[time_index] = (expm(matrix * time) @ initial)[:equation.order]

    if not np.isfinite(states).all():
        raise ValueError("Matrix-exponential reference produced non-finite states")

    return states
