"""Describe an ODE without solving it.

For order n, the initial state contains n values: y, y', ..., y^(n-1).
The implemented family is

    y^(n) + a[0]*y + a[1]*y' + ... + a[n-1]*y^(n-1)
        = amplitude * sin(frequency * elapsed_time + phase).

The coefficient of the highest derivative is normalized to 1 (a 'monic' ODE).
All values are nondimensional for now. A zero forcing amplitude gives a
homogeneous equation. The state conversion works for any positive order; high
orders can still be numerically difficult and require independent validation.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearODE:
    """One constant-coefficient scalar ODE and its complete initial state.

    Coefficients are ordered by increasing derivative, NOT by polynomial
    convention: (2, 3) means y'' + 3*y' + 2*y = forcing.
    State and coefficient lengths must agree. Forcing uses time since t0,
    allowing the same encoded equation to work at different absolute start times.
    """

    coefficients: tuple[float, ...]
    initial_state: tuple[float, ...]
    amplitude: float = 0.0
    frequency: float = 0.0
    phase: float = 0.0

    def __post_init__(self):
        """Canonicalize JSON lists to immutable numeric tuples and validate."""
        object.__setattr__(self, "coefficients", tuple(float(x) for x in self.coefficients))
        object.__setattr__(self, "initial_state", tuple(float(x) for x in self.initial_state))
        if not self.coefficients or len(self.coefficients) != len(self.initial_state):
            raise ValueError("Need one initial value per derivative below the ODE order")
        values = (*self.coefficients, *self.initial_state,
                  self.amplitude, self.frequency, self.phase)
        if not np.isfinite(values).all() or self.frequency < 0:
            raise ValueError("Equation values must be finite; frequency must be nonnegative")

    @property
    def order(self) -> int:
        """Number of initial values needed, e.g. 3 for a third-order ODE."""
        return len(self.coefficients)

    @property
    def family(self) -> str:
        """Human-readable family label, derived from the actual forcing."""
        return "linear_homogeneous" if self.amplitude == 0 else "linear_sinusoidal"

    def rhs(self, elapsed_time: float, state: np.ndarray) -> np.ndarray:
        """Return the time derivative of a state of shape (order,).

        The first slots just shift derivatives: d(y)/dt = y', etc.
        Only the last slot needs the original ODE, rearranged to isolate y^(n).
        This is what lets a first-order numerical solver handle higher orders.
        """
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self.order,):
            raise ValueError(f"Expected state shape {(self.order,)}, got {state.shape}")
        derivative = np.empty(self.order, dtype=np.float64)
        derivative[:-1] = state[1:]
        forcing = self.amplitude * np.sin(self.frequency * elapsed_time + self.phase)
        derivative[-1] = forcing - np.dot(self.coefficients, state)
        return derivative


def stable_coefficients(order: int, rng: np.random.Generator,
                        decay_range: tuple[float, float],
                        frequency_range: tuple[float, float]) -> tuple[float, ...]:
    """Sample coefficients from decaying modes, without solving any ODE.

    A mode is a basic exponential or oscillation composing the solution.
    Negative real parts make unforced modes decay. Complex-conjugate pairs
    give real-valued coefficients and damped oscillations. Odd orders get one
    extra purely decaying mode. Stability does NOT guarantee an easy problem:
    transient growth, stiffness and high-order conditioning still need checks.
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    for bounds in (decay_range, frequency_range):
        if len(bounds) != 2 or not np.isfinite(bounds).all() or not 0 < bounds[0] <= bounds[1]:
            raise ValueError("Mode ranges must be finite, positive and increasing")
    roots = []
    for _ in range(order // 2):
        decay = rng.uniform(*decay_range)
        frequency = rng.uniform(*frequency_range)
        roots.extend([complex(-decay, frequency), complex(-decay, -frequency)])
    if order % 2:
        roots.append(-rng.uniform(*decay_range))
    # np.poly uses descending powers. Our state uses ascending derivatives.
    coefficients = np.real_if_close(np.poly(roots)[:0:-1])
    if np.iscomplexobj(coefficients) or not np.isfinite(coefficients).all():
        raise ValueError("Unusable coefficients; reduce order or mode ranges")
    return tuple(float(value) for value in coefficients)
