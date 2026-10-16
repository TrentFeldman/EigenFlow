"""Describe linear ODEs and sample their coefficients for data generation.

The generator builds these equations before reference solving. The dataset
loader and predictor reuse the same equation identity and derivative ordering.
See MATH_GUIDE.md for the connection between a scalar ODE and its state.
"""

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np


def fingerprint(values) -> str:
    """Compute a stable content ID using the dataset's JSON serialization.

    Args:
        values: JSON-compatible parameters or initial conditions.
    Returns:
        Hexadecimal SHA-256 digest of the serialized values.
    Raises:
        ValueError: Values contain a non-finite number.
        TypeError: Values cannot be serialized as JSON.
    """
    return sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class LinearODE:
    """Describe y^(n) + sum(a[k] * y^(k)) = A * sin(omega*t + phase).

    Args:
        coefficients: n real values in increasing derivative order.
        initial_state: n real initial values; (2, 3) means y=2, y'=3.
        amplitude: Forcing strength A; zero selects the homogeneous family.
        frequency: Nonnegative omega, in radians per elapsed time unit.
        phase: Initial forcing phase in radians.
    Raises:
        ValueError: Values are non-finite, lengths disagree, or frequency < 0.
    """

    coefficients: tuple[float, ...]
    initial_state: tuple[float, ...]
    amplitude: float = 0.0
    frequency: float = 0.0
    phase: float = 0.0

    def __post_init__(self):
        """Canonicalize numeric values before validating the immutable equation."""
        # A frozen dataclass blocks normal assignment, even during canonicalization.
        object.__setattr__(
            self, "coefficients", tuple(float(value) for value in self.coefficients)
        )
        object.__setattr__(
            self, "initial_state", tuple(float(value) for value in self.initial_state)
        )

        for name in ("amplitude", "frequency", "phase"):
            object.__setattr__(self, name, float(getattr(self, name)))

        if not self.coefficients or len(self.coefficients) != len(self.initial_state):
            raise ValueError("Need one initial value per derivative below the ODE order")

        values = (
            *self.coefficients,
            *self.initial_state,
            self.amplitude,
            self.frequency,
            self.phase,
        )

        if not np.isfinite(values).all() or self.frequency < 0:
            raise ValueError("Equation values must be finite; frequency must be nonnegative")

    @property
    def order(self) -> int:
        """Count the derivatives stored in the initial state.

        Returns:
            Positive order n, such as 3 for a third-order equation.
        """
        return len(self.coefficients)

    @property
    def family(self) -> str:
        """Classify the equation by its forcing amplitude.

        Returns:
            'linear_homogeneous' for zero amplitude, else 'linear_sinusoidal'.
        """
        return "linear_homogeneous" if self.amplitude == 0 else "linear_sinusoidal"

    @property
    def parameter_id(self) -> str:
        """Identify coefficients and forcing independently of the initial state.

        Returns:
            SHA-256 string used to keep equation groups in one dataset split.
        """
        return fingerprint([self.coefficients, self.amplitude, self.frequency, self.phase])

    def rhs(self, elapsed_time: float, state: np.ndarray) -> np.ndarray:
        """Convert the scalar ODE into the derivative of its full state.

        Args:
            elapsed_time: Time since the initial condition.
            state: Real array of shape (n,), converted to float64.
        Returns:
            Float64 array of shape (n,), ordered [y', ..., y^(n)].
        Raises:
            ValueError: The state shape differs from (n,).
        """
        state = np.asarray(state, dtype=np.float64)

        if state.shape != (self.order,):
            raise ValueError(f"Expected state shape {(self.order,)}, got {state.shape}")

        derivative = np.empty(self.order, dtype=np.float64)
        # For y'' + 3y' + 2y = 0, state=[1, 4] gives derivative=[4, -14].
        derivative[:-1] = state[1:]
        forcing = self.amplitude * np.sin(self.frequency * elapsed_time + self.phase)
        derivative[-1] = forcing - np.dot(self.coefficients, state)

        return derivative


def stable_coefficients(
    order: int,
    rng: np.random.Generator,
    decay_range: tuple[float, float],
    frequency_range: tuple[float, float],
) -> tuple[float, ...]:
    """Sample decaying real modes and complex-conjugate mode pairs.

    Args:
        order: Positive equation order n.
        rng: NumPy generator supplying the reproducible random stream.
        decay_range: Positive decay bounds, in inverse time units.
        frequency_range: Positive frequency bounds, in radians per time unit.
    Returns:
        Tuple of n floats, ordered from the coefficient of y to y^(n-1).
    Raises:
        ValueError: Order/ranges are invalid or coefficients are unusable.
    Note: Decaying modes can still produce transient growth or stiff equations.
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

    # Roots -1 +/- 2i give (s+1)^2 + 4 = s^2 + 2s + 5. np.poly returns
    # [1, 2, 5]; [:0:-1] produces (5, 2), describing y'' + 2y' + 5y = 0.
    coefficients = np.real_if_close(np.poly(roots)[:0:-1])

    if np.iscomplexobj(coefficients) or not np.isfinite(coefficients).all():
        raise ValueError("Unusable coefficients; reduce order or mode ranges")

    return tuple(float(value) for value in coefficients)
