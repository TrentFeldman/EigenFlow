"""Solve and validate the trajectories requested by the data generator.

Each job uses adaptive integration twice, then an independent matrix exponential
at selected times. Accepted arrays go to the generator; failures carry a reason
and evaluation count into its ledger. Closed-form benchmarks reuse this path.
"""

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp

from .analytic import linear_reference
from .config import GeneratorConfig
from .equations import LinearODE


@dataclass(frozen=True)
class SolveJob:
    """Describe one independent, process-picklable trajectory request.

    Args:
        trajectory_id: Unique trajectory label.
        parameter_id: Equation identity excluding initial conditions.
        split: Dataset partition assigned before solving.
        equation: Coefficients, forcing and initial state to integrate.
    """

    trajectory_id: str
    parameter_id: str
    split: str
    equation: LinearODE


@dataclass
class SolveResult:
    """Carry validated arrays or a failure reason back to the coordinator.

    Args:
        job: Original request.
        times: Float64 array (T,) on success, otherwise None.
        states: Float64 array (T, n) on success, otherwise None.
        error: Failure description, or None on success.
        nfev: RHS call count across both solves, including failed calls.
        refinement_ratio: Maximum componentwise L2 error/threshold, or None.
        independent_ratio: Maximum sampled pointwise error/threshold, or None.
        stage_seconds: Worker wall times for completed numerical stages; excludes I/O.
    """

    job: SolveJob
    times: np.ndarray | None = None
    states: np.ndarray | None = None
    error: str | None = None
    nfev: int = 0
    refinement_ratio: float | None = None
    independent_ratio: float | None = None
    # Wall seconds per numerical stage, excluding executor/IPC and file writing.
    stage_seconds: dict = field(default_factory=dict)


class _IntegrationFailure(Exception):
    """Preserve the original exception and calls spent by an unsuccessful solve."""

    def __init__(self, error, nfev):
        super().__init__(str(error))
        self.error = error
        self.nfev = nfev


def _checked_states(solution, equation, config, times):
    """Verify completion, sample grid, state limits and the initial condition."""
    if not solution.success:
        raise RuntimeError(f"Solver failed: {solution.message}")

    states = np.asarray(solution.y).T

    if states.shape != (config.time_points, equation.order):
        raise RuntimeError("Solver returned an incomplete trajectory")

    if not np.array_equal(solution.t, times):
        raise RuntimeError("Solver returned unexpected sample times")

    if not np.isfinite(states).all() or np.max(np.abs(states)) > config.max_abs_state:
        raise RuntimeError("Returned states exceeded finite-value or magnitude limit")

    if not np.allclose(states[0], equation.initial_state, rtol=0, atol=config.atol):
        raise RuntimeError("Returned trajectory does not match its initial state")

    return states


def _integrate(equation, config, times, tolerance_divisor):
    """Return float64 states (T, n) and RHS calls for one bounded integration.

    The float64 times (T,) specify the saved grid. Dividing rtol/atol by 10,
    for example, requests ten times tighter local error control. Failures carry
    their call count so the ledger still accounts for unsuccessful work.
    """
    calls = 0

    def rhs(time, state):
        nonlocal calls
        calls += 1

        if calls > config.max_rhs_calls:
            raise RuntimeError("Right-hand-side call budget exceeded")

        if not np.isfinite(state).all() or np.max(np.abs(state)) > config.max_abs_state:
            raise RuntimeError("State exceeded finite-value or magnitude limit")

        return equation.rhs(time - config.t_start, state)

    try:
        solution = solve_ivp(
            rhs,
            (config.t_start, config.t_stop),
            equation.initial_state,
            method=config.method,
            t_eval=times,
            rtol=config.rtol / tolerance_divisor,
            atol=config.atol / tolerance_divisor,
        )

        return _checked_states(solution, equation, config, times), calls
    except Exception as error:
        raise _IntegrationFailure(error, calls) from error


def _refinement_ratio(coarse, refined, config) -> float:
    """Compare float64 trajectories (T, n) using per-component L2 norms.

    Return max_k ||coarse_k-refined_k|| / (atol*sqrt(T) + rtol*||refined_k||),
    using reference acceptance tolerances. For T=4 and atol=1e-8, the absolute
    contribution is 2e-8. Separate norms keep a large derivative from hiding y.
    """
    difference = np.linalg.norm(coarse - refined, axis=0)
    reference_norm = np.linalg.norm(refined, axis=0)
    component_threshold = (
        config.reference_atol * np.sqrt(config.time_points)
        + config.reference_rtol * reference_norm
    )

    return float(np.max(difference / component_threshold))


def _independent_ratio(refined, equation, times, config) -> float:
    """Check float64 states (T, n) against an independent selected-time solution.

    Return max_{t,k} |refined-reference| / (atol + rtol*|reference|), using the
    reference tolerances and selected entries of float64 times (T,). At a zero
    reference component, an error of 2e-8 with atol=1e-8 gives ratio 2 (failure).
    """
    indices = np.linspace(
        0,
        config.time_points - 1,
        min(config.independent_check_points, config.time_points),
        dtype=int,
    )
    independent = linear_reference(equation, times[indices] - config.t_start)
    pointwise_threshold = config.reference_atol + config.reference_rtol * np.abs(independent)

    return float(np.max(np.abs(refined[indices] - independent) / pointwise_threshold))


def solve_job(job: SolveJob, config: GeneratorConfig) -> SolveResult:
    """Integrate twice and accept only trajectories passing both checks.

    Args:
        job: Equation, initial state and preassigned dataset identity.
        config: Time grid, integration tolerances and acceptance/resource limits.
    Returns:
        SolveResult with float64 times (T,) and states (T, n), or error text.
    Note: KeyboardInterrupt and SystemExit propagate to stop the caller.
    """
    result = SolveResult(job)
    times = np.linspace(config.t_start, config.t_stop, config.time_points)

    try:
        started = perf_counter()
        coarse, calls = _integrate(job.equation, config, times, 1.0)
        result.nfev += calls

        result.stage_seconds["coarse"] = perf_counter() - started
        started = perf_counter()
        refined, calls = _integrate(job.equation, config, times, config.refinement_factor)
        result.nfev += calls

        result.stage_seconds["refined"] = perf_counter() - started
        result.refinement_ratio = _refinement_ratio(coarse, refined, config)

        if not np.isfinite(result.refinement_ratio) or result.refinement_ratio > 1:
            raise RuntimeError("Tolerance-refinement agreement check failed")

        started = perf_counter()
        result.independent_ratio = _independent_ratio(refined, job.equation, times, config)

        result.stage_seconds["matrix_exponential"] = perf_counter() - started
        if not np.isfinite(result.independent_ratio) or result.independent_ratio > 1:
            raise RuntimeError("Independent matrix-exponential agreement check failed")

        result.times, result.states = times, refined

    except Exception as error:
        # Every worker failure becomes a ledger entry so other jobs can finish.
        if isinstance(error, _IntegrationFailure):
            result.nfev += error.nfev
            error = error.error

        result.error = f"{type(error).__name__}: {error}"

    return result
