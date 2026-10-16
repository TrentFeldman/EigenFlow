"""Numerical reference solving, isolated from sampling and file writing.

A successful solve is checked again at tighter tolerances and against an
independent matrix-exponential calculation at selected times. Closed-form
benchmarks in validate_reference.py additionally check both numerical methods.
Importing this module never integrates an equation.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .analytic import linear_reference
from .config import GeneratorConfig
from .equations import LinearODE


@dataclass(frozen=True)
class SolveJob:
    """One independent trajectory request; immutable and process-picklable.

    parameter_id identifies the equation without its initial conditions.
    split membership is decided before any solving, never from solver success.
    """

    trajectory_id: str
    parameter_id: str
    split: str
    equation: LinearODE


@dataclass
class SolveResult:
    """Worker result: arrays on success, an explicit reason on failure.

    states has shape (time_points, order). Failure arrays are None, so partial
    trajectories cannot accidentally enter training. nfev includes refinement.
    """

    job: SolveJob
    times: np.ndarray | None = None
    states: np.ndarray | None = None
    error: str | None = None
    nfev: int = 0
    refinement_error: float | None = None
    independent_error: float | None = None


def solve_job(job: SolveJob, config: GeneratorConfig) -> SolveResult:
    """Integrate twice and independently check selected reference points.

    max_rhs_calls caps calls per solve (not wall time); max_abs_state rejects
    runaway intermediate states as well as stored points. Solver exceptions
    become failure records. KeyboardInterrupt/SystemExit still propagate.
    """
    result = SolveResult(job)
    times = np.linspace(config.t_start, config.t_stop, config.time_points)

    def integrate(factor):
        calls = 0

        def rhs(time, state):
            nonlocal calls
            calls += 1
            result.nfev += 1
            if calls > config.max_rhs_calls:
                raise RuntimeError("Right-hand-side call budget exceeded")
            if not np.isfinite(state).all() or np.max(np.abs(state)) > config.max_abs_state:
                raise RuntimeError("State exceeded finite-value or magnitude limit")
            return job.equation.rhs(time - config.t_start, state)

        solution = solve_ivp(
            rhs, (config.t_start, config.t_stop), job.equation.initial_state,
            method=config.method, t_eval=times,
            rtol=config.rtol / factor, atol=config.atol / factor,
        )
        if not solution.success:
            raise RuntimeError(f"Solver failed: {solution.message}")
        states = np.asarray(solution.y).T
        if states.shape != (config.time_points, job.equation.order):
            raise RuntimeError("Solver returned an incomplete trajectory")
        if not np.array_equal(solution.t, times):
            raise RuntimeError("Solver returned unexpected sample times")
        if not np.isfinite(states).all() or np.max(np.abs(states)) > config.max_abs_state:
            raise RuntimeError("Returned states exceeded finite-value or magnitude limit")
        if not np.allclose(states[0], job.equation.initial_state, rtol=0, atol=config.atol):
            raise RuntimeError("Returned trajectory does not match its initial state")
        return states

    try:
        coarse = integrate(1.0)
        refined = integrate(config.refinement_factor)
        # Componentwise checks prevent a large derivative hiding an inaccurate y.
        difference = np.linalg.norm(coarse - refined, axis=0)
        reference = np.linalg.norm(refined, axis=0)
        allowed = (config.reference_atol * np.sqrt(config.time_points)
                   + config.reference_rtol * reference)
        result.refinement_error = float(np.max(difference / allowed))
        if not np.isfinite(result.refinement_error) or result.refinement_error > 1:
            raise RuntimeError("Tolerance-refinement agreement check failed")
        indices = np.linspace(0, config.time_points - 1,
                              min(config.independent_check_points, config.time_points), dtype=int)
        independent = linear_reference(job.equation, times[indices] - config.t_start)
        # Pointwise checks catch a bad component even when its trajectory norm
        # is small. This ratio is distinct from the refinement L2 ratio above.
        allowance = config.reference_atol + config.reference_rtol * np.abs(independent)
        result.independent_error = float(np.max(np.abs(refined[indices] - independent) / allowance))
        if not np.isfinite(result.independent_error) or result.independent_error > 1:
            raise RuntimeError("Independent matrix-exponential agreement check failed")
        result.times, result.states = times, refined
    except Exception as error:
        result.error = f"{type(error).__name__}: {error}"
    return result
