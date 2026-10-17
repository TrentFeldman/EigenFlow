"""Validate numerical references against explicit closed-form ODE solutions.

Generation runs these benchmarks before writing trajectories. Expected states
come from differentiated exponential-polynomial formulas, which check both
adaptive integration and matrix exponentials. The CLI prints a JSON report.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from math import comb, factorial
from pathlib import Path
from time import perf_counter

import numpy as np

from .analytic import linear_reference
from ..config import CONFIG_DIR, GeneratorConfig, provenance
from .equations import LinearODE
from .reference import SolveJob, solve_job


# Each mode contributes Re(exp(rate*t) * polynomial(t)); powers ascend from t^0.
Mode = tuple[complex, tuple[complex, ...]]


def exponential_state(rate: complex, polynomial: tuple, times, order: int) -> np.ndarray:
    """Differentiate Re(exp(rate*t) * polynomial(t)) by the general Leibniz rule.

    Args:
        rate: Real or complex exponential rate.
        polynomial: Real/complex coefficients in ascending powers of time.
        times: Real numeric array-like of shape (T,).
        order: Number n of derivatives, starting with the undifferentiated value.
    Returns:
        Float64 states of shape (T, n).
    """
    times = np.asarray(times)
    result = np.zeros((len(times), order))

    for derivative in range(order):
        value = np.zeros(len(times), dtype=complex)

        for polynomial_order in range(min(derivative, len(polynomial) - 1) + 1):
            # For p(t)=1+2t+3t^2, polyder(p, 1) gives [2, 6], meaning 2+6t.
            derivative_coefficients = np.polynomial.polynomial.polyder(polynomial, polynomial_order)
            value += (
                comb(derivative, polynomial_order)
                * rate**(derivative - polynomial_order)
                * np.polynomial.polynomial.polyval(times, derivative_coefficients)
            )

        result[:, derivative] = (np.exp(rate * times) * value).real

    return result


@dataclass(frozen=True)
class BenchmarkCase:
    """Pair an equation with independently specified solution modes.

    Args:
        name: Stable report label.
        equation: Coefficients, forcing and initial conditions to check.
        modes: Tuples of exponential rates and ascending polynomial coefficients.
    """

    name: str
    equation: LinearODE
    modes: tuple[Mode, ...]

    def expected(self, elapsed_times):
        """Evaluate every derivative from the explicit solution modes.

        Args:
            elapsed_times: Real numeric array-like of shape (T,).
        Returns:
            Float64 state array of shape (T, n).
        """
        return sum(
            exponential_state(rate, polynomial, elapsed_times, self.equation.order)
            for rate, polynomial in self.modes
        )


def _homogeneous_cases(config):
    """Describe unforced equations, their roots and explicit solution modes."""
    orders = sorted(set(config.orders) | {min(6, config.max_order)})
    cases = []

    for order in orders:
        # (D+1)^n y=0 has root -1 repeated n times and solution
        # y=exp(-t)*sum((0.2*t)^k/k!, k=0..n-1); D means time derivative.
        coefficients = tuple(comb(order, power) for power in range(order))
        polynomial = tuple(0.2**power / factorial(power) for power in range(order))
        repeated_mode = (-1., polynomial)
        cases.append((f"repeated_real_order{order}", coefficients, (repeated_mode,)))

    if config.max_order >= 2:
        # y''+3y'+2y=0: roots -1,-2; y=exp(-t)-0.3*exp(-2t).
        distinct_modes = ((-1., (1.,)), (-2., (-0.3,)))
        cases.append(("distinct_real_order2", (2., 3.), distinct_modes))

        # y''+y'+1.25y=0: roots -0.5 +/- i;
        # y=exp(-0.5t)*(0.7*cos(t)+0.2*sin(t)).
        damped_mode = (complex(-0.5, 1), (0.7 - 0.2j,))
        cases.append(("damped_complex_order2", (1.25, 1.), (damped_mode,)))

    if config.max_order >= 3:
        # y'''+6y''+11y'+6y=0: roots -1,-2,-3;
        # y=exp(-t)+0.2*exp(-2t)-0.1*exp(-3t).
        third_order_modes = ((-1., (1.,)), (-2., (0.2,)), (-3., (-0.1,)))
        cases.append(("distinct_real_order3", (6., 11., 6.), third_order_modes))

    return cases


def _family_case(name, coefficients, homogeneous_modes, family):
    """Add a nonresonant sine particular solution and derive its initial state."""
    order = len(coefficients)
    modes = homogeneous_modes
    amplitude, frequency, phase = 0., 0., 0.

    if family == "linear_sinusoidal":
        amplitude, frequency, phase = 0.4, 1.3, 0.4
        forcing_rate = 1j * frequency
        characteristic = forcing_rate**order + sum(
            coefficient * forcing_rate**power
            for power, coefficient in enumerate(coefficients)
        )

        # For P(D)y=A*sin(omega*t+phase), B=A*exp(i*phase)/P(i*omega)
        # gives y_p=Im(B*exp(i*omega*t)). Since Im(w)=Re(-i*w), store -i*B
        # for exponential_state, which returns real parts of its modes.
        particular = -1j * amplitude * np.exp(1j * phase) / characteristic
        modes = (*modes, (forcing_rate, (particular,)))

    initial = sum(
        exponential_state(rate, polynomial, [0.], order)[0]
        for rate, polynomial in modes
    )
    equation = LinearODE(coefficients, tuple(initial), amplitude, frequency, phase)

    return BenchmarkCase(f"{family}-{name}", equation, modes)


def benchmark_cases(config: GeneratorConfig) -> list[BenchmarkCase]:
    """Build repeated, distinct, oscillatory and resonant closed-form cases.

    Args:
        config: Requested orders, families and maximum supported order.
    Returns:
        Ordered list of BenchmarkCase objects, also covering min(6, M).
    """
    cases = []

    for name, coefficients, homogeneous_modes in _homogeneous_cases(config):
        for family in config.families:
            cases.append(_family_case(name, coefficients, homogeneous_modes, family))

    if config.max_order >= 2 and "linear_sinusoidal" in config.families:
        # y''+y=sin(t) has roots +/- i, matching the forcing frequency.
        # y=-0.5*t*cos(t) solves it: y''+y=sin(t), y(0)=0, y'(0)=-0.5.
        # The real-part representation is Re((-0.5*t)*exp(i*t)).
        resonant_equation = LinearODE((1., 0.), (0., -0.5), 1., 1., 0.)
        resonant_mode = (1j, (0., -0.5))
        cases.append(BenchmarkCase("linear_sinusoidal-resonance", resonant_equation, (resonant_mode,)))

    return cases


def _validate_case(case, config, elapsed):
    """Report pointwise numerical/closed-form agreement for one benchmark."""
    job = SolveJob(case.name, case.name, "validation", case.equation)
    started = perf_counter()
    result = solve_job(job, config)
    seconds = perf_counter() - started
    case_report = {
        "name": case.name,
        "family": case.equation.family,
        "order": case.equation.order,
        "passed": False,
        "seconds": seconds,
        "nfev": result.nfev,
        "error": result.error,
    }

    if result.error is None:
        expected = case.expected(elapsed)
        independent = linear_reference(case.equation, elapsed)
        pointwise_threshold = config.reference_atol + config.reference_rtol * np.abs(expected)
        absolute_error = np.abs(result.states - expected)
        ratio = float(np.max(absolute_error / pointwise_threshold))
        independent_ratio = float(np.max(np.abs(independent - expected) / pointwise_threshold))
        case_report.update(
            max_abs_error=float(absolute_error.max()),
            error_ratio=ratio,
            per_component_max_abs_error=absolute_error.max(axis=0).tolist(),
            matrix_exponential_error_ratio=independent_ratio,
            passed=bool(
                np.isfinite(ratio)
                and np.isfinite(independent_ratio)
                and ratio <= 1
                and independent_ratio <= 1
            ),
        )

    return case_report


def validate_reference(config: GeneratorConfig) -> dict:
    """Compare both numerical methods with closed forms at every stored time.

    Args:
        config: Time grid, supported families, orders and reference tolerances.
    Returns:
        Dict with overall passed, per-case errors, timing and software provenance.
    Note: Each component uses the threshold reference_atol + reference_rtol*abs(exact).
    """
    report = {
        "passed": True,
        "cases": [],
        "method": config.method,
        "reference_rtol": config.reference_rtol,
        "reference_atol": config.reference_atol,
        "families": list(config.families),
        "config": asdict(config),
        **provenance(
            ("numpy", "scipy"),
            ("generation/equations.py", "generation/reference.py", "generation/analytic.py",
             "generation/validate_reference.py"),
            include_machine=True,
        ),
    }
    elapsed = np.linspace(0, config.t_stop - config.t_start, config.time_points)
    started = perf_counter()

    for case in benchmark_cases(config):
        case_report = _validate_case(case, config, elapsed)
        report["cases"].append(case_report)
        report["passed"] = report["passed"] and case_report["passed"]

    report["seconds"] = perf_counter() - started

    return report


def main(argv=None) -> int:
    """Print benchmark results and optionally save a new JSON report.

    Args:
        argv: CLI argument strings; None reads the process arguments.
    Returns:
        Exit code 0 if all cases pass, otherwise 1.
    Raises:
        FileExistsError: The requested report path already exists.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "generator.json",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = validate_reference(GeneratorConfig.load(args.config))
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"

    if args.report:
        with args.report.open("x") as output:
            output.write(serialized)

    print(serialized, end="")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
