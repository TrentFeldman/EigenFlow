"""Small closed-form benchmarks for homogeneous and sine-forced references.

These are deterministic validation cases, not a training dataset. Expected
answers come from derivatives of explicit exponential/polynomial formulas,
not from solve_ivp or the matrix-exponential checker. No trajectory files are
written. The CLI prints a report and can optionally save that JSON report.
"""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
import json
from math import comb, factorial
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from .analytic import linear_reference
from .config import GeneratorConfig
from .equations import LinearODE
from .reference import SolveJob, solve_job


def exponential_state(rate: complex, polynomial: tuple, times, order: int) -> np.ndarray:
    """Differentiate real(exp(rate*t) * polynomial(t)) through order-1.

    polynomial is in ascending powers. The product rule supplies the factors
    comb(k,j), rate^(k-j) and the jth derivative of the polynomial. A linear
    polynomial gives the t*exp(lambda*t) behavior required for a double root.
    Output is a real array of shape (len(times), order).
    """
    times = np.asarray(times)
    result = np.zeros((len(times), order))
    for derivative in range(order):
        value = np.zeros(len(times), dtype=complex)
        for j in range(min(derivative, len(polynomial) - 1) + 1):
            derivative_coefficients = [polynomial[k] * factorial(k) / factorial(k - j)
                                       for k in range(j, len(polynomial))]
            value += (comb(derivative, j) * rate**(derivative - j)
                      * np.polynomial.polynomial.polyval(times, derivative_coefficients))
        result[:, derivative] = (np.exp(rate * times) * value).real
    return result


@dataclass(frozen=True)
class BenchmarkCase:
    """Equation plus explicit solution modes; each mode is (rate, polynomial)."""

    name: str
    equation: LinearODE
    modes: tuple

    def expected(self, elapsed_times):
        """Closed-form states for all real derivatives, shape (T, order)."""
        return sum(exponential_state(rate, polynomial, elapsed_times, self.equation.order)
                   for rate, polynomial in self.modes)


def benchmark_cases(config: GeneratorConfig) -> list[BenchmarkCase]:
    """Build cases for each requested order, plus order 6 when supported.

    Both configured families are checked. Root patterns include repeated roots
    through the full ODE order, distinct real roots, damped complex roots and a
    resonant forced oscillator. These validate behavior the random sampler may
    not currently exercise; they do not expand that sampler's distribution.
    """
    orders = sorted(set(config.orders) | {min(6, config.max_order)})
    definitions = []
    for order in orders:
        # (D+1)^n: exact binomial coefficients and a polynomial of degree n-1.
        coefficients = tuple(comb(order, k) for k in range(order))
        polynomial = tuple(0.2**k / factorial(k) for k in range(order))
        definitions.append((f"repeated_real_order{order}", coefficients, ((-1., polynomial),)))
    if config.max_order >= 2:
        definitions.extend([
            ("distinct_real_order2", (2., 3.), ((-1., (1.,)), (-2., (-0.3,)))),
            ("damped_complex_order2", (1.25, 1.), ((complex(-0.5, 1), (0.7 - 0.2j,)),)),
        ])
    if config.max_order >= 3:
        definitions.append(("distinct_real_order3", (6., 11., 6.),
                            ((-1., (1.,)), (-2., (0.2,)), (-3., (-0.1,)))))
    cases = []
    for name, coefficients, homogeneous_modes in definitions:
        order = len(coefficients)
        for family in config.families:
            modes = homogeneous_modes
            amplitude, frequency, phase = 0., 0., 0.
            if family == "linear_sinusoidal":
                amplitude, frequency, phase = 0.4, 1.3, 0.4
                z = 1j * frequency
                characteristic = z**order + sum(a * z**k for k, a in enumerate(coefficients))
                # Imag(B*exp(i*w*t)) solves the sine-driven equation when
                # B = A*exp(i*phase)/P(i*w). Multiplication by -i takes its real form.
                particular = -1j * amplitude * np.exp(1j * phase) / characteristic
                modes = (*modes, (z, (particular,)))
            initial = sum(exponential_state(rate, poly, [0.], order)[0] for rate, poly in modes)
            equation = LinearODE(coefficients, tuple(initial), amplitude, frequency, phase)
            cases.append(BenchmarkCase(f"{family}-{name}", equation, modes))
    if config.max_order >= 2 and "linear_sinusoidal" in config.families:
        # y = -0.5*t*cos(t) gives y'' + y = sin(t), including exact resonance.
        cases.append(BenchmarkCase("linear_sinusoidal-resonance", LinearODE(
            (1., 0.), (0., -0.5), 1., 1., 0.), ((1j, (0., -0.5)),)))
    return cases


def validate_reference(config: GeneratorConfig) -> dict:
    """Solve small benchmarks and return pass/fail, component errors and timing.

    Allowed error is reference_atol + reference_rtol*abs(exact), pointwise for
    every derivative and saved time. The chosen tolerances belong to reference
    validation, not to the neural model's 5% accuracy target. Timings are local
    benchmark measurements and exclude dataset I/O and worker startup.
    """
    report = {"passed": True, "cases": [], "method": config.method,
              "reference_rtol": config.reference_rtol, "reference_atol": config.reference_atol,
              "families": list(config.families), "config": asdict(config),
              "python": platform.python_version(), "machine": platform.machine(),
              "dependencies": {name: version(name) for name in ("numpy", "scipy")},
              "source_sha256": {name: sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in ("equations.py", "reference.py", "analytic.py", "validate_reference.py")}}
    elapsed = np.linspace(0, config.t_stop - config.t_start, config.time_points)
    started = perf_counter()
    for case in benchmark_cases(config):
        job = SolveJob(case.name, case.name, "validation", case.equation)
        before = perf_counter()
        result = solve_job(job, config)
        seconds = perf_counter() - before
        item = {"name": case.name, "family": case.equation.family, "order": case.equation.order,
                "passed": False, "seconds": seconds, "nfev": result.nfev, "error": result.error}
        if result.error is None:
            expected = case.expected(elapsed)
            independent = linear_reference(case.equation, elapsed)
            allowance = config.reference_atol + config.reference_rtol * np.abs(expected)
            error = np.abs(result.states - expected)
            ratio = float(np.max(error / allowance))
            independent_ratio = float(np.max(np.abs(independent - expected) / allowance))
            item.update(max_abs_error=float(error.max()), error_ratio=ratio,
                        per_component_max_abs_error=error.max(axis=0).tolist(),
                        matrix_exponential_error_ratio=independent_ratio,
                        passed=bool(np.isfinite(ratio) and np.isfinite(independent_ratio)
                                    and ratio <= 1 and independent_ratio <= 1))
        report["cases"].append(item)
        report["passed"] = report["passed"] and item["passed"]
    report["seconds"] = perf_counter() - started
    return report


def main(argv=None) -> int:
    """Run validation only; --report writes a new report, never trajectories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/generator.json")
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
