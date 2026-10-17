"""Check derivative conversion and solver safeguards with fabricated replies.

Fabricated solver replies exercise failure handling. Analytic integration
benchmarks are tested separately in test_validation.py using real integration.
"""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ode.config import GeneratorConfig
from ode.generation.equations import LinearODE, stable_coefficients
from ode.generation.reference import SolveJob, solve_job


class ReferenceContractTests(unittest.TestCase):
    """Check derivative ordering, numerical safeguards and failed-solve evidence."""

    def test_higher_order_state_conversion(self):
        """Check the shift-and-combine RHS rule for several equation orders."""
        for order in (1, 2, 3, 6, 10):
            with self.subTest(order=order):
                coefficients = tuple(float(value) for value in range(1, order + 1))
                state = np.arange(order, dtype=float) + 1
                equation = LinearODE(coefficients, tuple(state))
                derivative = equation.rhs(0, state)
                np.testing.assert_array_equal(derivative[:-1], state[1:])
                self.assertEqual(derivative[-1], -np.dot(coefficients, state))

    def test_forcing_and_validation(self):
        """Verify sine forcing, canonical numeric identity and equation validation."""
        equation = LinearODE((2, 3), (1, 0), amplitude=4, frequency=1, phase=np.pi / 2)
        np.testing.assert_allclose(equation.rhs(0, np.array([1, 0])), [0, 2])
        integer_forcing = LinearODE((2, 3), (1, 0), 1, 2, 3)
        float_forcing = LinearODE((2., 3.), (1., 0.), 1., 2., 3.)
        self.assertEqual(integer_forcing.parameter_id, float_forcing.parameter_id)
        self.assertEqual(integer_forcing.parameter_id,
                         LinearODE((2, 3), (-1, 4), 1, 2, 3).parameter_id)

        for name in ("amplitude", "frequency", "phase"):
            self.assertIsInstance(getattr(integer_forcing, name), float)

        for coefficients, initial in (((), ()), ((1, 2), (0,)), ((float("nan"),), (0,))):
            with self.assertRaises(ValueError):
                LinearODE(coefficients, initial)

    def test_stable_modes_are_reproducible(self):
        """Check repeatable sampled coefficients with roots in the decaying half-plane."""
        for order in (2, 3, 6):
            first = stable_coefficients(order, np.random.default_rng(7), (0.2, 1), (0.5, 2))
            second = stable_coefficients(order, np.random.default_rng(7), (0.2, 1), (0.5, 2))
            self.assertEqual(first, second)
            roots = np.roots([1, *reversed(first)])
            self.assertTrue(np.all(roots.real < 0))

    def test_solver_refinement_contract_and_failure(self):
        """Check tighter solver tolerance, incomplete arrays and exception reporting."""
        config = replace(GeneratorConfig(), time_points=3)
        job = SolveJob("fixture", "parameters", "train", LinearODE((2, 3), (1, 0)))
        times = np.linspace(config.t_start, config.t_stop, 3)
        fabricated = np.array([[1., 0.], [0.5, -0.2], [0.2, -0.1]])
        reply = SimpleNamespace(success=True, y=fabricated.T, t=times)

        with patch("ode.generation.reference.solve_ivp", return_value=reply) as solver, patch(
                "ode.generation.reference.linear_reference", return_value=fabricated):
            result = solve_job(job, config)
            self.assertIsNone(result.error)
            self.assertEqual(solver.call_count, 2)
            self.assertEqual(solver.call_args.kwargs["rtol"], config.rtol / config.refinement_factor)

        bad = SimpleNamespace(success=True, y=np.ones((2, 1)), t=times[:1])

        with patch("ode.generation.reference.solve_ivp", return_value=bad):
            self.assertIn("incomplete", solve_job(job, config).error)

        with patch("ode.generation.reference.solve_ivp", side_effect=RuntimeError("fixture error")):
            failed = solve_job(job, config)
            self.assertIn("fixture error", failed.error)
            self.assertIsNone(failed.states)

    def test_refinement_disagreement_and_resource_limits(self):
        """Reject inconsistent solves and retain calls exceeding the RHS budget."""
        config = replace(GeneratorConfig(), time_points=2, max_rhs_calls=1)
        job = SolveJob("fixture", "parameters", "train", LinearODE((1,), (1,)))
        times = np.array([config.t_start, config.t_stop])
        replies = [SimpleNamespace(success=True, t=times, y=np.array([[1., end]]))
                   for end in (1., 2.)]

        with patch("ode.generation.reference.solve_ivp", side_effect=replies):
            self.assertIn("refinement", solve_job(job, config).error)

        def excessive_calls(rhs, *solver_args, **solver_options):
            rhs(0, np.array([1.]))
            rhs(0, np.array([1.]))

        with patch("ode.generation.reference.solve_ivp", side_effect=excessive_calls):
            result = solve_job(job, config)
            self.assertIn("budget", result.error)
            self.assertEqual(result.nfev, 2)

    def test_independent_check_catches_consistently_wrong_solves(self):
        """Catch matching adaptive solves that disagree with an independent reference."""
        # Two identical answers can agree with each other and still be wrong.
        config = replace(GeneratorConfig(), time_points=3)
        job = SolveJob("fixture", "parameters", "train", LinearODE((1,), (1,)))
        reply = SimpleNamespace(success=True, t=np.linspace(0, 5, 3), y=np.ones((1, 3)))

        with patch("ode.generation.reference.solve_ivp", return_value=reply):
            result = solve_job(job, config)

        self.assertIn("Independent", result.error)
        self.assertEqual(result.refinement_ratio, 0)
        self.assertGreater(result.independent_ratio, 1)
        self.assertIsNone(result.states)

    def test_failed_refined_solve_preserves_call_counts(self):
        """Keep calls from both solves and the original error type after failure."""
        config = replace(GeneratorConfig(), time_points=2)
        job = SolveJob("fixture", "parameters", "train", LinearODE((1,), (1,)))
        times = np.array([config.t_start, config.t_stop])

        def solver_reply(rhs, interval, initial, **solver_options):
            rhs(0, np.array([1.]))

            if solver_options["rtol"] < config.rtol:
                raise ValueError("refined fixture failure")

            rhs(0, np.array([1.]))

            return SimpleNamespace(success=True, t=times, y=np.array([[1., 1.]]))

        with patch("ode.generation.reference.solve_ivp", side_effect=solver_reply):
            result = solve_job(job, config)

        self.assertEqual(result.nfev, 3)
        self.assertEqual(result.error, "ValueError: refined fixture failure")
        self.assertIsNone(result.states)


if __name__ == "__main__":
    unittest.main()
