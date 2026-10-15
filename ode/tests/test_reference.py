"""Check derivative conversion and solver safeguards WITHOUT numerical solving.

Fabricated solver replies exercise failure handling. Analytic integration
benchmarks are deliberately pending until reference generation is authorized.
"""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ode.config import GeneratorConfig
from ode.equations import LinearODE, stable_coefficients
from ode.reference import SolveJob, solve_job


class ReferenceContractTests(unittest.TestCase):
    """Tests use hand-written states, not generated ODE trajectories."""

    def test_higher_order_state_conversion(self):
        for order in (1, 2, 3, 6, 10):
            with self.subTest(order=order):
                coefficients = tuple(float(x) for x in range(1, order + 1))
                state = np.arange(order, dtype=float) + 1
                equation = LinearODE(coefficients, tuple(state))
                derivative = equation.rhs(0, state)
                np.testing.assert_array_equal(derivative[:-1], state[1:])
                self.assertEqual(derivative[-1], -np.dot(coefficients, state))

    def test_forcing_and_validation(self):
        equation = LinearODE((2, 3), (1, 0), amplitude=4, frequency=1, phase=np.pi / 2)
        np.testing.assert_allclose(equation.rhs(0, np.array([1, 0])), [0, 2])
        for coefficients, initial in (((), ()), ((1, 2), (0,)), ((float("nan"),), (0,))):
            with self.assertRaises(ValueError):
                LinearODE(coefficients, initial)

    def test_stable_modes_are_reproducible(self):
        # This samples coefficient vectors only; it never integrates an ODE.
        for order in (2, 3, 6):
            a = stable_coefficients(order, np.random.default_rng(7), (0.2, 1), (0.5, 2))
            b = stable_coefficients(order, np.random.default_rng(7), (0.2, 1), (0.5, 2))
            self.assertEqual(a, b)
            roots = np.roots([1, *reversed(a)])
            self.assertTrue(np.all(roots.real < 0))

    def test_solver_refinement_contract_and_failure(self):
        config = replace(GeneratorConfig(), time_points=3)
        job = SolveJob("fixture", "parameters", "train", LinearODE((2, 3), (1, 0)))
        times = np.linspace(config.t_start, config.t_stop, 3)
        fabricated = np.array([[1., 0.], [0.5, -0.2], [0.2, -0.1]])
        reply = SimpleNamespace(success=True, y=fabricated.T, t=times)
        with patch("ode.reference.solve_ivp", return_value=reply) as solver:
            result = solve_job(job, config)
            self.assertIsNone(result.error)
            self.assertEqual(solver.call_count, 2)
            self.assertEqual(solver.call_args.kwargs["rtol"], config.rtol / config.refinement_factor)
        bad = SimpleNamespace(success=True, y=np.ones((2, 1)), t=times[:1])
        with patch("ode.reference.solve_ivp", return_value=bad):
            self.assertIn("incomplete", solve_job(job, config).error)
        with patch("ode.reference.solve_ivp", side_effect=RuntimeError("fixture error")):
            failed = solve_job(job, config)
            self.assertIn("fixture error", failed.error)
            self.assertIsNone(failed.states)

    def test_refinement_disagreement_and_resource_limits(self):
        config = replace(GeneratorConfig(), time_points=2, max_rhs_calls=1)
        job = SolveJob("fixture", "parameters", "train", LinearODE((1,), (1,)))
        times = np.array([config.t_start, config.t_stop])
        replies = [SimpleNamespace(success=True, t=times, y=np.array([[1., end]]))
                   for end in (1., 2.)]
        with patch("ode.reference.solve_ivp", side_effect=replies):
            self.assertIn("refinement", solve_job(job, config).error)

        def excessive_calls(rhs, *args, **kwargs):
            rhs(0, np.array([1.]))
            rhs(0, np.array([1.]))

        with patch("ode.reference.solve_ivp", side_effect=excessive_calls):
            self.assertIn("budget", solve_job(job, config).error)


if __name__ == "__main__":
    unittest.main()
