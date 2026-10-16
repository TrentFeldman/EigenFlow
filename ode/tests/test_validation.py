"""Compare numerical references with independently specified closed forms.

These integration tests cover both equation families, repeated/distinct roots,
damped oscillations, higher orders, resonance and shifted start times. Process
worker checks also verify result ordering across the reference-solving pipeline.
"""

from dataclasses import replace
import unittest

import numpy as np

from ode.analytic import linear_reference
from ode.config import GeneratorConfig
from ode.equations import LinearODE
from ode.generate_data import iter_results
from ode.reference import SolveJob, solve_job
from ode.validate_reference import benchmark_cases, validate_reference


class NumericalValidationTests(unittest.TestCase):
    """Independent closed-form expectations detect systematic solver mistakes."""

    def test_both_families_and_higher_orders_pass_closed_forms(self):
        """Check benchmark success and coverage of both families, order six and resonance."""
        report = validate_reference(GeneratorConfig())
        self.assertTrue(report["passed"], report)
        self.assertEqual({case["family"] for case in report["cases"]},
                         {"linear_homogeneous", "linear_sinusoidal"})
        self.assertIn(6, {case["order"] for case in report["cases"]})
        self.assertTrue(any("resonance" in case["name"] for case in report["cases"]))

    def test_double_root_and_resonance_explicit_solutions(self):
        """Verify matrix-exponential states against repeated-root and resonant formulas."""
        times = np.linspace(0, 5, 31)
        repeated = LinearODE((1., 2.), (1., 0.))
        states = linear_reference(repeated, times)
        np.testing.assert_allclose(states[:, 0], (1 + times) * np.exp(-times), atol=1e-12)
        np.testing.assert_allclose(states[:, 1], -times * np.exp(-times), atol=1e-12)
        resonant = LinearODE((1., 0.), (0., -0.5), 1., 1., 0.)
        states = linear_reference(resonant, times)
        np.testing.assert_allclose(states[:, 0], -0.5 * times * np.cos(times), atol=1e-12)

    def test_nonzero_start_time_preserves_elapsed_forcing(self):
        """Check that shifting absolute time preserves the elapsed-time solution."""
        equation = LinearODE((2., 3.), (0.4, -0.2), 0.7, 1.3, 0.4)
        job = SolveJob("time-shift-check", "fixture", "validation", equation)
        config = GeneratorConfig()
        first = solve_job(job, config)
        shifted = solve_job(job, replace(config, t_start=7., t_stop=12.))
        self.assertIsNone(first.error)
        self.assertIsNone(shifted.error)
        np.testing.assert_allclose(first.states, shifted.states, rtol=1e-7, atol=1e-9)

    def test_process_workers_solve_both_families(self):
        """Verify ordered process results against closed forms for both families."""
        config = replace(GeneratorConfig(), backend="process", workers=2)
        cases = benchmark_cases(config)[:2]
        jobs = [SolveJob(case.name, case.name, "validation", case.equation) for case in cases]
        results = list(iter_results(jobs, config))
        self.assertEqual([result.job.trajectory_id for result in results],
                         [job.trajectory_id for job in jobs])

        for case, result in zip(cases, results):
            self.assertIsNone(result.error)
            np.testing.assert_allclose(result.states, case.expected(result.times - config.t_start),
                                       rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
