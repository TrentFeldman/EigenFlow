"""Scientific regression checks for metrics, split ownership and model selection."""

from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ode.config import GeneratorConfig
from ode.dataset import TrajectoryDataset
from ode.evaluate import trajectory_metrics, summarize
from ode.generate_data import generate, iter_jobs
from ode.predict import domain_violations
from ode.equations import LinearODE
from ode.train import _fit


class EvaluationTests(unittest.TestCase):
    """Verify hand-computable metrics and explicit failure accounting."""

    def test_relative_norm_and_component_errors(self):
        reference = np.array([[1., 2.], [1., 2.]])
        result = trajectory_metrics(reference, reference + [1., 0.])
        self.assertAlmostEqual(result['relative_l2'], 1 / np.sqrt(5))
        self.assertEqual(result['components']['d0']['relative_l2'], 1.)
        self.assertEqual(result['components']['d1']['relative_l2'], 0.)
        self.assertAlmostEqual(result['absolute_rmse'], np.sqrt(.5))

    def test_near_zero_floor_and_uniform_time_weights(self):
        result = trajectory_metrics(np.zeros((4, 2)), np.full((4, 2), 2e-7))
        self.assertTrue(result['near_zero'])
        self.assertAlmostEqual(result['relative_l2'], .2)
        self.assertAlmostEqual(result['absolute_rmse'], 2e-7)
        result = trajectory_metrics(np.ones((4, 1)), np.array([[3.], [1.], [1.], [1.]]))
        self.assertEqual(result['relative_l2'], 1.)

    def test_statistics_keep_nonfinite_and_threshold_failures(self):
        rows = [trajectory_metrics(np.ones((2, 1)), np.full((2, 1), value))
                for value in [1., 1.02, 1.1]]
        result = summarize(rows)
        self.assertAlmostEqual(result['mean'], .04)
        self.assertEqual(result['failure_count'], 1)
        self.assertTrue(result['target_passed'])
        rows.append(trajectory_metrics(np.ones((2, 1)), np.full((2, 1), np.nan)))
        result = summarize(rows)
        self.assertEqual(result['count'], 4)
        self.assertEqual(result['nonfinite_count'], 1)
        self.assertIsNone(result['mean'])
        self.assertFalse(result['target_passed'])
        json.dumps(result, allow_nan=False)
        self.assertIsNone(summarize([])['mean'])

    def test_invalid_shapes_and_references_rejected(self):
        for reference, prediction in [(np.ones((2, 1)), np.ones((1, 2))),
                                      (np.empty((0, 1)), np.empty((0, 1))),
                                      (np.full((2, 1), np.nan), np.ones((2, 1)))]:
            with self.assertRaises(ValueError):
                trajectory_metrics(reference, prediction)

    def test_diagnostic_membership_and_loader_enforcement(self):
        config = GeneratorConfig(groups_per_split=(1, 1, 1), initial_conditions_per_group=1,
                                 diagnostic_groups=1, time_points=5, backend='serial')
        jobs = list(iter_jobs(config))
        self.assertEqual(len(jobs), config.trajectory_count)
        pools = {split: [job for job in jobs if job.split == split]
                 for split in ['train', 'validation', 'test']}
        for job in jobs:
            if job.split == 'validation_parameter':
                self.assertIn(job.parameter_id, {j.parameter_id for j in pools['validation']})
                self.assertIn(job.equation.initial_state, {j.equation.initial_state for j in pools['train']})
            elif job.split == 'validation_initial':
                self.assertIn(job.parameter_id, {j.parameter_id for j in pools['train']})
                self.assertIn(job.equation.initial_state, {j.equation.initial_state for j in pools['validation']})
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'data'
            generate(config, root)
            for split in ['train', 'validation', 'test', 'validation_parameter', 'validation_initial']:
                dataset = TrajectoryDataset(root, split, config.max_order)
                dataset.raw_trajectory(0)
            records = [json.loads(line) for line in (root / 'records.jsonl').read_text().splitlines()]
            bad = next(r for r in records if r['split'] == 'validation_parameter')
            bad['split'] = 'validation_initial'
            fixture = object.__new__(TrajectoryDataset)
            fixture.split, fixture.max_order = 'train', 6
            with self.assertRaisesRegex(ValueError, 'Diagnostic split'):
                fixture._select_records(records)

    def test_checkpoint_selection_uses_relative_l2_not_mse(self):
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(run=Path(directory), epochs=2, batch_size=4)
            report = {'epochs': [], 'dataset_manifest_sha256': 'fixture'}
            loader = SimpleNamespace(dataset=None)
            scores = [{'overall': {'mean': value}, 'by_family': {}} for value in [.2, .3]]
            def save(run, *unused):
                (run / 'best.pt').write_bytes(b'fixture')
            with patch('ode.train.run_epoch', side_effect=[1., .8, .5, .1]), \
                    patch('ode.train.evaluate_model', side_effect=scores), \
                    patch('ode.train.save_checkpoint', side_effect=save) as saver:
                _fit(None, None, (loader, loader), 'cpu', None, None, args, report)
            self.assertEqual(report['best_epoch'], 1)
            self.assertEqual(saver.call_count, 1)

    def test_inference_domain_checks_roots_and_horizon(self):
        config = asdict(GeneratorConfig())
        equation = LinearODE((1.25, 1.), (1., 0.), .5, 1., 0.)
        self.assertEqual(domain_violations(equation, [0., 5.], config), [])
        self.assertTrue(domain_violations(equation, [6.], config))
        self.assertTrue(domain_violations(replace(equation, coefficients=(-1., 1.)), [0.], config))


if __name__ == '__main__':
    unittest.main()
