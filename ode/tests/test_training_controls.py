"""Check schedule endpoints, independent model seeds and saved optimizer state."""

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ode.config import GeneratorConfig
from ode.generation.generate_data import generate
from ode.tools.neurons import main as neurons_main
from ode.tools.video import main as video_main
from ode.training.dataset import Normalizer, TrajectoryDataset, feature_width
from ode.training.model import OrderConditionedMLP, masked_mse
from ode.training.train import (RELATIVE_LOSS_FLOOR, TensorLoader, build_scheduler, main, parse_args,
                                relative_loss_weights, save_training_state)

TINY = GeneratorConfig(groups_per_split=(2, 1, 1), initial_conditions_per_group=2, time_points=6,
                       max_order=3, backend='serial')


class TrainingControlTests(unittest.TestCase):
    """Exercise behavior needed for fair, independently reproducible controls."""

    def test_cosine_uses_initial_and_minimum_rates_in_training_epochs(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=.001)
        args = SimpleNamespace(learning_rate_schedule='cosine', epochs=3, minimum_learning_rate=.00001)
        scheduler = build_scheduler(optimizer, args)
        rates = []
        for epoch in range(args.epochs):
            rates.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            if epoch < args.epochs - 1:
                scheduler.step()
        np.testing.assert_allclose(rates, [.001, .000505, .00001], rtol=1e-12)

    def test_constant_schedule_leaves_optimizer_rate_alone(self):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.))], lr=.0003)
        self.assertIsNone(build_scheduler(optimizer, SimpleNamespace(learning_rate_schedule='constant')))
        self.assertEqual(optimizer.param_groups[0]['lr'], .0003)

    def test_model_seed_is_independent_from_dataset_recipe(self):
        output = StringIO()
        with redirect_stdout(output), patch('ode.training.train.TrajectoryDataset') as dataset:
            main(['--training-seed', '123', '--epochs', '3'])
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['training_seed'], 123)
        self.assertEqual(GeneratorConfig().seed, 20261015)
        dataset.assert_not_called()

    def test_invalid_schedule_and_seed_rejected(self):
        for arguments in (['--training-seed', '-1'], ['--weight-decay', 'nan'],
                          ['--learning-rate-schedule', 'cosine', '--epochs', '1'],
                          ['--learning-rate-schedule', 'cosine', '--minimum-learning-rate', '.1']):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(arguments)

    def test_saved_state_preserves_adam_moments_for_next_update(self):
        config = GeneratorConfig(max_order=2, orders=(2,))
        model = OrderConditionedMLP(2, width=8, depth=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        args = SimpleNamespace(learning_rate_schedule='cosine', epochs=3, minimum_learning_rate=.00001)
        scheduler = build_scheduler(optimizer, args)
        features = torch.ones(4, feature_width(2))
        def step(network, opt):
            opt.zero_grad()
            network(features).square().mean().backward()
            opt.step()
        step(model, optimizer)
        scheduler.step()
        normalizer = Normalizer(np.zeros(feature_width(2)), np.ones(feature_width(2)), np.zeros(2), np.ones(2))
        loader = SimpleNamespace(generator=torch.Generator().manual_seed(42))
        report = {'dataset_manifest_sha256':'fixture', 'best_epoch':1, 'best_validation_relative_l2':.1}
        with TemporaryDirectory() as directory:
            save_training_state(Path(directory), model, optimizer, scheduler, loader, normalizer, config, report, 1)
            saved = torch.load(Path(directory)/'last.pt', weights_only=True)
            clone = OrderConditionedMLP(**saved['architecture'])
            clone.load_state_dict(saved['state_dict'])
            other_optimizer = torch.optim.AdamW(clone.parameters(), lr=.001)
            other_scheduler = build_scheduler(other_optimizer, args)
            other_optimizer.load_state_dict(saved['optimizer'])
            other_scheduler.load_state_dict(saved['scheduler'])
            step(model, optimizer)
            step(clone, other_optimizer)
            for first, second in zip(model.parameters(), clone.parameters()):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            torch.testing.assert_close(saved['rng']['loader'], loader.generator.get_state())
            self.assertEqual(saved['scheduler']['last_epoch'], scheduler.last_epoch)

    def test_tensor_loader_matches_dataloader_batches(self):
        """The fast loader must reproduce DataLoader's shuffled batches for a seed."""
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'data'
            with redirect_stderr(StringIO()):
                generate(TINY, root)
            dataset = TrajectoryDataset(root, 'train', TINY.max_order)
            dataset.normalizer = Normalizer.fit(dataset)
            dataset.preload()
            slow = torch.utils.data.DataLoader(dataset, batch_size=7, shuffle=True,
                                               generator=torch.Generator().manual_seed(5))
            fast = TensorLoader(dataset, 7, torch.device('cpu'), torch.Generator().manual_seed(5))
            for _ in range(2):  # two epochs: the generator must advance identically
                for expected, actual in zip(slow, fast, strict=True):
                    for name in expected:
                        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_resumed_training_matches_uninterrupted_training(self):
        """Stopping after 2 epochs and resuming to 3 must give the same weights as 3 straight."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with redirect_stderr(StringIO()):
                generate(TINY, root / 'data')
            common = ['--train', '--data', str(root / 'data'), '--device', 'cpu', '--threads', '1',
                      '--batch-size', '8', '--width', '8', '--depth', '1']
            with redirect_stdout(StringIO()):
                main([*common, '--run', str(root / 'straight'), '--epochs', '3'])
                main([*common, '--run', str(root / 'resumed'), '--epochs', '2'])
                main([*common, '--run', str(root / 'resumed'), '--epochs', '3', '--resume'])
                with self.assertRaisesRegex(ValueError, 'original --width/--depth'):
                    main([*common, '--run', str(root / 'resumed'), '--epochs', '4', '--resume', '--depth', '2'])
                # A 1e-30 learning rate leaves copied float32 weights untouched, so best.pt must equal the source.
                main([*common, '--run', str(root / 'copied'), '--epochs', '1', '--learning-rate', '1e-30',
                      '--initialize-from', str(root / 'straight' / 'best.pt')])
                with self.assertRaisesRegex(ValueError, 'pass its --width/--depth'):
                    main([*common, '--run', str(root / 'other'), '--epochs', '1', '--depth', '2',
                          '--initialize-from', str(root / 'straight' / 'best.pt')])
            source, copied = (torch.load(root / name / 'best.pt', weights_only=True) for name in ('straight', 'copied'))
            for key, value in source['state_dict'].items():
                torch.testing.assert_close(copied['state_dict'][key], value, rtol=0, atol=0)
            straight, resumed = (torch.load(root / name / 'last.pt', weights_only=True)
                                 for name in ('straight', 'resumed'))
            for key, value in straight['state_dict'].items():
                torch.testing.assert_close(resumed['state_dict'][key], value, rtol=0, atol=0)
            report = json.loads((root / 'resumed' / 'run.json').read_text())
            self.assertEqual([row['epoch'] for row in report['epochs']], [1, 2, 3])
            self.assertEqual(report['status'], 'complete')

            # --snapshot-every saves weights for the neurons animation, which must then render a page.
            with redirect_stdout(StringIO()):
                main([*common, '--run', str(root / 'snap'), '--epochs', '2', '--snapshot-every', '1'])
            self.assertEqual(sorted(p.name for p in (root / 'snap' / 'snapshots').iterdir()),
                             ['epoch-0000.pt', 'epoch-0001.pt', 'epoch-0002.pt'])
            equation = root / 'equation.json'
            equation.write_text(json.dumps({'equation': {'coefficients': [2.0, 0.6], 'initial_state': [1.0, 0.0]}}))
            with redirect_stdout(StringIO()):
                neurons_main(['--run', str(root / 'snap'), '--input', str(equation)])
            self.assertIn('Watching a neural network learn', (root / 'snap' / 'neurons.html').read_text())
            if shutil.which('ffmpeg'):
                with redirect_stdout(StringIO()):
                    video_main(['--run', str(root / 'snap'), '--input', str(equation)])
                self.assertGreater((root / 'snap' / 'training.mp4').stat().st_size, 0)

            # The hard-case sampler's weights live in last.pt, so resuming stays exact with it on.
            focused = [*common, '--loss', 'relative', '--focus-hard']
            with redirect_stdout(StringIO()):
                main([*focused, '--run', str(root / 'focus-straight'), '--epochs', '3'])
                main([*focused, '--run', str(root / 'focus-resumed'), '--epochs', '2'])
                main([*focused, '--run', str(root / 'focus-resumed'), '--epochs', '3', '--resume'])
            straight, resumed = (torch.load(root / name / 'last.pt', weights_only=True)
                                 for name in ('focus-straight', 'focus-resumed'))
            for key, value in straight['state_dict'].items():
                torch.testing.assert_close(resumed['state_dict'][key], value, rtol=0, atol=0)
            weights = straight['trajectory_weights']
            self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
            self.assertGreater(float(weights.max()), float(weights.min()))  # losses moved it off uniform

    def test_relative_loss_charges_each_trajectory_for_its_own_size(self):
        """A 1% miss everywhere must cost every trajectory the same under --loss relative."""
        with TemporaryDirectory() as directory:
            with redirect_stderr(StringIO()):
                generate(TINY, Path(directory) / 'data')
            dataset = TrajectoryDataset(Path(directory) / 'data', 'train', TINY.max_order)
            dataset.normalizer = Normalizer.fit(dataset)
            dataset.preload()
            target, mask = (torch.as_tensor(dataset._preloaded[name]) for name in ('target', 'mask'))
            raw = torch.as_tensor(dataset.raw_targets).reshape(target.shape)
            miss = 0.01 * raw / torch.as_tensor(dataset.normalizer.target_scale) * mask
            rows = masked_mse(target + miss, target, mask, torch.as_tensor(relative_loss_weights(dataset)), per_row=True)
            per_trajectory = rows.reshape(len(dataset.records), -1).mean(1)
            size = ((raw ** 2 * mask).sum(-1) / mask.sum(-1)).reshape(len(dataset.records), -1).mean(1).sqrt()
            above_floor = per_trajectory[size >= RELATIVE_LOSS_FLOOR]
            self.assertGreaterEqual(len(above_floor), 2)
            torch.testing.assert_close(above_floor, above_floor[:1].expand_as(above_floor), rtol=1e-5, atol=0)


if __name__ == '__main__':
    unittest.main()
