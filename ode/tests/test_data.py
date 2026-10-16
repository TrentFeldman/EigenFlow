"""Verify safety, configuration and data contracts without generating datasets.

Only temporary metadata and fabricated in-memory arrays are used. Numerical
integration is patched out wherever the orchestration layer is exercised.
"""

from contextlib import redirect_stdout
from dataclasses import asdict, replace
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ode.config import GeneratorConfig
from ode.dataset import Normalizer, TrajectoryDataset, encode
from ode.equations import LinearODE
from ode.generate_data import fingerprint, generate, iter_jobs, iter_results, main
from ode.reference import SolveJob, SolveResult


class DataContractTests(unittest.TestCase):
    """No test calls the real solve_ivp or persists reference trajectories."""

    def test_dry_run_never_samples_solves_or_writes(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            with patch("ode.generate_data.iter_jobs") as jobs, patch("ode.generate_data.generate") as writer:
                stream = StringIO()
                with redirect_stdout(stream):
                    self.assertEqual(main(["--output", str(output)]), 0)
                jobs.assert_not_called()
                writer.assert_not_called()
                self.assertFalse(output.exists())
                self.assertEqual(json.loads(stream.getvalue())["planned_trajectories"], 384)

    def test_invalid_config_rejected_and_higher_orders_supported(self):
        for settings in ({"orders": (0,)}, {"orders": (7,)}, {"workers": 0},
                         {"backend": "invalid"}, {"method": "LSODA"},
                         {"rtol": float("nan")}, {"time_points": 1}, {"seed": True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                GeneratorConfig(**settings)
        self.assertEqual(GeneratorConfig(orders=(2, 3, 8), max_order=8).max_order, 8)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"workres": 2}')
            with self.assertRaisesRegex(ValueError, "Unknown"):
                GeneratorConfig.load(path)

    def test_thread_workers_and_stable_result_order(self):
        # A barrier proves two workers overlap; these workers do NOT solve ODEs.
        barrier = threading.Barrier(2)
        seen_threads = set()
        jobs = [SolveJob(str(i), str(i), "train", LinearODE((1,), (1,))) for i in range(4)]

        def fabricated_worker(job, config):
            seen_threads.add(threading.get_ident())
            barrier.wait(timeout=5)
            return SolveResult(job, error="fabricated failure")

        with patch("ode.generate_data.solve_job", side_effect=fabricated_worker):
            results = list(iter_results(iter(jobs), replace(GeneratorConfig(), workers=2)))
        self.assertEqual([result.job.trajectory_id for result in results], [str(i) for i in range(4)])
        self.assertEqual(len(seen_threads), 2)

    def test_request_sampling_is_deterministic_and_split_disjoint(self):
        # Request descriptions only: no solver, arrays of trajectories or files.
        config = GeneratorConfig(orders=(2, 3, 7), max_order=7,
                                 groups_per_split=(1, 1, 1), initial_conditions_per_group=2)
        first, second = list(iter_jobs(config)), list(iter_jobs(config))
        self.assertEqual(first, second)
        self.assertEqual(len(first), config.trajectory_count)
        parameter_splits = {}
        initial_splits = {}
        for job in first:
            for seen, key in ((parameter_splits, job.parameter_id),
                              (initial_splits, job.equation.initial_state)):
                self.assertIn(seen.get(key), (None, job.split))
                seen[key] = job.split

    def test_failed_jobs_recorded_and_existing_output_preserved(self):
        config = GeneratorConfig(orders=(1,), families=("linear_homogeneous",),
                                 groups_per_split=(1, 1, 1), initial_conditions_per_group=1)
        jobs = [SolveJob(str(i), f"p{i}", split, LinearODE((i + 1.,), (i + 1.,)))
                for i, split in enumerate(("train", "validation", "test"))]
        with TemporaryDirectory() as directory:
            output = Path(directory) / "fixture-metadata"
            with patch("ode.generate_data.iter_jobs", return_value=iter(jobs)), patch(
                    "ode.generate_data.solve_job", side_effect=lambda job, cfg: SolveResult(job, error="fixture")), patch(
                    "ode.generate_data.validate_reference", return_value={"passed": True, "cases": []}):
                manifest = generate(replace(config, backend="serial"), output)
            self.assertEqual(manifest["failed"], 3)
            self.assertEqual(manifest["status"], "complete_with_failures")
            self.assertEqual(len((output / "records.jsonl").read_text().splitlines()), 3)
            self.assertFalse(list(output.rglob("*.npz")))
            with self.assertRaises(FileExistsError):
                generate(config, output)

    def test_failed_preflight_prevents_dataset_creation(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            with patch("ode.generate_data.validate_reference", return_value={
                    "passed": False, "cases": [{"name": "fixture", "passed": False}]}), patch(
                    "ode.generate_data.iter_jobs") as sampler:
                with self.assertRaisesRegex(RuntimeError, "preflight"):
                    generate(GeneratorConfig(), output)
                sampler.assert_not_called()
            self.assertFalse(output.exists())

    def test_encoding_masks_and_training_only_scaling(self):
        equation = LinearODE((2, 3), (1, -1))
        features, mask = encode(equation, [0, 1], 4)
        self.assertEqual(features.shape, (2, 17))
        np.testing.assert_array_equal(mask, [[1, 1, 0, 0]] * 2)
        target = np.array([[1, -1, 0, 0], [3, 1, 0, 0.]])
        fixture = SimpleNamespace(split="train", max_order=4, records=[{}],
                                  raw_trajectory=lambda index: (features, target, mask))
        scaler = Normalizer.fit(fixture)
        x, y = scaler.transform(features, target, mask)
        np.testing.assert_array_equal(x[:, 2:4], 0)
        np.testing.assert_array_equal(y[:, 2:], 0)
        np.testing.assert_allclose(y[:, :2].mean(0), 0)
        fixture.split = "validation"
        with self.assertRaises(ValueError):
            Normalizer.fit(fixture)
        with self.assertRaises(ValueError):
            encode(equation, [0], 1)

    def test_loader_rejects_cross_split_equations(self):
        # Metadata-only fixture: the loader must reject leakage before opening
        # any state archive. Identical numerical equations have the same ID.
        equation = LinearODE((2., 3.), (1., 0.))
        records = [{"trajectory_id": split, "parameter_id": fingerprint([
            equation.coefficients, 0., 0., 0.]), "equation": asdict(equation),
            "order": 2, "family": equation.family, "split": split, "status": "success"}
                   for split in ("train", "test")]
        ledger = "".join(json.dumps(record) + "\n" for record in records).encode()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "records.jsonl").write_bytes(ledger)
            (root / "manifest.json").write_text(json.dumps({
                "schema_version": 1, "status": "complete", "planned": 2,
                "succeeded": 2, "failed": 0, "records_sha256": sha256(ledger).hexdigest()}))
            with self.assertRaisesRegex(ValueError, "leakage"):
                TrajectoryDataset(root, "train", 4)

    def test_lazy_loader_and_checksum_with_fabricated_archive(self):
        # An empty file stands in for an archive; arrays are supplied by a mock.
        # This exercises real loader logic without writing numerical data.
        config = replace(GeneratorConfig(), time_points=3)
        equation = LinearODE((2., 3.), (1., 0.))
        record = {"trajectory_id": "fixture", "parameter_id": fingerprint([
            equation.coefficients, 0., 0., 0.]), "equation": asdict(equation),
            "order": 2, "family": equation.family, "split": "train", "status": "success",
            "path": "empty-fixture.npz", "sha256": sha256(b"").hexdigest()}
        ledger = (json.dumps(record) + "\n").encode()
        fabricated = {"times": np.linspace(config.t_start, config.t_stop, 3),
                      "states": np.array([[1., 0.], [0.5, -0.2], [0.2, -0.1]])}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "empty-fixture.npz").touch()
            (root / "records.jsonl").write_bytes(ledger)
            (root / "manifest.json").write_text(json.dumps({
                "schema_version": 1, "status": "complete", "planned": 1,
                "config": asdict(config), "succeeded": 1, "failed": 0,
                "records_sha256": sha256(ledger).hexdigest()}))
            dataset = TrajectoryDataset(root, "train", 4)
            self.assertEqual(len(dataset), 3)
            with patch("ode.dataset.np.load") as loader:
                loader.return_value.__enter__.return_value = fabricated
                np.testing.assert_array_equal(dataset[0]["target"], [1, 0, 0, 0])
                np.testing.assert_allclose(dataset[2]["target"], [0.2, -0.1, 0, 0])
                self.assertEqual(loader.call_count, 1)
                dataset.normalizer = Normalizer.fit(dataset)
                before = dataset[2]
                dataset.preload()
                for name, values in before.items():
                    np.testing.assert_allclose(dataset[2][name], values)
                self.assertEqual(loader.call_count, 1)
            with self.assertRaises(IndexError):
                dataset[3]
            dataset._cache.clear()
            del dataset._preloaded
            (root / "empty-fixture.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                dataset[0]


if __name__ == "__main__":
    unittest.main()
