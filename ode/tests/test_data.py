"""Verify configuration, generation orchestration and dataset integrity.

These contracts cover dry-run behavior, deterministic split assignment,
parallel result order, failed-job records, encoding and normalized loading.
Fixtures supply controlled metadata and arrays to isolate each check.
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
from ode.training.dataset import Normalizer, TrajectoryDataset, encode, feature_width
from ode.generation.equations import LinearODE
from ode.generation.generate_data import generate, iter_jobs, iter_results, main
from ode.generation.reference import SolveJob, SolveResult


def _write_fixture_dataset(root, records, config=None):
    """Write a ledger and matching manifest for controlled loader checks."""
    ledger = "".join(json.dumps(record) + "\n" for record in records).encode()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "planned": len(records),
        "succeeded": sum(record["status"] == "success" for record in records),
        "failed": sum(record["status"] == "failed" for record in records),
        "records_sha256": sha256(ledger).hexdigest(),
    }

    if config is not None:
        manifest["config"] = asdict(config)

    (root / "records.jsonl").write_bytes(ledger)
    (root / "manifest.json").write_text(json.dumps(manifest))


class DataContractTests(unittest.TestCase):
    """Check dataset identity, split isolation and the training input contract."""

    def test_dry_run_never_samples_solves_or_writes(self):
        """Verify that the default CLI only prints the planned trajectory count."""
        with TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"

            with patch("ode.generation.generate_data.iter_jobs") as jobs, patch("ode.generation.generate_data.generate") as writer:
                stream = StringIO()

                with redirect_stdout(stream):
                    self.assertEqual(main(["--output", str(output)]), 0)

                jobs.assert_not_called()
                writer.assert_not_called()
                self.assertFalse(output.exists())
                self.assertEqual(json.loads(stream.getvalue())["planned_trajectories"], 384)

    def test_invalid_config_rejected_and_higher_orders_supported(self):
        """Check invalid settings, unknown fields and configurable maximum order."""
        for settings in (
            {"orders": (0,)},
            {"orders": (7,)},
            {"workers": 0},
            {"backend": "invalid"},
            {"method": "LSODA"},
            {"rtol": float("nan")},
            {"time_points": 1},
            {"seed": True},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                GeneratorConfig(**settings)

        self.assertEqual(GeneratorConfig(orders=(2, 3, 8), max_order=8).max_order, 8)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"workres": 2}')

            with self.assertRaisesRegex(ValueError, "Unknown"):
                GeneratorConfig.load(path)

    def test_thread_workers_and_stable_result_order(self):
        """Verify concurrent threads still yield results in request order."""
        # The barrier only releases when two worker threads are active together.
        barrier = threading.Barrier(2)
        seen_threads = set()
        jobs = [SolveJob(str(i), str(i), "train", LinearODE((1,), (1,))) for i in range(4)]

        def fabricated_worker(job, config):
            seen_threads.add(threading.get_ident())
            barrier.wait(timeout=5)

            return SolveResult(job, error="fabricated failure")

        with patch("ode.generation.generate_data.solve_job", side_effect=fabricated_worker):
            results = list(iter_results(iter(jobs), replace(GeneratorConfig(), workers=2)))

        self.assertEqual([result.job.trajectory_id for result in results], [str(i) for i in range(4)])
        self.assertEqual(len(seen_threads), 2)

    def test_request_sampling_is_deterministic_and_split_disjoint(self):
        """Check repeatable jobs and disjoint parameter/initial-state partitions."""
        config = GeneratorConfig(
            orders=(2, 3, 7),
            max_order=7,
            groups_per_split=(1, 1, 1),
            initial_conditions_per_group=2,
        )
        first, second = list(iter_jobs(config)), list(iter_jobs(config))
        self.assertEqual(first, second)
        self.assertEqual(len(first), config.trajectory_count)
        parameter_splits = {}
        initial_splits = {}

        for job in first:
            for assignments, identity in (
                (parameter_splits, job.parameter_id),
                (initial_splits, job.equation.initial_state),
            ):
                self.assertIn(assignments.get(identity), (None, job.split))
                assignments[identity] = job.split

    def test_failed_jobs_recorded_and_existing_output_preserved(self):
        """Verify every failed request is recorded and existing output is protected."""
        config = GeneratorConfig(
            orders=(1,),
            families=("linear_homogeneous",),
            groups_per_split=(1, 1, 1),
            initial_conditions_per_group=1,
        )
        jobs = [
            SolveJob(str(i), f"p{i}", split, LinearODE((i + 1.,), (i + 1.,)))
            for i, split in enumerate(("train", "validation", "test"))
        ]

        with TemporaryDirectory() as directory:
            output = Path(directory) / "fixture-metadata"

            with (
                patch("ode.generation.generate_data.iter_jobs", return_value=iter(jobs)),
                patch(
                    "ode.generation.generate_data.solve_job",
                    side_effect=lambda job, cfg: SolveResult(job, error="fixture"),
                ),
                patch("ode.generation.generate_data.validate_reference", return_value={"passed": True, "cases": []}),
            ):
                manifest = generate(replace(config, backend="serial"), output)

            self.assertEqual(manifest["failed"], 3)
            self.assertEqual(manifest["status"], "complete_with_failures")
            self.assertEqual(len((output / "records.jsonl").read_text().splitlines()), 3)
            self.assertFalse(list(output.rglob("*.npz")))

            with self.assertRaises(FileExistsError):
                generate(config, output)

    def test_failed_preflight_prevents_dataset_creation(self):
        """Reject a failed reference preflight before sampling or output creation."""
        with TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"

            with (
                patch(
                    "ode.generation.generate_data.validate_reference",
                    return_value={
                        "passed": False,
                        "cases": [{"name": "fixture", "passed": False}],
                    },
                ),
                patch("ode.generation.generate_data.iter_jobs") as sampler,
            ):
                with self.assertRaisesRegex(RuntimeError, "preflight"):
                    generate(GeneratorConfig(), output)

                sampler.assert_not_called()

            self.assertFalse(output.exists())

    def test_encoding_masks_and_training_only_scaling(self):
        """Check padding, centered real targets and training-only normalization."""
        equation = LinearODE((2, 3), (1, -1))
        features, mask = encode(equation, [0, 1], 4)
        self.assertEqual(features.shape, (2, feature_width(4)))
        np.testing.assert_array_equal(mask, [[1, 1, 0, 0]] * 2)
        target = np.array([[1, -1, 0, 0], [3, 1, 0, 0.]])
        fixture = SimpleNamespace(
            split="train",
            max_order=4,
            records=[{}],
            raw_trajectory=lambda index: (features, target, mask),
        )
        scaler = Normalizer.fit(fixture)
        normalized_features, normalized_target = scaler.transform(features, target, mask)
        np.testing.assert_array_equal(normalized_features[:, 2:4], 0)
        np.testing.assert_array_equal(normalized_target[:, 2:], 0)
        np.testing.assert_allclose(normalized_target[:, :2].mean(0), 0)
        fixture.split = "validation"

        with self.assertRaises(ValueError):
            Normalizer.fit(fixture)

        with self.assertRaises(ValueError):
            encode(equation, [0], 1)

    def test_loader_rejects_cross_split_equations(self):
        """Reject equation leakage before opening a trajectory archive."""
        # Metadata-only fixture: the loader must reject leakage before opening
        # any state archive. Identical numerical equations have the same ID.
        equation = LinearODE((2., 3.), (1., 0.))
        records = [
            {
                "trajectory_id": split,
                "parameter_id": equation.parameter_id,
                "equation": asdict(equation),
                "order": 2,
                "family": equation.family,
                "split": split,
                "status": "success",
            }
            for split in ("train", "test")
        ]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture_dataset(root, records)

            with self.assertRaisesRegex(ValueError, "leakage"):
                TrajectoryDataset(root, "train", 4)

    def test_lazy_loader_and_checksum_with_fabricated_archive(self):
        """Verify lazy caching, equivalent preload output and checksum enforcement."""
        # Controlled arrays separate checksum/cache behavior from archive decoding.
        config = replace(GeneratorConfig(), time_points=3)
        equation = LinearODE((2., 3.), (1., 0.))
        record = {
            "trajectory_id": "fixture",
            "parameter_id": equation.parameter_id,
            "equation": asdict(equation),
            "order": 2,
            "family": equation.family,
            "split": "train",
            "status": "success",
            "path": "empty-fixture.npz",
            "sha256": sha256(b"").hexdigest(),
        }
        fabricated = {
            "times": np.linspace(config.t_start, config.t_stop, 3),
            "states": np.array([[1., 0.], [0.5, -0.2], [0.2, -0.1]]),
        }

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "empty-fixture.npz").touch()
            _write_fixture_dataset(root, [record], config)
            dataset = TrajectoryDataset(root, "train", 4)
            self.assertEqual(len(dataset), 3)

            with patch("ode.training.dataset.np.load") as loader:
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
            dataset._preloaded = None
            (root / "empty-fixture.npz").write_bytes(b"changed")

            with self.assertRaisesRegex(ValueError, "checksum"):
                dataset[0]


if __name__ == "__main__":
    unittest.main()
