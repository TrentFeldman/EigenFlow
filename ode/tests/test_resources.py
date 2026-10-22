"""Verify allocation-aware execution settings and explicit CLI precedence."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ode.resources import available_cpus, load_resources
from ode.training.train import parse_args
from ode.generation.generate_data import main as generate_main


class ResourceTests(unittest.TestCase):
    """Keep execution changes independent of scientific sampling settings."""

    def test_auto_respects_scheduler_and_affinity(self):
        with patch('ode.resources.os.cpu_count', return_value=64), \
                patch('ode.resources.os.sched_getaffinity', return_value=set(range(16)), create=True), \
                patch.dict('os.environ', {'SLURM_CPUS_PER_TASK': '8'}):
            self.assertEqual(available_cpus(), 8)

    def test_profile_and_cli_precedence(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'resources.json'
            path.write_text(json.dumps({'workers': 3, 'batch_size': 1024, 'threads': 2, 'preload': True}))
            args = parse_args(['--resources', str(path), '--batch-size', '16', '--no-preload'])
            self.assertEqual(args.batch_size, 16)
            self.assertEqual(args.threads, 2)
            self.assertFalse(args.preload)
            output = StringIO()
            with redirect_stdout(output):
                generate_main(['--resources', str(path), '--workers', '1'])
            plan = json.loads(output.getvalue())
            self.assertEqual(plan['config']['workers'], 1)
            self.assertEqual(plan['planned_trajectories'], 384)

    def test_invalid_profile_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'resources.json'
            for values in ({'workers': 0}, {'threads': True}, {'preload': 'yes'},
                           {'preload_limit_gib': float('nan')}, {'seed': 42}):
                path.write_text(json.dumps(values))
                with self.assertRaises(ValueError):
                    load_resources(path)


if __name__ == '__main__':
    unittest.main()
