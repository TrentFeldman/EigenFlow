"""Check the explorer page (`python -m ode viz`) against the Python reference.

The page runs the network and the exact solver in JavaScript. These tests run
that same script in Node (skipped when Node is not installed) and compare it
with PyTorch and generation/analytic.py, including the gradient behind the
input-importance scene. They also check how equations become slider modes and
that the worst-case reel only accepts an evaluation of the same checkpoint.
"""

from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from ode.evaluation.evaluate import trajectory_metrics
from ode.evaluation.plot_results import _equation_for
from ode.evaluation.predict import Predictor
from ode.generation.analytic import linear_reference
from ode.tools.viz import build, modes, worst_cases
from ode.config import GeneratorConfig
from ode.generation.equations import LinearODE

MODEL = Path(__file__).resolve().parents[1] / "models" / "current"
# Third order with a push, to cover the real mode, padding and forcing inputs:
# (s² + 0.6s + 2.05)(s + 0.8) = s³ + 1.4s² + 2.53s + 1.64, a pair d=0.3, w=1.4 and r=0.8.
EQUATION = LinearODE((1.64, 2.53, 1.4), (0.4, -0.3, 0.2), 0.6, 1.7, 0.4)
PROBE = 40  # time index where the gradient is compared


class ExplorerTests(unittest.TestCase):

    def test_modes_round_trip_and_reject_other_structures(self):
        """Pairs and real modes come back from the coefficients; overdamped or growing ones are refused."""
        result = modes(EQUATION)
        np.testing.assert_allclose(result["pairs"], [[0.3, 1.4]], atol=1e-9)
        np.testing.assert_allclose(result["real"], [0.8], atol=1e-9)
        for coefficients in ((2.0, 3.0), (2.0, -0.5)):  # roots -1, -2 (no pair); a growing pair
            with self.assertRaises(ValueError):
                modes(LinearODE(coefficients, (1.0, 0.0)))

    @unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
    def test_page_maths_matches_pytorch_and_exact_solver(self):
        """Exact states, predictions and the input gradient agree with Python to float32 precision."""
        with TemporaryDirectory() as folder:
            request = Path(folder) / "equation.json"
            request.write_text(json.dumps({"equation": EQUATION.__dict__}))
            html = build(MODEL, request, worst=0)
        core = re.search(r'<script id="core">(.*?)</script>', html, re.S).group(1)
        harness = core + f"""
const net = new Net(D.models[0]), run = net.run(D.start), F = net.F;
console.log(JSON.stringify({{check: selfCheck([net]), coefficients: coefficients(D.start),
  exact: exactStates(D.start, net.times), out: run.out,
  x: Array.from(run.x.slice({PROBE} * F, {PROBE + 1} * F)), grad: Array.from(net.gradient(run, 0).slice({PROBE} * F, {PROBE + 1} * F)),
  explained: pca2(run.act[run.act.length - 1], run.T, net.layers[0].out).explained,
  roles: Array.from(neuronRoles(run.act[0], run.T, net.layers[0].out, 1).role)}}));
"""
        page = json.loads(subprocess.run(["node"], input=harness, capture_output=True, text=True, check=True).stdout)

        predictor = Predictor(MODEL / "best.pt")
        times = np.linspace(0, 5, 101)
        np.testing.assert_allclose(page["coefficients"], EQUATION.coefficients, atol=1e-12)
        np.testing.assert_allclose(page["exact"], linear_reference(EQUATION, times), atol=1e-10)
        expected = predictor.predict(EQUATION, times)
        np.testing.assert_allclose(page["out"], expected, atol=1e-5 * np.abs(expected).max())
        self.assertLess(page["check"]["models"][0], 1e-5)

        # d(y in equation units)/d(normalized inputs) by PyTorch autograd at one time.
        x = torch.tensor(page["x"], dtype=torch.float32, requires_grad=True)
        (predictor.model(x)[0] * float(predictor.normalizer.target_scale[0])).backward()
        np.testing.assert_allclose(page["grad"], x.grad.numpy(), atol=1e-4 * float(x.grad.abs().max()))
        self.assertTrue(0 < page["explained"] <= 1)
        self.assertTrue(set(page["roles"]) <= set(range(6)))

    def test_reel_rebuilds_worst_cases_from_their_ids(self):
        """Cases come back worst first and must match the evaluated checkpoint and errors."""
        checkpoint = MODEL / "best.pt"
        predictor = Predictor(checkpoint)
        config = GeneratorConfig.from_dict(predictor.checkpoint["config"])
        times = np.linspace(0, 5, 101)
        rows = []
        for identity in ("validation-o2-linear_homogeneous-g0-i0", "validation-o3-linear_sinusoidal-g1-i2"):
            equation = _equation_for(identity, config)
            rows.append({"trajectory_id": identity, **trajectory_metrics(
                linear_reference(equation, times), predictor.predict(equation, times))})
        summary = {"checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
                   "splits": {"validation": {"trajectories": rows}}}
        with TemporaryDirectory() as folder:
            path = Path(folder) / "evaluation.json"
            path.write_text(json.dumps(summary))
            reel = worst_cases(predictor, checkpoint, path, 5)
            self.assertEqual([case["id"] for case in reel["cases"]],
                             [row["trajectory_id"] for row in sorted(rows, key=lambda r: -r["relative_l2"])])
            for broken in ({"checkpoint_sha256": "0" * 64}, {"splits": {"validation": {"trajectories": [
                    {**rows[0], "relative_l2": rows[0]["relative_l2"] * 2}]}}}):
                path.write_text(json.dumps({**summary, **broken}))
                with self.assertRaises(ValueError):
                    worst_cases(predictor, checkpoint, path, 5)


if __name__ == "__main__":
    unittest.main()
