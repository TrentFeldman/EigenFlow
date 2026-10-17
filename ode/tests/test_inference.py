"""Verify model outputs, masked loss and independent checkpoint inference.

The tests check gradient masking, mixed-order input dimensions, repeatable
checkpoint loading and CPU/CUDA selection. CLI checks protect dry-run behavior.
"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ode.training.dataset import Normalizer, feature_width
from ode.generation.equations import LinearODE
from ode.training.model import OrderConditionedMLP, masked_mse
from ode.evaluation.predict import Predictor, predict
from ode.generation.analytic import linear_reference
from ode.training.train import main, resolve_device


class InferenceContractTests(unittest.TestCase):
    """Check padding, gradients and independent weight loading."""

    def test_masked_loss_ignores_padding_and_balances_orders(self):
        """Check equal example weighting and zero gradients for padded outputs."""
        prediction = torch.tensor([[1., 1., 100.], [2., 2., 2.]], requires_grad=True)
        mask = torch.tensor([[1., 1., 0.], [1., 1., 1.]])
        loss = masked_mse(prediction, torch.zeros_like(prediction), mask)
        self.assertEqual(float(loss.detach()), 2.5)
        loss.backward()
        self.assertEqual(float(prediction.grad[0, 2]), 0)

    def test_mixed_order_model_shape(self):
        """Check the shared input width and padded output dimensions."""
        model = OrderConditionedMLP(max_order=8, width=16, depth=2)
        self.assertEqual(model(torch.zeros(3, feature_width(8))).shape, (3, 8))

    def test_checkpoint_round_trip_without_training(self):
        """Check repeatable inference and rejection of orders exceeding checkpoint M."""
        model = OrderConditionedMLP(4, width=8, depth=1)
        scaler = Normalizer(
            np.zeros(feature_width(4)),
            np.ones(feature_width(4)),
            np.zeros(4),
            np.ones(4),
        )
        checkpoint = {
            "schema_version": 1,
            "encoding": "linear-padded-v1",
            "architecture": model.architecture,
            "state_dict": model.state_dict(),
            "normalizer": scaler.to_dict(),
        }

        with TemporaryDirectory() as directory:
            path = Path(directory) / "untrained-fixture.pt"
            torch.save(checkpoint, path)
            equation = LinearODE((2., 3.), (1., 0.))
            first = predict(path, equation, [0, 0.5])
            second = predict(path, equation, [0, 0.5])
            self.assertEqual(first.shape, (2, 2))
            np.testing.assert_array_equal(first, second)

            with self.assertRaises(ValueError):
                predict(path, LinearODE((1.,) * 5, (0.,) * 5), [0])

    def test_rescaled_prediction_adds_no_error_of_its_own(self):
        """With an exact 'model', rescaling, superposition and stitching must stay exact."""
        predictor = Predictor.__new__(Predictor)
        predictor.checkpoint = {"config": {
            "orders": [1, 2, 3, 4], "families": ["linear_homogeneous", "linear_sinusoidal"],
            "decay_range": [0.1, 1.0], "frequency_range": [0.5, 2.0], "t_start": 0.0, "t_stop": 5.0}}
        predictor.predict = linear_reference
        times = np.linspace(0, 20, 201)
        for equation in [LinearODE((39.478418, 0.628319), (1., 0.)),  # 1 Hz spring, 3x too fast
                         LinearODE((1.5, 2.4, 2.1), (4., -3., 9.), 7., 3., 2.5)]:  # 3rd order, big push
            states, pieces = predictor.predict_rescaled(equation, times)
            self.assertGreater(pieces, 1)
            np.testing.assert_allclose(states, linear_reference(equation, times), rtol=1e-7, atol=1e-7)
        # First and fourth order (one real mode; two oscillating pairs) rescale exactly too.
        for equation in [LinearODE((8.,), (3.,), 2., 5., 1.),
                         LinearODE((float(x) for x in np.real(np.poly([-.3+1j, -.3-1j, -.5+3j, -.5-3j]))[:0:-1]),
                                   (1., 0., 0., 0.))]:
            states, pieces = predictor.predict_rescaled(equation, times)
            np.testing.assert_allclose(states, linear_reference(equation, times), rtol=1e-6, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "oscillating pair"):
            predictor.predict_rescaled(LinearODE((2., 3.), (1., 0.)), times)  # overdamped

    def test_training_dry_run_has_no_side_effects(self):
        """Verify default training CLI execution does not load data or run epochs."""
        with patch("ode.training.train.TrajectoryDataset") as dataset, patch("ode.training.train.run_epoch") as epoch:
            with redirect_stdout(StringIO()):
                self.assertEqual(main([]), 0)

            dataset.assert_not_called()
            epoch.assert_not_called()

    def test_device_selection_without_gpu_execution(self):
        """Check explicit and automatic device choices for either availability state."""
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

        with patch("ode.training.train.torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))

            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                resolve_device("cuda")

        with patch("ode.training.train.torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device("cuda"), torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
