"""Model/serialization checks with synthetic tensors and untrained weights.

These checks never invoke the training loop or numerical data generator.
The temporary checkpoint is a format fixture, not a trained model artifact.
"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ode.dataset import Normalizer
from ode.equations import LinearODE
from ode.model import OrderConditionedMLP, masked_mse
from ode.predict import predict
from ode.train import main, resolve_device


class InferenceContractTests(unittest.TestCase):
    """Check padding, gradients and independent weight loading."""

    def test_masked_loss_ignores_padding_and_balances_orders(self):
        prediction = torch.tensor([[1., 1., 100.], [2., 2., 2.]], requires_grad=True)
        mask = torch.tensor([[1., 1., 0.], [1., 1., 1.]])
        loss = masked_mse(prediction, torch.zeros_like(prediction), mask)
        self.assertEqual(float(loss.detach()), 2.5)
        loss.backward()
        self.assertEqual(float(prediction.grad[0, 2]), 0)

    def test_mixed_order_model_shape(self):
        model = OrderConditionedMLP(max_order=8, width=16, depth=2)
        self.assertEqual(model(torch.zeros(3, 29)).shape, (3, 8))

    def test_checkpoint_round_trip_without_training(self):
        model = OrderConditionedMLP(4, width=8, depth=1)
        scaler = Normalizer(np.zeros(17), np.ones(17), np.zeros(4), np.ones(4))
        checkpoint = {"schema_version": 1, "encoding": "linear-padded-v1",
                      "architecture": model.architecture, "state_dict": model.state_dict(),
                      "normalizer": scaler.to_dict()}
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

    def test_training_dry_run_has_no_side_effects(self):
        with patch("ode.train.TrajectoryDataset") as dataset, patch("ode.train.run_epoch") as epoch:
            with redirect_stdout(StringIO()):
                self.assertEqual(main([]), 0)
            dataset.assert_not_called()
            epoch.assert_not_called()

    def test_device_selection_without_gpu_execution(self):
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))
        with patch("ode.train.torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                resolve_device("cuda")
        with patch("ode.train.torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device("cuda"), torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
