"""Verify physical-time derivatives independently of the neural baseline weights."""

from contextlib import redirect_stderr
from io import StringIO
import unittest

import numpy as np
import torch

from ode.training.dataset import Normalizer, feature_width
from ode.evaluation.diagnose import main, state_and_time_derivative
from ode.generation.equations import LinearODE


class ExactExponential(torch.nn.Module):
    """Represent y=exp(-t), y'=-exp(-t) in deliberately non-unit scaled coordinates."""
    max_order = 2

    def forward(self, features):
        time = features[:, -1]*3+2
        physical = torch.stack((torch.exp(-time), -torch.exp(-time)), dim=1)
        return (physical-torch.tensor([.3,-.4]))/torch.tensor([2.,4.])


class DiagnosticTests(unittest.TestCase):
    """Ensure normalization and separate derivative outputs cannot mask inconsistency."""

    def test_derivatives_include_input_and_output_scaling(self):
        width = feature_width(2)
        mean, scale = np.zeros(width), np.ones(width)
        mean[-1], scale[-1] = 2,3
        normalizer = Normalizer(mean,scale,[.3,-.4],[2.,4.])
        equation = LinearODE((1.,2.),(1.,-1.))
        times = np.array([0.,.2,1.,2.])
        state, derivative = state_and_time_derivative(ExactExponential(),normalizer,equation,times)
        expected = np.stack((np.exp(-times),-np.exp(-times)),axis=1)
        np.testing.assert_allclose(state,expected,rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(derivative,-expected,rtol=1e-6,atol=1e-7)
        rhs = np.stack([equation.rhs(time,row) for time,row in zip(times,state)])
        np.testing.assert_allclose(derivative,rhs,rtol=1e-6,atol=1e-7)

    def test_spent_test_split_is_excluded_from_cli(self):
        with redirect_stderr(StringIO()),self.assertRaises(SystemExit):
            main(['--checkpoint','unused.pt','--data','unused','--output','unused.json','--split','test'])


if __name__=='__main__':
    unittest.main()
