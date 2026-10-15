"""Pending: repeatedly feed predicted fluid states back into the predictor.

This is an autoregressive rollout: each prediction becomes the next input, so
errors can accumulate. Explain any correction for nonzero divergence and
measure stability independently of visual smoothness. No rollout exists yet.
"""
