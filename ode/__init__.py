"""EigenFlow ODE track: numerical reference data, a neural predictor, and its evaluation.

Run `python -m ode` from the repo root for the command list. Code is grouped by
stage: generation/ (equations -> solved datasets), training/ (dataset -> model
inputs -> trained network), evaluation/ (metrics, charts, prediction), with
config.py (scientific recipe) and resources.py (hardware settings) shared by all.
Symbols: T time points, M max_order (padding width), B batch size, n equation
order; state = [y, y', ..., y^(n-1)]. All values are nondimensional. See
MATH_GUIDE.md for the math and README.md for usage.
"""
