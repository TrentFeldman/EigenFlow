"""EigenFlow tools for generating ODE references, training and prediction.

Pipeline: config -> generate_data (equations/reference/analytic/validate_reference)
-> dataset -> train (model) -> predict. T is time-point count, M is max_order
(padding width), B is batch size, n is equation order; state=[y, y', ..., y^(n-1)].
All values are nondimensional. See MATH_GUIDE.md for math and README.md for usage.
"""
