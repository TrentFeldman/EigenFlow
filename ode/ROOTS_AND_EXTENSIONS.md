# Real roots, repeated roots, Euler-Cauchy and nonhomogeneous equations

This note records the explanation using plain ASCII math. A prime means a
derivative with respect to the independent variable: y' is the first derivative,
y'' the second, and y^(n) the nth. C1 and C2 are independent constants determined
by the initial conditions. exp(z) means e raised to the power z.

## Real and repeated roots

The numerical solver can handle distinct and repeated real roots, but the
current random sampler does not deliberately cover those cases. The neural
network has not been trained yet, so we cannot claim it has learned them.

For a double characteristic root lambda, the analytical solution is:

```text
y(t) = (C1 + C2*t) * exp(lambda*t)
```

The second constant is C2, rather than another occurrence of C1. For a root
with multiplicity m, the polynomial factor extends through t^(m-1):

```text
y_root(t) = exp(lambda*t) * (C0 + C1*t + ... + C_(m-1)*t^(m-1))
```

The numerical solver does not need us to insert that extra t explicitly.
We supply the equation and initial conditions. For example:

```text
Equation:       y'' + 3*y' + 2*y = 0
Roots:          -1 and -2
Solution form:  C1*exp(-t) + C2*exp(-2*t)

Equation:       y'' + 2*y' + y = 0
Roots:          -1, repeated twice
Solution form:  (C1 + C2*t)*exp(-t)
```

Our LinearODE representation accepts either equation. Numerical integration
follows its derivative rules, producing the corresponding behavior without
constructing a symbolic solution.

The distinction is what we choose to generate:

- Current second-order samples always use a complex-conjugate pair with
  negative real parts. These are damped oscillations, not purely imaginary roots.
- Current third-order samples use that pair plus one negative real root.
- We have not added deliberate sampling of all-real, repeated or near-repeated
  roots.

The network receives coefficients, rather than a prescribed exponential or
sinusoidal solution formula. It has no hardcoded t*exp(lambda*t) term. We need
representative training examples and separate tests for these root cases
before expecting reliable predictions.

Update: numerical-reference benchmarks now test distinct/repeated real roots,
complex roots and resonant forcing, including higher-order repeated roots.
They check the solver, not learned predictions, and do not change the random
sampler's distribution. See README.md and validate_reference.py.

## Euler-Cauchy equations

Euler-Cauchy support is possible, but the current implementation is restricted
to constant coefficients. One extension route is a change of variable.

For x > 0, define:

```text
s = ln(x)
u(s) = y(exp(s))
```

Here ln is the natural logarithm. The chain rule gives:

```text
x*y'     = u_s
x^2*y''  = u_ss - u_s
```

The subscripts u_s and u_ss mean derivatives with respect to s. Consequently:

```text
Original equation:
    x^2*y'' + a*x*y' + b*y = f(x)

Transformed equation:
    u_ss + (a - 1)*u_s + b*u = f(exp(s))
```

The transformed equation has constant coefficients. For its repeated root
lambda, transforming back gives the homogeneous solution form:

```text
y(x) = x^lambda * (C1 + C2*ln(x))
```

For a numerical implementation, we must also transform the initial derivatives
and outputs, transform the forcing, and keep the original x-domain away from
zero. For example, at x0 > 0:

```text
s0 = ln(x0)
u(s0) = y(x0)
u_s(s0) = x0*y'(x0)

To recover derivatives:
    y'  = u_s / x
    y'' = (u_ss - u_s) / x^2
```

Our current model does not perform these transformations. Euler-Cauchy
support is not implemented yet.

Source: [Kansas State University's Euler equation derivation](https://onlinehw.math.ksu.edu/math340book/chap4/eulerrev.php).

## Nonhomogeneous equations

Nonhomogeneous equations already have limited support. The current code includes:

```text
y^(n) + a[0]*y + a[1]*y' + ... + a[n-1]*y^(n-1)
    = A*sin(omega*tau + phi)

tau = t - t_start
```

A is amplitude, omega is angular frequency, phi is phase, and tau is elapsed
time since the initial condition. Setting A = 0 gives the homogeneous case;
nonzero A gives sinusoidal forcing.

The numerical solver handles the full response without manually separating
complementary and particular solutions.

To support other forcing, such as polynomials, exponentials or combinations,
we would extend both the forcing calculation and the model's input
representation, then generate and validate examples covering those additions.

No dataset generation or model training is performed by this note.
