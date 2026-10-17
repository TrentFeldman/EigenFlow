# ODE math for the coding workflow

This guide assumes Python knowledge, not prior differential-equation coursework. The same definitions apply to the code, configuration and tests.

## What are we predicting?

An ordinary differential equation (ODE) relates a quantity to how it changes over one independent variable, here time. We write the quantity as `y(t)`. A prime means a time derivative: `y'` is the rate of change of `y`, and `y''` is the rate of change of `y'`. For position, these would be velocity and acceleration. Our initial dataset uses nondimensional values, so do not assume meters or seconds.

The **order** is the highest derivative in the equation. A second-order equation needs two starting values, `y(0)` and `y'(0)`. A third-order equation needs three, adding `y''(0)`. An order-n equation needs n starting values. Order is different from the numerical solver's accuracy order or the neural network's depth.

## Why turn a higher-order equation into a state vector?

A numerical solver expects a function that receives the current state and returns its rate of change. Consider:

`y'' + 3*y' + 2*y = 0`

We store `state = [y, y']`. Rearranging the equation gives `y'' = -2*y - 3*y'`. Therefore the rate of change of the whole state is:

`state_derivative = [state[1], -2*state[0] - 3*state[1]]`

This is the job of `LinearODE.rhs`. `rhs` means “right-hand side” of the first-order state equation. The coefficients in our code are `(2, 3)`, in increasing derivative order.

For `y''' + 6*y'' + 11*y' + 6*y = 0`, the state is `[y, y', y'']`, coefficients are `(6, 11, 6)`, and its derivative is `[y', y'', -6*y - 11*y' - 6*y'']`. The same shift-and-combine rule extends to higher orders without a separate solver for each one.

## Forcing and stability

Forcing is an external input driving the system. We currently support `amplitude * sin(frequency * elapsed_time + phase)`. Amplitude controls strength; frequency controls oscillation rate in radians per nondimensional time unit; phase controls where the cycle begins. Amplitude zero means no forcing, also called a homogeneous equation.

We sample coefficients by first choosing decaying modes and then multiplying their polynomial factors. A mode describes a basic exponential decay or damped oscillation in the solution. This biases the first experiments toward stable unforced systems. It does not cover every stable system or guarantee small intermediate values. Higher order, widely separated time scales and large coefficients can make numerical solving difficult even when the unforced solution eventually decays.

## What does “accurate reference” mean?

The numerical trajectory is the answer we train against. `rtol` is a relative local-error tolerance and `atol` is an absolute tolerance for values near zero. Smaller settings request tighter numerical control; they are not guarantees of a global trajectory error.

Each trajectory is solved twice, with the second solve using tighter tolerances. The code compares them separately for each derivative. The allowed L2 difference is `reference_atol * sqrt(number of times) + reference_rtol * norm(refined component)`. A reported refinement ratio of 1 is the acceptance boundary; smaller is better. This ratio is **not** the model's relative trajectory error.

Each trajectory is also checked at selected times using a matrix exponential: a separate way to evolve a constant linear system directly, rather than taking adaptive steps. Sine forcing becomes two extra internal states, sine and cosine, so this approach checks both homogeneous and nonhomogeneous equations, including resonance. Its pointwise allowance is `reference_atol + reference_rtol * abs(reference component)`. This is a floating-point numerical check, not symbolic exact arithmetic or a proof between the selected times. See [SciPy's matrix-exponential documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html).

Before any dataset is written, `validate_reference.py` compares both numerical approaches with derivatives of explicit closed-form solutions. The current example passes 13 small cases covering distinct/repeated roots, damped oscillations, orders 2/3/6 and resonant forcing. These checks validate those cases; they do not certify every coefficient range or the neural model's accuracy.

The default DOP853 method is intended for high-accuracy non-stiff problems. A stiff problem has time scales so different that an explicit solver may need extremely small steps; Radau or BDF are configurable alternatives. See [SciPy's method and tolerance descriptions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

## How does one model accept different orders?

Set a maximum supported order, for example 6. A second-order initial state `[y, y']` is padded to `[y, y', 0, 0, 0, 0]`, and its mask is `[1, 1, 0, 0, 0, 0]`. A mask distinguishes a missing derivative from a real derivative whose value happens to be zero.

The model receives padded coefficients, padded initial values, that mask, forcing information and elapsed time. It predicts padded state values. Training ignores padding when computing loss and normalization. The output returned to a caller includes only the equation's actual derivatives.

Normalization rescales values using means and standard deviations measured on training data only. The checkpoint stores these values so inference uses exactly the same scaling. Per-example loss is averaged over real derivatives, preventing higher-order examples from automatically receiving more weight.

Supporting an order in the input format does not mean the model learned it. A model trained only on orders 2 and 3 has no demonstrated accuracy on order 4. Generalization across parameters, initial conditions, equation families and orders are separate questions that require separate held-out evaluations.

## Why keep full trajectories together?

Randomly assigning individual time points to train and test would expose nearly the same curve during training and evaluation. That can make apparent accuracy misleading. We assign whole equation-parameter groups to a split first; all their initial-condition trajectories stay there. Current partitions are a joint holdout of parameters and initial states. The pilot additionally crosses validation parameters with training ICs, and training parameters with validation ICs, to diagnose each source of generalization error. These diagnostics deliberately share one identity with training and never use test identities.


## The frozen neural error metric

For one trajectory, subtract reference from prediction, square every real state
component at every saved time, average, and take the square root. This is the
absolute root-mean-square error E. Compute R the same way from the reference
itself. The primary error is `E / max(R, 0.000001)`. For example, reference
`[[1,2],[1,2]]` and prediction `[[2,2],[2,2]]` give `E=sqrt(1/2)` and
`R=sqrt(5/2)`, so relative L2 is `1/sqrt(5)`, about 44.7%. The first component
is 100% wrong and the second is exact; the report includes those breakdowns.

All time samples (including endpoints) have equal weights. Nondimensional
components have unit scales, with no centering or training standard deviation
in this metric. The denominator floor keeps a zero reference from dividing by
zero: predicting `2e-7` for an all-zero trajectory gives absolute RMSE `2e-7`
and floored relative error 0.2. Interpret the absolute error for such cases.

Average per-trajectory errors with equal weights; never combine all trajectories
into one norm. Report tails and failures as well as the mean. A nonfinite
prediction prevents passing, and the report labels finite-only summaries.
The <5% release target applies to the reserved joint test split, while validation
chooses checkpoints. Standardized training MSE is a different quantity.
