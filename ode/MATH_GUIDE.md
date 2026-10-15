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

Each future trajectory will be solved twice, with the second solve using tighter tolerances. The code compares them separately for each derivative. The allowed L2 difference is `reference_atol * sqrt(number of times) + reference_rtol * norm(refined component)`. A reported refinement ratio of 1 is the acceptance boundary; smaller is better. This ratio is **not** the model's relative trajectory error. Analytic checks and convergence experiments remain necessary before reference data is scientifically validated. No real solver benchmark has been run yet.

The default DOP853 method is intended for high-accuracy non-stiff problems. A stiff problem has time scales so different that an explicit solver may need extremely small steps; Radau or BDF are configurable alternatives. See [SciPy's method and tolerance descriptions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

## How does one model accept different orders?

Set a maximum supported order, for example 6. A second-order initial state `[y, y']` is padded to `[y, y', 0, 0, 0, 0]`, and its mask is `[1, 1, 0, 0, 0, 0]`. A mask distinguishes a missing derivative from a real derivative whose value happens to be zero.

The model receives padded coefficients, padded initial values, that mask, forcing information and elapsed time. It predicts padded state values. Training ignores padding when computing loss and normalization. The output returned to a caller includes only the equation's actual derivatives.

Normalization rescales values using means and standard deviations measured on training data only. The checkpoint stores these values so inference uses exactly the same scaling. Per-example loss is averaged over real derivatives, preventing higher-order examples from automatically receiving more weight.

Supporting an order in the input format does not mean the model learned it. A model trained only on orders 2 and 3 has no demonstrated accuracy on order 4. Generalization across parameters, initial conditions, equation families and orders are separate questions that require separate held-out evaluations.

## Why keep full trajectories together?

Randomly assigning individual time points to train and test would expose nearly the same curve during training and evaluation. That can make apparent accuracy misleading. We assign whole equation-parameter groups to a split first; all their initial-condition trajectories stay there. Current partitions are a joint holdout of parameters and initial states. Separate parameter-only and initial-condition-only diagnostic tests remain future work.
