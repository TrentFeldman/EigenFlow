# Numerical validation foundation

`Validation.h/.cpp` establishes diagnostics, an exact periodic reference and a
rollout evaluator. **The numerical CFD solver is not implemented or validated.**
Current tests validate the reference formulas and checking tools. Their exact
formula stepper is a test double and supplies no evidence of integration accuracy.

## Reference problem and symbols

Use an unforced, constant-density, constant-viscosity Taylor–Green vortex on a
square periodic domain. The default side length is `L=2 m`, initial amplitude
`U=1 m/s`, density `rho=1 kg/m^3`, viscosity `nu=0.01 m^2/s`, and mode `m=1`.
Mode means the number of full spatial periods in each axis. Let
`k=2*pi*m/L` (radians per metre), `X=x-origin_x`, `Y=y-origin_y`,
and `A=U*exp(-2*nu*k*k*t)` where `t` is time in seconds. The exact fields are:

```text
u =  A * sin(k*X) * cos(k*Y)                       horizontal velocity, m/s
v = -A * cos(k*X) * sin(k*Y)                       vertical velocity, m/s
p = rho*A*A/4 * (cos(2*k*X) + cos(2*k*Y))          gauge pressure, Pa
omega = 2*k*A * sin(k*X) * sin(k*Y)                scalar vorticity, 1/s
E = A*A/4                                        mean kinetic energy/mass, m^2/s^2
```

Both forces are zero. `fill_taylor_green()` samples cell centers and resets every
field. It requires a square physical domain and at least four cells per
wavelength. Translation of the domain changes its origin, not the relative
phase. This exact reference differs from the existing solid-rotation visual
`vortex` preset and is not added to that preset enum.

The nonlinear acceleration is balanced by `-grad(p)/rho`; viscous diffusion
provides the exponential decay. The pressure sign above follows that balance.
The periodic pressure has zero spatial mean, but a numerical pressure field can
have any constant offset. Reference comparisons remove that offset.

The velocity parameterization/decay follows the
[solver author's Taylor–Green example](https://computational-chemical-engineering.github.io/peclet-examples/examples/taylor-green/index.html).
The pressure and vorticity here follow by differentiating the displayed velocity;
tests independently check both components of the continuous momentum balance.
[Burkardt's exact Navier–Stokes examples](https://people.sc.fsu.edu/~jburkardt/f_src/navier_stokes_2d_exact/navier_stokes_2d_exact.html)
provide another description of the time-dependent, unforced Taylor solution.

## Diagnostics and assumptions

`diagnose_flow(grid, dt)` checks every field is finite and checks that density
is positive/uniform and viscosity is nonnegative/uniform. Relative uniformity
tolerance is `1e-12`. The grid needs at least three cells per axis. It reports:

- Divergence `du/dx + dv/dy`, RMS and maximum absolute value, in 1/s.
- Vorticity `dv/dx - du/dy`, RMS in 1/s. `vorticity_field()` returns the entire
  derived field as a flat `[y*width+x]` array.
- Maximum speed, mean velocity components and mean kinetic energy per mass.
- Mass per unit depth `rho*length_x*length_y` (kg/m). Since density is stored
  explicitly and not evolved yet, constant mass alone proves little about flow.
- Advective Courant indicator `max(dt*(abs(u)/dx+abs(v)/dy))` and diffusion
  indicator `nu*dt*(1/dx^2+1/dy^2)`, both dimensionless.

Derivatives use the second-order centered stencil: `du/dx` means the right cell's
horizontal velocity minus the left cell's, divided by `2*dx`; vertical
stencils use up/down and `2*dy`. Neighbor indexing wraps periodically.
These are independent diagnostic stencils, not a commitment to the solver's
operators. On nonsmooth data or periodic seam jumps, these are stencil results,
not evidence of a differentiable physical solution.

Centered cell-centered derivatives miss alternating cell-to-cell checkerboard
modes. A small reported divergence cannot alone establish incompressibility;
the future solver must also report its own compatible divergence/pressure
residual and check for these modes. Vorticity has diagnostic truncation error
even on an exactly sampled reference. Courant/diffusion values are indicators;
no general stability guarantee is assigned before the integration method exists.
Energy monotonicity is appropriate for this unforced benchmark, not arbitrary
externally forced flows.

Invalid inputs throw `invalid_argument`, unrepresentable diagnostics throw
`overflow_error`. A zero reference norm produces zero relative error only for
zero absolute error, otherwise infinity. Long-decayed fields need absolute-error
reporting as well as relative error.

## Rollout evaluator and proposed acceptance gates

When a real `FluidStepper` exists:

```cpp
#include "Validation.h"

// stepper must implement the actual Navier-Stokes update.
auto result = validate_taylor_green_rollout(stepper, 64, 0.001, 1000);
```

This initializes the exact t=0 state, checks the initial frame and every subsequent
frame up to t=1 s, and returns final diagnostics, worst observed errors and
`passed`. A bad intermediate frame remains a failure even if the final frame
looks correct. Step failures/invalid data propagate; exceeding limits returns
`passed=false`. It also enforces the benchmark's viscosity and zero forcing.
Defaults below are provisional solver acceptance targets, not measured passes
and not the separate 5% ML prediction target:

| Quantity | Proposed gate |
|---|---:|
| Relative velocity L2, `sqrt(sum(du^2+dv^2)/sum(u_exact^2+v_exact^2))` | <= 1% |
| Maximum divergence scaled by `L/U` | <= 1e-6 |
| Pressure RMS after gauge removal, divided by `rho*U^2` | <= 0.02 |
| Discrete vorticity vs exact-vorticity relative L2 | <= 2% |
| Relative mean kinetic energy error | <= 2% |
| Relative mass per depth error | <= 1e-12 |

`ValidationLimits` allows explicit overrides; record every override and its
reason in the run. The current harness accepts the exact formula test double
and rejects frozen velocity and a transiently damaged field. This tests the
harness only. Full solver validation still requires:

1. Run the actual stepper at the default case above; record worst errors and
   failing times/cases, not just final fields. Add still/uniform-flow preservation.
2. Spatial refinement: 32², 64², 128² at matched physical time; reduce dt enough
   that time error does not dominate. Require error decreases and matches the
   chosen method's claimed spatial order. The current diagnostic curl stencil
   independently demonstrates second order, not solver convergence.
3. Temporal refinement: 64² or finer at dt=0.002, 0.001, 0.0005 to t=1 s, provided
   stable for the chosen method. Isolate temporal error from spatial error and
   require the chosen integrator's order; document any error floor.
4. Repeat at several viscosities and smooth periodic initial conditions. Record
   Courant/diffusion indicators, pressure residual, finite-state failures, mean
   momentum drift and unforced energy increase. Do not weaken tolerances merely
   to make a failed run pass.

Reference accuracy is not established until those actual numerical rollouts
pass. This task implements the diagnostics/reference/evaluator, not the CFD step.

## Checks and evidence

```bash
cmake --build build/debug -j
ctest --test-dir build/debug --output-on-failure
cmake --build build/release -j
ctest --test-dir build/release --output-on-failure
./build/release/fluid_validation_tests
```

The new suite checks constant fields, invalid states, pressure gauge invariance,
known reference values, continuous momentum residuals, decay, curl convergence,
and harness rejection of incorrect/transient updates. All eight suites pass in
both Debug and Release. Recorded evidence lives in
[the foundation run](../runs/20261020-validation-foundation-01/README.md).
