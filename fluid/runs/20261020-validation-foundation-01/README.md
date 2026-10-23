# Validation foundation — October 20, 2026

Question: do the analytical reference, periodic diagnostics and rollout checking
harness behave correctly before a numerical solver is implemented?

Dataset: deterministic constant fields, exact cell-center Taylor–Green samples,
analytical pressure gauge shifts, and deliberately corrupted/frozen test-double
rollouts. No external data, model, training, test split or RNG is used.

Configuration/environment are in `environment.json`; full measured values and
status are in `checks.txt` and `status.txt`. Compiler GCC 16.2.1, C++17, Release.
Both Debug/Release CTest runs passed all eight suites, including existing tests.

| Grid | Curl relative L2 vs analytic | Observed order |
|---|---:|---:|
| 16² | 0.02550464 | — |
| 32² | 0.00641315 | 1.9917 |
| 64² | 0.00160561 | 1.9979 |
| 128² | 0.00040155 | 1.9995 |

Analytical time t=0.5 s. Maximum divergence remains below 2.6e-14 1/s.
Energy error is below 9e-17 relative. These are sampled exact fields, not a
numerical solver's output. Second-order convergence here belongs to the
centered vorticity diagnostic only.

The independent continuous momentum residual check passed 1e-6 m/s² tolerance
at three points using finite differences with spatial/time steps 1e-4 in SI
units. Known cell-center values anchor reference signs and phase. Constant-flow
mass, energy and Courant/diffusion quantities match their known values.
Pressure offset is ignored while spatial pressure and velocity errors are found.
Nonfinite fields, nonuniform/invalid material, invalid geometry/reference inputs
are rejected. The rollout harness rejects frozen flow and an intermediate
error even when later frames are exact.

Expected failure cases: frozen and transiently damaged test doubles fail solver
acceptance gates. They are negative controls, not unexpected test failures.
Unexpected test failures: none.

Limits: no Navier–Stokes integration method or pressure projection exists.
Analytical/reference tests cannot establish numerical solver accuracy or
stability. Centered divergence can miss checkerboard modes; the future solver
must also supply compatible residual diagnostics. See the validation protocol
for tolerances and the pending spatial/time convergence matrix.

Decision: keep the validation foundation. Numerical solver validation remains
NOT RUN until the real FluidStepper is implemented and passes that protocol.

Reproduce from fluid/:

```bash
cmake --build build/release -j
./build/release/fluid_validation_tests
ctest --test-dir build/release --output-on-failure
```

Protocol: [numerical validation](../../docs/numerical-validation.md).
