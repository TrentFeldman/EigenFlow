# Decision register

## Established constraints

- October ODE delivery is the critical path; fluid implementation follows it.
- Use supervised PyTorch trajectory prediction for the initial ODE baseline.
- Keep new work in root-level track folders and preserve the experimental archive.
- The initial scaffold has progressed to a documented ODE framework. Actual reference generation and model training remain deferred by the current request.

## October 15 implementation decisions

- Represent scalar higher-order ODEs as states ordered `[y, y', ..., y^(n-1)]`. The solver path accepts arbitrary positive orders; scientific validation is still required for each supported regime.
- Begin with normalized constant-coefficient linear equations, with optional sinusoidal forcing. Sample decaying real/complex-conjugate modes to obtain stable unforced coefficient sets. This does not imply coverage of arbitrary nonlinear equations.
- Use a shared configurable masked-order MLP as the first framework baseline, superseding the earlier proposal for separate per-family models. `max_order` fixes encoding/checkpoint size; supported input orders do not establish learned order generalization.
- Use a deterministic seeded sampler and bounded threaded execution, with serial and spawned-process alternatives. Keep file writing in the coordinator. No throughput benchmark has run.
- Default to DOP853 with a second tighter-tolerance solve. Analytic benchmarks are pending; tolerances/refinement agreement alone do not establish reference correctness.
- Use joint parameter/initial-condition holdout partitions now. Separate parameter-only and initial-condition-only diagnostic sets remain planned.
- Reuse the existing isolated `.venv`; add SciPy while retaining the installed NumPy and PyTorch stack. Default training to CPU and record the tested environment.
- Add team documentation rules to `AGENTS.md` and explain all scaffold modules, math symbols, shapes and limitations for both collaborators.

## Open decisions for October 18

| Decision | Proposed starting point | Must resolve before reference generation/release |
|---|---|---|
| Second-order family | Damped linear oscillator | Forcing policy, coefficient bounds, initial-condition bounds |
| Third-order family | Stable linear constant-coefficient system | Parameterization, stability constraints, coefficient bounds |
| Model arrangement | Shared masked-order MLP framework | Real-data adequacy versus family-specific baselines |
| Prediction mode | Initial conditions and elapsed time to full state | Time horizon, sampling density, extrapolation policy |
| Numerical reference | Configurable adaptive integration plus refinement | Analytic checks and scientific validation of draft tolerances |
| Dataset | Complete trajectories grouped by configuration | Counts, seeds, storage format and partition proportions |
| Accuracy | Per-trajectory relative L2 with aggregate reporting | Primary output, scaling, denominator floor and near-zero reporting |
| Compute budget | Small reproducible baseline first | Available hardware, maximum runtime and initial dataset size |

Example configuration values are draft experiment choices, not finalized scientific specifications. Record accepted choices with date, rationale, alternatives considered and the evidence that would justify revisiting them. Implementation decisions above can be revised when measured evidence warrants it.
