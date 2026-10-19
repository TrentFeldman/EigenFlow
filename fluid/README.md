# Fluid prototype

Status: deferred implementation until the October ODE release. Target: working demo December 4 and evaluated release December 11, 2026.

Scope: two-dimensional incompressible flow, periodic boundaries, initial 64x64 grid, configurable viscosity and initial velocity fields. Domain size, discretization, time step, viscosity bounds and validation tolerances remain open scientific decisions.

## Reserved files

| Documentation-only file | Future responsibility |
|---|---|
| `solver.py` | Numerical Navier-Stokes reference solver |
| `validate_solver.py` | Taylor-Green or benchmark validation and refinement studies |
| `generate_data.py` | Trajectories, provenance and grouped partitions |
| `dataset.py` | Field loading, time-step pairs and training-only preprocessing |
| `model.py` | Initial U-Net-style single-step predictor |
| `train.py` | Reproducible training and validation |
| `predict.py` | Independent checkpoint-based field prediction |
| `rollout.py` | Repeated prediction and documented stabilization |
| `evaluate.py` | Short- and long-horizon errors and physical diagnostics |
| `visualize.py` | Reference, prediction and error comparisons |
| `tests/test_reference.py` | Future benchmark, divergence and refinement checks |
| `tests/test_rollout.py` | Future horizon, stability and saved-model checks |

`configs/`, ignored `data/` and `runs/` reserve settings, generated fields and run artifacts, respectively.

## Stage gates

| Due | Gate and evidence |
|---|---|
| Nov 6 | Reference solver agrees with a benchmark; spatial/time refinement and divergence are measured |
| Nov 13 | Versioned trajectories include velocity, vorticity, viscosity, grid/domain, times, initial-condition IDs and solver settings; partitions isolate configurations |
| Nov 20 | Saved single-step estimator beats documented simple baselines on validation data; held-out errors are reported |
| Nov 27 | Autoregressive tests report error versus horizon, instability rate and the effects of stabilization |
| Dec 4 | Reproducible reference/prediction/error visualization runs end to end |
| Dec 11 | Released artifacts, independent inference, quantitative comparisons and limitations pass the shared release gate |

## Evaluation contract to finalize in November

Record relative velocity-field error, vorticity error, divergence norm, runtime, failure rate and error versus physical rollout time. Freeze norm definitions, near-zero handling, test horizons and instability criteria before evaluation. Report field-derived vorticity consistently with the chosen discretization.

Separate teacher-forced single-step evaluation from autonomous rollout. Compare stabilization with the unstabilized model under matching test conditions. State exactly which viscosity and initial-condition ranges were tested. The less-than-5% short-rollout velocity-error goal is a stretch target, not a claimed result.

If stable rollout is not achieved, deliver the validated reference solver plus trained single-step estimator and explicitly documented rollout limits. Consider Fourier Neural Operators only if baseline evidence justifies the added work.
