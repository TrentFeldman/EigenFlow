# ODE model blueprint

Target: reproducible higher-order trajectory prediction release by October 31, 2026. Design choices below are proposals until the October 18 specification is frozen.

Implementation update: the generator, data/model interfaces, trainer and checkpoint inference now exist; see [README.md](README.md) for their safe dry-runs and current limits. Numerical generation and real training have not run. Read [MATH_GUIDE.md](MATH_GUIDE.md) for the plain-language mathematical contract.

## Intended data flow

Family and sampling configuration → validated numerical trajectories → grouped train/validation/test partitions → training-only normalization → supervised baseline → saved model and preprocessing → standalone prediction → evaluation and plots.

## Reserved files

| File | Responsibility |
|---|---|
| `equations.py` | Family definitions and conversion to first-order state systems |
| `generate_data.py` | Reference trajectories, solver checks and dataset manifest |
| `dataset.py` | Load partitions and fit/apply saved normalization |
| `model.py` | Modest PyTorch trajectory baseline |
| `train.py` | Seeded training, validation and checkpoint selection |
| `evaluate.py` | Frozen-set metrics, failure analysis and evaluation report |
| `predict.py` | Load model, configuration and preprocessing for independent inference |
| `plot_results.py` | Reference/prediction overlays and error plots |
| `tests/test_reference.py` | Future analytic and numerical-convergence checks |
| `tests/test_data.py` | Future partition isolation and data-contract checks |
| `tests/test_inference.py` | Future saved-artifact inference and reproducibility checks |

`configs/` contains a draft generator configuration. `data/` and `runs/` remain empty artifact locations for later generation/training. Evaluation and plotting modules remain documented placeholders. Test modules currently check software contracts using fabricated states and mocked solvers, not analytic numerical accuracy.

## Proposed data contract

Each trajectory has a stable ID, family ID, named equation parameters, initial state, time samples and reference states. A second-order state is `(y, y')`; a third-order state is `(y, y', y'')`. State arrays have shape `(number of times, order)`. Record component units or nondimensionalization explicitly.

Implemented model input: padded coefficients and initial state, an order mask, sinusoidal forcing descriptors and elapsed time. Output: padded full state, sliced to the actual equation order at inference. A shared modest MLP is the initial framework baseline; see `planning/DECISIONS.md` for the change from the earlier separate-model proposal. Representability of higher orders is separate from measured generalization.

The dataset manifest records schema version, generation configuration, seed, solver and tolerances, bounds, successful and failed solves, and split membership. Do not silently discard failed solves. Store exact sample times; record any interpolation used to obtain them.

Split complete trajectories, never individual time samples. Hold parameter combinations out together for parameter-generalization tests, and initial conditions out together for initial-condition tests. Add a joint holdout where both are unseen. Record overlap rules explicitly. Keep extrapolation outside the training bounds separate from in-domain results.

## Reference and evaluation gates

1. Compare reference integration with an analytic case for each family where available. Tighten tolerances or refine sampling to quantify numerical convergence. Set acceptable reference error well below the model target before dataset generation.
2. Validate finite values, initial conditions, state dimensions, unique IDs and partition isolation.
3. Select checkpoints using validation data and report performance separately by family and generalization split.
4. Compute trajectory L2 error as the norm of prediction minus reference divided by the reference norm with a documented denominator floor. Freeze time weighting, component scaling and that floor before test evaluation. Report absolute error for near-zero references and report per-component errors so derivatives do not hide poor position predictions.
5. Report mean, median, 95th percentile, worst case and failure counts. The initial target is less than 5% mean relative trajectory L2 error on the defined held-out set; report both combined and per-family results, with the primary metric fixed by October 18.
6. Measure inference runtime with documented hardware and batch size. Verify checkpoint reload gives consistent predictions and requires no training data.

## October release contents

Provide implemented scripts, a recorded environment, dataset-generation and split recipes, saved configuration and normalization, accessible checkpoint, evaluation report, representative and worst-case plots, and exact setup/training/inference instructions. Record tested domains and unsupported inputs. No accuracy claim is valid until measured.
