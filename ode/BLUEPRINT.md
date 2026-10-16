# ODE model blueprint

Target: reproducible higher-order trajectory predictor by October 31, 2026.
The scientific v1 contract was resolved October 16, before real-data generation.
See [the decision register](../planning/DECISIONS.md) for bounds, rationale,
fallbacks and the explicitly unresolved target-hardware allocation.

## Implemented data flow

Frozen equation recipe → independently checked numerical trajectories → whole
trajectory splits → training-only normalization → shared MLP → validation
relative-L2 checkpoint selection → independent inference → evaluation and plots.

| File | Responsibility |
|---|---|
| `config.py`, `equations.py` | Validated recipe, equation representation and stable-root sampling |
| `generate_data.py`, `reference.py` | Bounded generation, dual solves, independent checks, archives and failure ledger |
| `analytic.py`, `validate_reference.py` | Matrix exponentials and 13 closed-form preflight cases |
| `dataset.py` | Archive/partition integrity, padded features, training-only normalization |
| `model.py`, `train.py` | Shared MLP, masked MSE optimization, release-metric validation selection |
| `predict.py` | CPU/CUDA checkpoint loading and dataset-independent JSON CLI |
| `evaluate.py` | Per-trajectory metrics, split/family/component summaries and failure accounting |
| `plot_results.py` | Standalone SVG overlays/error panels for family medians and worst cases |
| `benchmark.py` | Generation stage timings and warmed, synchronized inference runtime |
| `check_environment.py`, `target_smoke.sh` | Environment/allocation evidence and target CUDA smoke workflow |
| `tests/` | Discoverable numerical, leakage, metric, selection and inference contracts |

## Frozen data contract

State is `[y,y']` for order 2 and `[y,y',y'']` for order 3. Reference arrays have
shape `(T,n)`, with `T=101`, `t=0..5`, spacing `0.05`. All units are
nondimensional. Coefficients multiply ascending derivatives; the highest
coefficient is one. Both orders include equal homogeneous/sine-forced samples.

Roots are `-d +/- iw` and an additional `-r` for order 3: `d,r in [0.1,1]`,
`w in [0.5,2]`. Initial derivatives are in `[-1,1]`; sine amplitude `[0.1,1]`,
frequency `[0.5,2]`, phase `[-pi,pi]`. Coefficients are induced and correlated;
exact formulas and coefficient bounds are in the decision register. Extrapolation
past these distributions or time 5 is outside October acceptance.

Model input is padded coefficients, IC, order mask, forcing descriptors and time.
With padding width M=6, input is `(B,23)` and output `(B,6)`; prediction returns
only the actual n components. Untrained orders remain unsupported despite padding.

Archives retain float64 reference states. Every trajectory passes DOP853
`rtol=1e-9/atol=1e-11`, a tenfold tolerance refinement and five independent
matrix-exponential checks with `reference_rtol=1e-6/reference_atol=1e-8`.
Preflight must pass before creation. Manifests record resolved configuration,
seeds, environment/source hashes, failures, identities and archive checksums.

## Splits and metrics

Core train/validation/test are disjoint in both parameters and ICs. The pilot is
6,656 cases: 4,096 train, 1,024 validation, 1,024 reserved test, and two diagnostics
of 256 each. Parameter-only validation uses validation parameters with training
ICs. IC-only validation uses training parameters with validation ICs. The loader
checks those intentional overlaps; no test identity enters a diagnostic.
Normalization fits only the core training partition.

For one trajectory compute `E=RMS(prediction-reference)` and `R=RMS(reference)`
over all times and real derivatives. Primary error is `E/max(R,1e-6)`: relative
L2 with a norm denominator floor `1e-6*sqrt(T*n)`. Time weights and component
scales are all one, with no centering. Each trajectory has equal aggregate weight.
Report mean, median, p95, worst, >=5% errors and nonfinite failures, plus breakdowns
by order/forcing family, split and component. Report absolute RMSE/max errors;
references with RMS<=1e-6 also receive a separate near-zero absolute-error summary.
If any prediction is nonfinite, primary mean/worst are null and passing is false;
finite-only statistics remain labeled. MSE is the optimization loss only.

The target is **<5% mean relative L2 on the reserved joint test split**. Validation
progress alone does not establish that target. Test evaluation is explicit,
recorded and reserved for milestones. Diagnostics guide development and never
join the primary release average. No out-of-range test set is required for v1.

## Evidence and remaining gates

See [STATUS.md](../planning/STATUS.md) and the dated `runs/` notes for measured
CPU generation/training, checkpoint reload, plots and runtime. The 100,032 recipe
is an unused capacity option. Actual HPC allocation and NVIDIA execution remain
unverified until an allocated target job provides evidence.

Release still needs a selected checkpoint, a recorded reserved-test evaluation,
representative/worst-case plots and runtime, artifact distribution/checksums,
and reproduction from a clean tested environment. Neither reference accuracy
nor a completed training run alone establishes neural model acceptance.
