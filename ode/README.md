# ODE prediction framework

This folder contains EigenFlow's numerical-reference generator and supervised
PyTorch trajectory predictor. The current presets cover second- and third-order
linear ODEs, both homogeneous and sinusoidally forced.

Generation, training, release-metric evaluation, standalone inference, diagnostic
splits, plots and runtime benchmarking are implemented. The scientific contract
was frozen October 16. Recorded CPU smoke/pilot runs are linked from
[project status](../planning/STATUS.md); no October release is declared yet.
Target CUDA execution and actual HPC allocation still await hardware access.

The [error-reduction plan](error_reduction/README.md) reviews saved validation
evidence and prepares staged comparisons. Its draft settings are proposals;
no experiments were executed for that planning task.

All commands here run from the **repository root**, using `.venv/bin/python`.
See the [environment guide](ENVIRONMENT.md) for installation and the
[math guide](MATH_GUIDE.md) for a fuller explanation of the equations.

## Contents

- [Check the installation](#check-the-installation)
- [Equations and state ordering](#equations-and-state-ordering)
- [Run the small example](#run-the-small-example)
- [Full-run preset: 100,032 trajectories](#full-run-preset-100032-trajectories)
- [Configuration and sampling](#configuration-and-sampling)
- [Reference validation](#reference-validation)
- [Dataset format and split policy](#dataset-format-and-split-policy)
- [Model inputs, normalization and loss](#model-inputs-normalization-and-loss)
- [Training options and saved artifacts](#training-options-and-saved-artifacts)
- [Code map](#code-map)
- [Troubleshooting](#troubleshooting)
- [Verification and remaining work](#verification-and-remaining-work)

## Check the installation

```bash
.venv/bin/python -m ode.check_environment
.venv/bin/python -m pip check
.venv/bin/python -m ode.validate_reference
.venv/bin/python -m unittest discover -s ode/tests -v
```

The environment check reports Python, dependencies and GPU availability. A GPU
is optional. Reference validation runs small deterministic numerical benchmarks;
unit tests exercise numerical safeguards, data contracts and inference behavior.
These commands do not retain a training dataset or trained checkpoint.

Preview the default experiment without generating data or fitting a model:

```bash
.venv/bin/python -m ode.generate_data --config ode/configs/generator.json
.venv/bin/python -m ode.train --config ode/configs/generator.json
```

Generation planning prints settings and counts without sampling or solving.
Training planning prints settings without loading a dataset. Actual execution
requires the corresponding `--generate` or `--train` flag.

## Equations and state ordering

The supported equation has the form:

```text
y^(n) + a[n-1] y^(n-1) + ... + a[1] y' + a[0] y
    = A sin(ω τ + φ)
```

Here `n` is the equation's order, `y^(k)` means the kth derivative of `y`, and
`τ = t - t_start` is elapsed time from the initial condition. The highest
derivative has coefficient one. `A` is the forcing amplitude, `ω` its angular
frequency, and `φ` its phase in radians. All current state and time values are
nondimensional; there is no conversion from physical units in this pipeline.

- **Homogeneous:** `A = 0`, so the right-hand side is zero.
- **Sinusoidally forced:** a sine wave drives the equation over time.

`LinearODE` stores coefficients in **ascending derivative order** and stores the
initial value of every state component:

```python
from ode.equations import LinearODE

# y'' + y' + 1.25 y = 0, with y(0) = 0.5 and y'(0) = 0.
equation = LinearODE(
    coefficients=(1.25, 1.0),  # Coefficients of y, then y'.
    initial_state=(0.5, 0.0),  # Initial y, then initial y'.
)
```

For order two, a state is `[y, y']`; for order three, it is `[y, y', y'']`.
The solver evolves this first-order state vector, computing the highest
derivative from the original equation. A trajectory with `T` sample times and
order `n` therefore has shape `(T, n)`.

For example, adding `amplitude=0.4, frequency=1.2, phase=0.0` to the constructor
changes the right-hand side to `0.4 sin(1.2 τ)`.

## Run the small example

These commands create local artifacts. Choose new dataset and run names if the
example paths already exist; overwriting and automatic resume are not supported.
The scientific recipe is frozen; counts and execution profiles identify each experiment.

### 1. Generate reference trajectories

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m ode.generate_data \
  --config ode/configs/generator.json \
  --generate \
  --output ode/data/linear-v1
```

The preset uses six process workers. Add `--workers 4` for a smaller CPU allocation
or `--backend serial` for sequential execution. The environment variables limit
native-library threads per process so competing thread pools do not multiply the
requested parallelism.

The requested split sizes are **256 training, 64 validation and 64 test
trajectories**, each with 101 saved times. Inspect the printed manifest or
`ode/data/linear-v1/manifest.json`: clean completion has `status: "complete"`
and `failed: 0`. Failed solves are recorded, not substituted with new samples.

### 2. Train a baseline

```bash
.venv/bin/python -m ode.train \
  --config ode/configs/generator.json \
  --data ode/data/linear-v1 \
  --run ode/runs/linear-v1-baseline \
  --device cpu \
  --epochs 20 \
  --train
```

The trainer fits normalization on the training split, optimizes the model, and
saves the checkpoint with the lowest mean validation trajectory relative L2. It does not score the test
set. A successful run creates `best.pt` and `run.json`; completing 20 epochs does
not itself establish acceptable prediction accuracy.

### 3. Load the checkpoint and predict

Inference has both a Python API and a JSON CLI; neither requires the training dataset.
The API defaults to CPU; `Predictor(path, device="cuda")` loads once for repeated
GPU calls. The CLI rejects requests outside the saved scientific domain unless
`--allow-extrapolation` explicitly requests an unsupported probe.

```bash
.venv/bin/python -m ode.predict \
  --checkpoint ode/runs/linear-v1-baseline/best.pt \
  --input ode/configs/predict_example.json \
  --output ode/runs/linear-v1-baseline/prediction.json
```

```bash
.venv/bin/python - <<'PY'
from pathlib import Path

import numpy as np

from ode.analytic import linear_reference
from ode.equations import LinearODE
from ode.predict import predict

equation = LinearODE(
    coefficients=(1.25, 1.0),
    initial_state=(0.5, 0.0),
)
elapsed_times = np.linspace(0.0, 5.0, 101)
checkpoint = Path("ode/runs/linear-v1-baseline/best.pt")

predicted = predict(checkpoint, equation, elapsed_times)
reference = linear_reference(equation, elapsed_times)

print("State shape:", predicted.shape)  # (101, 2): columns are y and y'.
print("First five predicted states:\n", predicted[:5])
print("Maximum absolute error per component:")
print(np.max(np.abs(predicted - reference), axis=0))
PY
```

`predict(checkpoint_path, equation, elapsed_times)` returns a float64 NumPy array
of shape `(T, n)` in the original nondimensional scale. Model computation uses
float32; returning float64 does not imply double-precision prediction accuracy.
Elapsed times must be finite and nonnegative, and `n` must fit the checkpoint's
maximum order. The caller must check the saved training ranges and time horizon.

The numerical comparison above is a single illustrative case, not a held-out
evaluation. The network does not enforce the initial condition exactly and can
produce finite but inaccurate states.

## Resource profiles and selected pilot

Scientific recipes and execution settings are separate. Use
`--resources ode/configs/resources_local.json` locally, or copy/edit
[resources_cuda.example.json](configs/resources_cuda.example.json) for the
allocation obtained from the cluster. Explicit CLI flags override profile values;
profile values override defaults. All resolved settings are recorded in artifacts.

| Profile key / CLI flag | Meaning |
|---|---|
| `workers` / `--workers` | Generation processes; profile also accepts `"auto"` |
| `backend` / `--backend` | `process`, `thread`, or `serial` generation |
| `threads` / `--threads` | Training CPU threads; profile also accepts `"auto"` |
| `loader_workers` / `--loader-workers` | DataLoader subprocesses; zero for preload |
| `device` / `--device` | `cpu`, `cuda`, or `auto` training |
| `batch_size` / `--batch-size` | Training points per update; changes optimization |
| `preload` / `--preload`, `--no-preload` | Keep normalized training/validation in RAM |
| `preload_limit_gib` / `--preload-limit-gib` | Upper bound for preload arrays, excluding other process memory |

CPU `"auto"` uses the minimum of affinity, logical CPUs and `SLURM_CPUS_PER_TASK`
when present. It never uses total cluster capacity. For non-SLURM allocations
without enforced affinity, set explicit counts. The example CUDA profile is
editable and untested on the target. No scientific ranges change when switching
profiles. Batch size/device changes still define a new training experiment.

The selected next-size recipe is `generator_pilot.json`: **6,656 trajectories**
(4,096 train, 1,024 validation, 1,024 reserved test, 256 in each diagnostic).
The original 384-case preset stays available for wiring and device checks.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m ode.generate_data --config ode/configs/generator_pilot.json \
  --resources ode/configs/resources_local.json --generate --output ode/data/pilot-v1
.venv/bin/python -m ode.train --config ode/configs/generator_pilot.json \
  --resources ode/configs/resources_local.json --data ode/data/pilot-v1 \
  --run ode/runs/pilot-v1 --epochs 100 --train
```

## Evaluate, plot and benchmark

The primary metric is `RMS(pred-ref)/max(RMS(ref),1e-6)` over real state
components and all stored times. Component scales and time weights are one;
trajectories have equal aggregate weight. `evaluate.py` reports mean, median,
p95, worst, >=5% and numerical failures, absolute errors, near-zero summaries,
and per-order/forcing-family/component breakdowns. See [MATH_GUIDE.md](MATH_GUIDE.md)
for a worked example. Nonfinite outputs cannot produce a passing result.

```bash
.venv/bin/python -m ode.evaluate --checkpoint ode/runs/pilot-v1/best.pt \
  --data ode/data/pilot-v1 --splits validation validation_parameter validation_initial \
  --output ode/runs/pilot-v1/evaluation.json
.venv/bin/python -m ode.plot_results --checkpoint ode/runs/pilot-v1/best.pt \
  --data ode/data/pilot-v1 --report ode/runs/pilot-v1/evaluation.json \
  --output ode/runs/pilot-v1/plots --worst 3
.venv/bin/python -m ode.benchmark inference --checkpoint ode/runs/pilot-v1/best.pt \
  --batch-sizes 1 101 4096 --repeats 30 --threads 2 \
  --output ode/runs/pilot-v1/runtime.json
```

Plots include the median case in each family and worst cases overall. SVG files
contain component overlays and signed errors, labeled with equation parameters
and nondimensional axes; no plotting package is needed. Runtime measures warmed
loaded prediction, including transfers/encoding and excluding checkpoint load.
Use `--device cuda` on the tested GPU. Evaluation defaults to validation; request
`--splits test` only for a recorded milestone after model selection is finished.
Every report/plot verifies the dataset/checkpoint identity, and new paths prevent
accidental overwrite. Diagnostics are absent in the 384-case recipe.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m ode.benchmark generation --workers 1 6 12 --repeats 3 \
  --output ode/runs/local-throughput/report.json
```

This solves real data repeatedly in temporary directories and records all timing
and failure evidence. Match the worker sweep to the actual CPU allocation.

## Full-run preset: 100,032 trajectories

This is an unused capacity option, not the selected next run. It contains only
core splits; add and version diagnostics before promoting it to a release recipe.

[configs/generator_full.json](configs/generator_full.json) keeps the same
equation families, bounds, time grid and solver checks as the small preset. It
increases the number of equation-parameter groups.

| Split | Groups per order and family: small / full | Small trajectories | Full trajectories |
|---|---:|---:|---:|
| Train | 16 / 4,168 | 256 | 66,688 |
| Validation | 4 / 1,042 | 64 | 16,672 |
| Test | 4 / 1,042 | 64 | 16,672 |
| **Total** | **24 / 6,252** | **384** | **100,032** |

Each parameter group has four independently sampled initial states. The count is:

```text
number of orders × number of families × sum(groups_per_split)
    × initial_conditions_per_group

2 × 2 × (4168 + 1042 + 1042) × 4 = 100032
```

Preview first:

```bash
.venv/bin/python -m ode.generate_data --config ode/configs/generator_full.json
```

To execute with the preset's six process workers:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m ode.generate_data \
  --config ode/configs/generator_full.json \
  --generate \
  --output ode/data/linear-100k-v1
```

On the actual allocated node, set `--workers` to a measured appropriate count
and use a new output directory on the site's scratch storage. Use the site's Python environment and match the
worker count to the allocation. Sampling seeds are independent of worker
scheduling and backend choice.

Training must use the matching full configuration. For a compatible GPU host:

```bash
.venv/bin/python -m ode.train \
  --config ode/configs/generator_full.json \
  --data ode/data/linear-100k-v1 \
  --run ode/runs/linear-100k-baseline \
  --device cuda \
  --preload \
  --loader-workers 0 \
  --epochs 20 \
  --train
```

`--preload` requires enough host RAM for normalized examples; omit it to load
archives lazily. CPU execution is available with `--device cpu`. Local measured timings and unresolved target hardware are recorded
in [COMPUTE_ESTIMATES.md](COMPUTE_ESTIMATES.md).

## Configuration and sampling

All dataset presets use these frozen scientific settings:

| Setting | Value | Meaning |
|---|---|---|
| `seed` | `20261015` | Reproducible parameter and initial-state sampling |
| `orders` | `[2, 3]` | Equation orders included in the dataset |
| `max_order` | `6` | Padding width for shared model inputs and outputs |
| `families` | Homogeneous and sinusoidal | Two supported linear equation families |
| `initial_conditions_per_group` | `4` | Initial states sharing one parameter set |
| `decay_range` | `[0.1, 1.0]` | Positive decay magnitudes used to construct stable roots |
| `frequency_range` | `[0.5, 2.0]` | Oscillation frequencies and sinusoidal forcing frequencies |
| `initial_state_range` | `[-1.0, 1.0]` | Sampling bounds for each initial state component |
| `amplitude_range` | `[0.1, 1.0]` | Forcing amplitude for the sinusoidal family |
| `t_start`, `t_stop`, `time_points` | `0`, `5`, `101` | Uniform saved time grid, including both endpoints |
| `method` | `DOP853` | Adaptive numerical integration method |
| `rtol`, `atol` | `1e-9`, `1e-11` | Solver relative and absolute error controls |
| `refinement_factor` | `10` | Divide solver tolerances by this factor for the second solve |
| `reference_rtol`, `reference_atol` | `1e-6`, `1e-8` | Acceptance thresholds for reference comparisons |
| `independent_check_points` | `5` | Saved times checked using a matrix exponential |
| `max_abs_state` | `1e6` | Reject states exceeding this absolute component bound |
| `max_rhs_calls` | `100000` | Maximum right-hand-side evaluations per solve |
| `backend`, `workers` | `process`, `6` | Execution settings in the JSON presets |

Forcing phases are sampled in `[-π, π)`. Coefficients are constructed from
decaying complex-conjugate root pairs, with an additional negative real root for
odd orders. This controls stability of the sampled linear equations; it does not
cover all possible coefficient combinations. See
[ROOTS_AND_EXTENSIONS.md](ROOTS_AND_EXTENSIONS.md) for the mathematical reasoning.

The generator also accepts `thread` and `serial` backends. Supported integration
methods are `DOP853`, `RK45`, `Radau` and `BDF`. Configuration validation rejects
unknown fields and invalid values before execution.

To explore another order, add it to `orders` and keep `max_order` at least that
large. Changing the input/output width requires a compatible new model. Padding
to six does not train orders four through six. Nonlinear equations or new forcing
types require changes to the equation representation, sampler and encoding,
together with numerical validation.

## Reference validation

Generation runs closed-form benchmarks before creating an output directory.
With the default settings, the suite has 13 cases covering both families,
repeated and distinct real roots, damped oscillations, orders 2/3/6 and resonant
forcing. If preflight fails, no dataset is created.

Every requested trajectory then goes through these checks:

1. Integrate with the configured solver tolerances.
2. Integrate again with tolerances divided by `refinement_factor`.
3. Compare both trajectories over all saved times and state components.
4. Compare the refined solution with an independent matrix-exponential solution
   at the configured number of selected times, including the endpoints.
5. Require valid shapes, the requested time grid, finite values, the initial
   state and the configured state/evaluation bounds.

The saved trajectory comes from the tighter solve. Relative tolerance scales an
allowed difference with solution magnitude; absolute tolerance supplies a floor
near zero. These acceptance checks have separate settings from the integrator's
own local error controls.

For refinement, each component uses:

```text
||coarse - refined||₂
    / (reference_atol × sqrt(T) + reference_rtol × ||refined||₂)
```

The norm is taken over saved times for that component. Independent checks use
pointwise differences divided by
`reference_atol + reference_rtol × abs(independent_state)`. Each reported maximum
error ratio must be finite and at most one. These checks establish agreement at
the sampled times; they do not prove an error bound between them.

## Dataset format and split policy

```text
ode/data/linear-v1/
├── manifest.json
├── records.jsonl
├── train/
│   └── <trajectory-id>.npz
├── validation/
│   └── <trajectory-id>.npz
└── test/
    └── <trajectory-id>.npz
```

| Artifact | Contents |
|---|---|
| `manifest.json` | Schema version, resolved configuration, completion status, counts, reference preflight, environment/source provenance and ledger checksum |
| `records.jsonl` | One record per request: IDs, split, family, equation, success/failure, numerical checks and successful archive path/checksum |
| Each `.npz` | Float64 `times` with shape `(T,)` and `states` with shape `(T, n)` |

Archives store absolute times on the configured grid. Encoding subtracts
`t_start`, so the model and prediction API receive elapsed times.

Core splits hold out **both equation parameters and initial states**. All initial
conditions belonging to one equation-parameter group stay in the same split;
time samples from a trajectory are never distributed across splits. The parameter
ID hashes coefficients and forcing parameters, excluding the initial state.
Generation and loading detect cross-core-split parameter or initial-state collisions.
The pilot also has `validation_parameter` (new parameters, seen ICs) and
`validation_initial` (seen parameters, new ICs). They cross training/validation
identities intentionally; test identities remain isolated. The diagnostic group
count is `diagnostic_groups` per order/forcing family, with four ICs per group.

The loader verifies the manifest, ledger checksum, counts and record metadata.
Archive checksums and array contracts are checked when each trajectory is first
loaded. Failed solves remain in the ledger and are excluded from examples.

An interrupted dataset is marked `incomplete` and rejected by training. A
completed dataset containing failures is marked `complete_with_failures` and
requires explicit `--allow-failures` after inspection. Clean completion is
`complete` with zero failures. Generation does not resume an interrupted output.

## Model inputs, normalization and loss

Let `M` be the maximum padded order, `n` the current equation order, `T` the
trajectory length, and `B` the batch size. The encoding is named
`linear-padded-v1`, with `3M + 5` features per time point.

For the default `M = 6`, the input has 23 columns:

| Python slice/index | Contents |
|---|---|
| `0:6` | Equation coefficients in ascending derivative order, padded with zeros |
| `6:12` | Initial state, padded with zeros |
| `12:18` | Order mask: one for each active derivative, zero for padding |
| `18:22` | Amplitude, frequency, sine of phase, cosine of phase |
| `22` | Elapsed time |

Homogeneous equations use zeros for all four forcing features. Representing
phase by its sine and cosine avoids a jump in the input at the phase wraparound.

Each dataset item is one time point: float32 `features` of shape `(23,)`, plus
`target` and `mask` of shape `(6,)`. A batch has shapes `(B, 23)` and `(B, 6)`.
The small preset therefore provides 25,856 training examples if all 256 training
trajectories succeed; those points still belong to 256 grouped trajectories.

Normalization learns means and scales from the **training split only**. Padded
coefficient, initial-state and target slots are excluded from statistics; masks
remain unscaled. Constant or unseen components use safe fallback scales. The
same saved transformation is applied to validation and inference.

The baseline is a feed-forward network with three hidden layers of 128 neurons,
SiLU activations and six outputs. It predicts the state directly for each
requested time. Inference removes padded outputs after reversing normalization.

Training uses **masked mean squared error**: average squared error across active
derivatives within each example, then average across examples. Padding contributes
no loss, and higher-order equations do not receive extra weight merely for having
more components. This loss is in standardized state units. It is distinct from
the frozen relative trajectory L2 release metric; a validation MSE of `0.05`
does not mean 5% trajectory error.

## Training options and saved artifacts

| Option | Default | Purpose |
|---|---|---|
| `--epochs` | `20` | Training passes through the dataset |
| `--batch-size` | `256` | Time-point examples per batch |
| `--width`, `--depth` | `128`, `3` | Hidden-layer size and count |
| `--learning-rate` | `0.001` | AdamW learning rate |
| `--device` | `cpu` | Choose `cpu`, explicit `cuda`, or automatic `auto` |
| `--threads` | `4` | PyTorch CPU compute threads |
| `--loader-workers` | `0` | DataLoader subprocesses |
| `--preload` | Off | Keep normalized examples in host RAM; requires zero loader workers |
| `--allow-failures` | Off | Accept a completed dataset with inspected reference failures |

Generation `--workers`, training `--threads`, and training `--loader-workers`
control separate resources. Scientific settings must match the dataset manifest;
generation backend and worker count may differ from the training configuration.

Training seeds Python, NumPy and PyTorch and requests deterministic algorithms.
The optimizer uses AdamW, with gradient norm clipping at one. Identical seeds do
not guarantee identical results across hardware or library versions.

```text
ode/runs/linear-v1-baseline/
├── best.pt
└── run.json
```

`best.pt` contains weights, architecture, encoding/schema version, normalization,
generation configuration, selected epoch, validation loss and dataset-manifest
checksum. Checkpoint replacement is atomic after a complete temporary write.

`run.json` records configuration, seed, environment, device, source and dataset
identities, training options, epoch losses/timings and completion status. A
completed run records the checkpoint checksum. An interrupted run remains labeled
incomplete, even if an earlier best checkpoint exists. Training resume is not
implemented.

Add an experiment note describing the question, results, failure cases and next
decision, following [the project workflow](../planning/WORKFLOW.md). Checkpoints
and datasets stay outside Git; small reports and run notes provide reviewable
evidence.

## Code map

| Module | Responsibility |
|---|---|
| [config.py](config.py) | Validated settings, split helpers and provenance |
| [equations.py](equations.py) | Equation representation, parameter identity, root sampling and state-system conversion |
| [analytic.py](analytic.py) | Independent matrix-exponential reference states |
| [reference.py](reference.py) | Bounded integration, refinement, independent checks and failure results |
| [validate_reference.py](validate_reference.py) | Closed-form benchmark suite and optional JSON report |
| [generate_data.py](generate_data.py) | Deterministic jobs, bounded workers, archives and manifests |
| [dataset.py](dataset.py) | Integrity checks, feature encoding, loading and normalization |
| [model.py](model.py) | Mixed-order neural baseline and masked loss |
| [train.py](train.py) | Training, validation selection and saved artifacts |
| [predict.py](predict.py) | CPU/CUDA inference and independent JSON CLI |
| [check_environment.py](check_environment.py) | Interpreter, dependency and device readiness report |
| [evaluate.py](evaluate.py), [plot_results.py](plot_results.py) | Frozen metrics, breakdowns and standalone SVG plots |
| [benchmark.py](benchmark.py) | Generation/inference runtime with hardware and stage evidence |
| [resources.py](resources.py) | Separate execution profiles, CPU allocation detection and validation |
| [target_smoke.sh](target_smoke.sh) | Actual allocated NVIDIA end-to-end verification |

## Troubleshooting

| Symptom | What to check |
|---|---|
| `No module named ode` | Run from the repository root with `.venv/bin/python -m ode.<module>` |
| Missing NumPy, SciPy or PyTorch | Check the interpreter and follow [environment setup](ENVIRONMENT.md) |
| Command only prints a plan | Add `--generate` with `--output`, or `--train` with `--data` and `--run` |
| Output directory already exists | Choose a new dataset/run name; automatic overwrite and resume are unavailable |
| Dataset is incomplete | Inspect its manifest and ledger, resolve the interruption, and generate to a new path |
| Completed dataset has failed solves | Inspect failure records and family counts before considering `--allow-failures` |
| Scientific configuration mismatch | Use the exact generation recipe, including group counts and seed |
| Checksum mismatch | Check for edited, damaged or partially copied artifacts; retain the original evidence |
| CUDA is unavailable | Use `--device cpu` or a compatible PyTorch installation on the GPU host |
| Preloading conflicts with workers | Pair `--preload` with `--loader-workers 0` |
| Finite predictions are inaccurate | Check training bounds, horizon, run losses and per-component errors; finiteness is not an accuracy test |

## Verification and remaining work

Run `.venv/bin/python -m unittest discover -v` from the root. Tests cover
closed-form reference accuracy, partition isolation, near-zero/nonfinite metric
behavior, validation checkpoint selection, inference reload and execution
profile precedence. Historical reference evidence lives in
[runs/20261015-reference-validation-01](runs/20261015-reference-validation-01/README.md).
New smoke/pilot/benchmark evidence and current counts are in
[STATUS.md](../planning/STATUS.md). Reference error does not measure neural error.

Remaining release gates are target hardware verification, checkpoint selection
and distribution, documented milestone test evaluation, and clean-environment
reproduction. Padding width six does not establish neural accuracy on untrained
orders or all-real/repeated-root families. Fluid implementation remains deferred.
