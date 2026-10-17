# ODE prediction framework

EigenFlow's first track: generate exact-enough answers to linear ODEs with a
numerical solver, train a neural network to reproduce them, and measure how
far off it is on equations it never saw. The presets cover second- and
third-order linear ODEs, with and without a sine-wave forcing term.

Everything runs from the **repository root** with the project's Python
(`.venv/bin/python`, or `python` inside an activated environment). Setup is in
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md); the mathematics, in plain language,
is in [docs/MATH_GUIDE.md](docs/MATH_GUIDE.md). Run history and the current
milestone live in [planning/STATUS.md](../planning/STATUS.md).

## Quick start: one command

```bash
.venv/bin/python -m ode pipeline --name 20261017-pilot-02 --preset pilot --epochs 300 --batch-size 1024
```

That runs four stages and prints progress with time estimates:

| Stage | Writes | Notes |
|---|---|---|
| 1. generate | `ode/data/pilot-6656/` | Solves 6,656 equations on every CPU core (about 15 s here). Reused by later runs with the same preset |
| 2. train | `ode/runs/<name>/best.pt`, `last.pt`, `run.json` | Uses the GPU when PyTorch sees one (AMD ROCm counts as `cuda`) |
| 3. evaluate | `ode/runs/<name>/evaluation.json` | Validation splits only; the test split stays sealed |
| 4. plot | `ode/runs/<name>/plots/*.svg` | Start with `summary.svg`; open any SVG in a browser |

**If it stops** (Ctrl-C, crash, SLURM time limit), run the identical command
again. Finished stages are skipped and training resumes from the last completed
epoch with the same result as an uninterrupted run. Presets: `small` (384
equations, wiring checks), `pilot` (6,656), `full` (100,032), `wide` (480,000
with barely-damped to nearly critically damped springs and faster oscillation
and push; decay 0.01–1.5, frequency 0.5–4; 400,000 train, 40,000 validation). `final` (768,000, same ranges as `wide` plus orders 1–4;
640,000 train, 64,000 validation; the planned release recipe, not yet trained). Any option the
pipeline does not recognise goes to `train`, e.g. `--learning-rate-schedule cosine`.

On FSU's cluster, submit the same thing as a batch job (edit the two
`CHANGE ME` lines in the script once):

```bash
sbatch ode/slurm/pipeline.sbatch --name 20261101-full-01 --preset full --epochs 300 --batch-size 1024
```

## What is where

```text
ode/
├── README.md          you are here
├── __main__.py        `python -m ode <command>`: the single entry point
├── pipeline.py        generate → train → evaluate → plot, resumable
├── config.py          the scientific recipe (ranges, solver tolerances, split sizes)
├── resources.py       hardware settings (CPU/GPU counts), kept apart from the recipe
├── generation/        STAGE 1: make the dataset
│   ├── equations.py          what an equation is; random stable coefficients
│   ├── reference.py          solve one equation twice + an independent check
│   ├── analytic.py           matrix-exponential solution used by that check
│   ├── validate_reference.py test the solver against exact textbook formulas
│   └── generate_data.py      run all solves in parallel, write the dataset
├── training/          STAGE 2: learn from it
│   ├── dataset.py            load + verify a dataset; turn equations into model inputs
│   ├── model.py              the neural network and its loss
│   └── train.py              the training loop, checkpoints, resume
├── evaluation/        STAGE 3: judge and use the trained model
│   ├── evaluate.py           the official error metric, per split and equation type
│   ├── plot_results.py       charts for people (summary, training curve, examples)
│   ├── diagnose.py           deeper error breakdowns for choosing experiments
│   └── predict.py            load best.pt and predict any supported equation
├── tools/             machine checks and timing, not part of the science
├── configs/           JSON presets (generator_*.json) and hardware profiles (resources_*.json)
├── equations/         your own equations to try with `predict`; charts land next to them
├── slurm/             batch scripts for the HPC cluster
├── docs/              math guide, environment, design blueprint, compute estimates, error-reduction plan
├── tests/             `python -m unittest discover -s ode/tests`
├── models/current/    the released model (best.pt + MODEL.md with its scores); the only checkpoint in git
├── data/              generated datasets (not in git)
└── runs/              one folder per experiment: results, notes, charts (all local, not in git)
    └── archive/       finished history; `python -m ode runs` lists what is where
```

`python -m ode runs` prints every current run (status, dataset, epochs,
average error, size) and every dataset with the runs that use it. Move a run
you are done with into `ode/runs/archive/` to drop it from that list.

A step-by-step walkthrough of how a result is produced, file by file and with
one real equation traced through every stage, is at
[EigenFlow ODE Walkthrough](https://claude.ai/artifact/Ded3cmWmb3ymRzRN2zJEQA)
(shared page; ask the owner for access).

## Commands

`python -m ode` lists every command; `python -m ode <command> --help` lists its options.

| Command | What it does |
|---|---|
| `pipeline` | All four stages, resumable |
| `generate` | Print the plan for a preset; add `--generate --output DIR` to create it |
| `validate` | Check the solver against exact closed-form solutions |
| `train` | Print the plan; add `--train --data DIR --run DIR` to train (`--resume` continues) |
| `evaluate` | Score `best.pt` on chosen splits |
| `plot` | `--run DIR` draws the charts for a trained and evaluated run |
| `diagnose` | Error by regime, initial-state error, derivative consistency |
| `predict` | Predict one equation described in a JSON file |
| `runs` | List runs and datasets with status, score and size |
| `neurons` | Animated HTML page of the neurons changing during training (train with `--snapshot-every N`) |
| `check` | Python, package and GPU report |
| `benchmark` | Time generation or inference |

## Check the installation

```bash
.venv/bin/python -m ode check
.venv/bin/python -m pip check
.venv/bin/python -m ode validate
.venv/bin/python -m unittest discover -s ode/tests -v
```

The environment check reports Python, dependencies and GPU availability. A GPU
is optional. Reference validation runs small deterministic numerical benchmarks;
unit tests exercise numerical safeguards, data contracts, training resume and
inference behavior. These commands do not keep a dataset or checkpoint.

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
from ode.generation.equations import LinearODE

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

## Running the stages yourself

The pipeline is a wrapper around these commands; use them when you need one
stage with non-default options. Each refuses to overwrite an existing output.

### 1. Generate reference trajectories

```bash
.venv/bin/python -m ode generate --config ode/configs/generator.json \
  --generate --output ode/data/small-384
```

Workers default to every CPU this process may use (`"auto"`: the smallest of
the logical CPUs, the process affinity and `SLURM_CPUS_PER_TASK`). Override with
`--workers N`, or `--backend serial` for one process. Each worker is pinned to
one native math thread automatically; leaving OpenBLAS free to start its own
thread pool in every worker made generation 3.7 times slower (45 s instead of
12 s for the pilot). Progress prints every 10 seconds on stderr.

The small preset requests **256 training, 64 validation and 64 test
trajectories**, each with 101 saved times. `manifest.json` in the output shows
`status: "complete"` and `failed: 0` on a clean run. Failed solves are recorded,
not replaced with new samples.

### 2. Train

```bash
.venv/bin/python -m ode train --data ode/data/small-384 \
  --run ode/runs/small-baseline --epochs 20 --train
```

The scientific recipe is read from the dataset's manifest, so `--config` is
only needed to double-check it. Training fits normalization on the training
split, keeps the whole split in memory (on the GPU when training there) and
saves the epoch with the lowest mean validation error as `best.pt`. It never
scores the test set. Each epoch prints its validation error, the best so far,
seconds per epoch and an ETA.

To continue an interrupted run, repeat the command with `--resume`; to extend a
finished run, raise `--epochs` and add `--resume`.

### 3. Evaluate and plot

```bash
.venv/bin/python -m ode evaluate --checkpoint ode/runs/small-baseline/best.pt \
  --data ode/data/small-384 --output ode/runs/small-baseline/evaluation.json
.venv/bin/python -m ode plot --run ode/runs/small-baseline
```

The primary metric is `RMS(pred-ref)/max(RMS(ref),1e-6)` over real state
components and all stored times, so "2% error" means the miss is about 2% of
the size of the solution. Trajectories have equal weight. `evaluate` reports
mean, median, p95, worst, the count at or above 5%, numerical failures,
absolute errors, near-zero summaries and per-order/family/component breakdowns.
See [docs/MATH_GUIDE.md](docs/MATH_GUIDE.md) for a worked example. Request
`--splits test` only for a recorded milestone after model selection is final.

`plot` writes, into `<run>/plots/`:

| File | Shows |
|---|---|
| `summary.svg` | Headline average error, error per equation type, and how many equations fall in each error range |
| `training.svg` | Validation error after every epoch, best epoch marked |
| `example-typical-case-*.svg` | The median equation of each type: exact vs predicted curves and the gap |
| `example-worst-case-*.svg` | The worst predictions overall |

Charts have plain-English titles and labels, hover tooltips in a browser, and
follow the viewer's light/dark setting. No plotting package is needed.

### 4. Predict a new equation

Inference has a JSON CLI and a Python API; neither needs the training dataset.
Requests outside the trained range (faster/slower oscillation, bigger starting
values or push, longer times) are moved into it exactly: time is rescaled,
the free and forced parts are predicted separately, and long times are
stitched from several model runs (`Predictor.predict_rescaled`). Only the
*ratios* (decay vs frequency, push frequency vs natural frequency) must lie in
the trained range; if they don't, the CLI refuses unless
`--allow-extrapolation` is given.

```bash
.venv/bin/python -m ode predict --checkpoint ode/models/current/best.pt \
  --input ode/equations/predict_example.json --output ode/runs/prediction.json

# Try your own equation: copy a file in ode/equations/, edit it, and get a chart
# of exact vs predicted (try_equation.svg) and a table with both solutions
# (try_equation.csv) written next to it. --output adds exact_states to the JSON.
# coefficients go lowest derivative first: [2.0, 0.6] means y'' + 0.6 y' + 2 y.
# elapsed_times is optional; without it the model's trained 0–5 grid is used.
.venv/bin/python -m ode predict --checkpoint ode/models/current/best.pt \
  --input ode/equations/try_equation.json
```

```python
import numpy as np
from ode.generation.analytic import linear_reference
from ode.generation.equations import LinearODE
from ode.evaluation.predict import Predictor

equation = LinearODE(coefficients=(1.25, 1.0), initial_state=(0.5, 0.0))
times = np.linspace(0.0, 5.0, 101)
predicted = Predictor("ode/models/current/best.pt").predict(equation, times)  # (101, 2): y, y'
print(np.max(np.abs(predicted - linear_reference(equation, times)), axis=0))
```

`predict` returns float64 states of shape `(T, n)` in the original
nondimensional scale; the model computes in float32. The network does not
enforce the initial condition exactly and can produce finite but inaccurate
states.

## Hardware profiles and the cluster

Scientific recipes (`configs/generator_*.json`) and hardware settings
(`configs/resources_*.json`) are separate, so moving to a bigger machine never
changes the experiment. Pass a profile with `--resources FILE`; explicit flags
override it. Every resolved setting is recorded in the run's artifacts.

| Profile key / CLI flag | Meaning | Default |
|---|---|---|
| `workers` / `--workers` | Generation processes; `"auto"` = all allocated CPUs | `auto` in presets |
| `backend` / `--backend` | `process`, `thread`, or `serial` generation | `process` in presets |
| `threads` / `--threads` | Training CPU threads | all allocated CPUs |
| `device` / `--device` | `cpu`, `cuda`, or `auto` | `auto` |
| `batch_size` / `--batch-size` | Examples per update; changes optimization | 256 |
| `preload` / `--no-preload` | Hold the split in (GPU) memory | on |
| `preload_limit_gib` / `--preload-limit-gib` | Refuse to preload above this size | none |
| `loader_workers` / `--loader-workers` | DataLoader subprocesses, only with `--no-preload` | 0 |

On SLURM, `"auto"` follows `--cpus-per-task`, so
[slurm/pipeline.sbatch](slurm/pipeline.sbatch) needs no worker edits. Its
partition and environment lines are marked `CHANGE ME` because FSU's GPU
partition names are site-specific. [slurm/target_smoke.sh](slurm/target_smoke.sh)
is the separate NVIDIA readiness check (environment report, short train,
reload, inference benchmark).

**GPU utilisation.** The model is a small MLP, so at batch 1024 each step is
mostly kernel-launch overhead and the GPU is far from saturated even after the
data-loading fix (0.6 s per pilot epoch, previously 3.9 s). Larger batches use
the GPU much better (8192: about 0.2 s per epoch) but change the optimization,
so treat batch size as an experiment setting, not a speed knob.

## Full-run preset: 100,032 trajectories

[configs/generator_full.json](configs/generator_full.json) keeps the same
equation families, bounds, time grid and solver checks as the small preset and
increases the number of equation-parameter groups. It has no diagnostic splits;
add and version them before promoting it to a release recipe.

| Split | Groups per order and family: small / full | Small trajectories | Full trajectories |
|---|---:|---:|---:|
| Train | 16 / 4,168 | 256 | 66,688 |
| Validation | 4 / 1,042 | 64 | 16,672 |
| Test | 4 / 1,042 | 64 | 16,672 |
| **Total** | **24 / 6,252** | **384** | **100,032** |

Each parameter group has four independently sampled initial states:
`orders × families × sum(groups_per_split) × initial_conditions_per_group`
= `2 × 2 × (4168 + 1042 + 1042) × 4 = 100032`. Expect roughly four minutes of
generation on this 6-core machine and about 1.1 GiB of preloaded training and
validation examples. Sampling seeds are independent of worker count and
backend, so any machine produces identical trajectories. Measured timings are in
[docs/COMPUTE_ESTIMATES.md](docs/COMPUTE_ESTIMATES.md).

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
| `backend`, `workers` | `process`, `auto` | Execution settings in the JSON presets (not part of the science) |

Forcing phases are sampled in `[-π, π)`. Coefficients are constructed from
decaying complex-conjugate root pairs, with an additional negative real root for
odd orders. This controls stability of the sampled linear equations; it does not
cover all possible coefficient combinations. See
[docs/ROOTS_AND_EXTENSIONS.md](docs/ROOTS_AND_EXTENSIONS.md) for the mathematical reasoning.

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
ode/data/small-384/
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
| `--batch-size` | `256` | Time-point examples per update |
| `--width`, `--depth` | `128`, `3` | Hidden-layer size and count |
| `--learning-rate` | `0.001` | AdamW learning rate |
| `--learning-rate-schedule` | `constant` | `cosine` decays to `--minimum-learning-rate` by the last epoch |
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--training-seed` | dataset seed | Model initialisation and shuffle order |
| `--device` | `auto` | `cpu`, `cuda` (also AMD ROCm), or `auto` |
| `--threads` | all allocated CPUs | PyTorch CPU threads |
| `--preload` / `--no-preload` | on | Hold the split in memory; off streams archives through DataLoader |
| `--resume` | off | Continue the `--run` folder from its `last.pt` |
| `--snapshot-every N` | 0 (off) | Save the weights to `<run>/snapshots/` every N epochs (and epoch 0 of a fresh run) so `python -m ode neurons --run <run> --input <equation.json>` can animate the learning |
| `--initialize-from PATH` | off | Start from another run's `best.pt` (same dataset, width, depth) instead of random weights; use to train an under-trained model further with a fresh schedule |
| `--allow-failures` | off | Accept a completed dataset with inspected reference failures |

Training seeds Python, NumPy and PyTorch and requests deterministic algorithms.
The optimizer is AdamW with gradient-norm clipping at one. With preloading,
batches are slices of tensors already on the device; the shuffle consumes the
seed exactly as PyTorch's DataLoader did, so a seed reproduces runs made before
that change (checked to six decimals over six epochs of the GPU pilot). The
loss total is read back once per epoch rather than every batch, so a
non-finite gradient is reported at the end of its epoch. Identical seeds do
not guarantee identical results across hardware or library versions.

```text
ode/runs/<name>/
├── best.pt      weights of the best validation epoch (for prediction)
├── last.pt      latest epoch + optimizer/scheduler/RNG state (for --resume)
└── run.json     settings, environment, per-epoch errors and timings, status
```

`best.pt` contains weights, architecture, encoding/schema version, normalization,
generation configuration, selected epoch, validation scores and dataset-manifest
checksum. `last.pt` adds AdamW and scheduler state, every random-number state
and the run report; `--resume` restores all of it and refuses a different
dataset or recipe. Both are written atomically and both stay out of Git.

`run.json` records configuration, seed, environment, device, source and dataset
identities, training options, epoch losses/timings and completion status. A
completed run records the checkpoint checksum. An interrupted run stays labeled
incomplete until a resume finishes it.

Add an experiment note describing the question, results, failure cases and next
decision, following [the project workflow](../planning/WORKFLOW.md). Checkpoints
and datasets stay outside Git; small reports and run notes provide reviewable
evidence.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `No module named ode` | Run from the repository root: `.venv/bin/python -m ode ...` |
| `No module named ode.train` (old commands) | Modules moved into stage folders on Oct 16; use `python -m ode train` etc. |
| Missing NumPy, SciPy or PyTorch | Check the interpreter and follow [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) |
| Command only prints a plan | Add `--generate` with `--output`, or `--train` with `--data` and `--run` |
| Run stopped part-way | Rerun the same `pipeline` command, or the same `train` command plus `--resume` |
| Output directory already exists | Choose a new name; datasets and finished runs are never overwritten |
| Dataset is incomplete | Generation was interrupted; delete that folder and generate again (it takes minutes) |
| Completed dataset has failed solves | Inspect failure records and family counts before considering `--allow-failures` |
| Scientific configuration mismatch | Omit `--config` to use the dataset's own recipe, or pick the matching preset |
| Checksum mismatch | Check for edited, damaged or partially copied artifacts; retain the original evidence |
| Training ran on CPU | `python -m ode check` shows whether PyTorch sees the GPU; `--device cuda` makes that an error instead of a fallback |
| Finite predictions are inaccurate | Check training bounds, horizon, run losses and per-component errors; finiteness is not an accuracy test |

## Verification and remaining work

Run `.venv/bin/python -m unittest discover -s ode/tests -v` from the root. Tests
cover closed-form reference accuracy, partition isolation, near-zero/nonfinite
metric behavior, validation checkpoint selection, inference reload, execution
profile precedence, loader/DataLoader equivalence, bit-identical resume and
chart well-formedness. Historical reference evidence lives in
[runs/archive/20261015-reference-validation-01](runs/archive/20261015-reference-validation-01/README.md).
Run notes written before October 16 quote the old `python -m ode.<module>`
commands; they are historical records and are left as they were.
New smoke/pilot/benchmark evidence and current counts are in
[STATUS.md](../planning/STATUS.md). Reference error does not measure neural error.

Remaining release gates are target hardware verification, checkpoint selection
and distribution, documented milestone test evaluation, and clean-environment
reproduction. Padding width six does not establish neural accuracy on untrained
orders or all-real/repeated-root families. Fluid implementation remains deferred.

## First accuracy study: diagnostics and learning rate

`python -m ode diagnose` produces numeric development reports for initial-state error,
time-local error and physical-time derivative/RHS consistency. It supports
training/validation partitions and excludes the spent test split. See
[runs/archive/20261016-validation-diagnostics-01](runs/archive/20261016-validation-diagnostics-01/README.md)
for the measured baseline patterns. The autograd diagnostic currently assumes
the pointwise linear-padded-v1 MLP input path; update it explicitly if a new
feature encoding or cross-time architecture is introduced.

Training now accepts `--learning-rate-schedule constant|cosine`,
`--minimum-learning-rate`, `--weight-decay` and `--training-seed`. The model/shuffle
seed defaults to the scientific data seed but can vary without changing data.
Cosine runs use the initial rate in epoch 1 and the minimum in the final epoch;
constant scheduling remains the default. Each epoch records its actual rate.

`best.pt` remains the standalone inference checkpoint; `last.pt` is the resume
state described above. `--no-save-training-state` disables it (and therefore
`--resume`).
