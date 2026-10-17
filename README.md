# EigenFlow

EigenFlow is a machine learning research project for predicting solutions to
differential equations. We build reproducible models and measure their accuracy
against trusted numerical solutions.

The first deliverable is a higher-order **ordinary differential equation (ODE)**
predictor, targeted for **October 31, 2026**. The second is an ML-assisted
**two-dimensional Navier–Stokes fluid prototype**, with an evaluated release
targeted for **December 11, 2026**.

Our guiding principle: produce working, measurable systems first, then improve
them through controlled experiments.

## Current state

Implementation snapshot: **October 16, 2026**.

| Area | Available now | Still needed |
|---|---|---|
| ODE reference data | Deterministic sampling, numerical integration, independent checks, grouped splits and checksummed archives | Release dataset/checkpoint selection |
| ODE learning | Mixed-order training, release metrics, inference CLI, plots and resource profiles | Release checkpoint distribution and final reproduction |
| Verification | Discoverable tests, 13 closed-form cases, recorded CPU smoke/pilot and timing runs | Actual HPC allocation and target CUDA validation |
| Fluid simulation | Documented scope and module placeholders | Numerical solver, dataset, learned predictor, rollout and evaluation |

The scientific contract was frozen October 16. Recorded smoke and 6,656-case
pilot experiments exercise the full CPU workflow; see [STATUS.md](planning/STATUS.md)
for measured accuracy and limitations. The 100,032-case preset is unused, and
there is no October release yet. The release target is **less than 5% mean
relative trajectory L2 error** on the reserved joint holdout.

## Getting started

Run these commands from the repository root. They use the local virtual
environment explicitly, so shell activation is optional.

### 1. Check the environment

If `.venv` already exists:

```bash
.venv/bin/python -m ode check
.venv/bin/python -m pip check
```

For a fresh clone without a virtual environment, use the tested Python 3.14.8 interpreter:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r ode/requirements.txt
.venv/bin/python -m ode check
.venv/bin/python -m pip check
```

Dependencies are NumPy, SciPy and PyTorch. A GPU is optional; CPU execution is
the default. Direct dependencies and the Python version are pinned; GPU training
requires a PyTorch build compatible with the host. See the
[environment guide](ode/docs/ENVIRONMENT.md) for the locally checked versions.

### 2. Verify the numerical foundation

```bash
.venv/bin/python -m ode validate
.venv/bin/python -m unittest discover -s ode/tests -v
```

Reference validation solves small equations with known solutions. Passing these
checks establishes numerical behavior on those cases; neural prediction accuracy
must be measured separately.

### 3. Preview an experiment

```bash
.venv/bin/python -m ode generate --config ode/configs/generator.json
.venv/bin/python -m ode train --config ode/configs/generator.json
```

Both commands default to a dry run: they print settings without creating a
dataset or training a model. The small preset requests 384 trajectories across
second- and third-order equations, with and without sinusoidal forcing.

### 4. Generate, train, evaluate and plot

```bash
.venv/bin/python -m ode pipeline --name my-first-run --preset small --epochs 20
```

One command runs every stage and is safe to rerun if interrupted. The
[ODE README](ode/README.md#quick-start-one-command) explains each stage, the
individual commands, the charts it draws and how to submit it on the HPC cluster.
Presets are `small` (384 equations), `pilot` (6,656) and `full` (100,032); worker
and thread counts follow the CPUs allocated to the job automatically.

## What the ODE model learns

An ODE describes how a quantity changes over time. A second-order equation
includes its second derivative; for example:

```text
y'' + y' + 1.25 y = 0
y(0) = 0.5,  y'(0) = 0
```

Here `y` is the quantity, `y'` is its rate of change, and `y''` is the rate of
change of `y'`. The numerical solver produces reference states `[y, y']` at
sampled times. The neural model learns to estimate those states from the
equation coefficients, initial conditions and requested elapsed time.

For a third-order equation, the state becomes `[y, y', y'']`. The shared model
uses padding and a mask so both orders fit the same input/output format. Its
default padding width is six, but training on orders two and three does not
establish accuracy on unseen orders.

The current model predicts each requested time directly. All current variables
are nondimensional, and the presets cover linear equations over elapsed times
from zero to five. See the [math guide](ode/docs/MATH_GUIDE.md) for the derivation and
the [ODE README](ode/README.md) for array shapes, bounds and limitations.

```text
Equation configuration
        ↓
Validated numerical trajectories
        ↓
Grouped train / validation / test splits
        ↓
Training-only normalization → neural model training
        ↓
Validation-selected checkpoint → standalone prediction
        ↓
Held-out evaluation, failure reports, plots and runtime
```

## Repository map

```text
ODE/
├── README.md                 Project overview and starting point
├── ode/                      Active ODE implementation
│   ├── configs/              Small and full dataset recipes
│   ├── tests/                Numerical and software checks
│   ├── data/                 Local datasets; ignored by Git
│   ├── runs/                 Reports and local checkpoints
│   └── README.md             Commands, equations and artifact contracts
├── fluid/                    Planned fluid prototype; code is scaffold-only
├── planning/                 Status, designs, decisions and workflow
├── .venv/                    Local Python environment, when installed
└── .experiments/             Ignored archive of earlier work, when present
```

Dataset and run directories are created as needed. Data and model weights
(`*.pt`, `*.pth`) are ignored by Git. Keep small run notes and evaluation evidence
with their experiments. New work belongs in the active project, outside the
archived experiments.

## Delivery plan

All dates below are targets for **2026**; completion requires recorded evidence.

| Date | Deliverable |
|---|---|
| October 18 | Freeze ODE families, data format and evaluation protocol |
| October 23 | First functioning trained ODE predictor |
| October 27 | Higher-order tests and reliability improvements |
| **October 30–31** | **Reproducible ODE v1.0 with checkpoint and measured results** |
| November 6 | Validated 2D fluid reference solver |
| November 13 | Reproducible fluid training dataset |
| November 20 | First trained fluid-field predictor |
| November 27 | Evaluated multi-step rollout and stabilization |
| December 4 | Working fluid comparison demo |
| **December 11** | **Evaluated and documented semester prototype** |

The planned fluid scope is incompressible 2D flow with periodic boundaries on an
initial 64×64 grid. It will compare predicted velocity and vorticity with a
validated numerical reference. The ODE release remains the first priority; see
the [fluid blueprint](fluid/BLUEPRINT.md) for later stage gates.

## Documentation and collaboration

| Read this | For |
|---|---|
| [ODE README](ode/README.md) | Runnable examples, settings, data layout and troubleshooting |
| [ODE math guide](ode/docs/MATH_GUIDE.md) | Equations, derivatives and numerical reasoning |
| [Roots and extensions](ode/docs/ROOTS_AND_EXTENSIONS.md) | Characteristic roots and possible family extensions |
| [Environment guide](ode/docs/ENVIRONMENT.md) | Interpreter, dependencies and device setup |
| [Compute estimates](ode/docs/COMPUTE_ESTIMATES.md) | Measured local timings and unresolved target allocation |
| [Project status](planning/STATUS.md) | Dated milestone tracking and next actions |
| [Project blueprint](planning/BLUEPRINT.md) and [ODE blueprint](ode/docs/BLUEPRINT.md) | Architecture and frozen scientific contract |
| [Decision register](planning/DECISIONS.md) | Design choices and unresolved proposals |
| [Working system](planning/WORKFLOW.md) | Experiment records, reviews and release evidence |
| [Repository instructions](AGENTS.md) | Team conventions and mission |

This is a two-person project, including a collaborator learning the underlying
mathematics. Explain symbols, array shapes and numerical choices in plain
language. Preserve each experiment's configuration, seed, dataset identity,
environment, results and failures. Fit preprocessing on training data, select
models on validation data, and reserve the test set for documented milestone
evaluation.

A release needs executable code, accessible model artifacts, measured results
and reproduction instructions. Attractive plots alone do not establish accuracy.
