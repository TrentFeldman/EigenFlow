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
| ODE reference data | Deterministic sampling, numerical integration, independent checks, grouped splits and checksummed archives | Frozen experiment specification and production dataset |
| ODE learning | Mixed-order PyTorch model, CPU/CUDA training interface, saved normalization and checkpoint inference | Evaluated release checkpoint, held-out metrics and example plots |
| Verification | 25 passing tests, 13 passing closed-form reference cases, temporary CPU generation/training smoke checks | Full-scale throughput and GPU hardware validation |
| Fluid simulation | Documented scope and module placeholders | Numerical solver, dataset, learned predictor, rollout and evaluation |

Small CPU runs have exercised generation and training during development. These
are smoke checks; there is no released, evaluated ODE model yet. The **less-than-5%
mean relative trajectory L2 error** goal has not been established. The
100,032-trajectory preset has been previewed, but its full execution and performance
measurements remain pending.

## Getting started

Run these commands from the repository root. They use the local virtual
environment explicitly, so shell activation is optional.

### 1. Check the environment

If `.venv` already exists:

```bash
.venv/bin/python -m ode.check_environment
.venv/bin/python -m pip check
```

For a fresh clone without a virtual environment, use Python 3.11 or newer:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r ode/requirements.txt
.venv/bin/python -m ode.check_environment
.venv/bin/python -m pip check
```

Dependencies are NumPy, SciPy and PyTorch. A GPU is optional; CPU execution is
the default. Dependency ranges are not a locked environment, and GPU training
requires a PyTorch build compatible with the host. See the
[environment guide](ode/ENVIRONMENT.md) for the locally checked versions.

### 2. Verify the numerical foundation

```bash
.venv/bin/python -m ode.validate_reference
.venv/bin/python -m unittest discover -s ode/tests -v
```

Reference validation solves small equations with known solutions. Passing these
checks establishes numerical behavior on those cases; neural prediction accuracy
must be measured separately.

### 3. Preview an experiment

```bash
.venv/bin/python -m ode.generate_data --config ode/configs/generator.json
.venv/bin/python -m ode.train --config ode/configs/generator.json
```

Both commands default to a dry run: they print settings without creating a
dataset or training a model. The small preset requests 384 trajectories across
second- and third-order equations, with and without sinusoidal forcing.

### 4. Generate, train and predict

Follow the [complete ODE walkthrough](ode/README.md#run-the-small-example) for
generation, training and a Python prediction example. Execution requires
`--generate` or `--train` and a new output directory.

For the larger experiment, use the
[100,032-trajectory instructions](ode/README.md#full-run-preset-100032-trajectories).
Use the same scientific configuration for generation and training, and match
worker counts to the CPUs actually allocated to the job.

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
from zero to five. See the [math guide](ode/MATH_GUIDE.md) for the derivation and
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
Held-out evaluation and plots (implementation pending)
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
| [ODE math guide](ode/MATH_GUIDE.md) | Equations, derivatives and numerical reasoning |
| [Roots and extensions](ode/ROOTS_AND_EXTENSIONS.md) | Characteristic roots and possible family extensions |
| [Environment guide](ode/ENVIRONMENT.md) | Interpreter, dependencies and device setup |
| [Compute estimates](ode/COMPUTE_ESTIMATES.md) | HPC/GPU assumptions and unmeasured projections |
| [Project status](planning/STATUS.md) | Dated milestone tracking and next actions |
| [Project blueprint](planning/BLUEPRINT.md) and [ODE blueprint](ode/BLUEPRINT.md) | Architecture and proposed release contract |
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
