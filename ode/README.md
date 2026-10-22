# ODE predictor

A neural network that predicts the solution of a linear constant-coefficient
ODE of order 1 to 4, with or without a sine-wave forcing term:

```text
y^(n) + a[n-1] y^(n-1) + ... + a[1] y' + a[0] y = A sin(omega t + phi)
```

Training data comes from a numerical solver whose answers are checked twice.
Every result is scored against that solver. The math is explained in
[MATH_GUIDE.md](MATH_GUIDE.md).

All commands run from the **repository root**.

## Linux vs Windows

Examples use Linux/macOS (bash). On Windows use PowerShell and translate:

| Linux / macOS | Windows (PowerShell) |
|---|---|
| `python3 -m venv .venv` | `py -m venv .venv` |
| `.venv/bin/python` | `.venv\Scripts\python` |
| `source .venv/bin/activate` | `.venv\Scripts\Activate.ps1` |
| `ode/models/current/best.pt` | `ode\models\current\best.pt` (forward slashes also work) |
| `cmd \` (line continuation) | `` cmd ` `` (backtick) |
| `export VAR=1` | `$env:VAR = "1"` |
| `ls`, `cat`, `rm -r` | `dir`, `type`, `rmdir /s` |
| `sudo apt install ffmpeg` | `winget install ffmpeg` |
| `ode/flagship.sh` | No equivalent; use WSL or run its commands by hand |

Windows has no ROCm PyTorch build; AMD GPUs there run on CPU (or use WSL).

NVIDIA on Windows: plain `pip install torch` gives the CPU build. Install the
CUDA build first (pick the `cuXXX` suffix at [pytorch.org](https://pytorch.org/get-started/locally/)):

```powershell
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r ode\requirements.txt
.venv\Scripts\python -m ode check   # should list the GPU
```

## Setup

Tested with Python 3.14.8 (`.python-version`).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r ode/requirements.txt
.venv/bin/python -m ode check       # Python, packages, GPU
.venv/bin/python -m unittest discover -s ode/tests
```

A GPU is optional. For one, install the PyTorch build that matches your
hardware from [pytorch.org](https://pytorch.org/get-started/locally/) before
the requirements. AMD ROCm builds also use `--device cuda`.
`requirements-local.txt` pins the exact ROCm wheel used locally.
`python -m ode video` also needs the `ffmpeg` program.

## Predict with a released model

| Model | Covers | Mean error |
|---|---|---|
| [models/current](models/current/MODEL.md) | Orders 2-3 | 0.149% |
| [models/arc](models/arc/MODEL.md) | Orders 1-4, decaying oscillations | 0.227% |

Each `MODEL.md` lists the trained ranges, per-order scores and the exact
training commands.

Describe the equation in a JSON file (see `ode/equations/` for examples):

```json
{
  "equation": {
    "coefficients": [2.0, 0.6],
    "initial_state": [1.0, 0.0],
    "amplitude": 0.0,
    "frequency": 0.0,
    "phase": 0.0
  }
}
```

- `coefficients`: lowest derivative first; the highest is always 1.
  `[2.0, 0.6]` means `y'' + 0.6 y' + 2 y`.
- `initial_state`: `[y(0), y'(0), ...]`, one value per order.
- `amplitude`, `frequency`, `phase`: the sine forcing; all zero for none.
- `elapsed_times` (optional): times to predict; default is the trained grid.

```bash
.venv/bin/python -m ode predict --checkpoint ode/models/current/best.pt \
  --input ode/equations/try_equation.json
```

This writes `try_equation.svg` (exact vs predicted chart) and
`try_equation.csv` next to the input. `--output FILE.json` writes JSON instead.
Equations outside the trained ranges are rescaled into them automatically; if
that is impossible, the command refuses unless `--allow-extrapolation` is given.

From Python:

```python
import numpy as np
from ode.generation.equations import LinearODE
from ode.evaluation.predict import Predictor

equation = LinearODE(coefficients=(1.25, 1.0), initial_state=(0.5, 0.0))
times = np.linspace(0.0, 5.0, 101)
states = Predictor("ode/models/current/best.pt").predict(equation, times)  # (101, 2): y, y'
```

## Train a model

```bash
.venv/bin/python -m ode pipeline --name my-run --preset pilot --epochs 300 --batch-size 1024
```

| Stage | Writes |
|---|---|
| 1. generate | `ode/data/<preset>-<count>/`, reused by later runs |
| 2. train | `ode/runs/<name>/best.pt`, `last.pt`, `run.json` |
| 3. evaluate | `ode/runs/<name>/evaluation.json` (validation only) |
| 4. plot | `ode/runs/<name>/report.svg`, one chart of the model's accuracy |

If it stops, rerun the same command: finished stages are skipped and training
resumes. `python -m ode runs` lists every run and dataset with its score.

| Preset | Equations | Orders | Notes |
|---|---:|---|---|
| `small` | 384 | 2-3 | Wiring check |
| `pilot` | 6,656 | 2-3 | |
| `full` | 100,032 | 2-3 | |
| `wide` | 480,000 | 2-3 | Wider damping and frequency; trained `current` |
| `final` | 824,960 | 1-4 | Widest ranges, times 0-10 |

Any `ode/configs/generator_*.json` path also works as `--preset`
(`generator_orders4.json` trained `arc`).

Useful training options (`python -m ode train --help` lists all; the pipeline
passes them through):

| Option | Default | Purpose |
|---|---|---|
| `--epochs` | 20 | Passes through the data |
| `--batch-size` | 256 | Examples per update |
| `--width`, `--depth` | 128, 3 | Network size |
| `--learning-rate` | 0.001 | AdamW learning rate |
| `--learning-rate-schedule` | constant | `cosine` decays to `--minimum-learning-rate` |
| `--loss` | mse | `relative` weights each trajectory by 1/its size, like the score |
| `--focus-hard` | off | Train more on the trajectories with the largest error |
| `--initialize-from PATH` | off | Start from another run's `best.pt` (fine-tuning) |
| `--compile` | off | `torch.compile` on a GPU, about 1.6x faster |
| `--device` | auto | `cpu`, `cuda` or `auto` |
| `--resources FILE` | | Hardware profile (`configs/resources_*.json`); flags override it |
| `--snapshot-every N` | off | Save weights every N epochs for `neurons` / `video` |
| `--resume` | | Continue a run from its `last.pt` |

The released models were trained with a cosine schedule, then fine-tuned with
`--initialize-from`, a lower learning rate, `--loss relative` and `--focus-hard`.

## Error metric

For one trajectory, `E` is the root-mean-square of `prediction - reference`
over every saved time and state component, and `R` is the same for the
reference alone. The error is `E / max(R, 1e-6)`, so 2% means the miss is about
2% of the size of the solution. Reported numbers average this over trajectories
with equal weight (metric version `trajectory-rms-v1`). The test split is never
scored during development.

## Commands

`python -m ode` lists them; add `--help` after any one.

| Command | What it does |
|---|---|
| `pipeline` | Generate, train, evaluate and plot in one resumable run |
| `predict` | Predict one equation from a JSON file |
| `generate`, `train`, `evaluate`, `plot` | One stage at a time; `generate` and `train` only print a plan without `--generate` / `--train` |
| `runs` | List runs and datasets with scores and sizes |
| `diagnose` | Break validation error down by regime |
| `validate` | Check the solver against exact formulas |
| `neurons`, `video` | Animate training as HTML or MP4 (needs `--snapshot-every`) |
| `check`, `benchmark` | Environment report; generation or inference timing |

## Layout

```text
ode/
    generation/    make datasets (sampling, solver, checks)
    training/      dataset loading, model, training loop
    evaluation/    scoring, charts, prediction
    configs/       dataset presets and hardware profiles
    models/        released models (the only checkpoints in git)
    equations/     example equations for predict
    tests/         unit tests
    data/, runs/   generated datasets and runs (local only)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named ode` | Run from the repository root |
| Command only prints a plan | Add `--generate --output DIR` or `--train --data DIR --run DIR` |
| Output already exists | Choose a new name; nothing is overwritten |
| Dataset is incomplete | Generation was interrupted; delete the folder and rerun |
| Dataset has failed solves | Inspect `records.jsonl`, then pass `--allow-failures` |
| Training ran on CPU | `python -m ode check`; `--device cuda` makes a missing GPU an error |
