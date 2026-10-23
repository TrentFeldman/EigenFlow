# ODE predictor

A neural network that predicts the solution of a linear constant-coefficient
ODE of order 1 to 4, with or without a sine-wave forcing term:

```text
y^(n) + a[n-1] y^(n-1) + ... + a[1] y' + a[0] y = A sin(omega t + phi)
```

Training data comes from a numerical solver whose answers are checked twice.
Every result is scored against that solver. The math is explained in
[MATH_GUIDE.md](MATH_GUIDE.md).

## Setup

Needs Python 3.14 (tested with 3.14.8) and git. Every command below, and in the
rest of this README, runs from the **repository root** (the `EigenFlow`
folder), never from inside `ode/`.

**Linux / macOS**

```bash
git clone https://github.com/TrentFeldman/EigenFlow.git
cd EigenFlow
python3 -m venv .venv                                  # 1. private Python in .venv/
.venv/bin/python -m pip install -r ode/requirements.txt  # 2. packages
.venv/bin/python -m ode check                          # 3. Python, packages, GPU
.venv/bin/python -m unittest discover -s ode/tests     # 4. tests pass?
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/TrentFeldman/EigenFlow.git
cd EigenFlow
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r ode\requirements.txt #install gpu torches before this!
.venv\Scripts\python -m ode check
.venv\Scripts\python -m unittest discover -s ode\tests
```

On Windows the venv's Python is `.venv\Scripts\python`, not `.venv/bin/python`.

**GPU (optional).** Install the matching PyTorch build *before* step 2:

- NVIDIA: `<venv python> -m pip install torch --index-url https://download.pytorch.org/whl/cu128`
  (pick the `cuXXX` suffix for your driver at [pytorch.org](https://pytorch.org/get-started/locally/)).
- AMD (Linux only, ROCm): see pytorch.org; `requirements-local.txt` pins the exact
  ROCm wheel used locally. ROCm still uses `--device cuda`. Windows has no ROCm build,
  so AMD there runs on CPU (or use WSL).

`ode check` should then list the GPU. `python -m ode video` also needs the
`ffmpeg` program (`sudo apt install ffmpeg` / `winget install ffmpeg`).

### Reading the commands on Windows

The rest of this README uses Linux syntax. Translate:

| Linux / macOS | Windows (PowerShell) |
|---|---|
| `.venv/bin/python` | `.venv\Scripts\python` |
| `ode/models/current/best.pt` | `ode\models\current\best.pt` (forward slashes also work) |
| `\` at line end (continue command) | `` ` `` at line end |
| `export VAR=1` | `$env:VAR = "1"` |
| `ode/flagship.sh` | No equivalent; use WSL or run its commands by hand |

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

## Look inside a model (the demo explorer)

```bash
.venv/bin/python -m ode viz --compare ode/models/arc     # writes ode/models/current/explorer.html
```

This writes one self-contained HTML file. Open it in any browser: it needs no
internet, Python or server, so it can be emailed or opened on Windows by
double-clicking. The model's weights are inside the page and the page runs the
network itself, so moving a slider (fade rate, wiggle speed, starting values,
push) re-solves the equation instantly, next to the exact answer. Sliders show
the trained range as a darker band, and the page warns when you leave it.

Scenes (tabs): **network map** (every neuron and the strongest connections at
one moment), **trace** (click any neuron or output to see which neurons and
inputs fed it), **lighting up** (neuron grids over time), **sweep** (one setting
from far below to far above the trained range, with the error at each value),
**phase portrait** (hidden layers squeezed to 2D next to y against y′),
**neuron roles** (silent, rising, wiggling, ...), **input importance**
(gradient × input over time), **compare models** (`--compare`) and **worst
cases** (the worst validation equations from the model's evaluation file, with
likely reasons). **Record video** plays the scene once and saves a WebM
(`ffmpeg -i in.webm out.mp4` converts it); **Save picture** saves a PNG.
The **Colours** menu switches palettes: the default, two colour-blind friendly ones (red–green
safe for protanopia and deuteranopia, blue–yellow safe for tritanopia), and for fun garnet and gold,
ocean, forest, neon, midnight and sunset.
The browser remembers the choice for that file. Palettes are one table (`PALETTES`) in `ode/tools/viz.html`.

Options: `--model` (default `ode/models/current`), `--input` (equation JSON to
start on), `--worst N` (reel length, 0 for none), `--output`. With `--compare`
the file is about 9 MB, otherwise about 2 MB. On load, the page recomputes the
starting equation and checks it against PyTorch's numbers stored in the file.

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
| `viz` | Interactive explorer of a trained model: one portable HTML file ([details](#look-inside-a-model-the-demo-explorer)) |
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
