# Released ODE model

`best.pt`: a 336,646-parameter network (6 hidden layers of 256 SiLU units, encoding `linear-padded-v1`) that predicts the solution
of linear constant-coefficient ODEs of order 2 or 3, homogeneous or forced by a sine wave.
SHA-256 `4aa0cca3a9f6282644bdb68d3d023c87385efd800adfc88f29c3dc9e0e33b04e`.

## Accuracy (validation split, 40,000 equations held out from training)

Chart (open in a browser): [report.svg](report.svg).

Metric: relative L2 error of the whole trajectory, `trajectory-rms-v1` (defined in `ode/README.md`).
**The sealed test split has not been scored for this model.**

| | Error |
|---|---|
| Average | **0.149%** |
| Median | 0.123% |
| 95% of equations under | 0.327% |
| Worst case | 7.6% |
| Equations at or above 5% | 1 of 40,000 |

| Family | Average | 95% under | Worst |
|---|---|---|---|
| order2 / homogeneous | 0.079% | 0.140% | 2.0% |
| order2 / sinusoidal | 0.183% | 0.357% | 7.6% |
| order3 / homogeneous | 0.118% | 0.222% | 1.9% |
| order3 / sinusoidal | 0.217% | 0.425% | 2.7% |

Per-equation numbers (`evaluation_summary.json`, 54 MB) stay local; `python -m ode evaluate` regenerates them.

## Where it works

Trained on `ode/configs/generator_wide.json` (480,000 equations): decay 0.01-1.5, oscillation speed 0.5-4,
starting values within +-1, forcing strength 0.1-1, times 0-5. Inside that range `Predictor.predict` is used directly.
Outside it, `Predictor.predict_rescaled` (used automatically by `python -m ode predict`) rescales time, splits the answer into
a starting-value part and a forced part, and stitches several model runs. That wrapper is exact algebra for linear equations
but is not part of the learned model and was checked only on informal test sets. Not supported: other orders, nonlinear equations,
growing or purely undamped modes.

## How it was trained

Stage 1, from scratch, dataset `wide` (hash `f29324eed0ae391d...`):
`python -m ode pipeline --name wide-w256-d6-cosine --preset wide --epochs 280 --batch-size 1024 --width 256 --depth 6 --learning-rate-schedule cosine`

Stage 2, 100 more epochs from the stage-1 weights (`wide-finetune-relative-focus`), same flags as stage 3 below.

Stage 3, 250 more epochs from the stage-2 weights (9,437 s on the local 7900 XTX):
`python -m ode pipeline --name wide-finetune-2 --preset wide --epochs 250 --batch-size 1024 --width 256 --depth 6 --learning-rate-schedule cosine --learning-rate 0.00005 --minimum-learning-rate 0.000001 --initialize-from <stage-2 best.pt> --compile --loss relative --focus-hard`

## Try it

`python -m ode predict --checkpoint ode/models/current/best.pt --input ode/equations/try_equation.json`
