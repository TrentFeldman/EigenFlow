# Pilot baseline — October 16, 2026

Question: can the unchanged 3x128 raw-time shared MLP meet the frozen pooled
<5% target with a bounded, reproducible CPU pilot before a 100k run?

## Provenance and configuration

Base revision `339fe2a6315333d576d7ac47363cf9ed51d77776`, with this session's uncommitted
scientific-freeze, metrics/inference/diagnostics/profiling implementation.
`run.json` and the generation manifest record source SHA-256 hashes at execution;
resource-profile support and documentation were added after this training started.
The separate resource smoke rerun verifies that this execution-only change
preserves weights when options match. No commit or push was made.

Dataset `ode/data/20261016-pilot-6656`; manifest SHA-256
`502e37df1d6175eb0a7d1be18e0ad7ce282c1193a6f6e88931196567a4e36834`. Resolved recipe and a manifest copy are
retained alongside this note. Seed 20261015; orders 2/3, homogeneous and sine
forcing, frozen bounds, 101 times on [0,5]. DOP853/refinement/matrix-exponential
checks remained enabled. All **6656 references passed**, no failures;
generation took **13.056s** with six process workers/native threads one.

Splits: 4,096 train; 1,024 joint validation; 1,024 joint test; 256 parameter-only
and 256 IC-only validation diagnostics. Diagnostics never use test identities.
Normalization fits only training. Source identities/seeds are stable across workers.

Model: 3x128 SiLU, six padded outputs, AdamW lr 1e-3, batch 1024, 100 epochs,
two CPU threads, zero loader workers, RAM preloading. Python 3.14.8,
NumPy 2.5.2, SciPy 1.18.1, PyTorch 2.14.1+rocm7.14, Ryzen 5 5600.
Total recorded training/validation time **374.581s**.
The five-epoch smoke preceded this pilot; no architecture alternatives were run.

## Selection, results and test-use record

Select **epoch 96** using mean validation relative L2 only.
Checkpoint `best.pt` is local and gitignored; SHA-256
`d2afedb7cb9cce80ccf4de1baf56dccc821912fc444c0839077f56eb575b7ca6`. The one pilot milestone test evaluation was
recorded in DECISIONS.md before reading its result. No model fitting followed
that result; the independent resource smoke only reproduces earlier smoke weights.
`run.json:test_evaluated=false` describes training; `evaluation.json` records the
subsequent explicit test use. No further test-driven tuning is authorized by this
experiment protocol. Future candidates need an explicitly reserved evaluation plan.

| Split | Count | Mean relative L2 |
|---|---:|---:|
| validation | 1024 | 3.578% |
| validation_parameter | 256 | 3.993% |
| validation_initial | 256 | 2.938% |
| test | 1024 | 3.689% |

Pooled test target **passed for this pilot checkpoint and defined domain**:
mean 3.689%, median 2.704%, p95
8.976%, worst 36.908%. **216/1,024**
trajectories were at or above 5%; nonfinite predictions: 0.
This is not a claim that every trajectory or family is below 5%.

| Test family | Mean relative L2 |
|---|---:|
| order2/linear_homogeneous | 2.253% |
| order2/linear_sinusoidal | 4.842% |
| order3/linear_homogeneous | 2.090% |
| order3/linear_sinusoidal | 5.573% |

Third-order sine forcing misses 5% as a family. See per-component, absolute,
near-zero and every-trajectory records in `evaluation.json`. The worst trajectory
is `test-o2-linear_sinusoidal-g51-i1`: its small-amplitude reference and late-time
derivative error yield a large relative error. This case is not near zero under
the frozen 1e-6 rule. The SVG overlay/error panels in `plots/` were rendered and
visually inspected. Include these failures in any presentation of the mean.

`reload-check.json` confirms bitwise repeatable independent reload and agreement
with the JSON prediction CLI. `runtime.json` records warmed CPU prediction
latencies: roughly 0.086 ms for one point, 0.187 ms for 101 points and 2.434 ms
for 4,096 points (30 repetitions, five warmups, two threads). These include
encoding/forward/de-normalization and exclude loading. They are not GPU timings
or a comparison with numerical solver speed.

## Decision and reproduction

Keep the modest baseline and 6,656-case pilot. No 100k run, larger network or
Fourier encoding is needed to demonstrate pooled feasibility. Tail and forced
family reliability remain work for the next controlled validation study. The
October release still needs artifact packaging and clean/target-environment
reproduction. Allocation/CUDA details await the user's meeting.

From the repository root, use fresh output names; never overwrite this run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m ode.generate_data --config ode/configs/generator_pilot.json \
  --resources ode/configs/resources_local.json --generate --output ode/data/pilot-reproduction
.venv/bin/python -m ode.train --config ode/configs/generator_pilot.json \
  --resources ode/configs/resources_local.json --data ode/data/pilot-reproduction \
  --run ode/runs/pilot-reproduction --epochs 100 --train
.venv/bin/python -m ode.evaluate --checkpoint ode/runs/pilot-reproduction/best.pt \
  --data ode/data/pilot-reproduction --splits validation validation_parameter validation_initial \
  --output ode/runs/pilot-reproduction/evaluation.json
```

The recorded milestone command used `--splits validation validation_parameter
validation_initial test`. Do not keep repeating test evaluation during selection.
For plots, prediction and timing use the exercised commands in `ode/README.md`.
Generation seconds and manifest digest change on regeneration; the recorded
ledger also includes timings. Sampling and numerical arrays remain deterministic.
