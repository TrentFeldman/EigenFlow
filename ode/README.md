# ODE framework

This is an implemented foundation for data generation and mixed-order training. Small numerical benchmarks now check both supported families against closed-form solutions. No training dataset has been generated and no model has been trained; model accuracy and HPC/GPU throughput remain unmeasured.

Read [MATH_GUIDE.md](MATH_GUIDE.md) for the equations and [ENVIRONMENT.md](ENVIRONMENT.md) for interpreter setup. Commands below run from the repository root.

## Safe checks available now

```bash
.venv/bin/python -m ode.check_environment
.venv/bin/python -m ode.validate_reference
.venv/bin/python -m ode.generate_data --config ode/configs/generator.json
.venv/bin/python -m ode.train --config ode/configs/generator.json
.venv/bin/python -m unittest discover -s ode/tests -v
```

Generation and training default to dry-run. Generation planning validates settings and prints counts without sampling, solving or writing. Training planning prints settings without loading a dataset or training. `validate_reference` explicitly integrates a small deterministic benchmark suite and prints its numerical errors; it never writes trajectory archives. Tests combine fabricated fixtures with these real numerical checks and retain no dataset or trained-model artifacts.

## File responsibilities

| File | Role |
|---|---|
| `config.py`, `configs/generator.json` | Validated sampling ranges, solver settings and worker controls |
| `equations.py` | Equation description and order-n to first-order state conversion |
| `reference.py` | Bounded integration, tighter-tolerance and independent checks, failure handling |
| `analytic.py` | Independent matrix-exponential states for both supported families |
| `validate_reference.py` | Closed-form numerical benchmarks and optional JSON report |
| `generate_data.py` | Deterministic sampling, worker coordination, archives and manifest |
| `dataset.py` | Integrity checks, lazy loading, mixed-order encoding and normalization |
| `model.py` | Configurable masked-order feed-forward baseline and loss |
| `train.py` | CPU/CUDA training, validation selection and checkpoint/run provenance |
| `predict.py` | Saved-checkpoint inference without training data |
| `check_environment.py` | Interpreter and import readiness report |
| `evaluate.py`, `plot_results.py` | Documented placeholders; implementation pending |

## Configuration and extension points

The proposed example has orders 2 and 3, `max_order=6`, homogeneous and sinusoidally forced linear equations, a nondimensional time interval `[0, 5]`, and 101 stored times. `groups_per_split` means train/validation/test groups **per order and family**. Each group has four initial conditions: the default plan is 384 trajectories, each requiring two solves plus five matrix-exponential check points. `independent_check_points` controls that last count, capped at the number of saved times. These are draft experiment choices, not a frozen release specification.

To include order 5 in a future experiment, add 5 to `orders` and keep `max_order >= 5`. The equation and solver interfaces need no order-specific rewrite. A checkpoint's maximum order and encoding are fixed; changing them requires a new compatible model. Positive decay/frequency bounds sample decaying and oscillatory modes. The sampling distribution covers this linear family only. Nonlinear equations or more complicated forcing need an explicit equation description, sampler and encoder extension plus their own validation.

`backend` may be `thread`, `process` or `serial`; `workers` bounds concurrent requests and at most twice that many are submitted. The example JSON uses 12 process workers for the local Ryzen 5 5600; the configuration class retains conservative thread-based fallback defaults. Worker scheduling never controls sampling seeds. Thread speedup depends on Python/native execution; the process option addresses CPU-heavy Python callbacks. No throughput claim has been tested. Only DOP853, RK45, Radau and BDF are accepted; LSODA is excluded from this concurrent framework. See [SciPy thread-safety notes](https://docs.scipy.org/doc/scipy/tutorial/thread_safety.html).

## Future execution, explicitly deferred

### Full-run preset: 100,032 trajectories

Use [configs/generator_full.json](configs/generator_full.json) when you want the
100k-plus run. It increases equation-group counts while retaining the same
orders, families, parameter ranges, times and numerical validation settings.
`generator.json` remains the smaller 384-trajectory example.

| Split | Equation groups per order and family | Planned trajectories |
|---|---:|---:|
| Train | 4,168 | 66,688 |
| Validation | 1,042 | 16,672 |
| Test | 1,042 | 16,672 |
| Total | 6,252 | **100,032** |

The count formula is:

```text
number of orders * number of families * sum(groups_per_split)
    * initial_conditions_per_group

2 * 2 * (4168 + 1042 + 1042) * 4 = 100032 trajectories
```

This means 200,064 adaptive solves, plus five independent checks per trajectory.
These are requested counts; failed solves remain in the failure ledger and do
not become training examples. No large dataset has been generated during setup.

From the repository root, preview safely:

```bash
.venv/bin/python -m ode.generate_data --config ode/configs/generator_full.json
```

When ready, execute locally using the preset's 12 process workers:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m ode.generate_data \
  --config ode/configs/generator_full.json \
  --generate --output ode/data/linear-100k-v1
```

On a single allocated 32-core HPC node, use the same command with `--workers 32`
and replace the output path with a new directory on your scratch storage.
Use the site's Python environment. Worker count must match your allocation;
the total number of cores in the whole cluster is not this job's allocation.
Native-library thread limits keep each process from creating competing thread
pools. All commands keep automatic reference preflight and per-sample checks.

The output directory must be new: choose a different run name for retries.
Review `manifest.json` for `status`, `succeeded`, `failed` and `family_counts`.
`complete` with zero failures is the clean completion state. An interrupted
dataset is incomplete and cannot be loaded for training; resuming partial
generation is not implemented.

To scale beyond this, increase the three `groups_per_split` values. With two
orders, two families and four initial-state sets, each additional group in
their sum adds 16 trajectories. Adding orders or families also changes the
count; run the dry-run again after edits.

Later training must select the matching full-run configuration, for example on
the NVIDIA GPU host (this command has not been run):

```bash
python -m ode.train --config ode/configs/generator_full.json \
  --data ode/data/linear-100k-v1 --run ode/runs/linear-100k-baseline \
  --device cuda --preload --loader-workers 0 --epochs 20 --train
```

Adjust the dataset path if using scratch storage. HPC worker/backend overrides
do not need to match training, but scientific settings and group counts do.
The rough 100k estimates are 2-15 minutes for generation on one 32-core node
and 10-45 minutes for 20 training epochs on one RTX PRO 6000 Blackwell; these
are unmeasured projections. See [COMPUTE_ESTIMATES.md](COMPUTE_ESTIMATES.md).

### Small example

The commands in this block are examples for a later authorized generation/training session; they have **not** been run:

```bash
.venv/bin/python -m ode.generate_data --config ode/configs/generator.json --generate --output ode/data/linear-v1
.venv/bin/python -m ode.train --config ode/configs/generator.json --train --data ode/data/linear-v1 --run ode/runs/linear-v1-baseline
```

Freeze the data/evaluation specification before production generation. Every generation run first checks both configured families against closed forms and refuses to create a dataset if preflight fails. Every sample then passes tolerance refinement and independent checks. No existing output directory is overwritten. Dataset files are compressed NumPy archives loaded without pickle; equation metadata and all failure records live in `records.jsonl`. The manifest stores the benchmark report, per-family counts, resolved configuration, timing, dependency versions, source hashes and ledger checksum. Each successful archive has its own checksum. Partial/interrupted datasets are rejected by the loader. A completed dataset with failures requires explicit opt-in after inspection.

The trainer uses only training data for normalization and only validation data for checkpoint selection. It saves `best.pt` and a `run.json` recording configuration, seed, device/build, environment, data/source identities, epoch timing and losses. The default is deterministic CPU execution, four compute threads and zero DataLoader workers. `--device cuda` requests a GPU and fails clearly if unavailable; `--device auto` selects an available GPU or CPU. A compatible CUDA PyTorch build is needed for NVIDIA hardware. `--preload` stores scaled examples in host RAM to avoid repeated archive decompression; keep `--loader-workers 0` with this option. Test scoring and the standalone release evaluation report remain pending.

Use `--workers` and `--backend` on the generation CLI to match an HPC allocation without editing JSON. Training checks scientific configuration equality but permits backend/worker differences. See [COMPUTE_ESTIMATES.md](COMPUTE_ESTIMATES.md) for single-node HPC and RTX PRO 6000 Blackwell assumptions, commands and rough timings.

The Python inference entry point is `ode.predict.predict(checkpoint_path, equation, elapsed_times)`. It returns an array of shape `(number of times, equation.order)`, in the original nondimensional scale. It does not enforce the initial condition exactly; that behavior and the trajectory error must be evaluated before a release.

## Verification boundaries

Current checks establish software contracts plus agreement with closed-form solutions for bounded cases covering both families, repeated/distinct real roots, damped complex roots, order 6 and resonant forcing. Every generated sample also receives independent checks at selected times, not a proof of error bounds between those times. These checks do **not** establish correctness for every input domain, real-data trainability, GPU performance or the 5% model target. Dedicated parameter-only, initial-condition-only and unseen-order evaluation sets are still needed alongside the current joint holdout. The sampler still does not deliberately generate repeated/all-real root families; benchmark coverage is distinct from training-distribution coverage.
