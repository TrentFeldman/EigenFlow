# ODE framework

This is an implemented foundation for future data generation and mixed-order training. No reference trajectories have been generated and no model has been trained. Numerical correctness and prediction accuracy remain unmeasured.

Read [MATH_GUIDE.md](MATH_GUIDE.md) for the equations and [ENVIRONMENT.md](ENVIRONMENT.md) for interpreter setup. Commands below run from the repository root.

## Safe checks available now

```bash
.venv/bin/python -m ode.check_environment
.venv/bin/python -m ode.generate_data --config ode/configs/generator.json
.venv/bin/python -m ode.train --config ode/configs/generator.json
.venv/bin/python -m unittest discover -s ode/tests -v
```

Both CLI commands default to dry-run. Generation planning validates settings and prints counts without sampling, solving or writing. Training planning prints settings without loading a dataset or training. Tests use fabricated in-memory states, mocked solvers, temporary metadata and an untrained checkpoint fixture; they do not integrate ODEs or retain model/data artifacts.

## File responsibilities

| File | Role |
|---|---|
| `config.py`, `configs/generator.json` | Validated sampling ranges, solver settings and worker controls |
| `equations.py` | Equation description and order-n to first-order state conversion |
| `reference.py` | Bounded integration, tighter-tolerance comparison and failure handling |
| `generate_data.py` | Deterministic sampling, worker coordination, archives and manifest |
| `dataset.py` | Integrity checks, lazy loading, mixed-order encoding and normalization |
| `model.py` | Configurable masked-order feed-forward baseline and loss |
| `train.py` | CPU training, validation selection and checkpoint/run provenance |
| `predict.py` | Saved-checkpoint inference without training data |
| `check_environment.py` | Interpreter and import readiness report |
| `evaluate.py`, `plot_results.py` | Documented placeholders; implementation pending |

## Configuration and extension points

The proposed example has orders 2 and 3, `max_order=6`, homogeneous and sinusoidally forced linear equations, a nondimensional time interval `[0, 5]`, and 101 stored times. `groups_per_split` means train/validation/test groups **per order and family**. Each group has four initial conditions: the default plan is 384 trajectories, each requiring two solves. These are draft experiment choices, not a frozen release specification.

To include order 5 in a future experiment, add 5 to `orders` and keep `max_order >= 5`. The equation and solver interfaces need no order-specific rewrite. A checkpoint's maximum order and encoding are fixed; changing them requires a new compatible model. Positive decay/frequency bounds sample decaying and oscillatory modes. The sampling distribution covers this linear family only. Nonlinear equations or more complicated forcing need an explicit equation description, sampler and encoder extension plus their own validation.

`backend` may be `thread`, `process` or `serial`; `workers` bounds concurrent requests and at most twice that many are submitted. Worker scheduling never controls sampling seeds. Thread speedup depends on Python/native execution; the process option addresses CPU-heavy Python callbacks. No throughput claim has been tested. Only DOP853, RK45, Radau and BDF are accepted; LSODA is excluded from this concurrent framework. See [SciPy thread-safety notes](https://docs.scipy.org/doc/scipy/tutorial/thread_safety.html).

## Future execution, explicitly deferred

The commands in this block are examples for a later authorized generation/training session; they have **not** been run:

```bash
.venv/bin/python -m ode.generate_data --config ode/configs/generator.json --generate --output ode/data/linear-v1
.venv/bin/python -m ode.train --config ode/configs/generator.json --train --data ode/data/linear-v1 --run ode/runs/linear-v1-baseline
```

First validate the numerical solver against analytic cases and freeze the data/evaluation specification. No existing output directory is overwritten. Dataset files are compressed NumPy archives loaded without pickle; equation metadata and all failure records live in `records.jsonl`. The manifest stores the resolved configuration, dependency versions, source hashes and ledger checksum. Each successful archive has its own checksum. Partial/interrupted datasets are rejected by the loader. A completed dataset with failures requires explicit opt-in after inspection.

The trainer uses only training data for normalization and only validation data for checkpoint selection. It saves `best.pt` and a `run.json` recording configuration, seed, environment, data/source identities and epoch losses. The default is deterministic CPU execution, four compute threads and zero DataLoader workers. `--loader-workers` can enable data-loader processes later. Test scoring and the standalone release evaluation report remain pending.

The Python inference entry point is `ode.predict.predict(checkpoint_path, equation, elapsed_times)`. It returns an array of shape `(number of times, equation.order)`, in the original nondimensional scale. It does not enforce the initial condition exactly; that behavior and the trajectory error must be evaluated before a release.

## Verification boundaries

Current checks establish software contracts: derivative ordering, masks, failure capture, checksums/split rules, default dry-run safety, concurrent thread use, and checkpoint serialization. They do **not** establish analytic solver accuracy, real-data trainability, runtime gains or the 5% accuracy target. Dedicated parameter-only, initial-condition-only and unseen-order evaluation sets are still needed alongside the current joint holdout.
