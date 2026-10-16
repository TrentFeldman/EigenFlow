# Project status

Updated: October 15, 2026.

Current phase: both-family numerical-reference validation and CPU/CUDA training preparation. Small numerical benchmarks have passed; no training dataset has been generated and no model has been trained. Model accuracy and actual HPC/GPU throughput remain unmeasured.

## Verified foundation

- Configurable positive-order state conversion and stable linear-family parameter sampler.
- Dry-run-by-default generator, bounded thread/process/serial coordination, deterministic job seeds, failure ledger and integrity manifest. CLI worker/backend overrides support a single-node allocation.
- Both configured families pass closed-form preflight. Each future trajectory must additionally pass tolerance refinement and matrix-exponential checks at five configured times.
- Mixed-order padded features, masked loss, training-only normalization, CPU/CUDA trainer, optional host-RAM preloading and independent checkpoint inference. GPU execution is not hardware-tested locally.
- Existing isolated Python environment checked; SciPy added; NumPy/PyTorch retained; dependency compatibility check passed.
- Generator example configured for the local Ryzen 5 5600 with 12 process workers; numerical generation and performance benchmarking remain deferred.
- Team documentation rules, plain-language math guide and documented roles for every Python file, including fluid placeholders.
- All 24 unit checks passed, including real closed-form numerical checks, process-worker execution, dry-run safety, independent-check failure detection and preload/device contracts.
- All 13 deterministic benchmark cases passed across homogeneous/sine-forced equations, repeated/distinct roots, damped oscillations, orders 2/3/6 and resonance. Worst absolute state-component error was about 5.6e-10; this is reference accuracy on these cases, not neural accuracy.
- Validation evidence: `ode/runs/20261015-reference-validation-01/report.json` and its run note. No dataset or trained checkpoint was produced.
- `ode/COMPUTE_ESTIMATES.md` records explicitly unmeasured single-node HPC / RTX PRO 6000 Blackwell projections, including queue/I/O and hardware assumptions.
- Added `ode/configs/generator_full.json` and root/ODE README instructions for 100,032 planned trajectories (66,688 train; 16,672 validation; 16,672 test). Only the preset dry-run was checked; full generation remains user-triggered.

## Immediate milestone — October 18

| Task | State | Completion evidence |
|---|---|---|
| Choose second- and third-order families | planned | Equations, parameter bounds and rationale recorded |
| Freeze inputs, outputs and time domain | planned | Complete data contract in ODE blueprint |
| Define reference validation | planned | Solver choice, tolerances and convergence checks specified |
| Freeze evaluation protocol | planned | Split recipe, metrics, near-zero handling and target scope recorded |
| Estimate initial experiment size | planned | Hardware budget, dataset sizes and training budget recorded |

Next action: freeze the October 18 data/evaluation specification and determine the actual HPC CPU allocation. Bulk dataset generation and training remain deferred by the user's explicit clarification; the current task performed only code changes and small validation cases.

## Later milestones

| Due | Deliverable | State |
|---|---|---|
| Oct 23 | Trained ODE baseline | planned |
| Oct 27 | Higher-order evaluation and reliability improvements | planned |
| Oct 30–31 | Reproducible ODE v1.0 | planned |
| Nov 6 | Validated fluid reference solver | planned |
| Nov 13 | Reproducible fluid dataset | planned |
| Nov 20 | Trained single-step fluid predictor | planned |
| Nov 27 | Evaluated autoregressive rollout and stabilization | planned |
| Dec 4 | Working comparison demo | planned |
| Dec 11 | Evaluated semester release | planned |

## Risks and measured progress

- Equation ranges, compute budget and numerical tolerances are not yet settled.
- Reference quality and leakage-free splits must be established before model results are meaningful.
- The 5% model accuracy goal is untested. Only small local reference benchmarks are measured; HPC generation and Blackwell training estimates still need target-hardware benchmarks.
- If timing becomes threatened, reduce family scope explicitly before reducing validation or documentation.

## Review entry template

Date; verified tasks and evidence; measured results; blockers and risks; scope changes; next deliverable; target date.
