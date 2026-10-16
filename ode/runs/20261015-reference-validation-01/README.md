# Both-family reference validation

Purpose: check the homogeneous and sinusoidally forced numerical reference path
against explicit closed-form solutions, including repeated roots and resonance.
This is a validation run, not training-data generation or model training.

Command, from the repository root:

```bash
.venv/bin/python -m ode.validate_reference --report ode/runs/20261015-reference-validation-01/report.json
```

The output path must be new to reproduce the run; the CLI refuses to overwrite
an existing report. Use another run directory or omit --report to print only.

Configuration: `ode/configs/generator.json`, orders 2/3 plus order 6 numerical
checks, interval [0, 5], 101 stored times, DOP853 at rtol=1e-9 and atol=1e-11,
refinement factor 10, independent reference allowance 1e-8 + 1e-6*abs(reference).
The resolved settings, dependency versions and numerical-source hashes are in
[report.json](report.json).

Hardware: local AMD Ryzen 5 5600; validation cases run serially, not on the HPC
cluster or Blackwell GPU. Outcome: all 13 closed-form cases passed. Worst absolute
state-component error was about 5.6e-10. This is bounded numerical-reference
evidence, not a neural-model accuracy result or proof over all parameters.

Related verification: all 24 unit tests passed, including process-worker solving,
independent-check rejection of consistently wrong solver replies, preload parity,
device selection and dry-run safety. CUDA execution remains hardware-untested.

Decision: retain both supported families and require closed-form preflight plus
per-trajectory refinement/independent checks on future generation runs. Next:
freeze scientific data/evaluation scope, then benchmark an authorized dataset
and training run on the actual allocated hardware. Current compute estimates are
in [COMPUTE_ESTIMATES.md](../../COMPUTE_ESTIMATES.md).
