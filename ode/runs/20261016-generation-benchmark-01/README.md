# Generation benchmark — October 16, 2026

Question: is the 12-worker default justified before selecting a larger dataset?
Base revision `339fe2a6315333d576d7ac47363cf9ed51d77776` plus this session's numerical-stage
instrumentation and benchmark code; manifests in report.json retain source hashes
and resolved configuration. Seed 20261015, 384 trajectories, same frozen families,
range/time/tolerance recipe in every repetition. No trained model or test scoring.

On Ryzen 5 5600, Python 3.14.8, NumPy 2.5.2, SciPy 1.18.1: three full generations
per setting gave median 4.039s serial, 1.174s six processes, 1.448s twelve.
All 3,456 generated references passed. Temporary datasets were removed; report
contains their manifests/checksums and stage timing totals. Native threads were
one. Includes preflight/startup/archive I/O; /tmp storage/cache behavior is local.

Decision: use six local workers, preserve both numerical checks, select 6,656
pilot trajectories before 100k. No target-HPC timing or 32-core allocation claim.

Reproduction (fresh report path):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m ode.benchmark generation --workers 1 6 12 --repeats 3 \
  --output ode/runs/generation-reproduction/report.json
```

See `ode/COMPUTE_ESTIMATES.md` for stage-cost interpretation and CPU allocation
limits. Do not remove refinement just because it dominates validation cost.
