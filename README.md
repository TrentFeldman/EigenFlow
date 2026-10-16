# EigenFlow

Machine learning for numerical prediction of differential equation systems.

Current phase: ODE generator and mixed-order training framework implemented, with dry-run defaults. Small reference benchmarks passed for homogeneous and sinusoidally forced equations; no training dataset has been generated and no model has been trained.

- [Mission and repository rules](AGENTS.md)
- [Project blueprint](planning/BLUEPRINT.md)
- [Current status and next milestone](planning/STATUS.md)
- [Development and experiment workflow](planning/WORKFLOW.md)
- [Decision register](planning/DECISIONS.md)
- [ODE model plan — October 31, 2026](ode/BLUEPRINT.md)
- [ODE framework commands and documentation](ode/README.md)
- [ODE math guide for our two-person team](ode/MATH_GUIDE.md)
- [Python environment setup](ode/ENVIRONMENT.md)
- [HPC generation and Blackwell training estimates](ode/COMPUTE_ESTIMATES.md)
- [Fluid prototype plan — December 11, 2026](fluid/BLUEPRINT.md)

Next milestone: finalize ODE equation families, data format and evaluation protocol by October 18. Generation and training remain deferred. Fluid files contain documented placeholders. All new work must be understandable to both collaborators, including the programmer still learning ODE and Navier–Stokes mathematics.

## Generate the full ODE dataset when ready

From the repository root, this command requests **100,032 trajectories**, covering
homogeneous and sinusoidally forced equations of orders 2 and 3. It runs reference
validation automatically. It has not been executed as part of setup.

```bash
source .venv/bin/activate

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m ode.generate_data \
  --config ode/configs/generator_full.json \
  --generate \
  --output ode/data/linear-100k-v1
```

To preview the count/settings without generating, run:

```bash
python -m ode.generate_data --config ode/configs/generator_full.json
```

The preset uses 12 process workers locally. On an allocated 32-core HPC node,
add `--workers 32` and set `--output` to a new directory on your scratch storage.
Match workers to your actual CPU allocation. The output directory must not
already exist. Completed counts and failures are recorded in `manifest.json`.

See the [full-run instructions](ode/README.md#full-run-preset-100032-trajectories)
for the split sizes, how to scale further and the matching training configuration.
