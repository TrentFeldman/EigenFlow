# EigenFlow

Neural networks that predict solutions to differential equations, measured
against trusted numerical solutions.

| Track | What it is | State |
|---|---|---|
| [ODE](ode/README.md) | Predicts solutions of linear constant-coefficient ODEs of order 1 to 4, with or without a sine-wave forcing term | Two trained models in [ode/models/](ode/models/) |
| [Fluid](fluid/README.md) | ML-assisted 2D Navier-Stokes fluid simulation | Planned, target December 11, 2026 |
| [Sim](sim/) | Sussman's AMReX multiphase solver ([upstream](https://github.com/msussman42/amrex_implicit_interfaces)), kept here as a git subtree | Modified for speed: [code changes](https://github.com/TrentFeldman/EigenFlow/commit/6ee3c18493ab8952063ad7cb5dfeed8ff5694562) |

## Sim speed changes

Trent Feldman made two changes to the solver in `sim/`
([commit 6ee3c184](https://github.com/TrentFeldman/EigenFlow/commit/6ee3c18493ab8952063ad7cb5dfeed8ff5694562)).
Both are switches set in the inputs file or on the command line.

| Switch | Default | What it does |
|---|---|---|
| `ns.dt_regrow_after_clip` | `1` | The time step is shortened to land on each frame time. The next step now grows back from the normal step size, not from the shortened one, which before could take hundreds of steps at 1% per step. `0` gives the old behaviour. |
| `ns.loop_check` | `reserved` | Checks the per-loop cell counts once per step instead of after every loop, and drops redundant barriers. The output is bit-identical. `full` gives the old per-loop check; `off` skips the check. |

The same commit fixes a bug that stopped `amr.plot_per` frames from being
written for t < 1.

**Results** (RCC, 32 cores):
- **2D Rayleigh–Taylor:** 1.65× faster. This was measured with `loop_check=off`; on a laptop, `reserved` ran at the same speed.
- **3D bubble rise:** 7.7× faster to reach the same simulation time. Time per step is 16% lower; the rest comes from taking fewer steps.

**Accuracy:** in a 2D test, the new step sizes landed closer to a reference
run without frame clipping than the old ones did.

`ns.dt_regrow_after_clip=0 ns.loop_check=full` reproduces the original
results.

**3D on RCC:** gcc 13 (`module load gnu/13`) miscompiles
`NS_fluids_lib/NAVIERSTOKES_3D.F90` at `-O3`. The run then stops at step 0
with "operation_flag invalid". This also happens with the original code.
Workaround: in `amrex-master/Tools/GNUMake/comps/gnu.mak`, change
`F90FLAGS += -g -O3` to `-g -O2`, run `touch NAVIERSTOKES_3D.F90`, then run
`make`. Newer gcc (16) on a laptop is fine at `-O3`.

## Setup

Run everything from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r ode/requirements.txt
.venv/bin/python -m ode check
```

GPU setup is in [ode/README.md](ode/README.md#setup).
