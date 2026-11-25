# EigenFlow

Neural networks that predict solutions to differential equations, measured
against trusted numerical solutions.

| Track | What it is | State |
|---|---|---|
| [ODE](ode/README.md) | Predicts solutions of linear constant-coefficient ODEs of order 1 to 4, with or without a sine-wave forcing term | Two trained models in [ode/models/](ode/models/) |
| [Fluid](fluid/README.md) | ML-assisted 2D Navier-Stokes fluid simulation | Planned, target December 11, 2026 |
| [Sim](sim/) | Sussman's AMReX multiphase solver ([upstream](https://github.com/msussman42/amrex_implicit_interfaces)), kept here as a git subtree | Modified for speed: [code changes](https://github.com/TrentFeldman/EigenFlow/commit/6ee3c18493ab8952063ad7cb5dfeed8ff5694562) |

## Sim speed changes

Trent Feldman made five changes to the solver in `sim/`. The first two are in
[commit 6ee3c184](https://github.com/TrentFeldman/EigenFlow/commit/6ee3c18493ab8952063ad7cb5dfeed8ff5694562);
the other three came later. Each is a switch set in the inputs file or on the
command line.

| # | Switch | Default | What it does |
|---|---|---|---|
| 1 | `ns.dt_regrow_after_clip` | `1` | The time step is shortened to land on each frame time. The next step now grows back from the normal step size, not from the shortened one, which before could take hundreds of steps at 1% per step. `0` gives the old behaviour. |
| 2 | `ns.loop_check` | `reserved` | Checks the per-loop cell counts once per step instead of after every loop, and drops redundant barriers. `full` gives the old per-loop check; `off` skips the check. |
| 3 | `ns.inner_totals` | `reserved` | Computes the diagnostic totals inside a step once per step instead of several times. With solids or user-defined sums, every call still computes them. The totals printed at the end of a step are unchanged. `full` gives the old behaviour; `off` computes them only where something reads them. |
| 4 | `ns.mf_check` | `reserved` | Every rank still checks its arrays at every call, but the ranks compare results once per step instead of after each check. `full` gives the old behaviour; `off` skips the checks. |
| 5 | `ns.facefrac` | `reserved` | Reuses the face and cell fractions in `ColorSum` when their inputs are byte-identical, and rebuilds and compares them every `ns.facefrac_audit` steps (20). `full` always rebuilds; `check` compares at every reuse. |

Commit 6ee3c184 also fixes a bug that stopped `amr.plot_per` frames from being
written for t < 1.

**Results** against the original code (RCC, 32 cores, default switches, one run
each, times are the sum of the step times):

| Case | Original | Changes 1–2 | All five |
|---|---|---|---|
| 2D Rayleigh–Taylor, t = 0.144 to 0.16 | 5,787 s | 3,556 s (1.63×) | 2,648 s (2.19×, 54% less time) |
| 3D bubble rise, to t = 0.15 | 8,491 s | 2,870 s (2.96×) | 2,230 s (3.81×, 74% less time) |

Much of the 3D gain is change 1 taking fewer steps (90 to 36), which depends on
where the frame times fall. In an earlier test of changes 1–2 that ran further,
to t = 0.169, it reached 7.7×. Changes 3–5 alone save 26% (2D) and 22% (3D) of
the time left by changes 1–2.

**Accuracy:**
- Changes 2–5 gave bit-identical plot frames in every comparison. For changes 3–5 that covers 2D and 3D runs, a 2D restart and four 2D cases with solids; 3D restarts and 3D solids are untested.
- Change 1 changes the step sizes, so its results differ slightly from the original. In 2D and 3D tests they were closer to a reference run without frame clipping than the original's were. The reference is not a converged solution.

To switch everything back to the original behaviour:
`ns.dt_regrow_after_clip=0 ns.loop_check=full ns.inner_totals=full ns.mf_check=full ns.facefrac=full`.
The frame bug fix stays on.

**3D on RCC:** gcc 13 (`module load gnu/13`) miscompiles
`NS_fluids_lib/NAVIERSTOKES_3D.F90` at `-O3`. The run then stops at step 0
with "operation_flag invalid". This also happens with the original code.
Workaround: in `amrex-master/Tools/GNUMake/comps/gnu.mak`, change
`F90FLAGS += -g -O3` to `-g -O2`, then in `NS_fluids_lib` run
`touch NAVIERSTOKES_3D.F90` and `make`. Newer gcc (16) on a laptop is fine at `-O3`.

## Setup

Run everything from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r ode/requirements.txt
.venv/bin/python -m ode check
```

GPU setup is in [ode/README.md](ode/README.md#setup).
