# Multiphase solver

Sussman's AMReX multimaterial flow solver
([upstream](https://github.com/msussman42/amrex_implicit_interfaces)), kept as a
git subtree and modified for speed by Trent Feldman. The modifications reduce
run time by 54% in a 2D test and 74% in a 3D test. Each one except the
`plot_per` fix has a runtime switch that restores the upstream behavior.

## Requirements

- GNU make, a C++ compiler, gfortran and MPI. Tested with gcc 13 and gcc 16 and
  OpenMPI 4.1.6 and 5.
- On RCC: `module load gnu/13 openmpi/4.1.6`. 3D builds there need the
  [`-O2` workaround](#troubleshooting).

## Build

Run from `sim/NS_fluids_lib/`:

```bash
make -j6 DIM=2 USE_MPI=TRUE USE_OMP=FALSE    # amr2d.gnu.MPI.ex
make -j6 DIM=3 USE_MPI=TRUE USE_OMP=FALSE    # amr3d.gnu.MPI.ex
```

2D and 3D builds use separate object directories.

## Run

Run in a directory outside the repository. `SIM` is the absolute path of `sim/`:

```bash
SIM=/path/to/EigenFlow/sim
cp $SIM/NS_fluids_lib/run2d/inputs.rayleigh_taylor inputs
mpirun -np 4 $SIM/NS_fluids_lib/amr2d.gnu.MPI.ex inputs \
    max_step=10 amr.plot_int=10 amr.check_int=-1 ns.visual_nddata_format=1 > run.out
```

This writes field (`nddataPLT`) and interface (`MOF_PLT`) plotfiles and a
checkpoint (`chk`) at steps 0 and 10.
`key=value` arguments override the inputs file. Example inputs are in `run2d/`
and `run3d/`; not all of them start with the current code. See
[README_LOCAL](../docs/README_LOCAL.md) for input keys and outputs and
[README_HPC](../docs/README_HPC.md) for Slurm jobs.

## Performance switches

Set the switches in the inputs file or on the command line.

| Switch | Default | Other values |
|---|---|---|
| `ns.dt_regrow_after_clip` | `1`: grow the time step from its unclipped value after a frame | `0`: grow from the clipped value (upstream) |
| `ns.loop_check` | `reserved`: one batched cell-count check per step | `full`: check after every loop (upstream); `off`: no check |
| `ns.inner_totals` | `reserved`: compute in-step totals once per step | `full`: every call (upstream); `off`: only when a consumer exists |
| `ns.mf_check` | `reserved`: local array checks, one reduction per step | `full`: one reduction per check (upstream); `off`: no check |
| `ns.facefrac` | `reserved`: rebuild face fractions only when an input changed | `full`: always rebuild (upstream); `check`: verify every reuse |
| `ns.facefrac_audit` | `20`: verify every reuse on each 20th step | `0`: never |

Restore the upstream behavior, apart from the `plot_per` fix:

```text
ns.dt_regrow_after_clip=0 ns.loop_check=full ns.inner_totals=full ns.mf_check=full ns.facefrac=full
```

`ns.loop_check_selftest=N`, `ns.mf_check_selftest=N` and `ns.facefrac_selftest=1`
inject a fault to confirm that the corresponding check aborts the run.

## Architecture of the changes

**Time step after a frame.** `computeNewDt` clips the step to land on frame,
checkpoint and stop times. Upstream measures the 1% growth limit
(`ns.change_max` = 1.01) from the clipped step, so recovery can take hundreds of
steps. The modified code measures it from the unclipped step. The stability
limit still applies to every step.

**Frame trigger.** Upstream evaluates `(Long) cumtime/plot_per`, which casts
before dividing, so `amr.plot_per` frames are not written for t < 1. The
modified code writes a frame when a step crosses a multiple of `plot_per`.

**Loop check.** Upstream runs an `Allreduce` after almost every `MFIter` loop to
compare processed cells with the grid size, and barriers around each
`ParallelCopy`. `reserved` accumulates the counts, reduces once per step and
omits the barriers.

**In-step totals.** `volWgtSumALL(fast_mode=1)` integrates all diagnostics at
several points inside a step. The only value read there is the maximum
temperature of a rigid solid. `reserved` computes the totals at one call per
step, rotating through the call sites, and batches the reductions. With solids
or user-defined sums, every call computes them. The step-end output
(`fast_mode=0`) is unchanged.

**Array checks.** `debug_ngrow`, `debug_ixType` and `debug_boxArray` call
`FabArray::ok()`, which ends in an `Allreduce`. `reserved` keeps the test on each
rank and compares a count and hash across ranks once per step.

**Face fractions.** `ColorSum` rebuilds `FACEFRAC_MM`, `FACEFRAC_SOLVE_MM` and
`CELLFRAC_MM` on each call. `reserved` reuses them when the reconstruction,
mask, grids, distribution map, ghost widths, time and tessellation setting are
unchanged. The saved inputs are plain memory: AMReX's KNAPSACK distribution
counts FAB bytes per rank, so saved FABs would change box placement and
floating-point summation order.

| File | Contents |
|---|---|
| `NS_fluids_lib/local_thread_class.*` | Loop check, array check, all switches except the time step |
| `NS_fluids_lib/NavierStokes.cpp`, `.H` | Time step, in-step totals, array-check calls |
| `NS_fluids_lib/NavierStokes3.cpp` | Face fractions |
| `amrlib/AMReX_AmrCore.cpp` | Frame trigger |
| `amrlib/{AmrLevel,FillPatchUtil,StateData}.cpp`, `mglib/ABecLaplacian.cpp` | Loop-check call sites and barriers |
| `amrex-master/Tools/GNUMake/sites/Make.unknown` | Excludes `/usr/include` from the MPI include path |

## Validation and limitations

Measured on RCC with default switches, 32 ranks on two Xeon E5-2650 v2 nodes.
Times are the sum of step times; each is a single run.

| Case | Upstream | Time step and loop check | All switches |
|---|---:|---:|---:|
| 2D Rayleigh–Taylor, t = 0.144 to 0.16 | 5,787 s | 3,556 s (1.63×) | 2,648 s (2.19×) |
| 3D bubble rise, t = 0 to 0.15 | 8,491 s | 2,870 s (2.96×) | 2,230 s (3.81×) |

- The time-step speedup depends on the position of frame times relative to
  steps. In the 3D case it reduces the step count from 90 to 36. An earlier
  run with the first two changes reached 7.7× at t = 0.169.
- The runs compared used different nodes; a few percent may be node variation.
- In every comparison, `reserved` was bit-identical to the same run with the
  switch at `full`: 2D and 3D runs, a 2D restart and four 2D cases with solids.
  3D restarts and 3D solids are untested.
- `ns.dt_regrow_after_clip=1` changes the step sequence, so results differ from
  upstream. In the 2D and 3D tests they were closer to a run without frame
  clipping than the upstream results were. That run is a reference, not a
  converged solution.
- Code that reads `NS_sumdata` inside a step without a solid requires
  `ns.inner_totals=full`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| 3D run stops at step 0 with `operation_flag invalid` | gcc 13 miscompiles `NAVIERSTOKES_3D.F90` at `-O3`, also in upstream code. Change `F90FLAGS += -g -O3` to `-g -O2` in `amrex-master/Tools/GNUMake/comps/gnu.mak`, then run `touch NAVIERSTOKES_3D.F90` and rebuild |
| `ranks disagree` abort from `flush_mf_check` | Rerun with `ns.mf_check=full` to locate the failing check |
| `facefrac audit` abort | A reused face fraction differs from its rebuild. Rerun with `ns.facefrac=full` and report the case |
| Abort naming `ns.loop_check` | Replace `ns.check_loop_counts` and `ns.fillpatch_barriers` with `ns.loop_check` |
