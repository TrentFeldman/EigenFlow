# README_LOCAL: how to use this code (on the laptop)

These are personal notes; git ignores this file and it isn't part of Sussman's repo. For running on the FSU supercomputer, see `README_HPC.md`.

> **The golden rule: this is Sussman's repo, and we never edit it.** We only build it with flags on the `make` line, feed it inputs files, and override settings on the command line. Keep runs and job files **outside** the repo (e.g. `~/runs/`, `~/hpc_jobs/`).

Unless marked **untested**, everything here was checked on this laptop on 2026-11-13: Ryzen 5 5600 (6 cores), OpenMPI 5, GCC 16.

---

## The 30-second picture

- **What the code is.** One program that simulates fluids with sharp interfaces: water/air, drops, bubbles, boiling, flow past solids. It's built on **AMReX** (grids + adaptive mesh refinement) and uses **MPI** to split the work across cores.
- **How a run works.** You run the executable with an **inputs file** (plain text, `key = value` lines). `ns.probtype` in that file picks which built-in problem runs. The run writes snapshots and save points into the folder you started it from.
- **Changing a setting.** Add `key=value` after the inputs file name on the command line; it overrides the file. So you never edit the stock inputs files.
- **Where the examples are.** Ready-made inputs files live in `NS_fluids_lib/run2d/` and `run3d/` (hundreds of them, not all still working; see Gotchas).

---

## Quick start

```bash
# 1. Build (once). DIM=2 -> amr2d.gnu.MPI.ex, DIM=3 -> amr3d.gnu.MPI.ex
cd ~/amrex_implicit_interfaces/NS_fluids_lib
make -j6 DIM=2 USE_MPI=TRUE USE_OMP=FALSE

# 2. Make a run folder OUTSIDE the repo and copy an inputs file in
mkdir -p ~/runs/cone2d_test && cd ~/runs/cone2d_test
cp ~/amrex_implicit_interfaces/NS_fluids_lib/run2d/inputs.cone2d inputs

# 3. Run on 6 cores in the background
nohup mpirun -np 6 ~/amrex_implicit_interfaces/NS_fluids_lib/amr2d.gnu.MPI.ex inputs \
    ns.enable_spectral=0 amr.time_blocking_factor=1 amr.space_blocking_factor="1 1 1 1" \
    > run.out 2> run.err &

# 4. Watch / stop
grep "^STEP = " run.out | tail -n 1   # latest step and sim time
pkill -f prterun                      # stops it (this kills ALL your mpirun jobs)
```

- **The one allowed local edit.** `amrex-master/Tools/GNUMake/sites/Make.unknown`, line 46, wraps `mpicxx -showme:incdirs` in `filter-out /usr/include`. On Arch Linux, OpenMPI reports `/usr/include`, and GCC then fails with "stdlib.h: No such file or directory". Keep this patch and never commit it. RCC doesn't need it.
- **2D and 3D builds can live side by side.** Each one has its own build folder, so switching `DIM` doesn't rebuild everything.
- **Command-line overrides are verified to work.** For settings with several numbers, like `amr.n_cell=64 256`, see Gotchas.

---

## Making it fast: the settings that matter

Some stock inputs (e.g. the cone) turn on **"spectral" high-order mode**. It's ~10× slower per step, and the code itself warns it isn't implemented for walls (`run.err`: *"no slip BC not implemented for space order>2"*). The code's default is off, so files that don't mention it (like `inputs.rayleigh_taylor`) are already fast.

| Setting | Cone stock | Fast | Notes |
|---|---|---|---|
| `ns.enable_spectral` | 1 | 0 | **Change all three together**, or it aborts |
| `amr.time_blocking_factor` | 2 (3D) / 4 (2D) | 1 | |
| `amr.space_blocking_factor` | 2 2 2 2 / 4 4 4 4 | 1 1 1 1 | One number per level |
| `amr.plot_int`, `amr.check_int` | 4, 4 | 50–100+ | Affects disk, not speed: ~190 MB per 3D snapshot |
| `ns.visual_nddata_format` | 0 | 1 | Binary output: faster and smaller |
| `ns.v` | 1 | 0 | Log size (~200 KB/step at 1) |

**Before making real data, ask Sussman whether spectral mode is wanted,** and use the same choice across a whole training set.

### How long things take (6 cores)

| Case | Cells | s/step | Time step | Steps | Wall time |
|---|---|---|---|---|---|
| cone 3D, spectral (stock) | 125k | ~335 | 0.032 | ~640 | ~60 h |
| cone 3D, fast | 125k | ~34 | 0.05 | ~425 | ~4 h |
| cone 2D, fast | 5.9k | ~1.25 | 0.021 | ~380 | ~10 min |
| boiling 2D-RZ (`inputs.Earth_Grav_Water_Dhir_2001_Vahab`) | 9.6k | ~2.2 | 3.7e-4 | ~16,000 | ~10–11 h |

- **Rule of thumb:** `core-hours ≈ cells × steps × 1.5 ms ÷ 3600`.
- **Interface problems are slow because of the step count, not the cost per step.** Surface tension, heat transfer and boiling force tiny time steps.
- **The boiling row is ±2–3×.** It's extrapolated from 30 steps.

### Why it's slow even in fast mode (profiled; FYI only, we don't change the code)

- **Cores wait on each other.** After almost every loop, all cores synchronize (`MPI_Allreduce`). With uneven grid pieces, some cores sit idle 20–55% of the time. That's why piling more cores onto one small run doesn't help much.
- **Per-cell overhead.** Coordinates get recomputed for every cell on every operation.
- **Repeated work.** The solid geometry is rebuilt ~29× per step, and diagnostics run more often than asked.

---

## Inputs cheat sheet

| Key | Meaning |
|---|---|
| `max_step`, `stop_time` | Stop at whichever comes first |
| `amr.n_cell` | Base grid size (cells in x y [z]) |
| `amr.max_level` | How many levels of refinement on top (0 = uniform grid; each level halves the cell size) |
| `amr.max_grid_size` | Biggest grid piece. Smaller = more pieces to share among cores. Must be ≥ 16 (2D) / 8 (3D) |
| `amr.plot_int` / `amr.plot_per` | Snapshot every N steps / every T of sim time (set the other to -1) |
| `amr.check_int` | Save point every N steps. A final one is always written |
| `amr.restart=chkNNNNN` | Continue from a save point |
| `ns.visual_nddata_format` | 0 = ASCII Tecplot (slow), **1 = AMReX plotfiles (use this)**, 2 = Tecplot supercells |
| `ns.probtype` | Which problem runs (222 = cone/cylinder/sphere, 602 = Rayleigh–Taylor, 55 = boiling) |
| `ns.num_materials` | Number of materials (water, air, solid, …) |
| `ns.denconst`, `ns.viscconst`, `ns.tension` | Density, viscosity, surface tension per material |
| `ns.gravity` | Gravity (sign = direction) |
| `geometry.coord_sys` | 0 = normal x-y(-z), 1 = axisymmetric (r-z, 2D build) |
| `ns.lo_bc`, `ns.hi_bc` | Boundaries per direction: 0 periodic/interior, 1 inflow, 2 outflow, 3 symmetry, 4 slip wall, 5 no-slip wall |

To find any setting's default, grep its name in `NS_fluids_lib/NavierStokes.cpp` or `amrlib/AMReX_AmrCore.cpp` (look for `queryAdd("name"`).

### Problem: cone / cylinder / sphere (probtype 222, `cone3D.F90`)

| `ns.axis_dir` | Shape | Parameters | Inflow |
|---|---|---|---|
| 0 | cone | apex `xblob yblob zblob`, axis `xblob2 yblob2 zblob2`, half-angle `radblob` (radians) | `ns.adv_vel` in +x |
| 1 | cylinder (3D only) | center `xblob3..zblob3`, axis `xblob4..zblob4`, radius `radblob3` | `xblob10 yblob10 zblob10` |
| 2 | sphere (circle in 2D) | center `xblob3..zblob3`, radius `radblob3` | `xblob10 yblob10 zblob10` |

Optional inflow wobble: `xblob9` (duration), `yblob9`, `radblob9`, `yblob8`, `radblob8` (see the comment at the top of `cone3D.F90`).

### Problem: Rayleigh–Taylor (probtype 602, `run2d/inputs.rayleigh_taylor`)

Water (material 1) on top of air (material 2), in cgs units. The interface starts at `y = yblob + radblob·cos(2π·xblob·x / width)`.

| Key | Stock | Meaning |
|---|---|---|
| `ns.yblob` | 16 | Starting interface height (cm) in a 32 cm tall box |
| `ns.radblob` | 0.008 | Ripple amplitude (cm). Bigger = takes off sooner |
| `ns.xblob` | 1 | Number of ripple wavelengths across the 8 cm width |
| `ns.denconst` | 1.0 0.001225 | Densities: water, air (g/cm³) |
| `ns.tension` | 72.8 | Surface tension (dyn/cm) |

The high-res HPC version of this run is in `README_HPC.md`, Step 2.

---

## Outputs: what lands in the run folder

| File | What it is | Open with |
|---|---|---|
| `run.out` / `run.err` | Log / warnings and errors | `grep "^STEP = " run.out` |
| `nddataPLTNNNNNNNN/` (format 1) | Snapshot of all fields | Python `yt` (`pip install yt`; `yt.load(dir)`), ParaView, VisIt |
| `MOF_PLTNNNNNNNN/` (format 1) | Where each material is (volume fractions) | same |
| `nddataNNNNNNNN.plt` (format 0) | Same snapshot as ASCII Tecplot, ~59 MB in 3D | VisIt / Tecplot |
| `chkNNNNN/` | Save point | `amr.restart=` |
| `matNNlsNNNNNNNN.tec` | Interface surface for material NN | VisIt |
| `uniformNNNNNNNN.tec` | Fields on a plain uniform grid | quick looks |
| `grdlog` | Which core owns which grid piece each step (`:: N` = core) | checking balance |

**Fields in a snapshot.** The full list is on line 3 onward of `nddataPLT*/Header`. Useful ones:
- `x_velocity`, `y_velocity`, `PRES_MG` (pressure)
- `F01 F02` (volume fraction of each material)
- `L0101` (distance to the interface)
- `D01` (density)
- `VORT01` (vorticity)

---

## Several runs in a row (local sweep)

Put one line of overrides per run in `params.txt`, then:

```bash
EXE=~/amrex_implicit_interfaces/NS_fluids_lib/amr2d.gnu.MPI.ex
FAST='ns.enable_spectral=0 amr.time_blocking_factor=1 amr.space_blocking_factor=1 1 1 1 ns.visual_nddata_format=1'
i=0; while read -r p; do
  d=~/runs/sweep/run_$i; mkdir -p $d; cp inputs $d/
  ( cd $d && mpirun -np 6 $EXE inputs $FAST $p > run.out 2> run.err )
  i=$((i+1)); done < params.txt
```

For hundreds of runs, use RCC (`README_HPC.md`, Appendix).

---

## Gotchas

- **Some stock inputs files no longer run.** For example, `inputs.boiling_crossflow` aborts. Prefer files Sussman touched recently: `git log -3 -- NS_fluids_lib/run2d/<file>`.
- **Errors are terse.** They show up as `amrex::Error::<rank>::<message>` in `run.err`, or as `... invalid` followed by `stop` in `run.out`. Grep the message in the source to see what it checks.
- **Settings with several numbers.** They need one value per level or direction (`amr.space_blocking_factor=1 1 1 1`, `amr.n_cell=64 256`). In scripts, keep them in a variable and expand it **unquoted** (`$FAST`, not `"$FAST"`) so the numbers split into separate words. That's verified. A single `=1` where 4 are needed aborts with `queryarr too many values requested`.
- **Leftover save points.** A run folder that already has `chk…/` folders will restart from them if your script passes `amr.restart`. Delete test runs before real ones.
- **Disk fills fast.** Raise `plot_int`/`check_int`, and use `ns.visual_nddata_format=1`.
- **More cores isn't always faster.** If `grdlog` shows fewer grid pieces than cores, some cores do nothing; lower `amr.max_grid_size`.

## Where things live in the repo (read-only reference)

| File | What |
|---|---|
| `NS_fluids_lib/run2d/`, `run3d/` | Example inputs files |
| `NS_fluids_lib/PROB_CPP_PARMS.F90` | Maps `probtype` → problem code |
| `NS_fluids_lib/cone3D.F90` | Cone/cylinder/sphere setup |
| `NS_fluids_lib/PROB.F90` | Many older problems' setups, incl. Rayleigh–Taylor (search `probtype.eq.602`) |
| `NS_fluids_lib/NavierStokes.cpp` | Where `ns.*` settings are read (and their defaults) |
| `NS_fluids_lib/NavierStokes3.cpp` | One time step: move interface → viscosity → pressure |
| `amrlib/AMReX_AmrCore.cpp` | `amr.*` settings (restart, plot/check intervals) |
